#!/usr/bin/env python3
"""Run the final CSBrain seed-42 ablation sequentially on one GPU.

This launcher reuses the validated smoke-ablation training implementation but
writes to a separate final output root and final-summary filenames.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_csbrain_smoke_ablation_seed42 import (  # noqa: E402
    DEFAULT_CKPT,
    DEFAULT_H5,
    DEFAULT_INDEX_NPZ,
    DEFAULT_ORIGINAL,
    DEFAULT_REPORT_DIR,
    ROWS,
    ROOT,
    train_one,
    read_index_npz,
    summary_row,
)


FINAL_ROW_ORDER = [
    "S2_original_scale_10000",
    "S6_unified_bipolar_mul10000",
    "S1_original_author_scale_1000",
    "S5_unified_bipolar_mul1000",
    "S3_original_scale_div100",
    "S4_unified_bipolar_div100",
    "S7_unified_endpoint_current",
]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "csbrain_unified60_final_ablation_seed42_1gpu"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_final_summary(report_dir: Path, output_root: Path, metrics_list: list[dict[str, Any]]) -> None:
    rows = [summary_row(metrics) for metrics in metrics_list]
    csv_path = report_dir / "csbrain_final_ablation_seed42_summary.csv"
    json_path = report_dir / "csbrain_final_ablation_seed42_summary.json"
    md_path = report_dir / "csbrain_final_ablation_seed42_summary.md"
    logs_path = report_dir / "csbrain_final_ablation_training_logs_index.md"
    write_csv(csv_path, rows)
    json_path.write_text(
        json.dumps(
            {
                "output_root": str(output_root),
                "row_order": FINAL_ROW_ORDER,
                "rows": rows,
                "metrics": metrics_list,
            },
            indent=2,
            allow_nan=True,
        )
        + "\n",
        encoding="utf-8",
    )
    lines = [
        "# CSBrain Final Ablation Seed42 Summary",
        "",
        "Sequential 1-GPU final ablation using the author-style CSBrain recipe from the smoke runner.",
        "",
        "| Row | Source | Montage | Scale | Ckpt | Keys | Epochs | Best Epoch | Best Val B-Acc | Test B-Acc | AUROC | AUPRC | Acc | Pos Rate | Collapse |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('row')} | {row.get('source')} | {row.get('montage')} | {row.get('scale')} | "
            f"{row.get('checkpoint_loaded')} | {row.get('loaded_key_count')} | {row.get('epochs_completed')} | "
            f"{row.get('best_epoch')} | {row.get('best_validation_balanced_accuracy')} | "
            f"{row.get('test_balanced_accuracy')} | {row.get('test_auroc')} | {row.get('test_auprc')} | "
            f"{row.get('test_accuracy')} | {row.get('prediction_positive_rate')} | {row.get('one_class_collapse')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- S3/S4 `/100` is the strongest smoke-identified fair pair but is diagnostic, not author-equivalent.",
            "- S1/S5 `*1000` is a scale-sanity control pair.",
            "- S2/S6 `*10000` is the strict author-scale control pair.",
            "- S7 endpoint-current is historical control only and should not be treated as fair unless later repo evidence supports endpoint-current semantics.",
            "",
            f"- CSV: `{csv_path}`",
            f"- JSON: `{json_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    log_lines = ["# CSBrain Final Ablation Training Logs Index", ""]
    for row_name in FINAL_ROW_ORDER:
        row_dir = output_root / row_name
        log_lines.append(f"- `{row_name}`: log `{row_dir / 'train.log'}`, metrics `{row_dir / 'metrics.json'}`")
    logs_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CSBrain final ablation sequentially on one GPU.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=int(os.environ.get("NUM_WORKERS", "8")))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--report_dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--original_processed_root", default=str(DEFAULT_ORIGINAL))
    parser.add_argument("--index_npz", default=str(DEFAULT_INDEX_NPZ))
    parser.add_argument("--h5", default=str(DEFAULT_H5))
    parser.add_argument("--csbrain_repo", default="/nicoletye/workspace/Noise Robustness Evaluation Protocol/EEG-FM/CSBrain")
    parser.add_argument("--checkpoint_path", default=str(DEFAULT_CKPT))
    parser.add_argument("--rows", nargs="*", default=FINAL_ROW_ORDER)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> int:
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    args = parse_args()
    output_root = Path(args.output_root)
    report_dir = Path(args.report_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    row_names = args.rows or FINAL_ROW_ORDER
    specs = [ROWS[name] for name in row_names]
    run_config = {
        "mode": "final_ablation_seed42_1gpu",
        "sequential": True,
        "max_parallel": 1,
        "args": vars(args),
        "row_order": row_names,
        "rows": [spec.__dict__ for spec in specs],
    }
    (report_dir / "csbrain_final_ablation_seed42_run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "START", "rows": row_names, "output_root": str(output_root)}, indent=2), flush=True)
    if args.dry_run:
        write_final_summary(report_dir, output_root, [])
        print(json.dumps({"status": "DRY_RUN", "config": str(report_dir / "csbrain_final_ablation_seed42_run_config.json")}, indent=2))
        return 0

    index = read_index_npz(Path(args.index_npz))
    metrics_list: list[dict[str, Any]] = []
    for spec in specs:
        metrics = train_one(spec, args, index)
        metrics_list.append(metrics)
        write_final_summary(report_dir, output_root, metrics_list)
    write_final_summary(report_dir, output_root, metrics_list)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output_root": str(output_root),
                "summary": str(report_dir / "csbrain_final_ablation_seed42_summary.md"),
                "rows_completed": [m.get("row") for m in metrics_list],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
