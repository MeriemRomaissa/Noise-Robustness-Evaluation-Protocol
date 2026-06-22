#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize LaBraM unified60 3-strategy 4-seed epoch-50 runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any


STRATEGIES = ["full_finetune", "linear_probe", "lora_meriem_exact"]
SEEDS = [42, 123, 256, 512]
FIELDS = [
    "strategy",
    "seed",
    "status",
    "epochs_completed",
    "best_epoch",
    "best_val_balanced_accuracy",
    "test_loss",
    "test_accuracy",
    "test_balanced_accuracy",
    "test_auroc",
    "test_auprc",
    "confusion_matrix",
    "recipe_classification",
    "base_recipe_classification",
    "seed_variant",
    "reference_seed",
    "run_seed",
    "base_recipe_strict_components_pass",
    "strategy_param_audit_status",
    "lora_comparability",
    "checkpoint_loaded",
    "exact_split_used",
    "layer_decay_active",
    "lr_schedule_active",
    "warmup_active",
    "output_root",
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}


def finite_float(value: Any) -> float | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


def row_for(root: Path, strategy: str, seed: int) -> dict[str, Any]:
    out = root / strategy / f"seed_{seed}"
    metrics = read_json(out / "metrics.json")
    return {
        "strategy": strategy,
        "seed": seed,
        "status": metrics.get("status", "MISSING"),
        "epochs_completed": metrics.get("epochs_completed", ""),
        "best_epoch": metrics.get("best_epoch", ""),
        "best_val_balanced_accuracy": metrics.get("best_val_balanced_accuracy", ""),
        "test_loss": metrics.get("test_loss", ""),
        "test_accuracy": metrics.get("test_accuracy", ""),
        "test_balanced_accuracy": metrics.get("test_balanced_accuracy", ""),
        "test_auroc": metrics.get("test_auroc", ""),
        "test_auprc": metrics.get("test_auprc", ""),
        "confusion_matrix": json.dumps(metrics.get("test_confusion_matrix", "")),
        "checkpoint_loaded": metrics.get("checkpoint_loaded", ""),
        "exact_split_used": metrics.get("exact_split_used", ""),
        "recipe_classification": metrics.get("recipe_classification", ""),
        "base_recipe_classification": metrics.get("base_recipe_classification", ""),
        "seed_variant": metrics.get("seed_variant", ""),
        "reference_seed": metrics.get("reference_seed", ""),
        "run_seed": metrics.get("run_seed", ""),
        "base_recipe_strict_components_pass": metrics.get("base_recipe_strict_components_pass", ""),
        "strategy_param_audit_status": metrics.get("strategy_param_audit_status", ""),
        "lora_comparability": metrics.get("lora_comparability", ""),
        "layer_decay_active": metrics.get("layer_decay_active", ""),
        "lr_schedule_active": metrics.get("lr_schedule_active", ""),
        "warmup_active": metrics.get("warmup_active", ""),
        "output_root": str(out),
    }


def grouped(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = []
    for strategy in STRATEGIES:
        items = [row for row in rows if row["strategy"] == strategy and row["status"] == "PASS"]
        bal = [(row["seed"], finite_float(row["test_balanced_accuracy"])) for row in items]
        auroc = [finite_float(row["test_auroc"]) for row in items]
        auprc = [finite_float(row["test_auprc"]) for row in items]
        bal_vals = [(seed, val) for seed, val in bal if val is not None]
        auroc_vals = [val for val in auroc if val is not None]
        auprc_vals = [val for val in auprc if val is not None]
        best = max(bal_vals, key=lambda x: x[1]) if bal_vals else ("", "")
        worst = min(bal_vals, key=lambda x: x[1]) if bal_vals else ("", "")
        groups.append(
            {
                "strategy": strategy,
                "pass_count": len(items),
                "mean_test_balanced_accuracy": mean([v for _, v in bal_vals]) if bal_vals else "",
                "std_test_balanced_accuracy": stdev([v for _, v in bal_vals]) if len(bal_vals) > 1 else 0.0 if bal_vals else "",
                "mean_auroc": mean(auroc_vals) if auroc_vals else "",
                "std_auroc": stdev(auroc_vals) if len(auroc_vals) > 1 else 0.0 if auroc_vals else "",
                "mean_auprc": mean(auprc_vals) if auprc_vals else "",
                "std_auprc": stdev(auprc_vals) if len(auprc_vals) > 1 else 0.0 if auprc_vals else "",
                "best_seed_by_test_balanced_accuracy": best[0],
                "best_test_balanced_accuracy": best[1],
                "worst_seed_by_test_balanced_accuracy": worst[0],
                "worst_test_balanced_accuracy": worst[1],
            }
        )
    return groups


def write_reports(rows: list[dict[str, Any]], group_rows: list[dict[str, Any]], report_root: Path) -> None:
    report_root.mkdir(parents=True, exist_ok=True)
    extra_fields = sorted({key for row in rows for key in row if key not in FIELDS})
    fieldnames = FIELDS + extra_fields
    with (report_root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    payload = {"jobs": rows, "grouped_by_strategy": group_rows}
    (report_root / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# LaBraM Unified60 3-Strategy 4-Seed Epoch-50 Summary",
        "",
        "## Jobs",
        "",
        "| Strategy | Seed | Status | Epochs | Best Val Bal | Test Bal | AUROC | AUPRC | Output |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['strategy']} | {row['seed']} | {row['status']} | {row['epochs_completed']} | "
            f"{row['best_val_balanced_accuracy']} | {row['test_balanced_accuracy']} | {row['test_auroc']} | "
            f"{row['test_auprc']} | `{row['output_root']}` |"
        )
    lines.extend(["", "## Grouped Statistics", "", "| Strategy | PASS | Mean Test Bal | Std Test Bal | Mean AUROC | Mean AUPRC | Best Seed | Worst Seed |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in group_rows:
        lines.append(
            f"| {row['strategy']} | {row['pass_count']} | {row['mean_test_balanced_accuracy']} | "
            f"{row['std_test_balanced_accuracy']} | {row['mean_auroc']} | {row['mean_auprc']} | "
            f"{row['best_seed_by_test_balanced_accuracy']} | {row['worst_seed_by_test_balanced_accuracy']} |"
        )
    (report_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default="outputs/labram_full_unified60_3strategy_4seed_epoch50_v1")
    parser.add_argument("--report_root", default="reports/labram_full_unified60_3strategy_4seed_epoch50_v1")
    args = parser.parse_args()
    root = Path(args.output_root)
    rows = [row_for(root, strategy, seed) for strategy in STRATEGIES for seed in SEEDS]
    group_rows = grouped(rows)
    write_reports(rows, group_rows, Path(args.report_root))
    print(f"summary={Path(args.report_root) / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
