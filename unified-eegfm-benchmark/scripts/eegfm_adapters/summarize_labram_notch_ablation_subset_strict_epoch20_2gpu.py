#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize strict epoch-20 LaBraM notch ablation subset branch metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "labram_notch_ablation_subset_v1_strict_epoch20_2gpu"
DEFAULT_REPORT_ROOT = PROJECT_ROOT / "reports" / "labram_notch_ablation_subset_v1_strict_epoch20_2gpu"
BRANCHES = ["original_pkl_50hz", "unified_50hz", "unified_60hz"]
METRICS = ["test_balanced_accuracy", "test_accuracy", "test_auroc", "test_auprc", "test_f1", "test_sensitivity", "test_specificity"]
FIELDS = [
    "branch",
    "status",
    "branch_status",
    "recipe_classification",
    "epochs_completed",
    "best_epoch",
    "best_val_balanced_accuracy",
    *METRICS,
    "test_confusion_matrix",
    "metrics_path",
    "notes",
]
COMPARISON_FIELDS = [
    "comparison",
    "status",
    "reason",
    *[f"delta_{metric}_right_minus_left" for metric in METRICS],
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}


def delta_row(rows: dict[str, dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    out = {"comparison": f"{left} vs {right}", "status": "PENDING", "reason": ""}
    if left not in rows or right not in rows:
        out["reason"] = "missing comparison branch row"
        return out
    left_status = rows[left].get("status")
    right_status = rows[right].get("status")
    if left_status != "PASS" or right_status != "PASS":
        out["reason"] = f"{left} status={left_status}; {right} status={right_status}"
        return out
    out["status"] = "PASS"
    for metric in METRICS:
        try:
            out[f"delta_{metric}_right_minus_left"] = float(rows[right].get(metric)) - float(rows[left].get(metric))
        except Exception:
            out[f"delta_{metric}_right_minus_left"] = ""
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--report_root", default=str(DEFAULT_REPORT_ROOT))
    args = parser.parse_args()
    output_root = Path(args.output_root)
    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)

    rows = []
    by_branch = {}
    for branch in BRANCHES:
        metrics_path = output_root / branch / "metrics.json"
        metrics = read_json(metrics_path)
        status_path = output_root / branch / "branch_status.json"
        branch_status = read_json(status_path)
        status = metrics.get("status") or branch_status.get("status", "MISSING")
        notes = metrics.get("notes") or metrics.get("error") or branch_status.get("reason") or branch_status.get("error", "")
        row = {
            "branch": branch,
            "status": status,
            "branch_status": branch_status.get("status", ""),
            "recipe_classification": metrics.get("recipe_classification", ""),
            "epochs_completed": metrics.get("epochs_completed", ""),
            "best_epoch": metrics.get("best_epoch", ""),
            "best_val_balanced_accuracy": metrics.get("best_val_balanced_accuracy", ""),
            "test_confusion_matrix": metrics.get("test_confusion_matrix", ""),
            "metrics_path": str(metrics_path),
            "notes": notes,
        }
        for metric in METRICS:
            row[metric] = metrics.get(metric, "")
        rows.append(row)
        by_branch[branch] = row

    with (report_root / "summary_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    comparisons = [
        delta_row(by_branch, "original_pkl_50hz", "unified_50hz"),
        delta_row(by_branch, "original_pkl_50hz", "unified_60hz"),
        delta_row(by_branch, "unified_50hz", "unified_60hz"),
    ]
    with (report_root / "comparison_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        writer.writerows(comparisons)

    payload = {
        "branches": rows,
        "comparisons": comparisons,
        "partial_summary": {
            "ablation_1_original_vs_unified_50hz": next(row for row in comparisons if row["comparison"] == "original_pkl_50hz vs unified_50hz")["status"],
            "ablation_2_original_vs_unified_60hz": next(row for row in comparisons if row["comparison"] == "original_pkl_50hz vs unified_60hz")["status"],
            "ablation_3_unified_50hz_vs_unified_60hz": next(row for row in comparisons if row["comparison"] == "unified_50hz vs unified_60hz")["status"],
        },
    }
    (report_root / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# LaBraM Strict Epoch-20 Notch Ablation Subset Summary",
        "",
        "| Branch | Status | Branch Status | Recipe | Best Val Bal Acc | Test Bal Acc | AUROC | AUPRC | Notes |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['branch']} | {row['status']} | {row['branch_status']} | {row['recipe_classification']} | "
            f"{row['best_val_balanced_accuracy']} | {row['test_balanced_accuracy']} | {row['test_auroc']} | {row['test_auprc']} | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "## Comparisons",
            "",
            "| Comparison | Status | Reason |",
            "|---|---|---|",
        ]
    )
    for comp in comparisons:
        lines.append(f"| {comp['comparison']} | {comp['status']} | {comp['reason']} |")
    lines.extend(
        [
            "",
            "A comparison is `PENDING` when one branch is missing, skipped, still running, or build-failed. This is expected for partial-safe runs where `unified_50hz` is deferred.",
        ]
    )
    (report_root / "summary_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"summary_table={report_root / 'summary_table.csv'}")
    print(f"comparison_table={report_root / 'comparison_table.csv'}")
    print(f"summary_json={report_root / 'summary.json'}")
    print(f"summary_report={report_root / 'summary_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
