#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize the all-6 unified60 subset epoch-15 dev benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


MODELS = ["labram", "biot", "eegpt", "cbramod", "csbrain", "codebrain"]
STRATEGIES = ["full_finetune", "linear_probe", "lora"]
SEEDS = [0, 42, 123, 256, 512]
FIELDS = [
    "model",
    "strategy",
    "seed",
    "status",
    "epochs_completed",
    "best_epoch",
    "best_val_balanced_accuracy",
    "test_accuracy",
    "test_balanced_accuracy",
    "test_auroc",
    "test_auprc",
    "checkpoint_loaded",
    "checkpoint_load_status",
    "strategy_classification",
    "lora_placement",
    "metrics_path",
    "log_path",
    "output_path",
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "READ_ERROR", "error": f"{type(exc).__name__}: {exc}"}


def finite_float(value: Any) -> float | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


def row_for(root: Path, model: str, strategy: str, seed: int) -> dict[str, Any]:
    out = root / model / strategy / f"seed_{seed}"
    metrics_path = out / "metrics.json"
    data = read_json(metrics_path)
    checkpoint = data.get("checkpoint_load_report", {}) or {}
    status = data.get("status", "INCOMPLETE" if not metrics_path.exists() else "UNKNOWN")
    lora_placement = data.get("lora_placement", "")
    if model == "codebrain" and strategy == "lora" and not lora_placement:
        lora_placement = "MERIEM_STRICT_ARCHITECTURE_NATIVE_EQUIVALENT_PARTIAL"
    return {
        "model": data.get("model", model),
        "strategy": data.get("finetune_strategy", data.get("strategy", strategy)),
        "seed": seed,
        "status": status,
        "epochs_completed": data.get("epochs_completed", ""),
        "best_epoch": data.get("best_epoch", ""),
        "best_val_balanced_accuracy": data.get("best_val_balanced_accuracy", ""),
        "test_accuracy": data.get("test_accuracy", ""),
        "test_balanced_accuracy": data.get("test_balanced_accuracy", ""),
        "test_auroc": data.get("test_auroc", data.get("auroc", "")),
        "test_auprc": data.get("test_auprc", data.get("auprc", "")),
        "checkpoint_loaded": data.get("checkpoint_loaded", checkpoint.get("checkpoint_loaded", "")),
        "checkpoint_load_status": data.get("checkpoint_load_status", checkpoint.get("checkpoint_status", "")),
        "strategy_classification": data.get("strategy_classification", data.get("strict_status", data.get("classification", ""))),
        "lora_placement": lora_placement,
        "metrics_path": str(metrics_path),
        "log_path": data.get("log_path", str(out / "train.log")),
        "output_path": str(out),
    }


def stats(values: list[Any]) -> dict[str, Any]:
    vals = [finite_float(v) for v in values]
    vals = [v for v in vals if v is not None]
    return {
        "mean": mean(vals) if vals else "",
        "std": stdev(vals) if len(vals) > 1 else 0.0 if vals else "",
        "n": len(vals),
    }


def grouped(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_model = []
    by_model_strategy = []
    for model in MODELS:
        items = [r for r in rows if str(r["model"]).lower() == model and r["status"] == "PASS"]
        by_model.append(
            {
                "model": model,
                "pass_count": len(items),
                "job_count": len([r for r in rows if str(r["model"]).lower() == model]),
                "test_accuracy": stats([r["test_accuracy"] for r in items]),
                "test_balanced_accuracy": stats([r["test_balanced_accuracy"] for r in items]),
                "test_auroc": stats([r["test_auroc"] for r in items]),
                "test_auprc": stats([r["test_auprc"] for r in items]),
                "best_val_balanced_accuracy": stats([r["best_val_balanced_accuracy"] for r in items]),
            }
        )
        for strategy in STRATEGIES:
            subset = [
                r
                for r in rows
                if str(r["model"]).lower() == model and r["strategy"] == strategy and r["status"] == "PASS"
            ]
            by_model_strategy.append(
                {
                    "model": model,
                    "strategy": strategy,
                    "pass_count": len(subset),
                    "job_count": 5,
                    "mean_test_accuracy": stats([r["test_accuracy"] for r in subset])["mean"],
                    "std_test_accuracy": stats([r["test_accuracy"] for r in subset])["std"],
                    "mean_test_balanced_accuracy": stats([r["test_balanced_accuracy"] for r in subset])["mean"],
                    "std_test_balanced_accuracy": stats([r["test_balanced_accuracy"] for r in subset])["std"],
                    "mean_test_auroc": stats([r["test_auroc"] for r in subset])["mean"],
                    "std_test_auroc": stats([r["test_auroc"] for r in subset])["std"],
                    "mean_test_auprc": stats([r["test_auprc"] for r in subset])["mean"],
                    "std_test_auprc": stats([r["test_auprc"] for r in subset])["std"],
                    "mean_best_val_balanced_accuracy": stats([r["best_val_balanced_accuracy"] for r in subset])["mean"],
                    "std_best_val_balanced_accuracy": stats([r["best_val_balanced_accuracy"] for r in subset])["std"],
                }
            )
    return by_model, by_model_strategy


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default="outputs/all6_unified60_subset_epoch15_5seed_1gpu_dev_v1")
    parser.add_argument("--report_root", default="reports/all6_unified60_subset_epoch15_5seed_1gpu_dev_v1")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    report_root = Path(args.report_root)
    aggregate = output_root / "aggregate"
    rows = [row_for(output_root, model, strategy, seed) for model in MODELS for strategy in STRATEGIES for seed in SEEDS]
    by_model, by_model_strategy = grouped(rows)
    status_counts = Counter(row["status"] for row in rows)

    for root in [report_root, aggregate]:
        root.mkdir(parents=True, exist_ok=True)
        write_csv(root / "summary.csv", rows, FIELDS)
        write_csv(root / "summary_by_model_strategy.csv", by_model_strategy)
        payload = {
            "job_count": len(rows),
            "status_counts": dict(status_counts),
            "jobs": rows,
            "by_model": by_model,
            "by_model_strategy": by_model_strategy,
            "codebrain_lora_caveat": "CodeBrain LoRA is MERIEM_STRICT_ARCHITECTURE_NATIVE_EQUIVALENT_PARTIAL, not exact LaBraM q/k/v + MLP placement.",
        }
        (root / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        lines = [
            "# All-6 Unified60 Subset Epoch-15 Dev Benchmark Summary",
            "",
            f"- Total jobs: {len(rows)}",
            f"- PASS: {status_counts.get('PASS', 0)}",
            f"- FAIL: {status_counts.get('FAIL', 0)}",
            f"- INCOMPLETE: {status_counts.get('INCOMPLETE', 0)}",
            "",
            "## Per-Job Metrics",
            "",
            "| Model | Strategy | Seed | Status | Epochs | Best Epoch | Best Val Bal | Test Acc | Test Bal | AUROC | AUPRC |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                f"| {row['model']} | {row['strategy']} | {row['seed']} | {row['status']} | "
                f"{row['epochs_completed']} | {row['best_epoch']} | {row['best_val_balanced_accuracy']} | "
                f"{row['test_accuracy']} | {row['test_balanced_accuracy']} | {row['test_auroc']} | {row['test_auprc']} |"
            )
        lines.extend(
            [
                "",
                "## Per-Model/Strategy PASS Means",
                "",
                "| Model | Strategy | PASS | Mean Test Acc | Mean Test Bal | Mean AUROC | Mean AUPRC | Mean Best Val Bal |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in by_model_strategy:
            lines.append(
                f"| {row['model']} | {row['strategy']} | {row['pass_count']}/{row['job_count']} | "
                f"{row['mean_test_accuracy']} | {row['mean_test_balanced_accuracy']} | "
                f"{row['mean_test_auroc']} | {row['mean_test_auprc']} | {row['mean_best_val_balanced_accuracy']} |"
            )
        lines.extend(
            [
                "",
                "## Caveat",
                "",
                "CodeBrain LoRA is `MERIEM_STRICT_ARCHITECTURE_NATIVE_EQUIVALENT_PARTIAL`; do not report it as exact LaBraM q/k/v + MLP placement.",
            ]
        )
        (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report_root / "summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
