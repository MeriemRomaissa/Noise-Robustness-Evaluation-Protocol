#!/usr/bin/env python3
"""Create NMT OOD plots from saved Benchmark inference outputs.

This script intentionally does not run inference. It reads
``nmt_ood_results.json`` files produced by ``Benchmark/Inference`` and turns
their saved metrics into tables and figures.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_OUTPUT_DIR = Path("Benchmark/Outputs/nmt_ood_plots")
OOD_METRICS = [
    ("accuracy", "Accuracy"),
    ("balanced_accuracy", "Balanced Accuracy"),
    ("roc_auc", "ROC-AUC"),
    ("pr_auc", "PR-AUC"),
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric_value(metrics: dict[str, Any], name: str):
    candidates = {
        "accuracy": ("accuracy", "test_accuracy"),
        "balanced_accuracy": ("balanced_accuracy", "test_balanced_accuracy"),
        "roc_auc": ("roc_auc", "auc", "test_roc_auc", "test_auroc"),
        "pr_auc": ("pr_auc", "auprc", "test_pr_auc", "test_auprc"),
        "loss": ("loss", "test_loss"),
    }
    for key in candidates[name]:
        if key in metrics:
            return metrics[key]
    return None


def normalize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "accuracy": metric_value(metrics, "accuracy"),
        "balanced_accuracy": metric_value(metrics, "balanced_accuracy"),
        "roc_auc": metric_value(metrics, "roc_auc"),
        "pr_auc": metric_value(metrics, "pr_auc"),
        "loss": metric_value(metrics, "loss"),
        "n_samples": metrics.get("n_samples"),
    }


def infer_model_name(path: Path, data: dict[str, Any]) -> str:
    description = str(data.get("description", ""))
    for model in ("LaBraM", "EEGPT", "BIOT", "CBraMod", "CSBrain", "CodeBrain"):
        if model.lower() in description.lower() or model.lower() in str(path).lower():
            return model
    return path.parent.name


def records_from_result(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    records: list[dict[str, Any]] = []

    if "ft" in data or "lora" in data or "full_ft" in data or "eegnet" in data:
        primary_model = infer_model_name(path, data)
        labels = {
            "ft": "LaBraM Full FT",
            "lora": "LaBraM LoRA",
            "full_ft": f"{primary_model} Full FT",
            "eegnet": "EEGNet",
        }
        for key, label in labels.items():
            if key not in data:
                continue
            entry = data[key]
            row = {
                "model": "EEGNet" if key == "eegnet" else primary_model,
                "condition": label,
                "result_path": str(path),
                "checkpoint": entry.get("checkpoint") or entry.get("source_json"),
            }
            row.update({f"nmt_ood_{k}": v for k, v in normalize_metrics(entry.get("nmt_ood", {})).items()})
            row.update({f"clean_{k}": v for k, v in normalize_metrics(entry.get("tuab_clean", {})).items()})
            records.append(row)
        return records

    model = data.get("model") or infer_model_name(path, data)
    condition = data.get("condition") or model
    row = {
        "model": model,
        "condition": condition,
        "result_path": str(path),
        "checkpoint": data.get("checkpoint"),
    }
    row.update({f"nmt_ood_{k}": v for k, v in normalize_metrics(data.get("nmt_ood", {})).items()})
    records.append(row)
    return [row]


def collect_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(records_from_result(path))
    return records


def load_clean_metrics(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    data = load_json(path)
    clean = {}
    for name, metrics in data.items():
        clean[name] = normalize_metrics(metrics)
    return clean


def attach_clean_metrics(records: list[dict[str, Any]], clean_metrics: dict[str, dict[str, Any]]):
    for row in records:
        keys = [row["condition"], row["model"]]
        for key in keys:
            if key not in clean_metrics:
                continue
            for metric, value in clean_metrics[key].items():
                row.setdefault(f"clean_{metric}", value)
            break


def as_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_tables(records: list[dict[str, Any]], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in records for key in row})
    with (output_dir / "nmt_ood_plot_records.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    with (output_dir / "nmt_ood_plot_records.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def plot_ood_bars(records: list[dict[str, Any]], output_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["condition"] for row in records]
    x = np.arange(len(records))
    for metric, title in OOD_METRICS:
        values = [as_float(row.get(f"nmt_ood_{metric}")) for row in records]
        if all(value is None for value in values):
            continue
        fig, ax = plt.subplots(figsize=(max(6, len(records) * 0.8), 4))
        heights = [np.nan if value is None else value for value in values]
        bars = ax.bar(x, heights, color="#4c78a8", edgecolor="black", linewidth=0.5)
        for bar, value in zip(bars, values):
            if value is None:
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.005,
                    f"{value:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(title)
        ax.set_title(f"NMT OOD {title}")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(output_dir / f"nmt_ood_{metric}.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_clean_vs_ood(records: list[dict[str, Any]], output_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    available = [
        row for row in records
        if as_float(row.get("clean_accuracy")) is not None and as_float(row.get("nmt_ood_accuracy")) is not None
    ]
    if not available:
        return

    labels = [row["condition"] for row in available]
    x = np.arange(len(available))
    width = 0.35
    for metric, title in OOD_METRICS:
        clean = [as_float(row.get(f"clean_{metric}")) for row in available]
        ood = [as_float(row.get(f"nmt_ood_{metric}")) for row in available]
        if all(value is None for value in clean) or all(value is None for value in ood):
            continue
        fig, ax = plt.subplots(figsize=(max(6, len(available) * 0.9), 4))
        clean_vals = [np.nan if value is None else value for value in clean]
        ood_vals = [np.nan if value is None else value for value in ood]
        ax.bar(x - width / 2, clean_vals, width, label="TUAB Clean",
               color="#5b9bd5", edgecolor="black", linewidth=0.5)
        ax.bar(x + width / 2, ood_vals, width, label="NMT OOD",
               color="#ed7d31", edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(title)
        ax.set_title(f"Clean TUAB vs NMT OOD {title}")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(output_dir / f"clean_vs_nmt_ood_{metric}.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)


def records_with_clean_and_ood_accuracy(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row for row in records
        if as_float(row.get("clean_accuracy")) is not None and as_float(row.get("nmt_ood_accuracy")) is not None
    ]


def plot_relative_robustness(records: list[dict[str, Any]], output_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    available = records_with_clean_and_ood_accuracy(records)
    if not available:
        return

    labels = [row["condition"] for row in available]
    values = [
        as_float(row["nmt_ood_accuracy"]) / as_float(row["clean_accuracy"])
        if as_float(row["clean_accuracy"]) not in (None, 0.0) else None
        for row in available
    ]
    if all(value is None for value in values):
        return

    x = np.arange(len(available))
    fig, ax = plt.subplots(figsize=(max(6, len(available) * 0.8), 4))
    heights = [np.nan if value is None else value for value in values]
    bars = ax.bar(x, heights, color="#70ad47", edgecolor="black", linewidth=0.5)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.0)
    for bar, value in zip(bars, values):
        if value is None:
            continue
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01,
                f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Relative Accuracy")
    ax.set_title("Relative OOD Robustness (NMT OOD / TUAB Clean)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"nmt_ood_relative_robustness.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_drop(records: list[dict[str, Any]], output_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    available = records_with_clean_and_ood_accuracy(records)
    if not available:
        return

    labels = [row["condition"] for row in available]
    drops = [
        as_float(row["clean_accuracy"]) - as_float(row["nmt_ood_accuracy"])
        for row in available
    ]
    x = np.arange(len(available))
    fig, ax = plt.subplots(figsize=(max(6, len(available) * 0.8), 4))
    bars = ax.bar(x, drops, color="#c55a11", edgecolor="black", linewidth=0.5)
    ax.axhline(0.0, color="gray", linewidth=0.8)
    for bar, drop in zip(bars, drops):
        ax.text(bar.get_x() + bar.get_width() / 2, drop + 0.005,
                f"{drop:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Accuracy Drop")
    ax.set_title("Accuracy Drop (TUAB Clean - NMT OOD)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"nmt_ood_accuracy_drop.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_input_paths(args) -> list[Path]:
    paths = [Path(p) for p in args.input]
    for pattern in args.input_glob:
        paths.extend(Path(p) for p in glob.glob(pattern, recursive=True))
    unique = sorted({path.resolve() for path in paths if path.exists()})
    if not unique:
        raise FileNotFoundError("no input nmt_ood_results.json files found")
    return unique


def parse_args():
    parser = argparse.ArgumentParser(description="Plot saved NMT OOD inference results.")
    parser.add_argument("--input", nargs="*", default=[], help="One or more nmt_ood_results.json files.")
    parser.add_argument("--input_glob", nargs="*", default=[],
                        help="Glob pattern(s), for example 'outputs/**/nmt_ood_results.json'.")
    parser.add_argument("--clean_metrics_json", type=Path, default=None,
                        help="Optional JSON mapping model/condition names to clean TUAB metrics.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    input_paths = parse_input_paths(args)
    records = collect_records(input_paths)
    attach_clean_metrics(records, load_clean_metrics(args.clean_metrics_json))

    if args.dry_run:
        print(f"inputs={len(input_paths)}")
        for path in input_paths:
            print(path)
        print(f"records={len(records)}")
        print(f"output_dir={args.output_dir}")
        return

    write_tables(records, args.output_dir)
    plot_ood_bars(records, args.output_dir)
    plot_clean_vs_ood(records, args.output_dir)
    plot_relative_robustness(records, args.output_dir)
    plot_accuracy_drop(records, args.output_dir)
    print(f"Wrote plot tables and figures to {args.output_dir}")


if __name__ == "__main__":
    main()
