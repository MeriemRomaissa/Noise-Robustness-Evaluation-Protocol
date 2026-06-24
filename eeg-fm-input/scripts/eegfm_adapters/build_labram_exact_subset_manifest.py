#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Select a fixed exact-key subset for LaBraM notch ablations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXACT_SPLIT = PROJECT_ROOT / "reports" / "labram_exact_original_processed_split" / "canonical_h5_labram_exact_original_processed_split_index.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "labram_notch_ablation_subset_v1"
SPLITS = ["train", "val", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact_split_csv", default=str(DEFAULT_EXACT_SPLIT))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--train_n", type=int, default=8192)
    parser.add_argument("--val_n", type=int, default=2048)
    parser.add_argument("--test_n", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["h5_index"] = int(row["h5_index"])
        row["label"] = int(row["label"])
        row["window_idx"] = int(row.get("window_idx") or round(float(row["start_time"]) / 10.0))
    return rows


def select_split(rows: list[dict[str, Any]], split: str, target: int, seed: int) -> list[dict[str, Any]]:
    pool = [row for row in rows if row["canonical_split"] == split]
    if target <= 0 or target >= len(pool):
        return sorted(pool, key=lambda r: r["h5_index"])
    by_label = {label: [row for row in pool if row["label"] == label] for label in [0, 1]}
    selected = []
    rng = np.random.default_rng(seed)
    for label, label_rows in by_label.items():
        natural_n = int(round(target * len(label_rows) / max(1, len(pool))))
        natural_n = min(natural_n, len(label_rows))
        idx = np.arange(len(label_rows))
        rng.shuffle(idx)
        selected.extend(label_rows[i] for i in idx[:natural_n])
    while len(selected) < target:
        selected_keys = {row["h5_index"] for row in selected}
        remaining = [row for row in pool if row["h5_index"] not in selected_keys]
        if not remaining:
            break
        selected.append(remaining[int(rng.integers(0, len(remaining)))])
    return sorted(selected[:target], key=lambda r: r["h5_index"])


def main() -> int:
    args = parse_args()
    rows = load_rows(Path(args.exact_split_csv))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = {"train": args.train_n, "val": args.val_n, "test": args.test_n}
    selected = []
    for offset, split in enumerate(SPLITS):
        selected.extend(select_split(rows, split, targets[split], args.seed + offset * 1009))
    manifest_path = output_dir / "labram_notch_ablation_exact_subset_manifest.csv"
    fields = [
        "canonical_split",
        "label",
        "h5_index",
        "recording_id",
        "window_idx",
        "subject_id",
        "sample_id",
        "start_time",
        "end_time",
        "source_path",
        "raw_hash",
    ]
    if not args.dry_run:
        write_csv(manifest_path, selected, fields)
    summary = {
        "status": "DRY_RUN" if args.dry_run else "PASS",
        "exact_split_csv": str(args.exact_split_csv),
        "manifest_path": str(manifest_path),
        "targets": targets,
        "selected_total": len(selected),
        "counts": {
            split: {
                "n": sum(1 for row in selected if row["canonical_split"] == split),
                "label_0": sum(1 for row in selected if row["canonical_split"] == split and row["label"] == 0),
                "label_1": sum(1 for row in selected if row["canonical_split"] == split and row["label"] == 1),
            }
            for split in SPLITS
        },
        "seed": args.seed,
    }
    summary_path = output_dir / "labram_notch_ablation_exact_subset_manifest_summary.json"
    if not args.dry_run:
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
