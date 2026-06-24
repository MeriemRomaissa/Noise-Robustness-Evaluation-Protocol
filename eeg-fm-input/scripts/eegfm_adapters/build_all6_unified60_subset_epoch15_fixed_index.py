#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a fixed exact-split subset index NPZ for the all-6 dev benchmark.

This helper reads only the split CSV and stores H5 row indices/labels. It does
not read /eeg, modify the canonical H5, rebuild preprocessing, or touch EDFs.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


CHANNEL_NAMES = np.asarray(
    [
        "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8",
        "T3", "T4", "T5", "T6", "A1", "A2", "FZ", "CZ", "PZ", "T1", "T2",
    ],
    dtype=object,
)


def count_labels(labels: np.ndarray) -> dict[str, int]:
    counts = Counter(labels.astype(int).tolist())
    return {"0": int(counts.get(0, 0)), "1": int(counts.get(1, 0))}


def select_ratio_preserving(indices: np.ndarray, labels: np.ndarray, target: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if target <= 0 or target >= len(indices):
        order = np.argsort(indices)
        return indices[order], labels[order]
    rng = np.random.default_rng(seed)
    selected_parts: list[np.ndarray] = []
    remaining = target
    label_values = sorted(set(labels.astype(int).tolist()))
    for i, label in enumerate(label_values):
        label_pos = np.where(labels == label)[0]
        if i == len(label_values) - 1:
            take = remaining
        else:
            take = int(round(target * len(label_pos) / len(labels)))
            take = max(0, min(take, len(label_pos), remaining))
        chosen = rng.choice(label_pos, size=take, replace=False) if take else np.asarray([], dtype=np.int64)
        selected_parts.append(chosen)
        remaining -= take
    selected = np.concatenate(selected_parts) if selected_parts else np.asarray([], dtype=np.int64)
    if len(selected) < target:
        missing = target - len(selected)
        already = set(selected.astype(int).tolist())
        pool = np.asarray([i for i in range(len(indices)) if i not in already], dtype=np.int64)
        selected = np.concatenate([selected, rng.choice(pool, size=missing, replace=False)])
    elif len(selected) > target:
        selected = rng.choice(selected, size=target, replace=False)
    out_indices = indices[selected]
    out_labels = labels[selected]
    order = np.argsort(out_indices)
    return out_indices[order], out_labels[order]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_csv", default="reports/labram_exact_original_processed_split/canonical_h5_labram_exact_original_processed_split_index.csv")
    parser.add_argument("--output_npz", default="reports/all6_unified60_subset_epoch15_5seed_1gpu_dev_v1/all6_fixed_subset_seed42_index.npz")
    parser.add_argument("--summary_json", default="reports/all6_unified60_subset_epoch15_5seed_1gpu_dev_v1/all6_fixed_subset_seed42_index_summary.json")
    parser.add_argument("--train_n", type=int, default=8192)
    parser.add_argument("--val_n", type=int, default=2048)
    parser.add_argument("--test_n", type=int, default=2048)
    parser.add_argument("--subset_seed", type=int, default=42)
    args = parser.parse_args()

    rows: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"indices": [], "labels": []})
    split_csv = Path(args.split_csv)
    if not split_csv.exists():
        raise FileNotFoundError(f"Missing split CSV: {split_csv}")
    with split_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"h5_index", "canonical_split", "label"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Split CSV missing columns: {missing}")
        for row in reader:
            split = row["canonical_split"].strip().lower()
            if split == "eval":
                split = "test"
            if split in {"train", "val", "test"}:
                rows[split]["indices"].append(int(row["h5_index"]))
                rows[split]["labels"].append(int(row["label"]))

    targets = {"train": args.train_n, "val": args.val_n, "test": args.test_n}
    arrays: dict[str, np.ndarray] = {"channel_names": CHANNEL_NAMES}
    split_counts: dict[str, dict] = {}
    source_counts: dict[str, dict] = {}
    for offset, split in enumerate(["train", "val", "test"]):
        all_indices = np.asarray(rows[split]["indices"], dtype=np.int64)
        all_labels = np.asarray(rows[split]["labels"], dtype=np.int64)
        selected_indices, selected_labels = select_ratio_preserving(
            all_indices,
            all_labels,
            int(targets[split]),
            int(args.subset_seed) + offset * 1009,
        )
        arrays[f"{split}_h5_indices"] = selected_indices
        arrays[f"{split}_y"] = selected_labels
        split_counts[split] = {"n": int(len(selected_indices)), "labels": count_labels(selected_labels)}
        source_counts[split] = {"n": int(len(all_indices)), "labels": count_labels(all_labels)}

    output_npz = Path(args.output_npz)
    summary_json = Path(args.summary_json)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    arrays["split_counts_json"] = np.asarray([json.dumps(split_counts, sort_keys=True)], dtype=object)
    arrays["tiny_counts_json"] = np.asarray([json.dumps(split_counts, sort_keys=True)], dtype=object)
    np.savez_compressed(output_npz, **arrays)

    summary = {
        "status": "PASS",
        "split_csv": str(split_csv),
        "output_npz": str(output_npz),
        "subset_seed": int(args.subset_seed),
        "targets": targets,
        "split_counts": split_counts,
        "source_counts": source_counts,
        "channel_count": int(len(CHANNEL_NAMES)),
        "notes": "Fixed index NPZ contains H5 row indices and labels only; all jobs use train_n/val_n/test_n=0 so training seed cannot alter subset membership.",
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
