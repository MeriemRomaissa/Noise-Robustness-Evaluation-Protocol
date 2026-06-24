#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tiny canonical TUAB H5 subset loader and input adapters for EEG-FM smoke tests."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch


DEFAULT_H5 = "/nicoletye/workspace/unified_tuab/data/canonical_tuab_full.h5"
DEFAULT_SPLIT_INDEX = (
    "/nicoletye/workspace/unified_tuab/reports/"
    "canonical_h5_labram_referenced_max_coverage_split_index.csv"
)

TINY_TARGETS = {"train": 32, "val": 16, "test": 16}

# The 16-channel TUAB order used by CBraMod/CSBrain/CodeBrain TUAB wrappers.
TUAB_16_CHANNEL_NAMES = [
    "FP1", "F7", "T3", "T5",
    "FP2", "F8", "T4", "T6",
    "FP1", "F3", "C3", "P3",
    "FP2", "F4", "C4", "P4",
]


def _decode(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _ratio_indices(rows: List[dict], target_size: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    by_label: Dict[int, List[dict]] = defaultdict(list)
    for row in rows:
        by_label[int(row["label"])].append(row)
    for group in by_label.values():
        rng.shuffle(group)

    n_total = len(rows)
    if target_size <= 0 or target_size >= n_total:
        selected = rows[:]
        rng.shuffle(selected)
        return selected

    labels = sorted(by_label)
    raw_targets = {label: target_size * len(by_label[label]) / max(n_total, 1) for label in labels}
    take = {label: min(int(np.floor(raw_targets[label])), len(by_label[label])) for label in labels}
    remaining = target_size - sum(take.values())
    order = sorted(
        labels,
        key=lambda label: (raw_targets[label] - np.floor(raw_targets[label]), len(by_label[label])),
        reverse=True,
    )
    while remaining > 0:
        progressed = False
        for label in order:
            if remaining <= 0:
                break
            if take[label] < len(by_label[label]):
                take[label] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break

    selected: List[dict] = []
    for label in labels:
        selected.extend(by_label[label][:take[label]])
    rng.shuffle(selected)
    return selected


def read_split_index(split_index_csv: str) -> Dict[str, List[dict]]:
    rows_by_split: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    with open(split_index_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"h5_index", "canonical_split", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Split index missing required columns: {sorted(missing)}")
        for row in reader:
            split = row["canonical_split"].strip().lower()
            if split == "eval":
                split = "test"
            if split not in rows_by_split:
                continue
            rows_by_split[split].append(
                {
                    "h5_index": int(row["h5_index"]),
                    "label": int(row["label"]),
                    "sample_id": row.get("sample_id", ""),
                }
            )
    return rows_by_split


def class_counts(labels: Iterable[int]) -> Dict[str, int]:
    labels_arr = np.asarray(list(labels), dtype=np.int64)
    return {
        "normal_0": int((labels_arr == 0).sum()),
        "abnormal_1": int((labels_arr == 1).sum()),
    }


def select_tiny_rows(rows_by_split: Dict[str, List[dict]], seed: int = 42) -> Dict[str, List[dict]]:
    return {
        split: _ratio_indices(rows, TINY_TARGETS[split], seed + offset)
        for offset, (split, rows) in enumerate(rows_by_split.items())
    }


def load_train_batch(
    h5_path: str = DEFAULT_H5,
    split_index_csv: str = DEFAULT_SPLIT_INDEX,
    batch_size: int = 2,
    seed: int = 42,
) -> dict:
    import h5py

    rows_by_split = read_split_index(split_index_csv)
    tiny_rows = select_tiny_rows(rows_by_split, seed=seed)
    train_rows = tiny_rows["train"][:batch_size]
    h5_indices = [row["h5_index"] for row in train_rows]
    labels = [row["label"] for row in train_rows]

    with h5py.File(h5_path, "r") as f:
        channel_names = [_decode(ch) for ch in f.attrs["channel_names"][:]]
        sorted_pairs = sorted(enumerate(h5_indices), key=lambda pair: pair[1])
        sorted_h5_indices = [idx for _, idx in sorted_pairs]
        raw_sorted = f["eeg"][sorted_h5_indices].astype(np.float32)
        inverse = np.argsort([pos for pos, _ in sorted_pairs])
        raw = raw_sorted[inverse]

    x = torch.from_numpy(raw)
    y = torch.tensor(labels, dtype=torch.long)
    split_counts = {
        split: {"n": len(rows), **class_counts(row["label"] for row in rows)}
        for split, rows in rows_by_split.items()
    }
    tiny_counts = {
        split: {"n": len(rows), **class_counts(row["label"] for row in rows)}
        for split, rows in tiny_rows.items()
    }
    return {
        "x": x,
        "y": y,
        "h5_indices": h5_indices,
        "channel_names": channel_names,
        "split_counts": split_counts,
        "tiny_counts": tiny_counts,
    }


def save_batch_npz(batch: dict, path: str) -> None:
    np.savez_compressed(
        path,
        x=batch["x"].numpy(),
        y=batch["y"].numpy(),
        h5_indices=np.asarray(batch["h5_indices"], dtype=np.int64),
        channel_names=np.asarray(batch["channel_names"], dtype=object),
        split_counts_json=np.asarray([__import__("json").dumps(batch["split_counts"])]),
        tiny_counts_json=np.asarray([__import__("json").dumps(batch["tiny_counts"])]),
    )


def load_batch_npz(path: str) -> dict:
    import json

    data = np.load(path, allow_pickle=True)
    return {
        "x": torch.from_numpy(data["x"].astype(np.float32)),
        "y": torch.from_numpy(data["y"].astype(np.int64)),
        "h5_indices": data["h5_indices"].astype(np.int64).tolist(),
        "channel_names": [str(x) for x in data["channel_names"].tolist()],
        "split_counts": json.loads(str(data["split_counts_json"][0])),
        "tiny_counts": json.loads(str(data["tiny_counts_json"][0])),
    }


def select_tuab_16(x: torch.Tensor, channel_names: List[str]) -> Tuple[torch.Tensor, List[int], List[str]]:
    name_to_index = {name.upper(): idx for idx, name in enumerate(channel_names)}
    indices = [name_to_index[name.upper()] for name in TUAB_16_CHANNEL_NAMES]
    return x[:, indices, :], indices, TUAB_16_CHANNEL_NAMES[:]


def to_patches(x: torch.Tensor, patch_size: int = 200) -> torch.Tensor:
    if x.shape[-1] % patch_size != 0:
        raise ValueError(f"Cannot patch length {x.shape[-1]} by patch_size={patch_size}")
    return x.reshape(x.shape[0], x.shape[1], x.shape[-1] // patch_size, patch_size)
