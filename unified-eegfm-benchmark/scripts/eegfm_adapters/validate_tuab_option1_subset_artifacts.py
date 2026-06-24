#!/usr/bin/env python3
"""Validate TUAB Option 1 raw EDF symlink and unified H5 NPZ artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_RECORDING_MANIFEST = (
    "/nicoletye/workspace/unified_tuab/reports/"
    "tuab_original_vs_unified_matched_recording_manifest.csv"
)
DEFAULT_WINDOW_MANIFEST = (
    "/nicoletye/workspace/unified_tuab/reports/"
    "tuab_original_vs_unified_matched_window_manifest.csv"
)
DEFAULT_RAW_ROOT = "/nicoletye/workspace/unified_tuab/data/tuab_option1_raw_edf_subset"
DEFAULT_NPZ = "/nicoletye/workspace/unified_tuab/reports/tuab_option1_unified_h5_subset.npz"
DEFAULT_REPORT_DIR = "/nicoletye/workspace/unified_tuab/reports"
SUMMARY_JSON = "tuab_option1_subset_artifacts_validation_summary.json"
REPORT_MD = "tuab_option1_subset_artifacts_validation_report.md"

SPLITS = ["train", "val", "test"]
LABEL_TO_NAME = {"0": "normal", "1": "abnormal", 0: "normal", 1: "abnormal"}
EXPECTED_NPZ_KEYS = {
    "train_x",
    "val_x",
    "test_x",
    "train_y",
    "val_y",
    "test_y",
    "train_h5_indices",
    "val_h5_indices",
    "test_h5_indices",
    "channel_names",
    "split_counts_json",
    "matched_manifest_json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate TUAB Option 1 subset artifacts.")
    parser.add_argument("--recording_manifest", default=DEFAULT_RECORDING_MANIFEST)
    parser.add_argument("--window_manifest", default=DEFAULT_WINDOW_MANIFEST)
    parser.add_argument("--raw_root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--npz", default=DEFAULT_NPZ)
    return parser.parse_args()


def nested_counter(counter: Counter[tuple[str, str]]) -> dict[str, dict[str, int]]:
    nested: dict[str, dict[str, int]] = defaultdict(dict)
    for split in SPLITS:
        for label in ["normal", "abnormal", "0", "1"]:
            value = counter.get((split, label), 0)
            if value:
                nested[split][label] = int(value)
    return dict(nested)


def read_recording_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_window_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def manifest_recording_counts(rows: list[dict[str, str]]) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        counts[(row["canonical_split"], LABEL_TO_NAME[row["label"]])] += 1
    return counts


def manifest_window_counts(rows: list[dict[str, str]]) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        counts[(row["canonical_split"], row["label"])] += 1
    return counts


def npz_array_headers(npz_path: Path) -> dict[str, dict[str, Any]]:
    headers: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(npz_path, "r") as zf:
        for name in zf.namelist():
            if not name.endswith(".npy"):
                continue
            key = name[:-4]
            with zf.open(name, "r") as f:
                version = np.lib.format.read_magic(f)
                if version == (1, 0):
                    shape, _fortran_order, dtype = np.lib.format.read_array_header_1_0(f)
                elif version in {(2, 0), (3, 0)}:
                    shape, _fortran_order, dtype = np.lib.format.read_array_header_2_0(f)
                else:
                    raise ValueError(f"unsupported npy version {version} for {name}")
            headers[key] = {"shape": list(shape), "dtype": str(dtype)}
    return headers


def validate_symlink_tree(raw_root: Path, recording_rows: list[dict[str, str]]) -> dict[str, Any]:
    counts: Counter[tuple[str, str]] = Counter()
    missing_tree_dirs = []
    missing_symlinks = []
    non_symlinks = []
    broken_symlinks = []
    duplicate_targets: Counter[str] = Counter()

    for split in SPLITS:
        for label_name in ["normal", "abnormal"]:
            d = raw_root / split / label_name
            if not d.is_dir():
                missing_tree_dirs.append(str(d))

    for row in recording_rows:
        label_name = LABEL_TO_NAME[row["label"]]
        target = raw_root / row["canonical_split"] / label_name / row["edf_basename"]
        duplicate_targets[str(target)] += 1
        if not target.exists() and not target.is_symlink():
            missing_symlinks.append(str(target))
            continue
        if not target.is_symlink():
            non_symlinks.append(str(target))
            continue
        if not target.exists():
            broken_symlinks.append(str(target))
            continue
        counts[(row["canonical_split"], label_name)] += 1

    duplicate_target_examples = {k: v for k, v in duplicate_targets.items() if v > 1}
    return {
        "raw_root_exists": raw_root.is_dir(),
        "missing_tree_dir_count": len(missing_tree_dirs),
        "missing_tree_dir_examples": missing_tree_dirs[:20],
        "symlink_counts_by_split_label": nested_counter(counts),
        "symlink_total": sum(counts.values()),
        "expected_symlink_total": len(recording_rows),
        "missing_symlink_count": len(missing_symlinks),
        "missing_symlink_examples": missing_symlinks[:20],
        "non_symlink_count": len(non_symlinks),
        "non_symlink_examples": non_symlinks[:20],
        "broken_symlink_count": len(broken_symlinks),
        "broken_symlink_examples": broken_symlinks[:20],
        "duplicate_target_count": len(duplicate_target_examples),
        "duplicate_target_examples": dict(list(duplicate_target_examples.items())[:20]),
    }


def validate_npz(npz_path: Path, window_rows: list[dict[str, str]]) -> dict[str, Any]:
    if not npz_path.exists():
        return {"npz_exists": False, "error": f"missing NPZ: {npz_path}"}

    manifest_counts = manifest_window_counts(window_rows)
    manifest_indices = {
        split: [int(row["h5_index"]) for row in window_rows if row["canonical_split"] == split]
        for split in SPLITS
    }
    manifest_labels = {
        split: [int(row["label"]) for row in window_rows if row["canonical_split"] == split]
        for split in SPLITS
    }

    headers = npz_array_headers(npz_path)
    keys = set(headers)
    missing_keys = sorted(EXPECTED_NPZ_KEYS - keys)
    extra_keys = sorted(keys - EXPECTED_NPZ_KEYS)
    shapes = {key: value["shape"] for key, value in headers.items()}
    dtypes = {key: value["dtype"] for key, value in headers.items()}
    shape_mismatches = []

    with np.load(npz_path, allow_pickle=True) as z:
        all_indices = []
        label_mismatch_count = 0
        index_mismatch_count = 0
        split_label_counts: Counter[tuple[str, str]] = Counter()

        for split in SPLITS:
            expected_n = len(manifest_labels[split])
            for key, expected_shape in [
                (f"{split}_x", [expected_n, 23, 2000]),
                (f"{split}_y", [expected_n]),
                (f"{split}_h5_indices", [expected_n]),
            ]:
                observed_shape = shapes.get(key)
                if observed_shape != expected_shape:
                    shape_mismatches.append({"key": key, "expected": expected_shape, "observed": observed_shape})
            if f"{split}_y" not in z or f"{split}_h5_indices" not in z:
                continue
            y = z[f"{split}_y"].astype(np.int64)
            indices = z[f"{split}_h5_indices"].astype(np.int64)
            all_indices.extend(indices.tolist())
            expected_y = np.asarray(manifest_labels[split], dtype=np.int64)
            expected_indices = np.asarray(manifest_indices[split], dtype=np.int64)
            if y.shape != expected_y.shape or not np.array_equal(y, expected_y):
                label_mismatch_count += 1
            if indices.shape != expected_indices.shape or not np.array_equal(indices, expected_indices):
                index_mismatch_count += 1
            labels = Counter(str(int(v)) for v in y.tolist())
            for label, value in labels.items():
                split_label_counts[(split, label)] += value

        duplicate_h5_index_count = len(all_indices) - len(set(all_indices))
        split_counts_json = {}
        if "split_counts_json" in z:
            try:
                split_counts_json = json.loads(str(z["split_counts_json"][0]))
            except Exception as exc:
                split_counts_json = {"parse_error": f"{type(exc).__name__}: {exc}"}

    return {
        "npz_exists": True,
        "missing_keys": missing_keys,
        "extra_keys": extra_keys,
        "shapes": shapes,
        "dtypes": dtypes,
        "shape_mismatches": shape_mismatches,
        "manifest_window_counts_by_split_label": nested_counter(manifest_counts),
        "npz_window_counts_by_split_label": nested_counter(split_label_counts),
        "label_array_mismatch_split_count": label_mismatch_count,
        "h5_index_array_mismatch_split_count": index_mismatch_count,
        "duplicate_h5_index_count": duplicate_h5_index_count,
        "split_counts_json": split_counts_json,
    }


def write_reports(summary: dict[str, Any], report_dir: Path) -> None:
    summary_path = report_dir / SUMMARY_JSON
    md_path = report_dir / REPORT_MD
    summary["output_paths"] = {
        "summary_json": str(summary_path),
        "report_md": str(md_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# TUAB Option 1 Subset Artifact Validation Report",
        "",
        f"- Status: `{summary['status']}`",
        f"- Raw root: `{summary['inputs']['raw_root']}`",
        f"- NPZ: `{summary['inputs']['npz']}`",
        "",
        "## Raw EDF Symlink Tree",
        "",
        json.dumps(summary["symlink_tree_validation"], indent=2),
        "",
        "## Unified H5 NPZ",
        "",
        json.dumps(summary["npz_validation"], indent=2),
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    recording_manifest = Path(args.recording_manifest)
    window_manifest = Path(args.window_manifest)
    raw_root = Path(args.raw_root)
    npz_path = Path(args.npz)
    report_dir = Path(DEFAULT_REPORT_DIR)
    report_dir.mkdir(parents=True, exist_ok=True)

    recording_rows = read_recording_manifest(recording_manifest)
    window_rows = read_window_manifest(window_manifest)
    symlink_validation = validate_symlink_tree(raw_root, recording_rows)
    npz_validation = validate_npz(npz_path, window_rows)

    errors = [
        symlink_validation.get("missing_tree_dir_count", 0),
        symlink_validation.get("missing_symlink_count", 0),
        symlink_validation.get("non_symlink_count", 0),
        symlink_validation.get("broken_symlink_count", 0),
        symlink_validation.get("duplicate_target_count", 0),
        0 if npz_validation.get("npz_exists") else 1,
        len(npz_validation.get("missing_keys", [])),
        len(npz_validation.get("shape_mismatches", [])),
        npz_validation.get("label_array_mismatch_split_count", 0),
        npz_validation.get("h5_index_array_mismatch_split_count", 0),
        npz_validation.get("duplicate_h5_index_count", 0),
    ]
    summary = {
        "status": "PASS" if not any(errors) else "FAIL",
        "inputs": {
            "recording_manifest": str(recording_manifest),
            "window_manifest": str(window_manifest),
            "raw_root": str(raw_root),
            "npz": str(npz_path),
        },
        "recording_manifest_rows": len(recording_rows),
        "window_manifest_rows": len(window_rows),
        "recording_manifest_counts_by_split_label": nested_counter(manifest_recording_counts(recording_rows)),
        "window_manifest_counts_by_split_label": nested_counter(manifest_window_counts(window_rows)),
        "symlink_tree_validation": symlink_validation,
        "npz_validation": npz_validation,
    }
    write_reports(summary, report_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
