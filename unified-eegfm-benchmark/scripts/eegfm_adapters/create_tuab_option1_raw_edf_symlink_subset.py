#!/usr/bin/env python3
"""Create a raw-EDF symlink tree from the TUAB Option 1 recording manifest.

This script creates symlinks only. It does not copy EDFs, run preprocessing,
launch training, or modify model repositories.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_RECORDING_MANIFEST = (
    "/nicoletye/workspace/unified_tuab/reports/"
    "tuab_original_vs_unified_matched_recording_manifest.csv"
)
DEFAULT_OUTPUT_ROOT = "/nicoletye/workspace/unified_tuab/data/tuab_option1_raw_edf_subset"
DEFAULT_REPORT_DIR = "/nicoletye/workspace/unified_tuab/reports"
SUMMARY_JSON = "tuab_option1_raw_edf_symlink_subset_summary.json"
REPORT_MD = "tuab_option1_raw_edf_symlink_subset_report.md"

SPLITS = ["train", "val", "test"]
LABEL_TO_NAME = {"0": "normal", "1": "abnormal", 0: "normal", 1: "abnormal"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create TUAB Option 1 raw EDF symlink subset tree.")
    parser.add_argument("--recording_manifest", default=DEFAULT_RECORDING_MANIFEST)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true", help="Remove an existing output tree before creating links.")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    required = {
        "canonical_split",
        "label",
        "subject_id",
        "recording_id",
        "edf_path",
        "edf_basename",
        "n_h5_windows",
    }
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(f"recording manifest missing columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"recording manifest is empty: {path}")
    return rows


def validate_manifest(rows: list[dict[str, str]]) -> dict[str, Any]:
    missing_sources = []
    duplicate_targets: Counter[tuple[str, str, str]] = Counter()
    recording_splits: dict[str, set[str]] = defaultdict(set)
    counts_by_split_label: Counter[tuple[str, str]] = Counter()
    windows_by_split_label: Counter[tuple[str, str]] = Counter()

    for row in rows:
        split = row["canonical_split"]
        label_name = LABEL_TO_NAME.get(row["label"])
        if split not in SPLITS:
            raise ValueError(f"unexpected canonical_split={split!r} for recording {row.get('recording_id')}")
        if label_name is None:
            raise ValueError(f"unexpected label={row['label']!r} for recording {row.get('recording_id')}")
        edf_path = Path(row["edf_path"])
        if not edf_path.exists():
            missing_sources.append(str(edf_path))
        target_key = (split, label_name, row["edf_basename"])
        duplicate_targets[target_key] += 1
        recording_splits[row["recording_id"]].add(split)
        counts_by_split_label[(split, label_name)] += 1
        windows_by_split_label[(split, label_name)] += int(row["n_h5_windows"])

    duplicate_target_examples = {
        "/".join(key): value for key, value in duplicate_targets.items() if value > 1
    }
    recordings_multi_split = {
        recording_id: sorted(splits)
        for recording_id, splits in recording_splits.items()
        if len(splits) > 1
    }
    return {
        "recording_manifest_rows": len(rows),
        "missing_source_count": len(missing_sources),
        "missing_source_examples": missing_sources[:20],
        "duplicate_target_filename_count": len(duplicate_target_examples),
        "duplicate_target_filename_examples": dict(list(duplicate_target_examples.items())[:20]),
        "recordings_in_multiple_canonical_splits_count": len(recordings_multi_split),
        "recordings_in_multiple_canonical_splits_examples": dict(list(recordings_multi_split.items())[:20]),
        "recording_counts_by_split_label": nested_counter(counts_by_split_label),
        "h5_window_counts_by_split_label": nested_counter(windows_by_split_label),
    }


def nested_counter(counter: Counter[tuple[str, str]]) -> dict[str, dict[str, int]]:
    nested: dict[str, dict[str, int]] = defaultdict(dict)
    for split in SPLITS:
        for label_name in ["normal", "abnormal"]:
            value = counter.get((split, label_name), 0)
            if value:
                nested[split][label_name] = int(value)
    return dict(nested)


def ensure_clean_output_root(output_root: Path, force: bool) -> None:
    if output_root.exists():
        if not force:
            raise FileExistsError(f"output_root already exists; use --force to replace it: {output_root}")
        if output_root.is_symlink() or output_root.is_file():
            output_root.unlink()
        else:
            shutil.rmtree(output_root)
    for split in SPLITS:
        for label_name in ["normal", "abnormal"]:
            (output_root / split / label_name).mkdir(parents=True, exist_ok=True)


def create_symlinks(rows: list[dict[str, str]], output_root: Path) -> list[dict[str, str]]:
    created = []
    for row in rows:
        label_name = LABEL_TO_NAME[row["label"]]
        target = output_root / row["canonical_split"] / label_name / row["edf_basename"]
        source = Path(row["edf_path"])
        target.symlink_to(source)
        created.append(
            {
                "source": str(source),
                "target": str(target),
                "split": row["canonical_split"],
                "label_name": label_name,
                "recording_id": row["recording_id"],
            }
        )
    return created


def validate_created_tree(output_root: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    symlink_count = 0
    non_symlink_paths = []
    broken_symlinks = []
    by_split_label: Counter[tuple[str, str]] = Counter()
    expected_targets = set()

    for row in rows:
        label_name = LABEL_TO_NAME[row["label"]]
        target = output_root / row["canonical_split"] / label_name / row["edf_basename"]
        expected_targets.add(str(target))
        if target.is_symlink():
            symlink_count += 1
            if not target.exists():
                broken_symlinks.append(str(target))
        elif target.exists():
            non_symlink_paths.append(str(target))
        else:
            broken_symlinks.append(str(target))
        by_split_label[(row["canonical_split"], label_name)] += 1

    return {
        "symlink_count": symlink_count,
        "expected_symlink_count": len(rows),
        "symlink_count_matches_manifest_rows": symlink_count == len(rows),
        "non_symlink_path_count": len(non_symlink_paths),
        "non_symlink_path_examples": non_symlink_paths[:20],
        "missing_or_broken_symlink_count": len(broken_symlinks),
        "missing_or_broken_symlink_examples": broken_symlinks[:20],
        "symlink_counts_by_split_label": nested_counter(by_split_label),
    }


def write_reports(summary: dict[str, Any], report_dir: Path) -> None:
    summary_path = report_dir / SUMMARY_JSON
    md_path = report_dir / REPORT_MD
    summary["output_paths"]["summary_json"] = str(summary_path)
    summary["output_paths"]["report_md"] = str(md_path)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# TUAB Option 1 Raw EDF Symlink Subset Report",
        "",
        "This report describes a raw EDF symlink tree created from the approved Option 1 recording manifest.",
        "",
        "No EDF files are copied. Each subset file is a symlink to the original raw EDF.",
        "",
        "## Inputs",
        "",
        f"- Recording manifest: `{summary['inputs']['recording_manifest']}`",
        f"- Output root: `{summary['output_root']}`",
        "",
        "## Validation",
        "",
        f"- Source EDF paths missing: `{summary['pre_validation']['missing_source_count']}`",
        f"- Duplicate target filenames: `{summary['pre_validation']['duplicate_target_filename_count']}`",
        f"- Recordings in multiple canonical splits: `{summary['pre_validation']['recordings_in_multiple_canonical_splits_count']}`",
        f"- Symlink count: `{summary['post_validation']['symlink_count']}` / `{summary['post_validation']['expected_symlink_count']}`",
        f"- Missing or broken symlinks: `{summary['post_validation']['missing_or_broken_symlink_count']}`",
        f"- Non-symlink paths: `{summary['post_validation']['non_symlink_path_count']}`",
        "",
        "## Symlink Counts By Split/Label",
        "",
        json.dumps(summary["post_validation"]["symlink_counts_by_split_label"], indent=2),
        "",
        "## Next Step",
        "",
        "Use the matched H5 NPZ builder to create the unified H5 subset, then run the combined artifact validator before any preprocessing or training.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.recording_manifest)
    output_root = Path(args.output_root)
    report_dir = Path(DEFAULT_REPORT_DIR)
    report_dir.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(manifest_path)
    pre_validation = validate_manifest(rows)
    if pre_validation["missing_source_count"]:
        raise FileNotFoundError(f"missing source EDF paths: {pre_validation['missing_source_examples']}")
    if pre_validation["duplicate_target_filename_count"]:
        raise ValueError(f"duplicate target filenames: {pre_validation['duplicate_target_filename_examples']}")
    if pre_validation["recordings_in_multiple_canonical_splits_count"]:
        raise ValueError(
            "recordings appear in multiple canonical splits: "
            f"{pre_validation['recordings_in_multiple_canonical_splits_examples']}"
        )

    ensure_clean_output_root(output_root, args.force)
    create_symlinks(rows, output_root)
    post_validation = validate_created_tree(output_root, rows)

    summary = {
        "status": "PASS"
        if post_validation["symlink_count_matches_manifest_rows"]
        and post_validation["missing_or_broken_symlink_count"] == 0
        and post_validation["non_symlink_path_count"] == 0
        else "FAIL",
        "matching_level": "EXACT",
        "inputs": {"recording_manifest": str(manifest_path)},
        "output_root": str(output_root),
        "pre_validation": pre_validation,
        "post_validation": post_validation,
        "output_paths": {},
    }
    write_reports(summary, report_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
