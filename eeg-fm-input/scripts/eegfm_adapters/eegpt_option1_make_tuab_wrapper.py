#!/usr/bin/env python3
"""Non-invasive EEGPT TUAB preprocessing wrapper for Option 1.

This wrapper preserves EEGPT's original TUAB EDF-to-PKL transform while
replacing the original hardcoded TUAB root and random train/val split with the
approved Option 1 matched raw-EDF subset and canonical train/val/test split.

It writes EEGPT-style window PKLs:
    {"X": np.ndarray[23, 2000], "y": int}

No original EEGPT repository files are modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_RAW_SUBSET_ROOT = "/nicoletye/workspace/unified_tuab/data/tuab_option1_raw_edf_subset"
DEFAULT_RECORDING_MANIFEST = (
    "/nicoletye/workspace/unified_tuab/reports/"
    "tuab_original_vs_unified_matched_recording_manifest.csv"
)
DEFAULT_OUTPUT_ROOT = "/nicoletye/workspace/unified_tuab/data/eegpt_option1_original_processed"
DEFAULT_REPORT_DIR = "/nicoletye/workspace/unified_tuab/reports"
SUMMARY_JSON = "eegpt_option1_preprocessing_wrapper_summary.json"
REPORT_MD = "eegpt_option1_preprocessing_wrapper_report.md"

SPLITS = ["train", "val", "test"]
LABEL_TO_NAME = {"0": "normal", "1": "abnormal", 0: "normal", 1: "abnormal"}

# Copied from EEGPT downstream_tueg/dataset_maker/make_TUAB.py.
DROP_CHANNELS = [
    "PHOTIC-REF",
    "IBI",
    "BURSTS",
    "SUPPR",
    "EEG ROC-REF",
    "EEG LOC-REF",
    "EEG EKG1-REF",
    "EMG-REF",
    "EEG C3P-REF",
    "EEG C4P-REF",
    "EEG SP1-REF",
    "EEG SP2-REF",
    "EEG LUC-REF",
    "EEG RLC-REF",
    "EEG RESP1-REF",
    "EEG RESP2-REF",
    "EEG EKG-REF",
    "RESP ABDOMEN-REF",
    "ECG EKG-REF",
    "PULSE RATE",
    "EEG PG2-REF",
    "EEG PG1-REF",
]
DROP_CHANNELS.extend([f"EEG {i}-REF" for i in range(20, 129)])

CH_ORDER_STANDARD = [
    "EEG FP1-REF",
    "EEG FP2-REF",
    "EEG F3-REF",
    "EEG F4-REF",
    "EEG C3-REF",
    "EEG C4-REF",
    "EEG P3-REF",
    "EEG P4-REF",
    "EEG O1-REF",
    "EEG O2-REF",
    "EEG F7-REF",
    "EEG F8-REF",
    "EEG T3-REF",
    "EEG T4-REF",
    "EEG T5-REF",
    "EEG T6-REF",
    "EEG A1-REF",
    "EEG A2-REF",
    "EEG FZ-REF",
    "EEG CZ-REF",
    "EEG PZ-REF",
    "EEG T1-REF",
    "EEG T2-REF",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EEGPT Option 1 TUAB preprocessing wrapper.")
    parser.add_argument("--raw_subset_root", default=DEFAULT_RAW_SUBSET_ROOT)
    parser.add_argument("--recording_manifest", default=DEFAULT_RECORDING_MANIFEST)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Remove output_root before writing.")
    parser.add_argument(
        "--limit_recordings_per_split_label",
        type=int,
        default=0,
        help="Limit recordings per canonical split/label. 0 means full Option 1 subset original preprocessing.",
    )
    parser.add_argument("--n_jobs_resample", type=int, default=5)
    return parser.parse_args()


def nested_counter(counter: Counter[tuple[str, str]]) -> dict[str, dict[str, int]]:
    nested: dict[str, dict[str, int]] = defaultdict(dict)
    for split in SPLITS:
        for label_name in ["normal", "abnormal"]:
            value = counter.get((split, label_name), 0)
            if value:
                nested[split][label_name] = int(value)
    return dict(nested)


def read_recording_manifest(path: Path) -> list[dict[str, Any]]:
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
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"recording manifest missing columns: {missing}")
        rows = []
        for row in reader:
            row["label"] = int(row["label"])
            row["n_h5_windows"] = int(row["n_h5_windows"])
            if row["canonical_split"] not in SPLITS:
                raise ValueError(f"unexpected canonical_split={row['canonical_split']!r}")
            if row["label"] not in (0, 1):
                raise ValueError(f"unexpected label={row['label']!r}")
            rows.append(row)
    return rows


def symlink_path_for_row(raw_subset_root: Path, row: dict[str, Any]) -> Path:
    label_name = LABEL_TO_NAME[row["label"]]
    return raw_subset_root / row["canonical_split"] / label_name / row["edf_basename"]


def select_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    counts: Counter[tuple[str, int]] = Counter()
    for row in sorted(rows, key=lambda r: (r["canonical_split"], r["label"], r["recording_id"])):
        key = (row["canonical_split"], row["label"])
        if limit and counts[key] >= limit:
            continue
        selected.append(row)
        counts[key] += 1
    return selected


def validate_inputs(raw_subset_root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    missing_paths = []
    broken_symlinks = []
    duplicate_output_basenames: Counter[tuple[str, str]] = Counter()
    counts_by_split_label: Counter[tuple[str, str]] = Counter()
    expected_windows_by_split_label: Counter[tuple[str, str]] = Counter()

    for row in rows:
        split = row["canonical_split"]
        label_name = LABEL_TO_NAME[row["label"]]
        edf_path = symlink_path_for_row(raw_subset_root, row)
        if not edf_path.exists():
            if edf_path.is_symlink():
                broken_symlinks.append(str(edf_path))
            else:
                missing_paths.append(str(edf_path))
        duplicate_output_basenames[(split, edf_path.stem)] += 1
        counts_by_split_label[(split, label_name)] += 1
        expected_windows_by_split_label[(split, label_name)] += int(row["n_h5_windows"])

    duplicates = {
        f"{split}/{basename}": count
        for (split, basename), count in duplicate_output_basenames.items()
        if count > 1
    }
    return {
        "raw_subset_root_exists": raw_subset_root.is_dir(),
        "selected_recordings": len(rows),
        "missing_path_count": len(missing_paths),
        "missing_path_examples": missing_paths[:20],
        "broken_symlink_count": len(broken_symlinks),
        "broken_symlink_examples": broken_symlinks[:20],
        "duplicate_output_basename_count": len(duplicates),
        "duplicate_output_basename_examples": dict(list(duplicates.items())[:20]),
        "recording_counts_by_split_label": nested_counter(counts_by_split_label),
        "expected_window_counts_by_split_label": nested_counter(expected_windows_by_split_label),
    }


def prepare_output_root(output_root: Path, force: bool) -> None:
    if output_root.exists():
        if not force:
            raise FileExistsError(f"output_root exists; use --force to replace: {output_root}")
        if output_root.is_file() or output_root.is_symlink():
            output_root.unlink()
        else:
            shutil.rmtree(output_root)
    for split in SPLITS:
        (output_root / split).mkdir(parents=True, exist_ok=True)


def process_one_recording(edf_path: Path, output_dir: Path, label: int, n_jobs_resample: int) -> dict[str, Any]:
    import mne

    raw = mne.io.read_raw_edf(str(edf_path), preload=True)
    try:
        useless_chs = [ch for ch in DROP_CHANNELS if ch in raw.ch_names]
        raw.drop_channels(useless_chs)
        if len(CH_ORDER_STANDARD) == len(raw.ch_names):
            raw.reorder_channels(CH_ORDER_STANDARD)
        if raw.ch_names != CH_ORDER_STANDARD:
            raise RuntimeError(f"channel order is wrong: {raw.ch_names}")

        raw.filter(l_freq=0.1, h_freq=75.0)
        raw.notch_filter(50.0)
        raw.resample(200, n_jobs=n_jobs_resample)
        channeled_data = raw.get_data(units="uV").copy()
    finally:
        raw.close()

    n_windows = channeled_data.shape[1] // 2000
    output_examples = []
    for i in range(n_windows):
        dump_path = output_dir / f"{edf_path.stem}_{i}.pkl"
        with dump_path.open("wb") as f:
            pickle.dump({"X": channeled_data[:, i * 2000 : (i + 1) * 2000], "y": label}, f)
        if len(output_examples) < 3:
            output_examples.append(str(dump_path))
    return {
        "edf_path": str(edf_path),
        "n_windows": int(n_windows),
        "output_examples": output_examples,
    }


def run_preprocessing(args: argparse.Namespace, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_subset_root = Path(args.raw_subset_root)
    output_root = Path(args.output_root)
    processed = []
    errors = []
    for row in rows:
        split = row["canonical_split"]
        edf_path = symlink_path_for_row(raw_subset_root, row)
        try:
            result = process_one_recording(edf_path, output_root / split, int(row["label"]), args.n_jobs_resample)
            result.update(
                {
                    "canonical_split": split,
                    "label": int(row["label"]),
                    "recording_id": row["recording_id"],
                    "expected_h5_windows": int(row["n_h5_windows"]),
                }
            )
            processed.append(result)
        except Exception as exc:
            errors.append(
                {
                    "canonical_split": split,
                    "label": int(row["label"]),
                    "recording_id": row["recording_id"],
                    "edf_path": str(edf_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return processed, errors


def write_reports(summary: dict[str, Any]) -> None:
    report_dir = Path(DEFAULT_REPORT_DIR)
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / SUMMARY_JSON
    report_path = report_dir / REPORT_MD
    summary["output_paths"]["summary_json"] = str(summary_path)
    summary["output_paths"]["report_md"] = str(report_path)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# EEGPT Option 1 Preprocessing Wrapper Report",
        "",
        f"- Status: `{summary['status']}`",
        f"- Dry run: `{summary['dry_run']}`",
        f"- Raw subset root: `{summary['inputs']['raw_subset_root']}`",
        f"- Recording manifest: `{summary['inputs']['recording_manifest']}`",
        f"- Output root: `{summary['inputs']['output_root']}`",
        "",
        "## Preserved EEGPT Behavior",
        "",
        "- Drops the same auxiliary channels as EEGPT `make_TUAB.py`.",
        "- Requires the same 23-channel TUAB order.",
        "- Applies bandpass `0.1-75 Hz`, notch `50 Hz`, resample `200 Hz`.",
        "- Writes non-overlapping 2000-sample PKL windows as `{'X': ..., 'y': ...}`.",
        "",
        "## Intentional Option 1 Change",
        "",
        "The wrapper preserves the approved canonical `train/val/test` split instead of EEGPT's original random subject train/val split.",
        "",
        "## Full Option 1 Subset Original Preprocessing Boundary",
        "",
        "Full mode means full Option 1 subset original preprocessing only. It does not read or process the full TUAB EDF root.",
        "",
        "## Planned / Actual Counts",
        "",
        json.dumps(summary["input_validation"]["expected_window_counts_by_split_label"], indent=2),
        "",
        "## Validation",
        "",
        json.dumps(summary["input_validation"], indent=2),
        "",
    ]
    if summary.get("processed_recordings"):
        lines.extend(["## Processed Recordings", "", json.dumps(summary["processed_recordings"][:20], indent=2), ""])
    if summary.get("errors"):
        lines.extend(["## Errors", "", json.dumps(summary["errors"][:20], indent=2), ""])
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    raw_subset_root = Path(args.raw_subset_root)
    recording_manifest = Path(args.recording_manifest)
    output_root = Path(args.output_root)

    all_rows = read_recording_manifest(recording_manifest)
    selected_rows = select_rows(all_rows, args.limit_recordings_per_split_label)
    validation = validate_inputs(raw_subset_root, selected_rows)

    serious_input_errors = [
        0 if validation["raw_subset_root_exists"] else 1,
        validation["missing_path_count"],
        validation["broken_symlink_count"],
        validation["duplicate_output_basename_count"],
    ]
    status = "DRY_RUN_PASS" if args.dry_run else "PASS"
    if any(serious_input_errors):
        status = "DRY_RUN_FAIL" if args.dry_run else "FAIL"

    processed: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    if not args.dry_run and status == "PASS":
        prepare_output_root(output_root, args.force)
        processed, errors = run_preprocessing(args, selected_rows)
        if errors:
            status = "FAIL"

    summary = {
        "status": status,
        "dry_run": bool(args.dry_run),
        "processing_launched": not args.dry_run and status in {"PASS", "FAIL"},
        "full_option1_subset_original_preprocessing": args.limit_recordings_per_split_label == 0,
        "inputs": {
            "raw_subset_root": str(raw_subset_root),
            "recording_manifest": str(recording_manifest),
            "output_root": str(output_root),
            "limit_recordings_per_split_label": args.limit_recordings_per_split_label,
        },
        "eegpt_original_preprocessing_choices": {
            "bandpass_hz": [0.1, 75.0],
            "notch_hz": 50.0,
            "resample_hz": 200,
            "window_samples": 2000,
            "window_seconds": 10.0,
            "channel_count": len(CH_ORDER_STANDARD),
            "channel_order": CH_ORDER_STANDARD,
        },
        "input_validation": validation,
        "processed_recordings": processed,
        "errors": errors,
        "output_paths": {"output_root": str(output_root)},
    }
    write_reports(summary)
    print(json.dumps(summary, indent=2))
    return 0 if status in {"PASS", "DRY_RUN_PASS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
