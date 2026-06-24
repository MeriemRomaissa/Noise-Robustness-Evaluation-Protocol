#!/usr/bin/env python3
"""CSBrain Option 1 TUAB preprocessing wrapper.

Project-local wrapper for the matched-subset comparison. CSBrain does not ship
a TUAB raw EDF maker in the inspected tree, so this preserves the repo's TUAB
PKL input contract while keeping the approved Option 1 train/val/test split and
never scanning the full TUAB root.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/nicoletye/workspace/unified_tuab")
DEFAULT_RAW_SUBSET_ROOT = ROOT / "data" / "tuab_option1_raw_edf_subset"
DEFAULT_RECORDING_MANIFEST = ROOT / "reports" / "tuab_original_vs_unified_matched_recording_manifest.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "csbrain_option1_original_processed"
REPORT_JSON = ROOT / "reports" / "csbrain_option1_preprocessing_wrapper_summary.json"
REPORT_MD = ROOT / "reports" / "csbrain_option1_preprocessing_wrapper_report.md"

SPLITS = ["train", "val", "test"]
LABEL_NAMES = {0: "normal", 1: "abnormal"}
BIPOLAR_PAIRS = [
    ("EEG FP1-REF", "EEG F7-REF"),
    ("EEG F7-REF", "EEG T3-REF"),
    ("EEG T3-REF", "EEG T5-REF"),
    ("EEG T5-REF", "EEG O1-REF"),
    ("EEG FP2-REF", "EEG F8-REF"),
    ("EEG F8-REF", "EEG T4-REF"),
    ("EEG T4-REF", "EEG T6-REF"),
    ("EEG T6-REF", "EEG O2-REF"),
    ("EEG FP1-REF", "EEG F3-REF"),
    ("EEG F3-REF", "EEG C3-REF"),
    ("EEG C3-REF", "EEG P3-REF"),
    ("EEG P3-REF", "EEG O1-REF"),
    ("EEG FP2-REF", "EEG F4-REF"),
    ("EEG F4-REF", "EEG C4-REF"),
    ("EEG C4-REF", "EEG P4-REF"),
    ("EEG P4-REF", "EEG O2-REF"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CSBrain Option 1 TUAB preprocessing wrapper.")
    parser.add_argument("--raw_subset_root", default=str(DEFAULT_RAW_SUBSET_ROOT))
    parser.add_argument("--recording_manifest", default=str(DEFAULT_RECORDING_MANIFEST))
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit_recordings_per_split_label", type=int, default=0)
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    required = {"canonical_split", "label", "edf_basename", "recording_id"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"recording manifest missing columns: {sorted(missing)}")
    return rows


def subset_edf_path(raw_subset_root: Path, row: dict[str, Any]) -> Path:
    label = int(row["label"])
    return raw_subset_root / row["canonical_split"] / LABEL_NAMES[label] / row["edf_basename"]


def selected_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return rows
    counts: Counter[tuple[str, int]] = Counter()
    selected = []
    for row in rows:
        key = (row["canonical_split"], int(row["label"]))
        if counts[key] < limit:
            selected.append(row)
            counts[key] += 1
    return selected


def summarize_rows(rows: list[dict[str, Any]], raw_subset_root: Path) -> dict[str, Any]:
    split_label_counts: Counter[str] = Counter()
    missing_sources = []
    for row in rows:
        split_label_counts[f"{row['canonical_split']}/label_{row['label']}"] += 1
        path = subset_edf_path(raw_subset_root, row)
        if not path.exists():
            missing_sources.append(str(path))
    return {
        "recording_count": len(rows),
        "split_label_recording_counts": dict(sorted(split_label_counts.items())),
        "missing_source_count": len(missing_sources),
        "missing_sources": missing_sources[:20],
    }


def prepare_output_root(output_root: Path, force: bool) -> None:
    if output_root.exists():
        if not force:
            raise FileExistsError(f"output_root exists; pass --force to replace: {output_root}")
        shutil.rmtree(output_root)
    for split in SPLITS:
        (output_root / split).mkdir(parents=True, exist_ok=True)


def channel_index(ch_names: list[str], wanted: str) -> int:
    if wanted in ch_names:
        return ch_names.index(wanted)
    lookup = {name.upper().replace("EEG ", "EEG "): idx for idx, name in enumerate(ch_names)}
    key = wanted.upper()
    if key in lookup:
        return lookup[key]
    raise KeyError(wanted)


def process_one_edf(edf_path: Path, output_dir: Path, label: int) -> dict[str, Any]:
    import mne

    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    raw.resample(200, verbose="ERROR")
    raw.filter(l_freq=0.3, h_freq=75, verbose="ERROR")
    raw.notch_filter((60), verbose="ERROR")
    raw_data = raw.get_data(units="uV")
    ch_names = raw.ch_names
    channeled = np.zeros((16, raw_data.shape[1]), dtype=np.float32)
    for idx, (left, right) in enumerate(BIPOLAR_PAIRS):
        channeled[idx] = raw_data[channel_index(ch_names, left)] - raw_data[channel_index(ch_names, right)]

    n_windows = int(channeled.shape[1] // 2000)
    written = []
    stem = edf_path.stem
    for i in range(n_windows):
        out_path = output_dir / f"{stem}_{i}.pkl"
        with out_path.open("wb") as f:
            pickle.dump({"X": channeled[:, i * 2000 : (i + 1) * 2000], "y": int(label)}, f)
        written.append(str(out_path))
    return {"edf_path": str(edf_path), "windows": n_windows, "written_examples": written[:3]}


def write_reports(summary: dict[str, Any]) -> None:
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    lines = [
        "# CSBrain Option 1 Preprocessing Wrapper Report",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Dry run: `{summary.get('dry_run')}`",
        f"- Raw subset root: `{summary.get('raw_subset_root')}`",
        f"- Output root: `{summary.get('output_root')}`",
        "",
        "## CSBrain TUAB-Compatible Transform",
        "",
        "- No CSBrain TUAB raw EDF maker was found in the inspected repo tree.",
        "- The repo TUAB loader expects PKL files under `train`, `val`, and `test`.",
        "- Each PKL contains `X` and `y`; the loader reshapes `X` to `(16, 10, 200)` and multiplies by 10000.",
        "- Resample to 200 Hz.",
        "- Bandpass filter: 0.3-75 Hz.",
        "- Notch filter: 60 Hz.",
        "- Convert to microvolts.",
        "- Build 16 bipolar channels.",
        "- Save 10-second windows as PKL dicts with `X` shape `(16, 2000)` and integer `y`.",
        "",
        "## Summary JSON",
        "",
        "```json",
        json.dumps(summary, indent=2, allow_nan=True),
        "```",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    raw_subset_root = Path(args.raw_subset_root)
    manifest_path = Path(args.recording_manifest)
    output_root = Path(args.output_root)
    try:
        rows_all = load_manifest(manifest_path)
        rows = selected_rows(rows_all, args.limit_recordings_per_split_label)
        pre_summary = summarize_rows(rows, raw_subset_root)
        summary: dict[str, Any] = {
            "status": "DRY_RUN_PASS" if args.dry_run else "PASS",
            "dry_run": bool(args.dry_run),
            "raw_subset_root": str(raw_subset_root),
            "recording_manifest": str(manifest_path),
            "output_root": str(output_root),
            "limit_recordings_per_split_label": args.limit_recordings_per_split_label,
            "full_option1_subset_only": args.limit_recordings_per_split_label <= 0,
            "pre_validation": pre_summary,
            "processed_recordings": [],
            "processed_window_counts": {},
            "errors": [],
        }
        if pre_summary["missing_source_count"]:
            raise FileNotFoundError(f"missing source EDF paths: {pre_summary['missing_sources'][:5]}")
        if args.dry_run:
            write_reports(summary)
            print(json.dumps(summary, indent=2, allow_nan=True))
            return 0

        prepare_output_root(output_root, args.force)
        window_counts: Counter[str] = Counter()
        for row in rows:
            split = row["canonical_split"]
            label = int(row["label"])
            src = subset_edf_path(raw_subset_root, row)
            try:
                result = process_one_edf(src, output_root / split, label)
                result.update({"split": split, "label": label, "recording_id": row["recording_id"]})
                summary["processed_recordings"].append(result)
                window_counts[f"{split}/label_{label}"] += int(result["windows"])
            except Exception as exc:
                summary["errors"].append(
                    {
                        "recording_id": row.get("recording_id"),
                        "edf_path": str(src),
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
        summary["processed_window_counts"] = dict(sorted(window_counts.items()))
        summary["status"] = "PASS" if not summary["errors"] else "FAIL_WITH_ERRORS"
        write_reports(summary)
        print(json.dumps(summary, indent=2, allow_nan=True))
        return 0 if summary["status"] == "PASS" else 1
    except Exception as exc:
        summary = {
            "status": "FAIL_WITH_REASON",
            "dry_run": bool(args.dry_run),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "raw_subset_root": str(raw_subset_root),
            "recording_manifest": str(manifest_path),
            "output_root": str(output_root),
        }
        write_reports(summary)
        print(json.dumps(summary, indent=2, allow_nan=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
