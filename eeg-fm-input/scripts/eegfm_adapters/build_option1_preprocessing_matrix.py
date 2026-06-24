#!/usr/bin/env python3
"""Build a lightweight preprocessing matrix for Option 1.

The matrix is derived from known wrapper/inspection documentation and source
text only. It does not inspect large processed data or run preprocessing.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("/nicoletye/workspace/unified_tuab")
SCRIPT_DIR = ROOT / "scripts" / "eegfm_adapters"
REPORT_DIR = ROOT / "reports"
OUT_CSV = REPORT_DIR / "eegfm_option1_preprocessing_matrix.csv"
OUT_MD = REPORT_DIR / "eegfm_option1_preprocessing_matrix.md"
OUT_JSON = REPORT_DIR / "eegfm_option1_preprocessing_matrix_summary.json"

MODELS = ["LaBraM", "EEGPT", "BIOT", "CBraMod", "CSBrain", "CodeBrain"]
FIELDS = [
    "model",
    "branch",
    "channel_count",
    "channel_type",
    "sampling_rate",
    "window_length_seconds",
    "samples_per_window",
    "bandpass",
    "notch",
    "normalization",
    "file_format",
    "input_shape",
    "split_behavior",
    "notes",
]


def file_exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if file_exists(path) else ""


def wrapper_notes(model: str) -> str:
    script = SCRIPT_DIR / f"{model.lower()}_option1_make_tuab_wrapper.py"
    text = read_text(script)
    notes = []
    if text:
        notes.append(f"wrapper={script.name}")
    if "raw EDF maker was found" in text or "does not ship" in text:
        notes.append("repo raw TUAB EDF maker not found; project-local format-compatible wrapper")
    return "; ".join(notes)


def unified_row(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "branch": "unified",
        "channel_count": 23,
        "channel_type": "canonical referential TUAB channels",
        "sampling_rate": "200 Hz",
        "window_length_seconds": 10,
        "samples_per_window": 2000,
        "bandpass": "0.1-75 Hz",
        "notch": "60 Hz",
        "normalization": "model adapter/training wrapper specific",
        "file_format": "matched subset NPZ from canonical H5",
        "input_shape": "[N, 23, 2000] before model adapter",
        "split_behavior": "fixed Option 1 matched train/val/test split",
        "notes": "Frozen canonical TUAB H5 pipeline; no preprocessing is run by this report.",
    }


ORIGINAL_ROWS: dict[str, dict[str, Any]] = {
    "LaBraM": {
        "channel_count": 23,
        "channel_type": "LaBraM original TUAB channel order",
        "sampling_rate": "200 Hz",
        "window_length_seconds": 10,
        "samples_per_window": 2000,
        "bandpass": "LaBraM original wrapper behavior",
        "notch": "LaBraM original wrapper behavior",
        "normalization": "LaBraM original downstream convention",
        "file_format": "PKL windows",
        "input_shape": "LaBraM original processed TUAB sample",
        "split_behavior": "Option 1 wrapper preserves fixed train/val/test",
        "notes": "Original LaBraM branch already completed; wrapper exists from earlier stage.",
    },
    "EEGPT": {
        "channel_count": 23,
        "channel_type": "EEGPT TUAB-compatible channels",
        "sampling_rate": "200 Hz",
        "window_length_seconds": 10,
        "samples_per_window": 2000,
        "bandpass": "EEGPT wrapper/repo-compatible preprocessing",
        "notch": "EEGPT wrapper/repo-compatible preprocessing",
        "normalization": "EEGPT wrapper/training convention",
        "file_format": "PKL/processed windows",
        "input_shape": "EEGPT downstream sample format",
        "split_behavior": "Option 1 wrapper preserves fixed train/val/test",
        "notes": "Original-vs-unified comparison complete.",
    },
    "BIOT": {
        "channel_count": 16,
        "channel_type": "bipolar channels",
        "sampling_rate": "200 Hz",
        "window_length_seconds": 10,
        "samples_per_window": 2000,
        "bandpass": "none",
        "notch": "none",
        "normalization": "per-channel 0.95-quantile normalization",
        "file_format": "PKL windows",
        "input_shape": "[N, 16, 2000]",
        "split_behavior": "Option 1 wrapper preserves fixed train/val/test",
        "notes": "BIOT original branch is better in final Option 1 result.",
    },
    "CBraMod": {
        "channel_count": 16,
        "channel_type": "bipolar channels",
        "sampling_rate": "200 Hz",
        "window_length_seconds": 10,
        "samples_per_window": 2000,
        "bandpass": "0.3-75 Hz",
        "notch": "60 Hz",
        "normalization": "divide by 100 in training wrapper",
        "file_format": "PKL windows",
        "input_shape": "[N, 16, 10, 200]",
        "split_behavior": "Option 1 wrapper preserves fixed train/val/test",
        "notes": "CBraMod original wrapper mirrors inspected TUAB transform.",
    },
    "CSBrain": {
        "channel_count": 16,
        "channel_type": "bipolar channels",
        "sampling_rate": "200 Hz",
        "window_length_seconds": 10,
        "samples_per_window": 2000,
        "bandpass": "0.3-75 Hz",
        "notch": "60 Hz",
        "normalization": "loader/training wrapper multiplies by 10000",
        "file_format": "PKL windows",
        "input_shape": "[N, 16, 10, 200]",
        "split_behavior": "Option 1 wrapper preserves fixed train/val/test",
        "notes": "No CSBrain TUAB raw EDF maker found; project-local format-compatible wrapper. Unified branch was chance-level.",
    },
    "CodeBrain": {
        "channel_count": 16,
        "channel_type": "bipolar channels",
        "sampling_rate": "200 Hz",
        "window_length_seconds": 10,
        "samples_per_window": 2000,
        "bandpass": "0.3-75 Hz",
        "notch": "60 Hz",
        "normalization": "loader/training wrapper divides by 100",
        "file_format": "PKL windows",
        "input_shape": "[N, 16, 10, 200]",
        "split_behavior": "Option 1 wrapper preserves fixed train/val/test",
        "notes": "No CodeBrain TUAB raw EDF maker found; wrapper uses adapter-side flattened TUAB forward.",
    },
}


def original_row(model: str) -> dict[str, Any]:
    row = {"model": model, "branch": "original"}
    row.update(ORIGINAL_ROWS[model])
    notes = wrapper_notes(model)
    if notes:
        row["notes"] = f"{row['notes']} {notes}"
    return row


def csv_value(value: Any) -> Any:
    return json.dumps(value, allow_nan=True) if isinstance(value, (list, dict)) else value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in FIELDS})


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# EEG-FM Option 1 Preprocessing Matrix",
        "",
        "Lightweight matrix from existing inspection reports and wrapper code. No processed data is inspected.",
        "",
        "| Model | Branch | Channels | Sampling | Window | Bandpass | Notch | Normalization | Input Shape | Notes |",
        "|---|---|---:|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['branch']} | {row['channel_count']} {row['channel_type']} | "
            f"{row['sampling_rate']} | {row['window_length_seconds']}s/{row['samples_per_window']} | "
            f"{row['bandpass']} | {row['notch']} | {row['normalization']} | `{row['input_shape']}` | {row['notes']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        rows.append(original_row(model))
        rows.append(unified_row(model))
    payload = {
        "status": "PASS",
        "outputs": {"csv": str(OUT_CSV), "md": str(OUT_MD), "json": str(OUT_JSON)},
        "source_policy": "Existing inspection reports and wrapper code only; no large processed data inspected.",
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    write_csv(OUT_CSV, rows)
    write_md(OUT_MD, rows)
    print(json.dumps(payload, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
