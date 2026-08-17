#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Preprocess NMT-Scalp-EEG as a LaBraM-style OOD test set.

Goal
----
Read all NMT EDF files from both train/eval folders, merge them into one OOD
`test` output folder, and save one pickle per non-overlapping 10-second segment.

Output pickle format per segment:
    {
        "X": np.ndarray, shape (23, 2000), dtype float32, unit microvolts,
        "y": int, 0 for normal, 1 for abnormal/pathological
    }

Default input root expected on Windows:
    E:\NMT-Scalp-EEG\nmt_scalp_eeg_dataset

Required packages:
    pip install mne pandas numpy tqdm
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import mne
except ImportError as exc:
    raise ImportError(
        "MNE is required to read EDF files. Install it with: pip install mne"
    ) from exc


# LaBraM/TUAB target channel order required by the technical instruction.
LABRAM_CHANNELS: List[str] = [
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "A1", "A2", "FZ", "CZ", "PZ",
    "T1", "T2",
]

# NMT is expected to have 19 standard scalp channels. These four are commonly
# missing when converting to the 23-channel LaBraM/TUAB order.
EXPECTED_ZERO_PADDED_CHANNELS = {"A1", "A2", "T1", "T2"}

# Mapping for possible modern/alternative EEG names to old 10-20 names used by TUAB/LaBraM.
# Example: T7/T8/P7/P8 are common modern names corresponding roughly to T3/T4/T5/T6.
CHANNEL_SYNONYMS: Dict[str, str] = {
    "FP1": "FP1", "FP2": "FP2", "F3": "F3", "F4": "F4", "C3": "C3", "C4": "C4",
    "P3": "P3", "P4": "P4", "O1": "O1", "O2": "O2", "F7": "F7", "F8": "F8",
    "T3": "T3", "T4": "T4", "T5": "T5", "T6": "T6", "A1": "A1", "A2": "A2",
    "FZ": "FZ", "CZ": "CZ", "PZ": "PZ", "T1": "T1", "T2": "T2",
    "T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6", "M1": "A1", "M2": "A2",
}

LABEL_MAP = {
    "normal": 0,
    "abnormal": 1,
    "pathological": 1,
    "pathologic": 1,
}


def normalize_channel_name(name: str) -> Optional[str]:
    """Return canonical LaBraM/TUAB channel name, or None if not an EEG target channel."""
    original = str(name).strip().upper()
    original = original.replace("EEG", " ")
    original = original.replace(".", "")
    original = original.replace(" ", "")

    # First try direct simplified name.
    if original in CHANNEL_SYNONYMS:
        return CHANNEL_SYNONYMS[original]

    # Then try tokens from names like "EEG FP1-REF", "FP1-A1", "FZ_AVG".
    tokens = re.split(r"[-_/\\:;,+()\s]+", str(name).strip().upper())
    for token in tokens:
        token = token.replace("EEG", "").replace(".", "").strip()
        if token in CHANNEL_SYNONYMS:
            return CHANNEL_SYNONYMS[token]

    # Finally try a regex search, longest first so FP1 is not confused with P1.
    for key in sorted(CHANNEL_SYNONYMS.keys(), key=len, reverse=True):
        if re.search(rf"(^|[^A-Z0-9]){re.escape(key)}([^A-Z0-9]|$)", str(name).upper()):
            return CHANNEL_SYNONYMS[key]

    return None


def build_channel_index(raw_channel_names: List[str]) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
    """Build canonical channel -> raw index mapping, also returning duplicate raw names."""
    canonical_to_index: Dict[str, int] = {}
    duplicates: Dict[str, List[str]] = defaultdict(list)

    for idx, raw_name in enumerate(raw_channel_names):
        canonical = normalize_channel_name(raw_name)
        if canonical is None:
            continue
        if canonical not in canonical_to_index:
            canonical_to_index[canonical] = idx
        else:
            duplicates[canonical].append(raw_name)

    return canonical_to_index, duplicates


def resolve_edf_path(nmt_root: Path, recordname: str, label: str, loc: str) -> Path:
    """Resolve EDF path using the NMT folder layout: root/label/loc/recordname."""
    label_dir = str(label).strip().lower()
    loc_dir = str(loc).strip().lower()
    return nmt_root / label_dir / loc_dir / str(recordname).strip()


def ensure_microvolts(raw: "mne.io.BaseRaw") -> np.ndarray:
    """
    Return raw data in microvolts.

    MNE internally stores EEG in volts. In newer MNE versions, get_data(units="uV")
    is available. For older versions, multiplying by 1e6 gives microvolts.
    """
    try:
        data_uv = raw.get_data(units="uV")
    except TypeError:
        data_uv = raw.get_data() * 1e6
    return data_uv


def load_preprocess_edf(
    edf_path: Path,
    target_sfreq: float = 200.0,
    l_freq: float = 0.1,
    h_freq: float = 75.0,
    notch_freq: float = 50.0,
) -> Tuple[np.ndarray, List[str], Dict[str, int], Dict[str, List[str]], float]:
    """Read one EDF, apply LaBraM-aligned preprocessing, and return full data in microvolts."""
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    raw_channel_names = list(raw.ch_names)

    # Channel mapping is detected before filtering/resampling so we can report original EDF names.
    canonical_to_index, duplicates = build_channel_index(raw_channel_names)

    # LaBraM-aligned preprocessing order.
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", n_jobs=1, verbose="ERROR")
    raw.notch_filter(freqs=[notch_freq], n_jobs=1, verbose="ERROR")
    if abs(float(raw.info["sfreq"]) - target_sfreq) > 1e-6:
        raw.resample(target_sfreq, npad="auto", n_jobs=1, verbose="ERROR")

    sfreq_after = float(raw.info["sfreq"])
    data_uv_all_channels = ensure_microvolts(raw)

    return data_uv_all_channels, raw_channel_names, canonical_to_index, duplicates, sfreq_after


def align_to_labram_23(
    data_uv_all_channels: np.ndarray,
    canonical_to_index: Dict[str, int],
    n_times: int,
) -> Tuple[np.ndarray, List[str], Dict[str, Optional[int]]]:
    """Create a 23-channel array in exact LaBraM order, zero-padding missing channels."""
    aligned = np.zeros((len(LABRAM_CHANNELS), n_times), dtype=np.float32)
    missing: List[str] = []
    used_indices: Dict[str, Optional[int]] = {}

    for out_idx, ch in enumerate(LABRAM_CHANNELS):
        src_idx = canonical_to_index.get(ch)
        used_indices[ch] = src_idx
        if src_idx is None:
            missing.append(ch)
            continue
        aligned[out_idx, :] = data_uv_all_channels[src_idx, :].astype(np.float32, copy=False)

    return aligned, missing, used_indices


def save_pickle(path: Path, X: np.ndarray, y: int) -> None:
    item = {"X": X.astype(np.float32, copy=False), "y": int(y)}
    with open(path, "wb") as f:
        pickle.dump(item, f, protocol=pickle.HIGHEST_PROTOCOL)


def write_text_report(report_path: Path, report: dict) -> None:
    lines: List[str] = []
    lines.append("NMT -> LaBraM OOD preprocessing verification report")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Summary of changes made:")
    for item in report["summary_of_changes"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append(f"Input root: {report['input_root']}")
    lines.append(f"Output test folder: {report['output_test_folder']}")
    lines.append(f"Total subjects/records processed: {report['total_subjects_processed']}")
    lines.append(f"Total segments saved: {report['total_segments_saved']}")
    lines.append(f"Output shape confirmed: {report['output_shape_confirmed']}")
    lines.append("")
    lines.append("Class distribution by record:")
    for k, v in report["class_distribution_records"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("Class distribution by segment:")
    for k, v in report["class_distribution_segments"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("Missing/zero-padded channel frequency across processed records:")
    for k, v in report["missing_channel_frequency"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append(f"Errors/skipped records: {len(report['errors'])}")
    if report["errors"]:
        for err in report["errors"][:20]:
            lines.append(f"- {err}")
        if len(report["errors"]) > 20:
            lines.append(f"... {len(report['errors']) - 20} more errors omitted in text report.")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess NMT EDF files into LaBraM/TUAB-style OOD test pickle segments."
    )
    parser.add_argument(
        "--nmt_root",
        type=str,
        default=r"E:\NMT-Scalp-EEG\nmt_scalp_eeg_dataset",
        help="Path to NMT root containing Labels.csv, normal/, abnormal/.",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default=r"E:\NMT-Scalp-EEG\nmt_scalp_eeg_labram_ood",
        help="Output root. Pickles will be saved under out_root/test.",
    )
    parser.add_argument("--target_sfreq", type=float, default=200.0)
    parser.add_argument("--segment_seconds", type=float, default=10.0)
    parser.add_argument("--l_freq", type=float, default=0.1)
    parser.add_argument("--h_freq", type=float, default=75.0)
    parser.add_argument("--notch_freq", type=float, default=50.0)
    parser.add_argument("--overwrite", type=int, default=0, help="1 = overwrite existing segment pickle files.")
    parser.add_argument("--max_records", type=int, default=None, help="Optional debug limit, e.g. 5.")
    parser.add_argument("--dry_run", type=int, default=0, help="1 = only check files/channel mapping, do not save pickles.")
    args = parser.parse_args()

    mne.set_log_level("ERROR")

    nmt_root = Path(args.nmt_root)
    out_root = Path(args.out_root)
    out_test = out_root / "test"
    out_test.mkdir(parents=True, exist_ok=True)

    labels_path = nmt_root / "Labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels.csv not found: {labels_path}")

    df = pd.read_csv(labels_path)
    required_cols = {"recordname", "label", "loc"}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise ValueError(f"Labels.csv is missing required columns: {sorted(missing_cols)}")

    if args.max_records is not None:
        df = df.head(args.max_records).copy()

    seg_len = int(round(args.target_sfreq * args.segment_seconds))
    expected_shape = (len(LABRAM_CHANNELS), seg_len)

    print("\n================ NMT -> LaBraM OOD preprocessing ================")
    print(f"Input root      : {nmt_root}")
    print(f"Output test dir : {out_test}")
    print(f"Records in CSV  : {len(df)}")
    print(f"Target channels : {LABRAM_CHANNELS}")
    print(f"Target shape    : {expected_shape}")
    print("==================================================================\n")

    processed_index_rows: List[dict] = []
    channel_rows: List[dict] = []
    errors: List[str] = []
    class_records = Counter()
    class_segments = Counter()
    missing_channel_frequency = Counter()
    processed_records = 0
    total_segments = 0
    first_mapping_printed = False

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing EDF records"):
        recordname = str(row["recordname"]).strip()
        label_text = str(row["label"]).strip().lower()
        loc = str(row["loc"]).strip().lower()
        y = LABEL_MAP.get(label_text)
        if y is None:
            errors.append(f"{recordname}: unknown label '{label_text}'")
            continue

        edf_path = resolve_edf_path(nmt_root, recordname, label_text, loc)
        if not edf_path.exists():
            errors.append(f"{recordname}: EDF not found at {edf_path}")
            continue

        try:
            data_uv_all, raw_ch_names, canonical_to_index, duplicates, sfreq_after = load_preprocess_edf(
                edf_path=edf_path,
                target_sfreq=args.target_sfreq,
                l_freq=args.l_freq,
                h_freq=args.h_freq,
                notch_freq=args.notch_freq,
            )

            n_times = data_uv_all.shape[1]
            aligned, missing_channels, used_indices = align_to_labram_23(
                data_uv_all_channels=data_uv_all,
                canonical_to_index=canonical_to_index,
                n_times=n_times,
            )

            for ch in missing_channels:
                missing_channel_frequency[ch] += 1

            if not first_mapping_printed:
                print("\n---------------- First EDF channel check ----------------")
                print(f"Record: {recordname}")
                print(f"NMT raw EDF channel names ({len(raw_ch_names)}):")
                print(raw_ch_names)
                print(f"\nDetected canonical NMT channels ({len(canonical_to_index)}):")
                print(sorted(canonical_to_index.keys()))
                print(f"\nLaBraM required channel order ({len(LABRAM_CHANNELS)}):")
                print(LABRAM_CHANNELS)
                print(f"\nMissing channels zero-padded for this record ({len(missing_channels)}):")
                print(missing_channels)
                if duplicates:
                    print("\nDuplicate canonical mappings detected:")
                    print(dict(duplicates))
                print("---------------------------------------------------------\n")
                first_mapping_printed = True

            # Record channel mapping for verification.
            for labram_ch in LABRAM_CHANNELS:
                src_idx = used_indices[labram_ch]
                source_name = raw_ch_names[src_idx] if src_idx is not None else "ZERO_PADDED"
                channel_rows.append({
                    "recordname": recordname,
                    "labram_channel": labram_ch,
                    "source_edf_channel": source_name,
                    "status": "mapped" if src_idx is not None else "zero_padded",
                })

            n_segments = n_times // seg_len
            if n_segments <= 0:
                errors.append(f"{recordname}: too short after preprocessing, n_times={n_times}")
                continue

            class_records[label_text] += 1
            processed_records += 1

            if args.dry_run:
                class_segments[label_text] += n_segments
                total_segments += n_segments
                continue

            record_stem = Path(recordname).stem
            for seg_idx in range(n_segments):
                start = seg_idx * seg_len
                end = start + seg_len
                X = aligned[:, start:end]

                if X.shape != expected_shape:
                    raise ValueError(f"Unexpected segment shape {X.shape}; expected {expected_shape}")
                if X.dtype != np.float32:
                    X = X.astype(np.float32)

                out_name = f"{record_stem}_seg{seg_idx:06d}.pkl"
                out_path = out_test / out_name
                if out_path.exists() and not args.overwrite:
                    # Count it as already available, but do not rewrite.
                    pass
                else:
                    save_pickle(out_path, X, y)

                processed_index_rows.append({
                    "recordname": recordname,
                    "source_edf_path": str(edf_path),
                    "label_text": label_text,
                    "y": y,
                    "loc_original": loc,
                    "segment_index": seg_idx,
                    "start_sample": start,
                    "end_sample": end,
                    "sfreq": sfreq_after,
                    "output_pickle": str(out_path),
                    "shape": str(tuple(X.shape)),
                    "dtype": str(X.dtype),
                    "unit": "microvolts",
                    "missing_zero_padded_channels": ";".join(missing_channels),
                })

            class_segments[label_text] += n_segments
            total_segments += n_segments

        except Exception as exc:  # Continue processing remaining records.
            errors.append(f"{recordname}: {type(exc).__name__}: {exc}")
            continue

    # Save verification artifacts.
    out_root.mkdir(parents=True, exist_ok=True)

    index_csv = out_root / "processed_index.csv"
    channel_csv = out_root / "channel_mapping_report.csv"
    errors_csv = out_root / "errors.csv"

    if processed_index_rows:
        pd.DataFrame(processed_index_rows).to_csv(index_csv, index=False, encoding="utf-8-sig")
    if channel_rows:
        pd.DataFrame(channel_rows).to_csv(channel_csv, index=False, encoding="utf-8-sig")
    with open(errors_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["error"])
        for err in errors:
            writer.writerow([err])

    report = {
        "summary_of_changes": [
            "Merged NMT train/eval folders into a single OOD test output folder.",
            "Mapped NMT EDF channel names to the exact 23-channel LaBraM/TUAB order.",
            "Zero-padded missing LaBraM channels instead of dropping or reordering target channels.",
            "Applied preprocessing in the required order: 0.1-75 Hz bandpass, 50 Hz notch, 200 Hz resampling.",
            "Kept EEG values in microvolts and did not apply normalization.",
            "Segmented each recording into non-overlapping 10-second windows of 2000 samples.",
            "Saved one pickle per segment using the TUAB-style dictionary format: {'X': (23, 2000) float32, 'y': int}.",
        ],
        "input_root": str(nmt_root),
        "output_root": str(out_root),
        "output_test_folder": str(out_test),
        "total_subjects_processed": int(processed_records),
        "total_segments_saved": int(total_segments),
        "class_distribution_records": dict(class_records),
        "class_distribution_segments": dict(class_segments),
        "missing_channel_frequency": dict(sorted(missing_channel_frequency.items())),
        "expected_zero_padded_channels": sorted(EXPECTED_ZERO_PADDED_CHANNELS),
        "labram_channel_order": LABRAM_CHANNELS,
        "output_shape_confirmed": str(expected_shape),
        "errors": errors,
        "dry_run": bool(args.dry_run),
    }

    report_json = out_root / "verification_report.json"
    report_txt = out_root / "verification_report.txt"
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_text_report(report_txt, report)

    print("\n================ Verification summary ================")
    print(f"Total records processed : {processed_records}")
    print(f"Total segments saved    : {total_segments}")
    print(f"Record class counts     : {dict(class_records)}")
    print(f"Segment class counts    : {dict(class_segments)}")
    print(f"Missing channel freq    : {dict(sorted(missing_channel_frequency.items()))}")
    print(f"Output shape confirmed  : {expected_shape}")
    print(f"Processed index         : {index_csv}")
    print(f"Channel report          : {channel_csv}")
    print(f"Verification report     : {report_txt}")
    print(f"Errors report           : {errors_csv} ({len(errors)} errors/skipped)")
    print("======================================================\n")


if __name__ == "__main__":
    main()
