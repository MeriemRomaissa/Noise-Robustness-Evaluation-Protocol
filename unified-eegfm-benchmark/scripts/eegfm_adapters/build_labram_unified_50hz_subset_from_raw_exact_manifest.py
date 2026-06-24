#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a manifest-exact unified LaBraM 50 Hz notch subset from raw TUAB EDFs.

This script reuses the canonical TUAB builder module that produced the working
unified 60 Hz H5. It changes only the module-level notch frequency before
calling the canonical EDF processor, then selects the exact windows requested by
the LaBraM exact subset manifest.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANONICAL_BUILDER = Path(
    "/nicoletye/workspace/Noise Robustness Evaluation Protocol/Unified Preprocessing/build_canonical_tuab.py"
)
if not DEFAULT_CANONICAL_BUILDER.exists():
    DEFAULT_CANONICAL_BUILDER = Path("/nicoletye/build_canonical_tuab.py")

DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "labram_notch_ablation_subset_v1" / "labram_notch_ablation_exact_subset_manifest.csv"
DEFAULT_RAW_ROOT = Path("/nicoletye/workspace/tuh_data/TUAB/v3.0.1/edf")
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "labram_notch_ablation_subset_v1" / "unified_50hz" / "unified_50hz_subset.npz"
DEFAULT_DEBUG_OUTPUT = PROJECT_ROOT / "outputs" / "labram_notch_ablation_subset_v1" / "unified_50hz" / "debug_unified_50hz_tiny_subset.npz"
SPLITS = ("train", "val", "test")


def load_canonical_builder(path: Path):
    spec = importlib.util.spec_from_file_location("canonical_tuab_builder_reused_for_50hz_subset", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load canonical builder from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("build_labram_unified_50hz_subset")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(out_dir / "build_unified_50hz_subset.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            item = dict(row)
            item["label"] = int(item["label"])
            item["h5_index"] = int(item["h5_index"])
            item["window_idx"] = int(item["window_idx"])
            item["start_time"] = float(item.get("start_time") or item.get("h5_start") or 0.0)
            item["end_time"] = float(item.get("end_time") or item.get("h5_end") or 0.0)
            rows.append(item)
    return rows


def apply_limit_per_split(rows: list[dict[str, Any]], limit_per_split: int | None) -> list[dict[str, Any]]:
    if not limit_per_split or limit_per_split <= 0:
        return rows
    kept: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for row in rows:
        split = row["canonical_split"]
        if counts[split] < limit_per_split:
            kept.append(row)
            counts[split] += 1
    return kept


def manifest_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_split = Counter(row["canonical_split"] for row in rows)
    by_split_label = Counter((row["canonical_split"], int(row["label"])) for row in rows)
    return {
        "total": len(rows),
        "by_split": {split: int(by_split.get(split, 0)) for split in SPLITS},
        "by_split_label": {
            f"{split}_label_{label}": int(by_split_label.get((split, label), 0))
            for split in SPLITS
            for label in (0, 1)
        },
    }


def parse_window_idx_from_sample_id(sample_id: str) -> int:
    try:
        return int(str(sample_id).rsplit("_", 1)[-1])
    except Exception as exc:
        raise ValueError(f"cannot parse window_idx from sample_id={sample_id!r}") from exc


def rows_by_edf(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source = row.get("source_path") or row.get("edf_path")
        if not source:
            raise ValueError(f"manifest row missing source_path/edf_path: {row}")
        grouped[str(source)].append(row)
    return dict(grouped)


def validate_source_paths(grouped: dict[str, list[dict[str, Any]]], raw_edf_root: Path) -> list[str]:
    missing = []
    root = raw_edf_root.resolve()
    for source in grouped:
        p = Path(source)
        if not p.exists():
            missing.append(source)
            continue
        try:
            p.resolve().relative_to(root)
        except Exception:
            # Keep this as a warning in metadata rather than rejecting older absolute manifests.
            pass
    return missing


def build_records(args: argparse.Namespace, logger: logging.Logger) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    builder_path = Path(args.canonical_builder)
    canonical = load_canonical_builder(builder_path)
    canonical.NOTCH_HZ = float(args.notch_freq)

    rows_all = read_manifest(Path(args.subset_manifest))
    rows = apply_limit_per_split(rows_all, args.limit_per_split)
    grouped = rows_by_edf(rows)
    missing_sources = validate_source_paths(grouped, Path(args.raw_edf_root))
    if missing_sources:
        raise FileNotFoundError(f"missing EDF source paths: {missing_sources[:10]}")

    logger.info("canonical_builder=%s", builder_path)
    logger.info("notch_hz=%s", canonical.NOTCH_HZ)
    logger.info("selected_manifest_rows=%d unique_edfs=%d", len(rows), len(grouped))
    logger.info("selected_counts=%s", json.dumps(manifest_counts(rows), sort_keys=True))

    built: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for edf_i, (source, source_rows) in enumerate(sorted(grouped.items()), start=1):
        edf_path = Path(source)
        logger.info("PROCESS_EDF %d/%d path=%s selected_windows=%d", edf_i, len(grouped), edf_path, len(source_rows))
        result = canonical.process_edf(edf_path, logger)
        if result is None:
            failures.append({"source_path": source, "error": "canonical process_edf returned None"})
            continue
        windows, metadatas, missing_channels = result
        meta_by_window_idx = {parse_window_idx_from_sample_id(m["sample_id"]): (windows[i], m) for i, m in enumerate(metadatas)}

        for row in source_rows:
            window_idx = int(row["window_idx"])
            if window_idx not in meta_by_window_idx:
                failures.append({
                    "source_path": source,
                    "recording_id": row.get("recording_id", ""),
                    "window_idx": window_idx,
                    "error": "selected window not produced by canonical process_edf",
                })
                continue
            window, meta = meta_by_window_idx[window_idx]
            if int(row["label"]) != int(meta["label"]):
                failures.append({
                    "source_path": source,
                    "window_idx": window_idx,
                    "error": f"label mismatch manifest={row['label']} processed={meta['label']}",
                })
                continue
            if str(row["recording_id"]) != str(meta["recording_id"]):
                failures.append({
                    "source_path": source,
                    "window_idx": window_idx,
                    "error": f"recording mismatch manifest={row['recording_id']} processed={meta['recording_id']}",
                })
                continue
            built.append({
                "row": row,
                "window": window.astype(np.float32, copy=False),
                "processed_metadata": meta,
                "missing_channels": missing_channels,
            })

    if failures:
        raise RuntimeError(f"window extraction failures: {json.dumps(failures[:20], indent=2)}")

    order = {int(row["h5_index"]): i for i, row in enumerate(rows)}
    built.sort(key=lambda item: order[int(item["row"]["h5_index"])])

    summary = {
        "canonical_builder": str(builder_path),
        "canonical_preprocessing_reused": True,
        "changed_from_canonical": {"notch_hz": float(args.notch_freq)},
        "requested_rows": len(rows),
        "built_rows": len(built),
        "unique_edfs": len(grouped),
        "manifest_counts": manifest_counts(rows),
        "missing_source_count": len(missing_sources),
        "failures": failures,
    }
    return built, summary


def save_npz(args: argparse.Namespace, built: list[dict[str, Any]], summary: dict[str, Any], logger: logging.Logger) -> dict[str, Any]:
    out = Path(args.output_npz)
    out.parent.mkdir(parents=True, exist_ok=True)

    eeg = np.stack([item["window"] for item in built], axis=0).astype(np.float32)
    rows = [item["row"] for item in built]
    processed = [item["processed_metadata"] for item in built]

    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    splits = np.asarray([row["canonical_split"] for row in rows])
    h5_indices = np.asarray([int(row["h5_index"]) for row in rows], dtype=np.int64)
    window_idx = np.asarray([int(row["window_idx"]) for row in rows], dtype=np.int64)
    recording_id = np.asarray([row["recording_id"] for row in rows])
    subject = np.asarray([row.get("subject_id", "") for row in rows])
    sample_id = np.asarray([row.get("sample_id", meta["sample_id"]) for row, meta in zip(rows, processed)])
    source_path = np.asarray([row.get("source_path", meta["source_path"]) for row, meta in zip(rows, processed)])
    start_time = np.asarray([float(row.get("start_time", meta["start_time"])) for row, meta in zip(rows, processed)], dtype=np.float32)
    end_time = np.asarray([float(row.get("end_time", meta["end_time"])) for row, meta in zip(rows, processed)], dtype=np.float32)
    raw_hash = np.asarray([meta["raw_hash"] for meta in processed])

    payload: dict[str, Any] = {
        "eeg": eeg,
        "label": labels,
        "split": splits,
        "recording_id": recording_id,
        "window_idx": window_idx,
        "subject": subject,
        "subject_id": subject,
        "sample_id": sample_id,
        "source_path": source_path,
        "start_time": start_time,
        "end_time": end_time,
        "raw_hash": raw_hash,
        "h5_indices": h5_indices,
        "channel_names": np.asarray(summary["channel_names"]),
        "split_counts_json": np.asarray(json.dumps(summary["actual_counts"], sort_keys=True)),
        "matched_manifest_json": np.asarray(json.dumps([{k: str(v) for k, v in row.items()} for row in rows])),
    }

    for split in SPLITS:
        idx = np.where(splits == split)[0]
        payload[f"{split}_x"] = eeg[idx]
        payload[f"{split}_y"] = labels[idx]
        payload[f"{split}_h5_indices"] = h5_indices[idx]
        payload[f"{split}_recording_id"] = recording_id[idx]
        payload[f"{split}_window_idx"] = window_idx[idx]
        payload[f"{split}_source_path"] = source_path[idx]

    np.savez_compressed(out, **payload)
    logger.info("WROTE_NPZ path=%s shape=%s", out, list(eeg.shape))
    return {"output_npz": str(out), "shape": list(eeg.shape)}


def validate_output(args: argparse.Namespace, built: list[dict[str, Any]], canonical_module, summary: dict[str, Any]) -> dict[str, Any]:
    rows = [item["row"] for item in built]
    actual_counts = manifest_counts(rows)
    requested_counts = summary["manifest_counts"]
    h5_indices = [int(row["h5_index"]) for row in rows]
    duplicate_h5_indices = len(h5_indices) - len(set(h5_indices))
    expected_n = requested_counts["total"]
    actual_n = len(built)
    shape = [actual_n, int(canonical_module.N_CHANNELS), int(canonical_module.WINDOW_SAMPLES)]

    problems = []
    if actual_n != expected_n:
        problems.append(f"count mismatch actual={actual_n} expected={expected_n}")
    if actual_counts != requested_counts:
        problems.append("split/label counts differ from selected manifest rows")
    if duplicate_h5_indices:
        problems.append(f"duplicate h5 indices: {duplicate_h5_indices}")
    if shape[1:] != [23, 2000]:
        problems.append(f"unexpected channel/sample shape: {shape}")
    if not args.debug and not args.limit_per_split and shape != [12288, 23, 2000]:
        problems.append(f"full subset shape expected [12288, 23, 2000], got {shape}")

    return {
        "status": "PASS" if not problems else "FAIL",
        "problems": problems,
        "actual_counts": actual_counts,
        "requested_counts": requested_counts,
        "duplicate_h5_indices": duplicate_h5_indices,
        "shape": shape,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--subset_manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--raw_edf_root", default=str(DEFAULT_RAW_ROOT))
    parser.add_argument("--output_npz", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--notch_freq", type=float, default=50.0)
    parser.add_argument("--canonical_builder", default=str(DEFAULT_CANONICAL_BUILDER))
    parser.add_argument("--limit_per_split", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_npz = Path(args.output_npz)
    out_dir = output_npz.parent
    logger = setup_logger(out_dir)
    metadata_path = out_dir / "build_metadata.json"
    status_path = out_dir / "build_status.json"

    if output_npz.exists() and not args.force:
        raise FileExistsError(f"output exists; use --force to overwrite: {output_npz}")

    logger.info("START unified 50Hz subset build")
    logger.info("args=%s", json.dumps(vars(args), sort_keys=True))
    canonical = load_canonical_builder(Path(args.canonical_builder))
    canonical.NOTCH_HZ = float(args.notch_freq)

    built, summary = build_records(args, logger)
    summary.update({
        "channel_names": list(canonical.CANONICAL_CHANNELS),
        "sfreq": canonical.SFREQ,
        "bandpass_hz": list(canonical.BANDPASS_HZ),
        "notch_hz": float(args.notch_freq),
        "window_sec": canonical.WINDOW_SEC,
        "window_samples": canonical.WINDOW_SAMPLES,
        "units": canonical.UNITS,
        "debug": bool(args.debug),
        "limit_per_split": int(args.limit_per_split or 0),
        "output_npz": str(output_npz),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    validation = validate_output(args, built, canonical, summary)
    summary.update(validation)
    if validation["status"] != "PASS":
        write_json(metadata_path, summary)
        write_json(status_path, {"status": "FAIL", "output_npz": str(output_npz), "problems": validation["problems"]})
        raise RuntimeError(f"validation failed: {validation['problems']}")

    npz_info = save_npz(args, built, summary, logger)
    summary.update(npz_info)
    write_json(metadata_path, summary)
    status = {
        "status": "PASS",
        "output_npz": str(output_npz),
        "shape": npz_info["shape"],
        "notch_hz": float(args.notch_freq),
        "debug": bool(args.debug),
        "limit_per_split": int(args.limit_per_split or 0),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(status_path, status)
    logger.info("PASS shape=%s output=%s", npz_info["shape"], output_npz)
    return summary


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_npz).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        summary = run(args)
    except Exception as exc:
        status = {
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "output_npz": str(args.output_npz),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        write_json(out_dir / "build_status.json", status)
        logger = logging.getLogger("build_labram_unified_50hz_subset")
        if logger.handlers:
            logger.error("FAIL %s", status["error"])
            logger.error(status["traceback"])
        else:
            print(json.dumps(status, indent=2), file=sys.stderr)
        return 1
    print(json.dumps({
        "status": "PASS",
        "output_npz": summary["output_npz"],
        "shape": summary["shape"],
        "actual_counts": summary["actual_counts"],
        "notch_hz": summary["notch_hz"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
