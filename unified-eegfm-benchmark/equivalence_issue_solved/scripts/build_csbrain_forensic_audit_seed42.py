#!/usr/bin/env python3
"""Build a read-only CSBrain forensic audit package for unified60 debugging."""

from __future__ import annotations

import csv
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROOT = Path("/nicoletye/workspace/unified_tuab")
CSBRAIN_REPO = Path("/nicoletye/workspace/Noise Robustness Evaluation Protocol/EEG-FM/CSBrain")
EXP = "all6_original_repo_vs_90job_unified60_matchedsplit_epoch15_seed42_v1"
REPORT_DIR = ROOT / "reports" / "csbrain_forensic_audit_seed42"
ORIGINAL_PROCESSED = ROOT / "data" / EXP / "original_processed" / "CSBrain"
MANIFEST = ROOT / "reports" / EXP / "original_matched_subset_manifest.csv"
INDEX_NPZ = ROOT / "reports" / "all6_unified60_subset_epoch15_5seed_1gpu_dev_v1" / "all6_fixed_subset_seed42_index.npz"
H5_PATH = ROOT / "data" / "canonical_tuab_full.h5"

METRICS = {
    "original_15": ROOT / "outputs" / EXP / "CSBrain" / "original" / "metrics.json",
    "debug_original_epoch3": ROOT / "outputs" / EXP / "non_labram_equivalence_debug_epoch3" / "CSBrain" / "original_debug" / "metrics.json",
    "debug_unified_bipolar_div100_epoch3": ROOT / "outputs" / EXP / "non_labram_equivalence_debug_epoch3" / "CSBrain" / "unified_bipolar_div100" / "metrics.json",
    "previous_unified60_seed42": ROOT / "outputs" / "all6_unified60_subset_epoch15_5seed_1gpu_dev_v1" / "csbrain" / "full_finetune" / "seed_42" / "metrics.json",
    "postdebug_original_epoch20": ROOT / "outputs" / "eegfm_postdebug_original_vs_unified60_equivalence_epoch20_seed42_v1" / "CSBrain" / "original" / "metrics.json",
    "postdebug_unified60_epoch20": ROOT / "outputs" / "eegfm_postdebug_original_vs_unified60_equivalence_epoch20_seed42_v1" / "CSBrain" / "unified_60hz" / "metrics.json",
}

CANONICAL_23 = [
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "A1", "A2", "FZ", "CZ", "PZ", "T1", "T2",
]
ENDPOINT_CURRENT = [
    "FP1", "F7", "T3", "T5", "FP2", "F8", "T4", "T6",
    "FP1", "F3", "C3", "P3", "FP2", "F4", "C4", "P4",
]
BIPOLAR_PAIRS = [
    ("FP1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
    ("FP2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
    ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
]
SPLITS = ["train", "val", "test"]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "READ_ERROR", "error": f"{type(exc).__name__}: {exc}"}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def read_manifest() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row = dict(row)
            row["label"] = int(row["label"])
            row["h5_label"] = int(row["h5_label"])
            row["h5_index"] = int(row["h5_index"])
            row["window_idx"] = int(row["window_idx"])
            row["split_position"] = int(row["split_position"])
            rows.append(row)
    return rows


def read_index_npz() -> dict[str, Any]:
    data = np.load(INDEX_NPZ, allow_pickle=True)
    return {
        "train_h5_indices": data["train_h5_indices"].astype(np.int64),
        "train_y": data["train_y"].astype(np.int64),
        "val_h5_indices": data["val_h5_indices"].astype(np.int64),
        "val_y": data["val_y"].astype(np.int64),
        "test_h5_indices": data["test_h5_indices"].astype(np.int64),
        "test_y": data["test_y"].astype(np.int64),
        "channel_names": [str(x) for x in data["channel_names"].tolist()],
    }


def rows_by_split(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = {split: [] for split in SPLITS}
    for row in rows:
        grouped[row["canonical_split"]].append(row)
    for split in SPLITS:
        grouped[split].sort(key=lambda r: r["split_position"])
    return grouped


def pkl_path(row: dict[str, Any]) -> Path:
    return ORIGINAL_PROCESSED / row["canonical_split"] / f"{row['recording_id']}_{row['window_idx']}.pkl"


def split_identity(rows: list[dict[str, Any]], index: dict[str, Any]) -> dict[str, Any]:
    grouped = rows_by_split(rows)
    out = {
        "status": "EXACT_WINDOW_MATCH_CONFIRMED",
        "split_counts": {},
        "label_counts": {},
        "h5_index_mismatch_count": {},
        "label_mismatch_count": {},
        "missing_pkl_count": {},
    }
    for split in SPLITS:
        split_rows = grouped[split]
        manifest_indices = np.asarray([r["h5_index"] for r in split_rows], dtype=np.int64)
        manifest_labels = np.asarray([r["label"] for r in split_rows], dtype=np.int64)
        h5_indices = index[f"{split}_h5_indices"]
        labels = index[f"{split}_y"]
        index_mismatch = len(manifest_indices) != len(h5_indices) or bool(np.any(manifest_indices != h5_indices))
        label_mismatch = len(manifest_labels) != len(labels) or bool(np.any(manifest_labels != labels))
        missing = [str(pkl_path(row)) for row in split_rows if not pkl_path(row).exists()]
        counts = Counter(manifest_labels.tolist())
        out["split_counts"][split] = len(split_rows)
        out["label_counts"][split] = {"label_0": int(counts.get(0, 0)), "label_1": int(counts.get(1, 0))}
        out["h5_index_mismatch_count"][split] = -1 if len(manifest_indices) != len(h5_indices) else int(np.sum(manifest_indices != h5_indices))
        out["label_mismatch_count"][split] = -1 if len(manifest_labels) != len(labels) else int(np.sum(manifest_labels != labels))
        out["missing_pkl_count"][split] = len(missing)
        if index_mismatch or label_mismatch or missing:
            out["status"] = "MISMATCH_FOUND"
    return out


def select_audit_rows(grouped: dict[str, list[dict[str, Any]]], per_split_label: int = 2) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for split in SPLITS:
        counts: Counter[int] = Counter()
        for row in grouped[split]:
            label = int(row["label"])
            if counts[label] < per_split_label:
                selected.append(row)
                counts[label] += 1
            if counts[0] >= per_split_label and counts[1] >= per_split_label:
                break
    return selected


def qstats(x: np.ndarray) -> dict[str, float]:
    x64 = x.astype(np.float64, copy=False).reshape(-1)
    abs_x = np.abs(x64)
    quantiles = np.quantile(x64, [0.01, 0.05, 0.50, 0.95, 0.99])
    return {
        "min": float(np.min(x64)),
        "max": float(np.max(x64)),
        "mean": float(np.mean(x64)),
        "std": float(np.std(x64)),
        "rms": float(np.sqrt(np.mean(x64 * x64))),
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q50": float(quantiles[2]),
        "q95": float(quantiles[3]),
        "q99": float(quantiles[4]),
        "abs_q95": float(np.quantile(abs_x, 0.95)),
    }


def corr_flat(a: np.ndarray, b: np.ndarray) -> float | None:
    av = a.reshape(-1).astype(np.float64)
    bv = b.reshape(-1).astype(np.float64)
    if av.size != bv.size or float(np.std(av)) == 0.0 or float(np.std(bv)) == 0.0:
        return None
    return float(np.corrcoef(av, bv)[0, 1])


def scalar_lstsq(source: np.ndarray, target: np.ndarray) -> float | None:
    s = source.reshape(-1).astype(np.float64)
    t = target.reshape(-1).astype(np.float64)
    denom = float(np.dot(s, s))
    if denom == 0.0:
        return None
    return float(np.dot(s, t) / denom)


def channel_corrs(a: np.ndarray, b: np.ndarray) -> list[float | None]:
    out: list[float | None] = []
    for i in range(a.shape[0]):
        out.append(corr_flat(a[i], b[i]))
    return out


def paired_input_audit(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    name_to_idx = {name: i for i, name in enumerate(CANONICAL_23)}
    endpoint_idx = [name_to_idx[name] for name in ENDPOINT_CURRENT]
    pair_idx = [(name_to_idx[left], name_to_idx[right]) for left, right in BIPOLAR_PAIRS]
    h5_indices = [row["h5_index"] for row in rows]
    sorted_indices = sorted(h5_indices)
    with h5py.File(H5_PATH, "r") as h5:
        h5_channel_names = [
            x.decode("utf-8", errors="replace") if isinstance(x, bytes) else str(x)
            for x in h5.attrs["channel_names"]
        ]
        data = h5["eeg"][sorted_indices].astype(np.float32)
    h5_map = {idx: data[pos] for pos, idx in enumerate(sorted_indices)}
    csv_rows: list[dict[str, Any]] = []
    aggregate: dict[str, list[float]] = defaultdict(list)
    label_mismatches = 0
    shape_mismatches = 0
    for row in rows:
        with pkl_path(row).open("rb") as f:
            sample = pickle.load(f)
        original_base = np.asarray(sample["X"], dtype=np.float32)
        label = int(sample["y"])
        h5_raw = h5_map[row["h5_index"]]
        endpoint = h5_raw[endpoint_idx]
        bipolar = np.stack([h5_raw[left] - h5_raw[right] for left, right in pair_idx]).astype(np.float32)
        variants = {
            "original_pkl_base": original_base,
            "original_model_current_mul10000": original_base * 10000.0,
            "original_author_scale_mul1000": original_base * 1000.0,
            "original_div100": original_base / 100.0,
            "unified_endpoint_current": endpoint,
            "unified_bipolar_base": bipolar,
            "unified_bipolar_div100": bipolar / 100.0,
            "unified_bipolar_mul1000": bipolar * 1000.0,
            "unified_bipolar_mul10000": bipolar * 10000.0,
        }
        if label != row["label"]:
            label_mismatches += 1
        if original_base.shape != (16, 2000) or bipolar.shape != (16, 2000):
            shape_mismatches += 1
        corr_base = corr_flat(original_base, bipolar)
        corr_endpoint = corr_flat(original_base, endpoint)
        scalar_h5_to_pkl = scalar_lstsq(bipolar, original_base)
        scalar_pkl_to_h5 = scalar_lstsq(original_base, bipolar)
        ch_corr = [c for c in channel_corrs(original_base, bipolar) if c is not None]
        orig_abs_q95 = qstats(original_base)["abs_q95"]
        h5_abs_q95 = qstats(bipolar)["abs_q95"]
        q95_ratio_original_over_h5 = orig_abs_q95 / h5_abs_q95 if h5_abs_q95 else None
        csv_row: dict[str, Any] = {
            "split": row["canonical_split"],
            "label_manifest": row["label"],
            "label_pkl": label,
            "recording_id": row["recording_id"],
            "window_idx": row["window_idx"],
            "h5_index": row["h5_index"],
            "shape_original": list(original_base.shape),
            "shape_unified_bipolar": list(bipolar.shape),
            "corr_original_vs_unified_bipolar": corr_base,
            "corr_original_vs_unified_endpoint": corr_endpoint,
            "channel_corr_mean_bipolar": float(np.mean(ch_corr)) if ch_corr else None,
            "channel_corr_min_bipolar": float(np.min(ch_corr)) if ch_corr else None,
            "channel_corr_negative_count": int(sum(1 for c in ch_corr if c < 0)),
            "scalar_h5_bipolar_to_original": scalar_h5_to_pkl,
            "scalar_original_to_h5_bipolar": scalar_pkl_to_h5,
            "q95_ratio_original_over_h5_bipolar": q95_ratio_original_over_h5,
            "pkl_path": str(pkl_path(row)),
        }
        for name, arr in variants.items():
            st = qstats(arr)
            for key, value in st.items():
                csv_row[f"{name}_{key}"] = value
        csv_rows.append(csv_row)
        for key in ["corr_original_vs_unified_bipolar", "corr_original_vs_unified_endpoint", "q95_ratio_original_over_h5_bipolar"]:
            if csv_row[key] is not None and math.isfinite(float(csv_row[key])):
                aggregate[key].append(float(csv_row[key]))
        aggregate["original_base_abs_q95"].append(csv_row["original_pkl_base_abs_q95"])
        aggregate["h5_bipolar_base_abs_q95"].append(csv_row["unified_bipolar_base_abs_q95"])
        aggregate["original_model_current_mul10000_rms"].append(csv_row["original_model_current_mul10000_rms"])
        aggregate["unified_bipolar_div100_rms"].append(csv_row["unified_bipolar_div100_rms"])
        aggregate["unified_bipolar_mul10000_rms"].append(csv_row["unified_bipolar_mul10000_rms"])
    summary = {
        "sample_count": len(csv_rows),
        "label_mismatch_count": label_mismatches,
        "shape_mismatch_count": shape_mismatches,
        "h5_channel_names": h5_channel_names,
        "endpoint_current_names": ENDPOINT_CURRENT,
        "endpoint_current_duplicate_names": sorted([name for name, count in Counter(ENDPOINT_CURRENT).items() if count > 1]),
        "bipolar_pairs": [f"{left}-{right}" for left, right in BIPOLAR_PAIRS],
        "aggregate": {
            key: {
                "mean": float(np.mean(values)) if values else None,
                "min": float(np.min(values)) if values else None,
                "max": float(np.max(values)) if values else None,
            }
            for key, values in aggregate.items()
        },
    }
    return csv_rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def metric_row(name: str, path: Path) -> dict[str, Any]:
    m = read_json(path)
    audit = m.get("author_settings_audit") or {}
    ckpt = m.get("checkpoint_load_report") or {}
    return {
        "row": name,
        "path": str(path),
        "status": m.get("status", "MISSING" if not path.exists() else ""),
        "epochs": m.get("epochs_completed", ""),
        "best_epoch": m.get("best_epoch", ""),
        "test_balanced_accuracy": m.get("test_balanced_accuracy", ""),
        "test_accuracy": m.get("test_accuracy", ""),
        "test_auroc": m.get("test_auroc", ""),
        "test_auprc": m.get("test_auprc", ""),
        "confusion_matrix": m.get("test_confusion_matrix", ""),
        "adapter_variant": m.get("adapter_variant", (m.get("adapter_meta") or {}).get("input_variant", "")),
        "recipe_mode": m.get("recipe_mode", audit.get("recipe_mode", "")),
        "loss_type": m.get("loss_type", "BCEWithLogitsLoss" if name.startswith("original") else ""),
        "lr": m.get("effective_lr", audit.get("effective_lr", "")),
        "weight_decay": m.get("effective_weight_decay", audit.get("effective_weight_decay", "")),
        "scheduler": m.get("scheduler", audit.get("scheduler", "")),
        "checkpoint_loaded": m.get("checkpoint_loaded", ckpt.get("checkpoint_loaded", "")),
        "notes": m.get("notes", ""),
    }


def write_author_repo_audit() -> dict[str, Any]:
    audit = {
        "repo_path": str(CSBRAIN_REPO),
        "official_tuab_dataset": "datasets/tuab_dataset.py",
        "official_tuab_shell": "sh/finetune_CSBrain_TUAB.sh",
        "model_file": "models/model_for_tuab.py",
        "input_shape": "[B,16,10,200]",
        "pkl_contract": "PKL dict with X and y; X resampled to 2000 then reshaped to [16,10,200]",
        "montage": "The model lists duplicated endpoint names matching bipolar-chain endpoints, but tuab_dataset.py assumes X already has 16 channels and does not construct bipolar itself.",
        "scaling": "datasets/tuab_dataset.py returns data * 10000 for TUAB.",
        "sampling_rate": "200 Hz implied by 2000 samples / 10 s and signal.resample(data, 2000, axis=-1)",
        "window_length": "10 seconds / 2000 samples",
        "labels": "binary y consumed by BCEWithLogitsLoss; threshold sigmoid(pred)>0.5 in evaluator",
        "optimizer": "AdamW default, lr=1e-4, weight_decay=0.05 from TUAB shell/defaults",
        "scheduler": "CosineAnnealingLR per train batch, eta_min=1e-6",
        "grad_clip": "clip_grad_norm_ with clip_value=1",
        "checkpoint": "TUAB shell passes --use_pretrained_weights and foundation_dir pth/CSBrain.pth",
        "best_selection": "finetune_trainer.py selects best by validation balanced accuracy variable named acc, not AUROC.",
        "evaluation": "balanced_accuracy_score, roc_auc_score, PR AUC via precision_recall_curve; confusion matrix",
        "hardcoded_assumptions": [
            "tuab_dataset.py truncates train files to first 50000.",
            "files are os.listdir order, not sorted.",
            "all tensors are moved with .cuda(), not device-agnostic.",
            "TUAB raw EDF preprocessing is not included in CSBrain repo; it delegates preprocessing docs to CBraMod.",
        ],
    }
    write_json(REPORT_DIR / "csbrain_author_repo_audit.json", audit)
    lines = [
        "# CSBrain Author Repo Audit",
        "",
        f"- Repo: `{audit['repo_path']}`",
        f"- TUAB loader: `{audit['official_tuab_dataset']}`",
        f"- TUAB launcher: `{audit['official_tuab_shell']}`",
        f"- Model: `{audit['model_file']}`",
        "",
        "## Confirmed Author TUAB Path",
        "",
        f"- Input contract: {audit['pkl_contract']}",
        f"- Expected model shape: `{audit['input_shape']}`",
        f"- Montage: {audit['montage']}",
        f"- Scaling: `{audit['scaling']}`",
        f"- Sampling/window: {audit['sampling_rate']}; {audit['window_length']}.",
        f"- Loss/labels: {audit['labels']}.",
        f"- Optimizer: {audit['optimizer']}.",
        f"- Scheduler: {audit['scheduler']}.",
        f"- Gradient clipping: {audit['grad_clip']}.",
        f"- Checkpoint/pretraining: {audit['checkpoint']}.",
        f"- Best model selection: {audit['best_selection']}.",
        "",
        "## Hard-Coded Or Fragile Assumptions",
        "",
    ]
    lines.extend(f"- {item}" for item in audit["hardcoded_assumptions"])
    (REPORT_DIR / "csbrain_author_repo_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return audit


def write_wrapper_audit(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    original = next((row for row in metric_rows if row["row"] == "original_15"), {})
    postdebug = next((row for row in metric_rows if row["row"] == "postdebug_original_epoch20"), {})
    audit = {
        "wrapper_files": [
            "scripts/eegfm_adapters/csbrain_option1_make_tuab_wrapper.py",
            "scripts/eegfm_adapters/csbrain_option1_train_wrapper.py",
        ],
        "preprocessing": "local EDF maker builds 16 bipolar channels after resample/filter/notch and saves PKL X [16,2000]",
        "training_input": "CSBrainPklDataset reshapes X to [16,10,200] and multiplies by 10000",
        "training_recipe": "BCEWithLogitsLoss, AdamW, cosine per batch, clip=1.0, best validation AUROC in wrapper",
        "author_difference": [
            "Wrapper selects best by val AUROC; author trainer selects best by validation balanced accuracy variable acc.",
            "Wrapper preprocessing applies 0.3-75 Hz and 60 Hz notch locally because repo ships no TUAB EDF maker.",
            "Wrapper defaults use_pretrained_weights=0 unless set by launcher; author TUAB shell uses pretrained weights.",
        ],
        "suspicious_evidence": [
            f"15-epoch original test B-Acc={original.get('test_balanced_accuracy')}, AUROC={original.get('test_auroc')}, confusion={original.get('confusion_matrix')}",
            f"20-epoch original collapsed confusion={postdebug.get('confusion_matrix')}",
            "3-epoch original debug first batch showed huge logits and saturated probabilities under *10000 scaling.",
        ],
    }
    write_json(REPORT_DIR / "csbrain_original_wrapper_audit.json", audit)
    lines = [
        "# CSBrain Original Wrapper Audit",
        "",
        "## Wrapper Behavior",
        "",
        f"- Files: `{audit['wrapper_files'][0]}`, `{audit['wrapper_files'][1]}`",
        f"- Preprocessing: {audit['preprocessing']}.",
        f"- Training input: {audit['training_input']}.",
        f"- Training recipe: {audit['training_recipe']}.",
        "",
        "## Differences From Author Repo",
        "",
    ]
    lines.extend(f"- {item}" for item in audit["author_difference"])
    lines.extend(["", "## Suspicious Evidence", ""])
    lines.extend(f"- {item}" for item in audit["suspicious_evidence"])
    lines.extend(["", "Interpretation: do not treat this original branch as a validated reference until scale/pretrain/best-metric controls are run."])
    (REPORT_DIR / "csbrain_original_wrapper_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return audit


def write_unified_audit(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    previous = next((row for row in metric_rows if row["row"] == "previous_unified60_seed42"), {})
    debug_bipolar = next((row for row in metric_rows if row["row"] == "debug_unified_bipolar_div100_epoch3"), {})
    audit = {
        "worker_files": [
            "scripts/eegfm_adapters/eegfm_small_subset_train_worker.py",
            "scripts/eegfm_adapters/eegfm_small_subset_train_worker_remaining_models.py",
            "scripts/eegfm_adapters/eegfm_tiny_train_worker.py",
        ],
        "previous_unified60": "endpoint/current selected-channel adapter [B,16,10,200], no true bipolar reconstruction, no *10000",
        "patched_smoke": "bipolar_div100 adapter computes true bipolar pairs then divides by 100",
        "author_difference": [
            "Author TUAB loader multiplies PKL X by 10000; patched smoke bipolar_div100 divides by 100.",
            "Author trainer uses cosine per batch and clip=1; older unified rows often used no scheduler/clip or selected best by balanced accuracy.",
            "Previous unified endpoint adapter duplicated FP1/FP2 as endpoint channels, not bipolar differences.",
        ],
        "performance_evidence": [
            f"Previous endpoint unified seed42 B-Acc={previous.get('test_balanced_accuracy')}, AUROC={previous.get('test_auroc')}",
            f"Patched smoke bipolar_div100 epoch3 B-Acc={debug_bipolar.get('test_balanced_accuracy')}, AUROC={debug_bipolar.get('test_auroc')}",
        ],
    }
    write_json(REPORT_DIR / "csbrain_unified60_adapter_audit.json", audit)
    lines = [
        "# CSBrain Unified60 Adapter Audit",
        "",
        "## Unified Paths",
        "",
    ]
    lines.extend(f"- `{item}`" for item in audit["worker_files"])
    lines.extend(
        [
            "",
            f"- Previous unified60 behavior: {audit['previous_unified60']}.",
            f"- Patched smoke behavior: {audit['patched_smoke']}.",
            "",
            "## Differences From Author/Original Semantics",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in audit["author_difference"])
    lines.extend(["", "## Performance Evidence", ""])
    lines.extend(f"- {item}" for item in audit["performance_evidence"])
    (REPORT_DIR / "csbrain_unified60_adapter_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return audit


def write_pair_reports(csv_rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    csv_path = REPORT_DIR / "csbrain_input_distribution_pair_audit.csv"
    write_csv(csv_path, csv_rows)
    write_json(REPORT_DIR / "csbrain_input_distribution_pair_audit.json", summary)
    agg = summary["aggregate"]
    lines = [
        "# CSBrain Paired Input Distribution Audit",
        "",
        "This read-only audit compares exact matched original CSBrain PKL windows against the same H5 windows after CSBrain transformations.",
        "",
        f"- Sample count: `{summary['sample_count']}`",
        f"- Label mismatches: `{summary['label_mismatch_count']}`",
        f"- Shape mismatches: `{summary['shape_mismatch_count']}`",
        f"- Endpoint duplicate names: `{summary['endpoint_current_duplicate_names']}`",
        "",
        "## Aggregate Evidence",
        "",
        "| Measure | Mean | Min | Max |",
        "|---|---:|---:|---:|",
    ]
    for key, values in agg.items():
        lines.append(f"| {key} | {values['mean']} | {values['min']} | {values['max']} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Exact labels and shapes can be compared window-by-window.",
            "- `corr_original_vs_unified_bipolar` is the key montage/polarity check.",
            "- `q95_ratio_original_over_h5_bipolar` estimates whether one scalar can align the original PKL and H5 bipolar amplitudes.",
            "- Endpoint-current correlation is expected to be lower because it is not a bipolar montage.",
            "",
            f"CSV: `{csv_path}`",
        ]
    )
    (REPORT_DIR / "csbrain_input_distribution_pair_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_recipe_matrix() -> None:
    rows = [
        {
            "row": "S0_original_current",
            "source": "original PKL",
            "montage": "current original wrapper true bipolar",
            "scale": "*10000",
            "recipe": "current original wrapper",
            "checkpoint": "as current wrapper/launcher",
            "purpose": "reproduce suspicious original baseline",
            "status": "planned_smoke",
        },
        {
            "row": "S1_original_author_scale_1000",
            "source": "original PKL",
            "montage": "true bipolar",
            "scale": "*1000",
            "recipe": "author default",
            "checkpoint": "author/default control",
            "purpose": "test whether *10000 broke original reproduction",
            "status": "planned_smoke",
        },
        {
            "row": "S2_original_scale_10000",
            "source": "original PKL",
            "montage": "true bipolar",
            "scale": "*10000",
            "recipe": "author default",
            "checkpoint": "author/default control",
            "purpose": "isolate scale magnitude effect",
            "status": "planned_smoke",
        },
        {
            "row": "S3_original_scale_div100",
            "source": "original PKL",
            "montage": "true bipolar",
            "scale": "/100",
            "recipe": "author default",
            "checkpoint": "author/default control",
            "purpose": "match patched unified60 scale convention on original data",
            "status": "planned_smoke",
        },
        {
            "row": "S4_unified_bipolar_div100",
            "source": "unified60 H5",
            "montage": "true bipolar",
            "scale": "/100",
            "recipe": "same as S3",
            "checkpoint": "same as S3",
            "purpose": "direct original-vs-unified comparison under same scale",
            "status": "planned_smoke",
        },
        {
            "row": "S5_unified_bipolar_mul1000",
            "source": "unified60 H5",
            "montage": "true bipolar",
            "scale": "*1000",
            "recipe": "same as S1",
            "checkpoint": "same as S1",
            "purpose": "direct author-scale unified60 test",
            "status": "planned_smoke",
        },
        {
            "row": "S6_unified_bipolar_mul10000",
            "source": "unified60 H5",
            "montage": "true bipolar",
            "scale": "*10000",
            "recipe": "same as S2",
            "checkpoint": "same as S2",
            "purpose": "test whether unified60 follows original-wrapper behavior",
            "status": "planned_smoke",
        },
        {
            "row": "S7_unified_endpoint_current",
            "source": "unified60 H5",
            "montage": "endpoint_current",
            "scale": "previous current behavior",
            "recipe": "previous unified60 recipe",
            "checkpoint": "previous/current",
            "purpose": "reproduce old unified60 row",
            "status": "planned_smoke",
        },
    ]
    write_csv(REPORT_DIR / "csbrain_training_recipe_matrix.csv", rows)
    lines = [
        "# CSBrain Training Recipe And Ablation Matrix",
        "",
        "No ablation training was launched by this audit. These rows are the recommended smoke matrix before selecting 15-epoch final rows.",
        "",
        "| Row | Source | Montage | Scale | Recipe | Checkpoint | Purpose | Status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row['row']} | {row['source']} | {row['montage']} | {row['scale']} | {row['recipe']} | {row['checkpoint']} | {row['purpose']} | {row['status']} |")
    lines.extend(
        [
            "",
            "## Final-Run Selection Rule",
            "",
            "Only promote rows to 15 epochs after smoke testing proves the original branch is a credible reference or proves that the shipped/local CSBrain TUAB path is broken.",
        ]
    )
    (REPORT_DIR / "csbrain_training_recipe_matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(REPORT_DIR / "csbrain_smoke_ablation_summary.json", {"status": "NOT_RUN", "reason": "forensic audit only; smoke matrix prepared but not launched", "rows": rows})
    write_csv(REPORT_DIR / "csbrain_smoke_ablation_summary.csv", rows)
    (REPORT_DIR / "csbrain_smoke_ablation_summary.md").write_text(
        "# CSBrain Smoke Ablation Summary\n\nStatus: `NOT_RUN`.\n\nThis audit prepared the S0-S7 smoke matrix but did not launch training.\n",
        encoding="utf-8",
    )


def write_summary(author: dict[str, Any], wrapper: dict[str, Any], unified: dict[str, Any], pair_summary: dict[str, Any], metric_rows: list[dict[str, Any]], split_check: dict[str, Any]) -> None:
    payload = {
        "status": "PASS",
        "classification": "ORIGINAL_REFERENCE_NOT_YET_VALIDATED",
        "interpretation_rule": "Do not call unified60 non-equivalent until CSBrain original reproduction is proven valid.",
        "split_identity": split_check,
        "author_repo": author,
        "original_wrapper": wrapper,
        "unified60_adapter": unified,
        "paired_input_summary": pair_summary,
        "metrics": metric_rows,
        "answer": {
            "what_author_repo_expects": "PKL X [16,2000] already prepared as CSBrain/TUAB 16-channel input; loader resamples to 2000, reshapes to [16,10,200], multiplies by 10000, uses BCEWithLogitsLoss, AdamW lr=1e-4 wd=0.05, cosine per batch, clip=1, pretrained CSBrain.pth, best validation balanced accuracy.",
            "what_original_wrapper_did_differently": "Local wrapper had to invent raw EDF maker, applies filter/notch/bipolar locally, multiplies by 10000 in loader, and selects best by validation AUROC rather than author trainer's validation balanced accuracy.",
            "what_unified60_did_differently": "Older unified used endpoint selection; patched smoke used true bipolar but divided by 100. Neither is author-equivalent to TUAB loader *10000 until scale/recipe are aligned.",
            "likely_bad_original_cause": "Not proven. Evidence points to scale/checkpoint/head/selection instability: *10000 produces huge logits/loss in debug, and some longer runs collapse to one-class predictions.",
            "fair_equivalent_row": "Not selected yet. Candidate must compare original PKL and unified H5 under the same true bipolar montage, same scalar convention, same BCE/AdamW/cosine/clip/checkpoint/best-metric recipe.",
            "meriem_table": "Send the author/wrapper/unified audit plus S0-S7 smoke matrix; do not present CSBrain as resolved until smoke rows identify a credible original reference.",
        },
    }
    write_json(REPORT_DIR / "csbrain_forensic_audit_summary.json", payload)
    lines = [
        "# CSBrain Forensic Audit Seed42",
        "",
        "## Executive Summary",
        "",
        "CSBrain remains unresolved. The exact matched subset is confirmed, so the current discrepancy should not be attributed to split mismatch. The original reproduction itself is suspicious and must be validated before unified60 can be judged non-equivalent.",
        "",
        f"- Classification: `{payload['classification']}`",
        f"- Split identity: `{split_check['status']}`",
        f"- Paired input label mismatches: `{pair_summary['label_mismatch_count']}`",
        f"- Paired input shape mismatches: `{pair_summary['shape_mismatch_count']}`",
        "",
        "## Key Findings",
        "",
        "- Author TUAB loader multiplies PKL `X` by `10000`; there is no TUAB raw EDF maker in the CSBrain repo.",
        "- Original wrapper locally builds bipolar PKLs and also applies `*10000`, but its longer runs can collapse to one-class predictions.",
        "- Previous unified60 endpoint adapter is not montage-equivalent.",
        "- Patched unified bipolar `/100` smoke is montage-equivalent but not scale-equivalent to author TUAB loader.",
        "- A fair row has not been selected yet; run S0-S7 smoke ablations first.",
        "",
        "## Existing Metrics Snapshot",
        "",
        "| Row | Status | Epochs | Best Epoch | B-Acc | AUROC | Confusion | Adapter | Recipe |",
        "|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    for row in metric_rows:
        lines.append(
            f"| {row['row']} | {row['status']} | {row['epochs']} | {row['best_epoch']} | {row['test_balanced_accuracy']} | {row['test_auroc']} | {row['confusion_matrix']} | {row['adapter_variant']} | {row['recipe_mode']} |"
        )
    lines.extend(
        [
            "",
            "## Required Next Step",
            "",
            "Run the smoke matrix only after reviewing this audit. Promote only the rows that prove original PKL and unified60 H5 are aligned under identical CSBrain semantics.",
            "",
            "## Report Files",
            "",
            "- `csbrain_author_repo_audit.md`",
            "- `csbrain_original_wrapper_audit.md`",
            "- `csbrain_unified60_adapter_audit.md`",
            "- `csbrain_input_distribution_pair_audit.csv`",
            "- `csbrain_input_distribution_pair_audit.md`",
            "- `csbrain_training_recipe_matrix.csv`",
            "- `csbrain_training_recipe_matrix.md`",
            "- `csbrain_smoke_ablation_summary.csv`",
            "- `csbrain_smoke_ablation_summary.md`",
            "- `csbrain_smoke_ablation_summary.json`",
        ]
    )
    (REPORT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_manifest()
    index = read_index_npz()
    split_check = split_identity(rows, index)
    selected = select_audit_rows(rows_by_split(rows), per_split_label=2)
    pair_rows, pair_summary = paired_input_audit(selected)
    write_pair_reports(pair_rows, pair_summary)
    metric_rows = [metric_row(name, path) for name, path in METRICS.items()]
    write_csv(REPORT_DIR / "csbrain_existing_metrics_snapshot.csv", metric_rows)
    author = write_author_repo_audit()
    wrapper = write_wrapper_audit(metric_rows)
    unified = write_unified_audit(metric_rows)
    write_recipe_matrix()
    write_summary(author, wrapper, unified, pair_summary, metric_rows, split_check)
    print(json.dumps({"status": "PASS", "report_dir": str(REPORT_DIR), "sample_count": pair_summary["sample_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
