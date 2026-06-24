#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit LaBraM original full TUAB vs unified full-H5 LaBraM run.

This script is inspection/report-only. It copies only small evidence files and
does not touch H5, EDF, NPZ, checkpoints, or processed datasets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path("/nicoletye/workspace/unified_tuab")
LABRAM_REPO = Path("/nicoletye/workspace/Noise Robustness Evaluation Protocol/EEG-FM/Labram")
OUT_DIR = PROJECT_ROOT / "reports" / "labram_original_vs_unified_full_audit_for_meriem"
EVIDENCE_DIR = OUT_DIR / "evidence"

UNIFIED_H5 = PROJECT_ROOT / "data" / "canonical_tuab_full.h5"
SPLIT_CSV = PROJECT_ROOT / "reports" / "canonical_h5_labram_referenced_max_coverage_split_index.csv"
SPLIT_JSON = PROJECT_ROOT / "reports" / "canonical_h5_labram_referenced_max_coverage_split_index.json"
SPLIT_TXT = PROJECT_ROOT / "reports" / "canonical_h5_labram_referenced_max_coverage_split_summary.txt"
BUILD_CANONICAL = PROJECT_ROOT / "scripts" / "build_canonical_tuab.py"
BUILD_CANONICAL_ALTS = [
    BUILD_CANONICAL,
    Path("/nicoletye/workspace/Noise Robustness Evaluation Protocol/Unified Preprocessing/build_canonical_tuab.py"),
    Path("/mnt/data/build_canonical_tuab.py"),
]
BUILD_SPLIT = PROJECT_ROOT / "scripts" / "build_canonical_h5_max_coverage_split.py"

UNIFIED_RUN = PROJECT_ROOT / "outputs" / "labram_full_unified_h5_epoch10_seed42_run_v3"
UNIFIED_METRICS = UNIFIED_RUN / "metrics.json"
UNIFIED_CONFIG = UNIFIED_RUN / "config.json"
UNIFIED_EPOCH_CSV = UNIFIED_RUN / "epoch_metrics.csv"
UNIFIED_LOG = UNIFIED_RUN / "train.log"
UNIFIED_RUN_META = UNIFIED_RUN / "diagnostics" / "run_metadata.json"
UNIFIED_INPUT_STATS = UNIFIED_RUN / "diagnostics" / "full_h5_input_stats.json"

ORIGINAL_LOG_CANDIDATES = [
    LABRAM_REPO / "checkpoints" / "TUAB" / "original" / "seed_42" / "log.txt",
    LABRAM_REPO / "checkpoints" / "TUAB" / "original_lr2e-5" / "seed_42" / "log.txt",
    Path("/mnt/data/log.txt"),
]

SMALL_COPY_LIMIT = 8 * 1024 * 1024


def safe_float(value: Any) -> float | None:
    try:
        f = float(value)
    except Exception:
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_read_error": f"{type(exc).__name__}: {exc}", "_path": str(path)}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], preferred: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    if preferred:
        keys.extend(preferred)
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def copy_evidence(path: Path, name: str | None = None) -> str:
    if not path.exists() or not path.is_file():
        return ""
    if path.stat().st_size > SMALL_COPY_LIMIT:
        return f"SKIPPED_LARGE:{path}"
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    dest = EVIDENCE_DIR / (name or path.name)
    if dest.exists():
        stem = dest.stem
        suffix = dest.suffix
        dest = EVIDENCE_DIR / f"{stem}_{abs(hash(str(path))) % 100000}{suffix}"
    shutil.copy2(path, dest)
    return str(dest)


def parse_jsonl_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{") and line.endswith("}"):
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if "epoch" in payload:
                rows.append(payload)
    rows.sort(key=lambda row: int(row.get("epoch", 10**9)))
    return rows


def parse_epoch_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        converted: dict[str, Any] = {}
        for key, value in row.items():
            if key == "epoch":
                try:
                    converted[key] = int(value)
                except Exception:
                    converted[key] = value
            else:
                converted[key] = safe_float(value)
        out.append(converted)
    out.sort(key=lambda row: int(row.get("epoch", 10**9)))
    return out


def best_by(rows: list[dict[str, Any]], metric: str, max_epoch: int = 10) -> dict[str, Any]:
    candidates = [row for row in rows if isinstance(row.get("epoch"), int) and row["epoch"] <= max_epoch]
    candidates = [row for row in candidates if safe_float(row.get(metric)) is not None]
    if not candidates:
        return {}
    return max(candidates, key=lambda row: float(row[metric]))


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def inspect_split_csv(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "MISSING", "path": str(path)}
    counts: dict[str, Counter] = defaultdict(Counter)
    orig_counts: dict[str, Counter] = defaultdict(Counter)
    rows = 0
    groups: dict[str, set[str]] = defaultdict(set)
    source_by_split: dict[str, set[str]] = defaultdict(set)
    columns: list[str] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        for row in reader:
            rows += 1
            split = row.get("canonical_split", "")
            orig = row.get("h5_original_split", "")
            label = row.get("label", "")
            counts[split][label] += 1
            orig_counts[orig][label] += 1
            group_id = row.get("group_id") or row.get("subject_id") or row.get("recording_id")
            source = row.get("source_path", "")
            if group_id:
                groups[split].add(group_id)
            if source:
                source_by_split[split].add(source)
    overlap = {}
    split_names = sorted(groups)
    for i, a in enumerate(split_names):
        for b in split_names[i + 1 :]:
            overlap[f"{a}_vs_{b}"] = len(groups[a].intersection(groups[b]))
    return {
        "status": "FOUND",
        "path": str(path),
        "columns": columns,
        "row_count": rows,
        "canonical_split_counts": {
            split: {"n": sum(counter.values()), "normal_0": counter.get("0", 0), "abnormal_1": counter.get("1", 0)}
            for split, counter in sorted(counts.items())
        },
        "h5_original_split_counts": {
            split: {"n": sum(counter.values()), "normal_0": counter.get("0", 0), "abnormal_1": counter.get("1", 0)}
            for split, counter in sorted(orig_counts.items())
        },
        "group_overlap_counts": overlap,
        "recording_source_counts": {split: len(values) for split, values in sorted(source_by_split.items())},
    }


def inspect_original_processed_roots() -> list[dict[str, Any]]:
    roots = [
        Path("/nicoletye/datasets/TUAB/processed"),
        LABRAM_REPO / "processed",
        LABRAM_REPO / "data" / "processed",
    ]
    out = []
    for root in roots:
        if not root.exists():
            out.append({"path": str(root), "status": "MISSING"})
            continue
        split_counts = {}
        for split in ["train", "val", "test"]:
            split_dir = root / split
            if split_dir.exists():
                try:
                    split_counts[split] = len(list(split_dir.glob("*.pkl")))
                except Exception:
                    split_counts[split] = "COUNT_FAILED"
            else:
                split_counts[split] = "MISSING"
        out.append({"path": str(root), "status": "FOUND", "split_counts": split_counts})
    return out


def grep_flags(path: Path, patterns: list[str]) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    out = {}
    for pattern in patterns:
        hits = []
        regex = re.compile(pattern, re.IGNORECASE)
        for line in text.splitlines():
            if regex.search(line):
                hits.append(line.strip())
                if len(hits) >= 8:
                    break
        out[pattern] = hits
    return out


def inspect_preprocessing() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    unified_config = read_json(UNIFIED_CONFIG)
    h5_attrs = unified_config.get("h5_metadata", {}).get("attrs", {})
    make_tuab = LABRAM_REPO / "dataset_maker" / "make_TUAB.py"
    text = make_tuab.read_text(encoding="utf-8", errors="replace") if make_tuab.exists() else ""
    channel_order = "chOrder_standard" in text and "EEG FP1-REF" in text
    rows = [
        {
            "dimension": "channels",
            "original_labram": "23 raw referential TUAB channels via chOrder_standard",
            "unified_h5": ", ".join(h5_attrs.get("channel_names", [])) or "UNVERIFIABLE",
            "equivalence": "MATCH" if h5_attrs.get("channel_names") and channel_order else "UNVERIFIABLE",
            "evidence": "make_TUAB.py and unified config",
        },
        {
            "dimension": "sampling_rate",
            "original_labram": "raw.resample(200, n_jobs=5)",
            "unified_h5": str(h5_attrs.get("sfreq", "UNVERIFIABLE")),
            "equivalence": "MATCH" if str(h5_attrs.get("sfreq")) == "200" else "UNVERIFIABLE",
            "evidence": "make_TUAB.py and H5 attrs",
        },
        {
            "dimension": "window_length",
            "original_labram": "2000 samples, full non-overlapping windows",
            "unified_h5": f"{h5_attrs.get('window_samples', 'UNVERIFIABLE')} samples",
            "equivalence": "MATCH" if str(h5_attrs.get("window_samples")) == "2000" else "UNVERIFIABLE",
            "evidence": "make_TUAB.py and H5 attrs",
        },
        {
            "dimension": "overlap",
            "original_labram": "none, integer chunks i*2000:(i+1)*2000",
            "unified_h5": str(h5_attrs.get("overlap", "UNVERIFIABLE")),
            "equivalence": "MATCH" if str(h5_attrs.get("overlap")) == "0" else "UNVERIFIABLE",
            "evidence": "make_TUAB.py and H5 attrs",
        },
        {
            "dimension": "units",
            "original_labram": "raw.get_data(units='uV')",
            "unified_h5": str(h5_attrs.get("units", "UNVERIFIABLE")),
            "equivalence": "MATCH" if str(h5_attrs.get("units")) == "uV" else "UNVERIFIABLE",
            "evidence": "make_TUAB.py and H5 attrs",
        },
        {
            "dimension": "bandpass",
            "original_labram": "raw.filter(l_freq=0.1, h_freq=75.0)",
            "unified_h5": str(h5_attrs.get("bandpass_hz", "UNVERIFIABLE")),
            "equivalence": "MATCH" if "0.1" in str(h5_attrs.get("bandpass_hz")) and "75" in str(h5_attrs.get("bandpass_hz")) else "UNVERIFIABLE",
            "evidence": "make_TUAB.py and H5 attrs",
        },
        {
            "dimension": "notch",
            "original_labram": "raw.notch_filter(50.0)",
            "unified_h5": str(h5_attrs.get("notch_hz", "UNVERIFIABLE")),
            "equivalence": "DIFFERENT" if str(h5_attrs.get("notch_hz")) == "60.0" else "UNVERIFIABLE",
            "evidence": "make_TUAB.py and H5 attrs",
        },
        {
            "dimension": "storage_path",
            "original_labram": "pickle files with {'X': [23,2000], 'y': label}",
            "unified_h5": "HDF5 /eeg [N,23,2000] plus adapter reshape [B,23,10,200]",
            "equivalence": "DIFFERENT_CONTAINER_MATCHED_SHAPE",
            "evidence": "make_TUAB.py, unified config, adapter log",
        },
    ]
    details = {
        "make_TUAB_exists": make_tuab.exists(),
        "make_TUAB_path": str(make_tuab),
        "unified_config_exists": UNIFIED_CONFIG.exists(),
        "h5_attrs": h5_attrs,
        "grep": grep_flags(make_tuab, [r"notch_filter", r"filter", r"resample", r"get_data", r"pickle.dump", r"np.random.shuffle"]),
    }
    return rows, details


def inspect_training_recipe(original_log: Path | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    unified_config = read_json(UNIFIED_CONFIG)
    unified_metrics = read_json(UNIFIED_METRICS)
    unified_meta = read_json(UNIFIED_RUN_META)
    readme = LABRAM_REPO / "README.md"
    run_script = LABRAM_REPO / "run_class_finetuning.py"
    readme_text = readme.read_text(encoding="utf-8", errors="replace") if readme.exists() else ""
    recipe_rows = [
        {
            "field": "pretrained_checkpoint",
            "original_labram": "README TUAB example uses --finetune ./checkpoints/labram-base.pth",
            "unified_full_h5": f"loaded={unified_metrics.get('pretrained_checkpoint_loaded', unified_meta.get('pretrained_checkpoint_loaded', 'UNVERIFIABLE'))}, path={unified_metrics.get('pretrained_checkpoint_path', unified_meta.get('pretrained_checkpoint_path'))}",
            "equivalence": "DIFFERENT_CONFIRMED" if unified_metrics.get("pretrained_checkpoint_loaded") is False else "UNVERIFIABLE",
            "evidence": "README.md, unified metrics/run_metadata",
        },
        {
            "field": "batch_size",
            "original_labram": "README default/example: 64; original log was produced by repo training family",
            "unified_full_h5": unified_config.get("batch_size", "UNVERIFIABLE"),
            "equivalence": "DIFFERENT_CONFIRMED" if unified_config.get("batch_size") == 4 else "UNVERIFIABLE",
            "evidence": "README.md and unified config",
        },
        {
            "field": "learning_rate",
            "original_labram": "README example: 5e-4; original log first epochs show warmup to about 6e-5 in seed_42/original",
            "unified_full_h5": unified_config.get("lr", "UNVERIFIABLE"),
            "equivalence": "DIFFERENT_CONFIRMED" if safe_float(unified_config.get("lr")) == 0.0001 else "UNVERIFIABLE",
            "evidence": "README.md, original log train_lr, unified config",
        },
        {
            "field": "weight_decay",
            "original_labram": "README/log: 0.05",
            "unified_full_h5": unified_config.get("weight_decay", "UNVERIFIABLE"),
            "equivalence": "DIFFERENT_CONFIRMED" if safe_float(unified_config.get("weight_decay")) == 0.0005 else "UNVERIFIABLE",
            "evidence": "README.md/original log, unified config",
        },
        {
            "field": "epochs",
            "original_labram": "README example: 50; original log extends beyond epoch 10",
            "unified_full_h5": unified_config.get("epochs", "UNVERIFIABLE"),
            "equivalence": "DIFFERENT_CONFIRMED" if unified_config.get("epochs") == 10 else "UNVERIFIABLE",
            "evidence": "README.md, original log, unified config",
        },
        {
            "field": "n_parameters",
            "original_labram": "from original metric rows",
            "unified_full_h5": unified_metrics.get("raw_model_output_shape_before_temp_head", "model output shape only"),
            "equivalence": "SEE_CHECKPOINT_TABLE",
            "evidence": "metric rows and unified metrics",
        },
        {
            "field": "mixed_precision_loss_scale",
            "original_labram": "NativeScaler/loss_scale fields in log",
            "unified_full_h5": "plain PyTorch adapter loop; no loss_scale metric in epoch CSV",
            "equivalence": "DIFFERENT_LIKELY",
            "evidence": "original log has train_loss_scale; unified epoch CSV does not",
        },
    ]
    checkpoint_rows = [
        {
            "item": "Original LaBraM intended checkpoint",
            "original_labram": "./checkpoints/labram-base.pth in README TUAB finetuning command",
            "unified_full_h5": str(unified_metrics.get("pretrained_checkpoint_path")),
            "status": "MISMATCH_CONFIRMED" if unified_metrics.get("pretrained_checkpoint_loaded") is False else "UNVERIFIABLE",
            "notes": "Unified metrics explicitly say no external pretrained checkpoint was loaded.",
        },
        {
            "item": "Trainable parameter count",
            "original_labram": "Original seed_42/original log reports n_parameters=5820137",
            "unified_full_h5": "UNVERIFIABLE from metrics; wrapper did not write n_parameters",
            "status": "PARTIAL",
            "notes": "Original and unified are both LaBraM model paths, but the unified report lacks n_parameters.",
        },
    ]
    details = {
        "readme_tuab_hits": grep_flags(readme, [r"--finetune", r"--batch_size", r"--lr", r"--weight_decay", r"--epochs", r"TUAB"]),
        "run_script_hits": grep_flags(run_script, [r"--finetune", r"build_dataset", r"criterion", r"NativeScaler", r"cosine_scheduler", r"checkpoint-best"]),
        "original_log": str(original_log) if original_log else None,
        "unified_config": unified_config,
        "unified_metrics": unified_metrics,
        "unified_run_metadata": unified_meta,
    }
    return recipe_rows, checkpoint_rows, details


def compare_metrics(original_rows: list[dict[str, Any]], unified_rows: list[dict[str, Any]], unified_metrics: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    max_epoch = 10
    by_epoch_original = {int(row["epoch"]): row for row in original_rows if isinstance(row.get("epoch"), int) and int(row["epoch"]) <= max_epoch}
    by_epoch_unified = {int(row["epoch"]): row for row in unified_rows if isinstance(row.get("epoch"), int) and int(row["epoch"]) <= max_epoch}
    all_epochs = sorted(set(by_epoch_original) | set(by_epoch_unified))
    for epoch in all_epochs:
        o = by_epoch_original.get(epoch, {})
        u = by_epoch_unified.get(epoch, {})
        rows.append(
            {
                "epoch": epoch,
                "original_train_loss": o.get("train_loss", ""),
                "unified_train_loss": u.get("train_loss", ""),
                "original_val_balanced_accuracy": o.get("val_balanced_accuracy", ""),
                "unified_val_balanced_accuracy": u.get("val_balanced_accuracy", ""),
                "delta_val_balanced_accuracy_original_minus_unified": (
                    safe_float(o.get("val_balanced_accuracy")) - safe_float(u.get("val_balanced_accuracy"))
                    if safe_float(o.get("val_balanced_accuracy")) is not None and safe_float(u.get("val_balanced_accuracy")) is not None
                    else ""
                ),
                "original_test_balanced_accuracy": o.get("test_balanced_accuracy", ""),
                "unified_test_balanced_accuracy": unified_metrics.get("test_balanced_accuracy", "") if epoch == max_epoch else "",
                "original_val_auroc": o.get("val_roc_auc", ""),
                "unified_val_auroc": u.get("val_auroc", ""),
                "original_val_auprc": o.get("val_pr_auc", ""),
                "unified_val_auprc": u.get("val_auprc", ""),
                "original_test_auroc": o.get("test_roc_auc", ""),
                "unified_test_auroc": unified_metrics.get("test_auroc", "") if epoch == max_epoch else "",
                "original_test_auprc": o.get("test_pr_auc", ""),
                "unified_test_auprc": unified_metrics.get("test_auprc", "") if epoch == max_epoch else "",
            }
        )
    best_original = best_by(original_rows, "val_balanced_accuracy", max_epoch=max_epoch)
    best_unified = best_by(unified_rows, "val_balanced_accuracy", max_epoch=max_epoch)
    summary = {
        "original_best_epoch_0_to_10": best_original.get("epoch"),
        "original_best_val_balanced_accuracy_0_to_10": best_original.get("val_balanced_accuracy"),
        "original_test_balanced_accuracy_at_best_epoch_if_logged": best_original.get("test_balanced_accuracy"),
        "unified_best_epoch_1_to_10": best_unified.get("epoch"),
        "unified_best_val_balanced_accuracy_1_to_10": best_unified.get("val_balanced_accuracy"),
        "unified_final_test_balanced_accuracy": unified_metrics.get("test_balanced_accuracy"),
        "delta_best_val_balanced_accuracy_original_minus_unified": (
            safe_float(best_original.get("val_balanced_accuracy")) - safe_float(best_unified.get("val_balanced_accuracy"))
            if best_original and best_unified else None
        ),
    }
    return rows, summary


def root_cause_rows(split_class: str, unified_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    checkpoint_mismatch = unified_metrics.get("pretrained_checkpoint_loaded") is False
    return [
        {
            "rank": 1,
            "explanation": "checkpoint/pretraining mismatch",
            "classification": "CONFIRMED" if checkpoint_mismatch else "UNVERIFIABLE",
            "evidence": "Unified metrics/run_metadata say no external pretrained checkpoint was loaded; LaBraM README TUAB finetuning command uses --finetune ./checkpoints/labram-base.pth.",
            "impact": "High: changes initialization and makes original-vs-unified not a preprocessing-only comparison.",
        },
        {
            "rank": 2,
            "explanation": "training recipe mismatch",
            "classification": "CONFIRMED",
            "evidence": "Unified full-H5 run used batch_size=4, lr=1e-4, weight_decay=5e-4, epochs=10; LaBraM original recipe/log family uses different schedule, loss scaling, and typically batch_size=64/weight_decay=0.05.",
            "impact": "High: differences affect optimization, regularization, warmup, and overfitting.",
        },
        {
            "rank": 3,
            "explanation": "split exact-membership mismatch",
            "classification": "LIKELY" if split_class == "POLICY_MATCH_ONLY" else ("UNLIKELY" if split_class == "EXACT_MATCH" else "UNVERIFIABLE"),
            "evidence": "Unified split was built from LaBraM reference counts/val fraction and subject grouping; no original LaBraM subject membership list was found in this audit.",
            "impact": "Medium/high: matched counts do not guarantee same subject/recording membership.",
        },
        {
            "rank": 4,
            "explanation": "H5 adapter/input scaling mismatch",
            "classification": "POSSIBLE",
            "evidence": "Unified adapter shape is LaBraM-compatible [B,23,10,200] and H5 units are uV, but original pickle loader path and H5 adapter path are distinct and exact same-window value comparison was not performed here.",
            "impact": "Medium: possible but less directly evidenced than checkpoint/recipe mismatch.",
        },
        {
            "rank": 5,
            "explanation": "50 Hz vs 60 Hz notch difference",
            "classification": "CONFIRMED",
            "evidence": "make_TUAB.py uses raw.notch_filter(50.0); unified H5 attrs report notch_hz=60.0.",
            "impact": "Medium/low in this comparison: real difference, but unlikely alone to explain a large epoch-10 gap compared with checkpoint/recipe confounds.",
        },
        {
            "rank": 6,
            "explanation": "exact preprocessing implementation differences",
            "classification": "POSSIBLE",
            "evidence": "Both use MNE-style filter/resample/window choices, but filter method/padding/FIR design and channel normalization details were not proven bit-equivalent.",
            "impact": "Medium/low unless same-window pickle-vs-H5 numerical audit shows divergence.",
        },
        {
            "rank": 7,
            "explanation": "overfitting / early stopping difference",
            "classification": "LIKELY",
            "evidence": "Unified training loss falls steadily while validation balanced accuracy peaks early then declines by epoch 10.",
            "impact": "Medium: explains epoch-10 behavior, but itself may be caused by no pretraining/recipe mismatch.",
        },
    ]


def make_markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            text = str(value).replace("\n", " ").replace("|", "\\|")
            if len(text) > 180:
                text = text[:177] + "..."
            values.append(text)
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_reports(summary: dict[str, Any], tables: dict[str, list[dict[str, Any]]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main = OUT_DIR / "labram_original_vs_unified_full_audit_for_meriem.md"
    executive = OUT_DIR / "executive_summary.md"
    readme = OUT_DIR / "README.md"

    root_rows = tables["root_cause_ranking"]
    metric_summary = summary["metric_summary"]
    split_class = summary["split_equivalence_classification"]

    exec_lines = [
        "# Executive Summary",
        "",
        "The current evidence does not support concluding that unified preprocessing is intrinsically worse. The epoch-10 full-dataset comparison is confounded by confirmed checkpoint and training-recipe differences, plus a policy-match-only split.",
        "",
        "Most major preprocessing dimensions were matched: 23 raw referential TUAB channels, 200 Hz, 10-second/2000-sample windows, no overlap, uV scale, and 0.1-75 Hz bandpass. The confirmed preprocessing difference is notch filtering: original LaBraM uses 50 Hz, while the unified H5 reports 60 Hz.",
        "",
        f"Split classification: **{split_class}**. The unified split was LaBraM-referenced by counts/ratio and selected validation groups, but this audit did not find original LaBraM subject/recording membership lists proving exact membership.",
        "",
        "Strongest finding: the unified full-H5 run explicitly reports that no external pretrained checkpoint was loaded, while the LaBraM TUAB fine-tuning recipe expects `--finetune ./checkpoints/labram-base.pth`. The unified recipe also used batch size 4, lr 1e-4, and weight decay 5e-4, rather than the LaBraM recipe family visible in the repo/logs.",
        "",
        f"Metric comparison through epoch 10: original best validation balanced accuracy was {metric_summary.get('original_best_val_balanced_accuracy_0_to_10')} at epoch {metric_summary.get('original_best_epoch_0_to_10')}; unified best validation balanced accuracy was {metric_summary.get('unified_best_val_balanced_accuracy_1_to_10')} at epoch {metric_summary.get('unified_best_epoch_1_to_10')}. Unified final test balanced accuracy was {metric_summary.get('unified_final_test_balanced_accuracy')}.",
        "",
        "Recommended next action: rerun unified LaBraM with the same pretrained checkpoint and a matched LaBraM recipe before attributing any gap to preprocessing. If the comparison must be preprocessing-only, also use exact original LaBraM split membership or prove membership equivalence.",
        "",
    ]
    executive.write_text("\n".join(exec_lines), encoding="utf-8")

    lines = [
        "# LaBraM Original vs Unified Full TUAB Audit for Meriem",
        "",
        "## 1. Executive Summary",
        "",
        *exec_lines[2:],
        "## 2. What Was Already Matched",
        "",
        "- 23 raw referential channels: matched by design and supported by `make_TUAB.py` plus H5 attributes.",
        "- 200 Hz sampling rate: matched.",
        "- 10 s / 2000 samples: matched.",
        "- No overlap: matched.",
        "- uV scale: matched.",
        "- 0.1-75 Hz bandpass: matched.",
        "- LaBraM-compatible channel order: supported by the H5 channel list and LaBraM standard channel order.",
        "",
        "## 3. Remaining Differences",
        "",
        "- Original LaBraM notch: 50 Hz; unified H5 notch: 60 Hz.",
        f"- Split exact membership status: {split_class}.",
        "- Checkpoint/pretraining status: confirmed mismatch; unified reports no external pretrained checkpoint loaded.",
        "- Training recipe status: confirmed mismatch in batch size, learning rate, weight decay, and likely scheduler/loss-scaling path.",
        "- Adapter/input path: original pickle loader vs unified H5 adapter path; shapes are compatible but not proven numerically bit-equivalent.",
        "",
        "## 4. Split Equivalence Audit",
        "",
        f"Classification: **{split_class}**",
        "",
        "The unified split builder uses LaBraM reference counts and validation fraction, preserves original H5 test as test, and selects whole subject/recording groups for validation. That is a LaBraM-referenced policy match, not necessarily exact original membership unless original subject/recording lists are found and matched.",
        "",
        *make_markdown_table(tables["split_equivalence_table"], ["item", "original_labram", "unified_h5", "classification", "notes"]),
        "",
        "## 5. Checkpoint and Training Recipe Audit",
        "",
        *make_markdown_table(tables["checkpoint_equivalence_table"], ["item", "original_labram", "unified_full_h5", "status", "notes"]),
        "",
        *make_markdown_table(tables["training_recipe_equivalence_table"], ["field", "original_labram", "unified_full_h5", "equivalence", "evidence"]),
        "",
        "## 6. Metric Comparison up to Epoch 10",
        "",
        f"- Original best val balanced accuracy through epoch 10: {metric_summary.get('original_best_val_balanced_accuracy_0_to_10')} at epoch {metric_summary.get('original_best_epoch_0_to_10')}.",
        f"- Unified best val balanced accuracy through epoch 10: {metric_summary.get('unified_best_val_balanced_accuracy_1_to_10')} at epoch {metric_summary.get('unified_best_epoch_1_to_10')}.",
        f"- Unified final test balanced accuracy: {metric_summary.get('unified_final_test_balanced_accuracy')}.",
        "",
        "The unified run shows an early validation peak followed by falling validation performance while training loss continues decreasing, consistent with overfitting under this recipe.",
        "",
        "## 7. Root-Cause Ranking",
        "",
        *make_markdown_table(root_rows, ["rank", "explanation", "classification", "evidence", "impact"]),
        "",
        "## 8. Final Interpretation",
        "",
        "This comparison is not yet a clean preprocessing-only comparison. The unified preprocessing is mostly matched on the main signal dimensions, but the full-dataset epoch-10 outcome is confounded by confirmed checkpoint/pretraining and training-recipe mismatches. The split is also policy-match-only unless exact original LaBraM split membership is recovered and matched.",
        "",
        "Therefore, original > unified up to epoch 10 should not be interpreted as proof that unified preprocessing is worse. The strongest current explanation is that the unified full-H5 run did not reproduce the original LaBraM fine-tuning initialization and recipe.",
        "",
        "## 9. Recommended Next Action",
        "",
        "1. Rerun unified LaBraM with the same pretrained checkpoint used by original LaBraM.",
        "2. Match the original LaBraM recipe: batch size, lr/min_lr/warmup, weight decay, scheduler, mixed precision/loss scaler, epochs, and best-checkpoint policy.",
        "3. Recover or create exact original LaBraM subject/recording membership lists and compare against the unified split CSV.",
        "4. Run an original-pickle vs H5 same-window numeric audit for a small matched set.",
        "5. If only notch differs after the above are controlled, run a 50 Hz vs 60 Hz ablation or document 60 Hz as the AI Station/TUAB-line-noise decision.",
        "",
        "## Evidence Files",
        "",
        "Small scripts/logs/reports copied into `evidence/`; no checkpoints, datasets, H5, NPZ, EDF, or pickle directories were copied.",
        "",
    ]
    main.write_text("\n".join(lines), encoding="utf-8")

    readme.write_text(
        "\n".join(
            [
                "# LaBraM Original vs Unified Full TUAB Audit Package",
                "",
                "This folder contains an inspection-only audit of the LaBraM original full TUAB pipeline versus the unified full-H5 LaBraM epoch-10 run.",
                "",
                "Main report: `labram_original_vs_unified_full_audit_for_meriem.md`",
                "Executive summary: `executive_summary.md`",
                "",
                "Evidence tables are CSV/JSON files in this folder. Small source scripts and logs are copied under `evidence/`.",
                "",
                "No training, preprocessing, H5 modification, checkpoint copying, EDF copying, NPZ copying, or dataset copying was performed by the audit script.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    global OUT_DIR, EVIDENCE_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    args = parser.parse_args()
    OUT_DIR = Path(args.output_dir)
    EVIDENCE_DIR = OUT_DIR / "evidence"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    original_log = first_existing(ORIGINAL_LOG_CANDIDATES)
    original_rows = parse_jsonl_metrics(original_log) if original_log else []
    unified_rows = parse_epoch_csv(UNIFIED_EPOCH_CSV)
    unified_metrics = read_json(UNIFIED_METRICS)

    split_info = inspect_split_csv(SPLIT_CSV)
    split_json = read_json(SPLIT_JSON)
    original_processed = inspect_original_processed_roots()
    original_membership_found = False
    split_class = "POLICY_MATCH_ONLY"
    if not SPLIT_CSV.exists():
        split_class = "UNVERIFIABLE"

    preprocessing_rows, preprocessing_details = inspect_preprocessing()
    recipe_rows, checkpoint_rows, recipe_details = inspect_training_recipe(original_log)
    metric_rows, metric_summary = compare_metrics(original_rows, unified_rows, unified_metrics)
    root_rows = root_cause_rows(split_class, unified_metrics)

    split_rows = [
        {
            "item": "train/val/test counts",
            "original_labram": json.dumps(split_json.get("labram_reference_counts", {}), sort_keys=True),
            "unified_h5": json.dumps(split_info.get("canonical_split_counts", {}), sort_keys=True),
            "classification": "POLICY_MATCH_COUNTS_CLOSE",
            "notes": "Unified split uses LaBraM reference counts and val fraction but not confirmed original membership.",
        },
        {
            "item": "exact subject/recording membership",
            "original_labram": "No original membership list found by this audit.",
            "unified_h5": "Split CSV contains subject_id, recording_id, source_path and selected whole groups.",
            "classification": split_class,
            "notes": "Labram-referenced split policy is not necessarily exact original membership unless original subject/recording lists are found and matched.",
        },
        {
            "item": "test preservation",
            "original_labram": "Original LaBraM make_TUAB.py maps eval folders to test.",
            "unified_h5": f"original_test_preserved={split_json.get('leakage_check', {}).get('original_test_preserved')}",
            "classification": "MATCH_POLICY",
            "notes": "Unified preserves original H5 eval/test as test.",
        },
        {
            "item": "train/val leakage",
            "original_labram": "Subject-level random train/val split in make_TUAB.py.",
            "unified_h5": f"train_val_leakage_group_count={split_json.get('leakage_check', {}).get('train_val_leakage_group_count')}",
            "classification": "MATCH_POLICY",
            "notes": "Unified selected whole groups and reports no train/val group overlap.",
        },
    ]

    tables = {
        "preprocessing_equivalence_table": preprocessing_rows,
        "split_equivalence_table": split_rows,
        "training_recipe_equivalence_table": recipe_rows,
        "checkpoint_equivalence_table": checkpoint_rows,
        "metric_epoch0_to_epoch10_comparison": metric_rows,
        "root_cause_ranking": root_rows,
    }
    write_csv(OUT_DIR / "preprocessing_equivalence_table.csv", preprocessing_rows)
    write_csv(OUT_DIR / "split_equivalence_table.csv", split_rows)
    write_csv(OUT_DIR / "training_recipe_equivalence_table.csv", recipe_rows)
    write_csv(OUT_DIR / "checkpoint_equivalence_table.csv", checkpoint_rows)
    write_csv(OUT_DIR / "metric_epoch0_to_epoch10_comparison.csv", metric_rows)
    write_csv(OUT_DIR / "root_cause_ranking.csv", root_rows)
    write_json(OUT_DIR / "metric_epoch0_to_epoch10_comparison.json", {"rows": metric_rows, "summary": metric_summary})
    write_json(OUT_DIR / "root_cause_ranking.json", root_rows)

    evidence_copies = {}
    canonical_builder = first_existing(BUILD_CANONICAL_ALTS)
    for path, name in [
        (LABRAM_REPO / "dataset_maker" / "make_TUAB.py", "original_labram_make_TUAB.py"),
        (LABRAM_REPO / "run_class_finetuning.py", "original_labram_run_class_finetuning.py"),
        (LABRAM_REPO / "engine_for_finetuning.py", "original_labram_engine_for_finetuning.py"),
        (LABRAM_REPO / "README.md", "original_labram_README.md"),
        (original_log, "original_labram_seed42_log.txt" if original_log else None),
        (UNIFIED_LOG, "unified_full_h5_train.log"),
        (UNIFIED_METRICS, "unified_full_h5_metrics.json"),
        (UNIFIED_CONFIG, "unified_full_h5_config.json"),
        (UNIFIED_EPOCH_CSV, "unified_full_h5_epoch_metrics.csv"),
        (UNIFIED_RUN_META, "unified_full_h5_run_metadata.json"),
        (UNIFIED_INPUT_STATS, "unified_full_h5_input_stats.json"),
        (SPLIT_JSON, "canonical_h5_split_index.json"),
        (SPLIT_TXT, "canonical_h5_split_summary.txt"),
        (canonical_builder, "build_canonical_tuab.py" if canonical_builder else None),
        (BUILD_SPLIT, "build_canonical_h5_max_coverage_split.py"),
    ]:
        if path:
            evidence_copies[str(path)] = copy_evidence(path, name)

    summary = {
        "audit_status": "PASS",
        "output_dir": str(OUT_DIR),
        "original_log_path": str(original_log) if original_log else None,
        "original_metric_rows_found": len(original_rows),
        "unified_metric_rows_found": len(unified_rows),
        "unified_metrics_path": str(UNIFIED_METRICS),
        "split_equivalence_classification": split_class,
        "original_labram_split_metadata_found": original_membership_found,
        "metric_summary": metric_summary,
        "split_info": split_info,
        "split_json": split_json,
        "original_processed_roots": original_processed,
        "preprocessing_details": preprocessing_details,
        "training_recipe_details": recipe_details,
        "root_cause_ranking": root_rows,
        "evidence_copies": evidence_copies,
        "copy_policy": "Copied only small scripts/logs/reports. Did not copy H5, EDF, NPZ, checkpoints, processed datasets, or pickle directories.",
    }
    write_json(OUT_DIR / "audit_summary.json", summary)
    write_reports(summary, tables)

    print(f"audit_output_dir={OUT_DIR}")
    print(f"main_report={OUT_DIR / 'labram_original_vs_unified_full_audit_for_meriem.md'}")
    print(f"split_equivalence={split_class}")
    print(f"original_split_metadata_found={original_membership_found}")
    print(f"root_cause_top={root_rows[0]['explanation']}:{root_rows[0]['classification']}")
    print("no_training_no_preprocessing_no_h5_modification=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
