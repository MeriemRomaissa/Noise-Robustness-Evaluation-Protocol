#!/usr/bin/env python3
"""Run phased CSBrain smoke ablations for original-vs-unified60 debugging.

This script intentionally runs short diagnostics only. It does not rebuild H5,
does not preprocess EDFs, and does not launch 15-epoch final ablations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path("/nicoletye/workspace/unified_tuab")
EXP = "all6_original_repo_vs_90job_unified60_matchedsplit_epoch15_seed42_v1"
CSBRAIN_REPO = Path("/nicoletye/workspace/Noise Robustness Evaluation Protocol/EEG-FM/CSBrain")
DEFAULT_CKPT = CSBRAIN_REPO / "downloaded_weights" / "pth" / "CSBrain.pth"
DEFAULT_ORIGINAL = ROOT / "data" / EXP / "original_processed" / "CSBrain"
DEFAULT_INDEX_NPZ = ROOT / "reports" / "all6_unified60_subset_epoch15_5seed_1gpu_dev_v1" / "all6_fixed_subset_seed42_index.npz"
DEFAULT_H5 = ROOT / "data" / "canonical_tuab_full.h5"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "csbrain_unified60_smoke_ablation_seed42"
DEFAULT_REPORT_DIR = ROOT / "reports" / "csbrain_forensic_audit_seed42"

SPLITS = ["train", "val", "test"]
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


@dataclass(frozen=True)
class RowSpec:
    row: str
    phase: str
    source: str
    montage: str
    scale: str
    recipe: str
    checkpoint: bool
    purpose: str


ROWS: dict[str, RowSpec] = {
    "S0_original_current": RowSpec("S0_original_current", "phase0", "original_pkl", "true_bipolar", "mul10000", "current_original_wrapper", True, "reproduce suspicious original baseline"),
    "S1_original_author_scale_1000": RowSpec("S1_original_author_scale_1000", "phase2", "original_pkl", "true_bipolar", "mul1000", "author_style", True, "scale sanity original branch"),
    "S2_original_scale_10000": RowSpec("S2_original_scale_10000", "phase1", "original_pkl", "true_bipolar", "mul10000", "author_style", True, "strict author scale original branch"),
    "S3_original_scale_div100": RowSpec("S3_original_scale_div100", "phase3", "original_pkl", "true_bipolar", "div100", "author_style_except_scale", True, "diagnostic original /100 branch"),
    "S4_unified_bipolar_div100": RowSpec("S4_unified_bipolar_div100", "phase3", "unified_h5", "true_bipolar", "div100", "author_style_except_scale", True, "diagnostic unified /100 branch"),
    "S5_unified_bipolar_mul1000": RowSpec("S5_unified_bipolar_mul1000", "phase2", "unified_h5", "true_bipolar", "mul1000", "author_style", True, "scale sanity unified branch"),
    "S6_unified_bipolar_mul10000": RowSpec("S6_unified_bipolar_mul10000", "phase1", "unified_h5", "true_bipolar", "mul10000", "author_style", True, "strict author scale unified branch"),
    "S7_unified_endpoint_current": RowSpec("S7_unified_endpoint_current", "phase4", "unified_h5", "endpoint_current", "identity", "previous_unified60", True, "old endpoint-current control only"),
}
PHASES = {
    "phase1": ["S2_original_scale_10000", "S6_unified_bipolar_mul10000"],
    "phase2": ["S1_original_author_scale_1000", "S5_unified_bipolar_mul1000"],
    "phase3": ["S3_original_scale_div100", "S4_unified_bipolar_div100"],
    "phase4": ["S7_unified_endpoint_current"],
}


def scale_array(x: np.ndarray, scale: str) -> np.ndarray:
    if scale == "mul10000":
        return x * 10000.0
    if scale == "mul1000":
        return x * 1000.0
    if scale == "div100":
        return x / 100.0
    if scale == "identity":
        return x
    raise ValueError(f"unknown scale {scale}")


def tensor_stats(t: torch.Tensor | np.ndarray) -> dict[str, float]:
    arr = t.detach().float().cpu().numpy() if torch.is_tensor(t) else np.asarray(t, dtype=np.float32)
    flat = arr.reshape(-1).astype(np.float64)
    if flat.size == 0:
        return {}
    return {
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "rms": float(np.sqrt(np.mean(flat * flat))),
        "q95": float(np.quantile(flat, 0.95)),
        "abs_q95": float(np.quantile(np.abs(flat), 0.95)),
    }


def read_index_npz(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {
        "train_h5_indices": data["train_h5_indices"].astype(np.int64),
        "train_y": data["train_y"].astype(np.int64),
        "val_h5_indices": data["val_h5_indices"].astype(np.int64),
        "val_y": data["val_y"].astype(np.int64),
        "test_h5_indices": data["test_h5_indices"].astype(np.int64),
        "test_y": data["test_y"].astype(np.int64),
        "channel_names": [str(x) for x in data["channel_names"].tolist()],
    }


class OriginalPklDataset(Dataset):
    def __init__(self, root: Path, split: str, scale: str):
        self.files = sorted((root / split).glob("*.pkl"))
        if not self.files:
            raise FileNotFoundError(f"no PKL files under {root / split}")
        self.scale = scale

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        with self.files[idx].open("rb") as f:
            obj = pickle.load(f)
        x = np.asarray(obj["X"], dtype=np.float32)
        if x.shape != (16, 2000):
            raise ValueError(f"expected PKL X [16,2000], got {x.shape} in {self.files[idx]}")
        x = scale_array(x, self.scale).reshape(16, 10, 200).astype(np.float32)
        return torch.from_numpy(x), torch.tensor(int(obj["y"]), dtype=torch.float32)


class UnifiedH5Dataset(Dataset):
    def __init__(self, h5_path: Path, indices: np.ndarray, labels: np.ndarray, channel_names: list[str], montage: str, scale: str):
        self.h5_path = str(h5_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.channel_names = [name.upper() for name in channel_names]
        self.montage = montage
        self.scale = scale
        self._h5 = None
        lookup = {name: idx for idx, name in enumerate(self.channel_names)}
        self.endpoint_indices = [lookup[name] for name in ENDPOINT_CURRENT]
        self.pair_indices = [(lookup[left], lookup[right]) for left, right in BIPOLAR_PAIRS]

    def __len__(self) -> int:
        return len(self.indices)

    def _file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        eeg = self._file()["eeg"][int(self.indices[idx])].astype(np.float32)
        if self.montage == "true_bipolar":
            x = np.stack([eeg[left] - eeg[right] for left, right in self.pair_indices]).astype(np.float32)
        elif self.montage == "endpoint_current":
            x = eeg[self.endpoint_indices].astype(np.float32)
        else:
            raise ValueError(f"unknown montage {self.montage}")
        x = scale_array(x, self.scale).reshape(16, 10, 200).astype(np.float32)
        return torch.from_numpy(x), torch.tensor(int(self.labels[idx]), dtype=torch.float32)

    def __del__(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device_arg}, but CUDA is unavailable")
    return device


def build_model(repo: Path, checkpoint_path: Path, device: torch.device, dropout: float = 0.1) -> tuple[torch.nn.Module, dict[str, Any]]:
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from models.model_for_tuab import Model

    param = SimpleNamespace(
        use_pretrained_weights=False,
        model="CSBrain",
        use_SmallerToken=False,
        dropout=dropout,
        cuda=0,
        foundation_dir=str(checkpoint_path),
    )
    model = Model(param).to(device)
    report = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_exists": checkpoint_path.exists(),
        "checkpoint_loaded": False,
        "loaded_key_count": 0,
        "missing_key_count": None,
        "unexpected_key_count": 0,
        "status": "NOT_REQUESTED",
        "notes": [],
    }
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu")
        state = state.get("state_dict", state) if isinstance(state, dict) else state
        stripped = {str(k).replace("module.", ""): v for k, v in state.items() if hasattr(v, "shape")}
        target = model.backbone.state_dict()
        matched = {k: v for k, v in stripped.items() if k in target and tuple(v.shape) == tuple(target[k].shape)}
        target.update(matched)
        model.backbone.load_state_dict(target)
        report.update(
            {
                "checkpoint_loaded": bool(matched),
                "loaded_key_count": len(matched),
                "missing_key_count": len(target) - len(matched),
                "status": "LOADED_FILTERED_BACKBONE" if matched else "PRESENT_NO_COMPATIBLE_KEYS",
                "checkpoint_top_level_keys": list(stripped.keys())[:30],
            }
        )
    else:
        report["status"] = "MISSING"
    return model, report


def binary_metrics(labels: list[int], probs: list[float], preds: list[int]) -> dict[str, Any]:
    labels_arr = np.asarray(labels, dtype=np.int64)
    probs_arr = np.asarray(probs, dtype=np.float64)
    preds_arr = np.asarray(preds, dtype=np.int64)
    acc = float(np.mean(labels_arr == preds_arr)) if len(labels_arr) else float("nan")
    recalls = []
    for label in [0, 1]:
        mask = labels_arr == label
        if mask.any():
            recalls.append(float(np.mean(preds_arr[mask] == label)))
    cm = np.zeros((2, 2), dtype=np.int64)
    for y, p in zip(labels_arr.tolist(), preds_arr.tolist()):
        if y in {0, 1} and p in {0, 1}:
            cm[y, p] += 1
    auroc = float("nan")
    auprc = float("nan")
    if len(np.unique(labels_arr)) == 2:
        from sklearn.metrics import average_precision_score, roc_auc_score

        auroc = float(roc_auc_score(labels_arr, probs_arr))
        auprc = float(average_precision_score(labels_arr, probs_arr))
    return {
        "accuracy": acc,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else float("nan"),
        "auroc": auroc,
        "auprc": auprc,
        "confusion_matrix": cm.tolist(),
        "positive_rate": float(np.mean(preds_arr == 1)) if len(preds_arr) else float("nan"),
        "probability": tensor_stats(probs_arr),
    }


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, criterion: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    labels: list[int] = []
    probs: list[float] = []
    preds: list[int] = []
    logit_chunks: list[torch.Tensor] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x).reshape(-1)
        loss = criterion(logits, y.float())
        prob = torch.sigmoid(logits)
        pred = (prob > 0.5).long()
        losses.append(float(loss.detach().cpu()))
        labels.extend(y.detach().cpu().long().tolist())
        probs.extend(prob.detach().cpu().tolist())
        preds.extend(pred.detach().cpu().tolist())
        logit_chunks.append(logits.detach().cpu())
    metrics = binary_metrics(labels, probs, preds)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    metrics["logits"] = tensor_stats(torch.cat(logit_chunks)) if logit_chunks else {}
    return metrics


def grad_norm(parameters) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is not None:
            norm = float(param.grad.detach().data.norm(2).cpu())
            total += norm * norm
    return math.sqrt(total)


def make_loaders(spec: RowSpec, args: argparse.Namespace, index: dict[str, Any], device: torch.device) -> tuple[dict[str, DataLoader], dict[str, int]]:
    if spec.source == "original_pkl":
        datasets = {split: OriginalPklDataset(Path(args.original_processed_root), split, spec.scale) for split in SPLITS}
    else:
        datasets = {
            split: UnifiedH5Dataset(
                Path(args.h5),
                index[f"{split}_h5_indices"],
                index[f"{split}_y"],
                index["channel_names"],
                spec.montage,
                spec.scale,
            )
            for split in SPLITS
        }
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda", generator=generator),
        "val": DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda"),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda"),
    }
    return loaders, {split: len(datasets[split]) for split in SPLITS}


def class_counts(loader: DataLoader, max_batches: int | None = None) -> dict[str, int]:
    counts = Counter()
    for i, (_, y) in enumerate(loader):
        counts.update(int(v) for v in y.long().tolist())
        if max_batches is not None and i + 1 >= max_batches:
            break
    return {"label_0": int(counts.get(0, 0)), "label_1": int(counts.get(1, 0))}


def train_one(spec: RowSpec, args: argparse.Namespace, index: dict[str, Any]) -> dict[str, Any]:
    row_dir = Path(args.output_root) / spec.row
    row_dir.mkdir(parents=True, exist_ok=True)
    log_path = row_dir / "train.log"
    log_path.write_text("", encoding="utf-8")

    def log(message: str) -> None:
        print(f"[{spec.row}] {message}", flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(message.rstrip() + "\n")

    started = time.time()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    loaders, counts = make_loaders(spec, args, index, device)
    first_x, first_y = next(iter(loaders["train"]))
    input_summary = tensor_stats(first_x)
    model, ckpt_report = build_model(Path(args.csbrain_repo), Path(args.checkpoint_path), device, dropout=0.1)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs * len(loaders["train"]), 1), eta_min=args.eta_min)
    best_state = None
    best_epoch = 0
    best_val_bal = -float("inf")
    history: list[dict[str, Any]] = []
    grad_norms: list[float] = []
    log(f"START source={spec.source} montage={spec.montage} scale={spec.scale} epochs={args.epochs}")
    log(f"counts={counts} input_summary={json.dumps(input_summary, sort_keys=True)} checkpoint={ckpt_report['status']} keys={ckpt_report.get('loaded_key_count')}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for x, y in loaders["train"]:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x).reshape(-1)
            loss = criterion(logits, y.float())
            loss.backward()
            gnorm = grad_norm(model.parameters())
            grad_norms.append(gnorm)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
        val = evaluate(model, loaders["val"], criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "val_loss": val["loss"],
            "val_balanced_accuracy": val["balanced_accuracy"],
            "val_accuracy": val["accuracy"],
            "val_auroc": val["auroc"],
            "val_auprc": val["auprc"],
            "val_positive_rate": val["positive_rate"],
        }
        history.append(row)
        log(json.dumps(row, allow_nan=True))
        if val["balanced_accuracy"] > best_val_bal:
            best_val_bal = float(val["balanced_accuracy"])
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save({"model_state_dict": best_state, "epoch": epoch, "best_val_balanced_accuracy": best_val_bal, "spec": spec.__dict__, "args": vars(args)}, row_dir / "checkpoint_best.pt")
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "best_epoch": best_epoch, "spec": spec.__dict__, "args": vars(args)}, row_dir / "checkpoint_last.pt")
    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate(model, loaders["test"], criterion, device)
    grad_summary = {
        "count": len(grad_norms),
        "min": float(np.min(grad_norms)) if grad_norms else None,
        "max": float(np.max(grad_norms)) if grad_norms else None,
        "mean": float(np.mean(grad_norms)) if grad_norms else None,
    }
    one_class_collapse = bool(test["positive_rate"] in {0.0, 1.0})
    metrics = {
        "row": spec.row,
        "phase": spec.phase,
        "status": "PASS",
        "source": spec.source,
        "montage": spec.montage,
        "scale": spec.scale,
        "recipe": spec.recipe,
        "checkpoint_loaded": ckpt_report.get("checkpoint_loaded"),
        "loaded_key_count": ckpt_report.get("loaded_key_count"),
        "checkpoint_load_report": ckpt_report,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_balanced_accuracy": best_val_bal,
        "test_balanced_accuracy": test["balanced_accuracy"],
        "test_auroc": test["auroc"],
        "test_auprc": test["auprc"],
        "test_accuracy": test["accuracy"],
        "test_confusion_matrix": test["confusion_matrix"],
        "prediction_positive_rate": test["positive_rate"],
        "logit_summary": test["logits"],
        "probability_summary": test["probability"],
        "input_summary": input_summary,
        "grad_norm_summary": grad_summary,
        "train_loss_curve": [row["train_loss"] for row in history],
        "history": history,
        "one_class_collapse": one_class_collapse,
        "counts": counts,
        "label_counts_sample": {split: class_counts(loaders[split], max_batches=4) for split in SPLITS},
        "metrics_path": str(row_dir / "metrics.json"),
        "log_path": str(log_path),
        "notes": spec.purpose,
        "runtime_seconds": time.time() - started,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "scheduler": "CosineAnnealingLR_per_batch",
        "eta_min": args.eta_min,
        "grad_clip": args.grad_clip,
        "best_metric": "val_balanced_accuracy",
    }
    (row_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    log(f"DONE metrics={row_dir / 'metrics.json'}")
    return metrics


def summary_row(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "row": metrics.get("row"),
        "source": metrics.get("source"),
        "montage": metrics.get("montage"),
        "scale": metrics.get("scale"),
        "checkpoint_loaded": metrics.get("checkpoint_loaded"),
        "loaded_key_count": metrics.get("loaded_key_count"),
        "epochs_completed": metrics.get("epochs_completed"),
        "best_epoch": metrics.get("best_epoch"),
        "best_validation_balanced_accuracy": metrics.get("best_validation_balanced_accuracy"),
        "test_balanced_accuracy": metrics.get("test_balanced_accuracy"),
        "test_auroc": metrics.get("test_auroc"),
        "test_auprc": metrics.get("test_auprc"),
        "test_accuracy": metrics.get("test_accuracy"),
        "confusion_matrix": metrics.get("test_confusion_matrix"),
        "prediction_positive_rate": metrics.get("prediction_positive_rate"),
        "logit_min": (metrics.get("logit_summary") or {}).get("min"),
        "logit_max": (metrics.get("logit_summary") or {}).get("max"),
        "logit_mean": (metrics.get("logit_summary") or {}).get("mean"),
        "logit_std": (metrics.get("logit_summary") or {}).get("std"),
        "prob_min": (metrics.get("probability_summary") or {}).get("min"),
        "prob_max": (metrics.get("probability_summary") or {}).get("max"),
        "prob_mean": (metrics.get("probability_summary") or {}).get("mean"),
        "prob_std": (metrics.get("probability_summary") or {}).get("std"),
        "input_rms": (metrics.get("input_summary") or {}).get("rms"),
        "input_abs_q95": (metrics.get("input_summary") or {}).get("abs_q95"),
        "grad_norm_mean": (metrics.get("grad_norm_summary") or {}).get("mean"),
        "train_loss_first": (metrics.get("train_loss_curve") or [""])[0],
        "train_loss_last": (metrics.get("train_loss_curve") or [""])[-1],
        "one_class_collapse": metrics.get("one_class_collapse"),
        "metrics_path": metrics.get("metrics_path"),
        "notes": metrics.get("notes"),
    }


def write_phase_summary(report_dir: Path, phase: str, metrics_list: list[dict[str, Any]]) -> None:
    rows = [summary_row(m) for m in metrics_list]
    fields = list(rows[0].keys()) if rows else []
    csv_path = report_dir / f"csbrain_smoke_{phase}_summary.csv"
    json_path = report_dir / f"csbrain_smoke_{phase}_summary.json"
    md_path = report_dir / f"csbrain_smoke_{phase}_summary.md"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps({"phase": phase, "rows": rows, "metrics": metrics_list}, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    lines = [
        f"# CSBrain Smoke {phase.title()} Summary",
        "",
        "| Row | Source | Montage | Scale | Ckpt | Keys | Epochs | Best Epoch | Best Val B-Acc | Test B-Acc | AUROC | AUPRC | Pos Rate | Collapse |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['row']} | {row['source']} | {row['montage']} | {row['scale']} | {row['checkpoint_loaded']} | {row['loaded_key_count']} | {row['epochs_completed']} | {row['best_epoch']} | {row['best_validation_balanced_accuracy']} | {row['test_balanced_accuracy']} | {row['test_auroc']} | {row['test_auprc']} | {row['prediction_positive_rate']} | {row['one_class_collapse']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_all_summary(report_dir: Path, output_root: Path) -> None:
    metrics = []
    for path in sorted(output_root.glob("S*/metrics.json")):
        try:
            metrics.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            pass
    if not metrics:
        return
    rows = [summary_row(m) for m in metrics]
    fields = list(rows[0].keys())
    with (report_dir / "csbrain_smoke_all_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (report_dir / "csbrain_smoke_all_summary.json").write_text(json.dumps({"rows": rows, "metrics": metrics}, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    lines = ["# CSBrain Smoke All Summary", "", "| Row | Phase | B-Acc | AUROC | Pos Rate | Collapse | Metrics |", "|---|---|---:|---:|---:|---|---|"]
    for m in metrics:
        lines.append(f"| {m['row']} | {m['phase']} | {m['test_balanced_accuracy']} | {m['test_auroc']} | {m['prediction_positive_rate']} | {m['one_class_collapse']} | `{m['metrics_path']}` |")
    (report_dir / "csbrain_smoke_all_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_logs_index(report_dir: Path, output_root: Path) -> None:
    lines = ["# CSBrain Smoke Training Logs Index", ""]
    for path in sorted(output_root.glob("S*/train.log")):
        metrics_path = path.parent / "metrics.json"
        lines.append(f"- `{path.parent.name}`: log `{path}`, metrics `{metrics_path}`")
    (report_dir / "csbrain_smoke_training_logs_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["phase1", "phase2", "phase3", "phase4"], required=True)
    parser.add_argument("--rows", nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--report_dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--original_processed_root", default=str(DEFAULT_ORIGINAL))
    parser.add_argument("--index_npz", default=str(DEFAULT_INDEX_NPZ))
    parser.add_argument("--h5", default=str(DEFAULT_H5))
    parser.add_argument("--csbrain_repo", default=str(CSBRAIN_REPO))
    parser.add_argument("--checkpoint_path", default=str(DEFAULT_CKPT))
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> int:
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    args = parse_args()
    output_root = Path(args.output_root)
    report_dir = Path(args.report_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    row_names = args.rows if args.rows else PHASES[args.phase]
    specs = [ROWS[name] for name in row_names]
    config = {"args": vars(args), "rows": [spec.__dict__ for spec in specs]}
    (report_dir / f"csbrain_smoke_{args.phase}_run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "START", "phase": args.phase, "rows": row_names, "output_root": str(output_root)}, indent=2), flush=True)
    if args.dry_run:
        write_phase_summary(report_dir, args.phase, [])
        return 0
    index = read_index_npz(Path(args.index_npz))
    metrics_list = []
    for spec in specs:
        try:
            metrics_list.append(train_one(spec, args, index))
        except Exception as exc:
            row_dir = output_root / spec.row
            row_dir.mkdir(parents=True, exist_ok=True)
            fail = {
                "row": spec.row,
                "phase": spec.phase,
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "metrics_path": str(row_dir / "metrics.json"),
            }
            (row_dir / "metrics.json").write_text(json.dumps(fail, indent=2) + "\n", encoding="utf-8")
            metrics_list.append(fail)
    write_phase_summary(report_dir, args.phase, metrics_list)
    write_all_summary(report_dir, output_root)
    write_logs_index(report_dir, output_root)
    print(json.dumps({"status": "PASS", "phase": args.phase, "rows": [m.get("row") for m in metrics_list]}, indent=2), flush=True)
    return 0 if all(m.get("status") == "PASS" for m in metrics_list) else 1


if __name__ == "__main__":
    raise SystemExit(main())
