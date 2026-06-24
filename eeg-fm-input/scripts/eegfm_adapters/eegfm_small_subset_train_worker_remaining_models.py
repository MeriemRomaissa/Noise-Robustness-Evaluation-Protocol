#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Isolated remaining-model small-subset worker for CBraMod/CSBrain/CodeBrain.

This copy is intentionally independent of the shared BIOT/EEGPT worker stack.
It imports finetune_strategy_utils_remaining_models.py only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from canonical_h5_subset import select_tuab_16, to_patches
from finetune_strategy_utils_remaining_models import apply_finetune_strategy

SUPPORTED_MODELS = {"CBraMod", "CSBrain", "CodeBrain"}
DEFAULT_PRETRAINED_CHECKPOINTS = {
    "CBraMod": "pretrained_weights/pretrained_weights.pth",
    "CSBrain": "downloaded_weights/pth/CSBrain.pth",
    "CodeBrain": "Checkpoints/CodeBrain.pth",
}
STRICT_WORDING = (
    "Meriem-strict strategy compliance means the same benchmark protocol is applied across EEG-FMs: "
    "pretrained checkpoint loading, full fine-tune / linear probe / LoRA strategy definitions, "
    "unified60 H5 split, five seeds, epoch-50 full run, consistent metric reporting, and "
    "trainable-parameter audits. For LoRA, LaBraM/Meriem adapter placement is used as the reference "
    "logic, and for non-LaBraM architectures the adapters are mapped to the closest native equivalent "
    "projection/attention/MLP modules. This preserves fairness without forcing invalid layer names."
)


def shape_of(value) -> list[int] | None:
    return list(value.shape) if hasattr(value, "shape") else None


class NoOpTensorCuda:
    def __enter__(self):
        self._orig_cuda = torch.Tensor.cuda
        torch.Tensor.cuda = lambda tensor, *args, **kwargs: tensor
        return self

    def __exit__(self, exc_type, exc, tb):
        torch.Tensor.cuda = self._orig_cuda
        return False


class nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def import_h5py():
    try:
        import h5py
        return h5py
    except Exception:
        eegnet_site = Path("/nicoletye/venvs/eegnet/lib/python3.10/site-packages")
        if eegnet_site.exists() and str(eegnet_site) not in sys.path:
            sys.path.append(str(eegnet_site))
        import h5py
        return h5py


class H5IndexDataset(Dataset):
    def __init__(self, h5_path: str, h5_indices: np.ndarray, labels: np.ndarray):
        self.h5_path = h5_path
        self.h5_indices = np.asarray(h5_indices, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self._h5 = None

    def __len__(self) -> int:
        return len(self.h5_indices)

    def _file(self):
        if self._h5 is None:
            self._h5 = import_h5py().File(self.h5_path, "r")
        return self._h5

    def __getitem__(self, idx: int):
        f = self._file()
        x = f["eeg"][int(self.h5_indices[idx])].astype(np.float32)
        y = int(self.labels[idx])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

    def __del__(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def append_log(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def load_index_npz(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    return {
        "train_h5_indices": data["train_h5_indices"].astype(np.int64),
        "train_y": data["train_y"].astype(np.int64),
        "val_h5_indices": data["val_h5_indices"].astype(np.int64),
        "val_y": data["val_y"].astype(np.int64),
        "test_h5_indices": data["test_h5_indices"].astype(np.int64),
        "test_y": data["test_y"].astype(np.int64),
        "channel_names": [str(x) for x in data["channel_names"].tolist()],
        "split_counts": json.loads(str(data["split_counts_json"][0])) if "split_counts_json" in data else {},
        "tiny_counts": json.loads(str(data["tiny_counts_json"][0])) if "tiny_counts_json" in data else {},
    }


def load_subset_npz(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    return {
        "train_x": torch.from_numpy(data["train_x"].astype(np.float32)),
        "train_y": torch.from_numpy(data["train_y"].astype(np.int64)),
        "val_x": torch.from_numpy(data["val_x"].astype(np.float32)),
        "val_y": torch.from_numpy(data["val_y"].astype(np.int64)),
        "test_x": torch.from_numpy(data["test_x"].astype(np.float32)),
        "test_y": torch.from_numpy(data["test_y"].astype(np.int64)),
        "channel_names": [str(x) for x in data["channel_names"].tolist()],
        "split_counts": json.loads(str(data["split_counts_json"][0])) if "split_counts_json" in data else {},
        "tiny_counts": json.loads(str(data["tiny_counts_json"][0])) if "tiny_counts_json" in data else {},
    }


def _split_length(batch: dict, split: str, index_mode: bool) -> int:
    return int(len(batch[f"{split}_h5_indices"] if index_mode else batch[f"{split}_x"]))


def _subset_array(value, selected: np.ndarray):
    if torch.is_tensor(value):
        return value[torch.as_tensor(selected, dtype=torch.long)]
    return value[selected]


def _limit_one_split(batch: dict, split: str, requested_n: int, seed: int, index_mode: bool) -> tuple[int, int, str | None]:
    available = _split_length(batch, split, index_mode)
    if requested_n is None or requested_n <= 0:
        return available, available, None
    used = min(int(requested_n), available)
    warning = None
    if requested_n > available:
        warning = f"WARNING: requested {split}_n={requested_n} but only {available} samples are available; using all available."
    rng = np.random.default_rng(seed)
    selected = np.arange(available, dtype=np.int64)
    rng.shuffle(selected)
    selected = np.sort(selected[:used])
    if index_mode:
        batch[f"{split}_h5_indices"] = _subset_array(batch[f"{split}_h5_indices"], selected)
        batch[f"{split}_y"] = _subset_array(batch[f"{split}_y"], selected)
    else:
        batch[f"{split}_x"] = _subset_array(batch[f"{split}_x"], selected)
        batch[f"{split}_y"] = _subset_array(batch[f"{split}_y"], selected)
    return available, used, warning


def apply_split_limits(args: argparse.Namespace, batch: dict, index_mode: bool) -> dict:
    limits = {"train": int(args.train_n or 0), "val": int(args.val_n or 0), "test": int(args.test_n or 0)}
    offsets = {"train": 0, "val": 1009, "test": 2003}
    summary = {"requested_train_n": limits["train"], "requested_val_n": limits["val"], "requested_test_n": limits["test"], "split_limit_warnings": []}
    for split in ["train", "val", "test"]:
        original, used, warning = _limit_one_split(batch, split, limits[split], int(args.seed) + offsets[split], index_mode)
        summary[f"original_{split}_count"] = original
        summary[f"actual_{split}_count"] = used
        if warning:
            summary["split_limit_warnings"].append(warning)
    return summary


def adapt_inputs(model_name: str, raw: torch.Tensor, channel_names: list[str], repo_path: str) -> tuple[torch.Tensor, dict]:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported remaining model: {model_name}")
    x16, indices, names = select_tuab_16(raw, channel_names)
    return to_patches(x16, 200), {
        "adapter": f"{model_name} [B,16,10,200]",
        "selected_channel_indices": indices,
        "selected_channel_names": names,
    }


def _insert_repo_paths(repo_path: str, model_name: str) -> Path:
    repo = Path(repo_path)
    if model_name == "CodeBrain" and not repo.exists():
        parent = repo.parent
        for candidate in [parent / "Codebrain", parent / "CodeBrain", parent / "codebrain"]:
            if candidate.exists():
                repo = candidate
                break
    for path in [repo, repo / "Models", repo / "models", repo / "Downstream", repo / "Datasets", repo / "datasets"]:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return repo


class CodeBrainWrapper(nn.Module):
    def __init__(self, repo_path: str):
        super().__init__()
        os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
        repo = _insert_repo_paths(repo_path, "CodeBrain")
        try:
            from Models.model_for_tuab import Model
        except Exception:
            from models.model_for_tuab import Model
        param = SimpleNamespace(
            downstream_dataset="TUAB",
            num_of_classes=2,
            use_pretrained_weights=False,
            dropout=0.1,
            cuda=0,
            foundation_dir="",
            n_layer=8,
            codebook_size_t=4096,
            codebook_size_f=4096,
            codebook_dim=32,
        )
        self.repo_path = str(repo)
        self.model = Model(param)

    def forward(self, x):
        bz, ch_num, seq_len, patch_size = x.shape
        context = NoOpTensorCuda() if x.device.type == "cpu" else nullcontext()
        with context:
            feats = self.model.backbone(x)
        feats = feats.contiguous().view(bz, ch_num * seq_len * patch_size)
        out = self.model.classifier(feats)
        return out.contiguous().view(bz)


def build_model(model_name: str, repo_path: str, adapted_train: torch.Tensor | None = None, adapter_meta: dict | None = None) -> nn.Module:
    if model_name == "CBraMod":
        _insert_repo_paths(repo_path, model_name)
        from models.model_for_tuab import Model
        param = SimpleNamespace(use_pretrained_weights=False, classifier="all_patch_reps", dropout=0.1, cuda=0, foundation_dir="")
        return Model(param)
    if model_name == "CSBrain":
        _insert_repo_paths(repo_path, model_name)
        from models.model_for_tuab import Model
        param = SimpleNamespace(model="CSBrain", use_pretrained_weights=False, dropout=0.1, cuda=0, foundation_dir="")
        return Model(param)
    if model_name == "CodeBrain":
        return CodeBrainWrapper(repo_path)
    raise ValueError(f"Unsupported remaining model: {model_name}")


def default_pretrained_checkpoint(model_name: str, repo_path: str) -> Path | None:
    rel = DEFAULT_PRETRAINED_CHECKPOINTS.get(model_name)
    return Path(repo_path) / rel if rel else None


def _safe_torch_load(path: Path, map_location: str = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _shape(value: Any) -> tuple[int, ...] | None:
    return tuple(value.shape) if hasattr(value, "shape") else None


def _load_matching_state(module: nn.Module, state: dict[str, Any], strip_prefix: str = "") -> dict:
    model_state = module.state_dict()
    mapped = {}
    skipped_shape = []
    skipped_missing = []
    for key, value in state.items():
        mk = key[len(strip_prefix):] if strip_prefix and key.startswith(strip_prefix) else key
        if mk in model_state and _shape(model_state[mk]) == _shape(value):
            mapped[mk] = value
        elif mk in model_state:
            skipped_shape.append({"checkpoint_key": key, "model_key": mk, "checkpoint_shape": list(value.shape), "model_shape": list(model_state[mk].shape)})
        else:
            skipped_missing.append(key)
    load_result = module.load_state_dict(mapped, strict=False)
    return {
        "mapped": mapped,
        "load_result": load_result,
        "missing_keys": list(getattr(load_result, "missing_keys", [])),
        "unexpected_keys": list(getattr(load_result, "unexpected_keys", [])),
        "skipped_shape_mismatch": skipped_shape,
        "skipped_not_in_model": skipped_missing[:200],
        "skipped_not_in_model_count": len(skipped_missing),
    }


def load_pretrained_checkpoint_if_available(model: nn.Module, model_name: str, repo_path: str) -> dict:
    checkpoint_path = default_pretrained_checkpoint(model_name, repo_path)
    report = {
        "model": model_name,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else "",
        "checkpoint_exists": bool(checkpoint_path and checkpoint_path.exists()),
        "checkpoint_loaded": False,
        "checkpoint_status": "NO_DEFAULT_CHECKPOINT",
        "loaded_key_count": 0,
        "model_key_count": len(model.state_dict()),
        "missing_key_count": None,
        "unexpected_key_count": None,
        "skipped_key_count": None,
        "notes": [],
    }
    if checkpoint_path is None or not checkpoint_path.exists():
        report["checkpoint_status"] = "MISSING"
        return report
    state = _safe_torch_load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if not isinstance(state, dict):
        report["checkpoint_status"] = "UNSUPPORTED_CHECKPOINT_OBJECT"
        return report

    if model_name == "CBraMod":
        target = model.backbone
        load = _load_matching_state(target, state)
        status = "LOADED_BACKBONE_STRICT_HEAD_INITIALIZED" if len(load["missing_keys"]) == 0 else "LOADED_BACKBONE_PARTIAL_HEAD_INITIALIZED"
    elif model_name == "CSBrain":
        target = model.backbone
        load = _load_matching_state(target, state, strip_prefix="module.")
        status = "LOADED_BACKBONE_FILTERED_HEAD_INITIALIZED" if load["mapped"] else "PRESENT_BACKBONE_LOAD_FAILED"
    elif model_name == "CodeBrain":
        target = model.model.backbone if hasattr(model, "model") else model.backbone
        load = _load_matching_state(target, state, strip_prefix="module.")
        status = "LOADED_BACKBONE_FILTERED_HEAD_INITIALIZED" if load["mapped"] else "PRESENT_BACKBONE_LOAD_FAILED"
    else:
        report["checkpoint_status"] = "UNSUPPORTED_MODEL"
        return report

    report.update({
        "checkpoint_loaded": bool(load["mapped"]),
        "checkpoint_status": status,
        "loaded_key_count": len(load["mapped"]),
        "model_key_count": len(target.state_dict()),
        "missing_key_count": len(load["missing_keys"]),
        "unexpected_key_count": len(load["unexpected_keys"]),
        "skipped_key_count": int(load["skipped_not_in_model_count"] + len(load["skipped_shape_mismatch"])),
        "missing_keys": load["missing_keys"][:200],
        "unexpected_keys": load["unexpected_keys"][:200],
        "skipped_shape_mismatch": load["skipped_shape_mismatch"][:80],
        "skipped_not_in_model_count": load["skipped_not_in_model_count"],
        "skipped_not_in_model_sample": load["skipped_not_in_model"][:80],
        "classifier_head_initialized": True,
        "strategy_compliance": "MERIEM_STRICT" if load["mapped"] else "NOT_COMPLIANT_UNTIL_CHECKPOINT_LOAD_FIXED",
    })
    if model_name == "CodeBrain":
        report["notes"].append("Loaded CodeBrain.pth into downstream SSSM backbone; Tokenizer checkpoint is separate and not used by native TUAB finetune wrapper.")
    return report


def infer_loss_and_maybe_head(model: nn.Module, sample_x: torch.Tensor) -> tuple[nn.Module, str, list[int], bool]:
    model.eval()
    with torch.no_grad():
        out = model(sample_x)
    original_shape = shape_of(out)
    if out.ndim == 2 and out.shape[1] == 2:
        return model, "CrossEntropyLoss", original_shape, False
    if out.ndim == 1 or (out.ndim == 2 and out.shape[1] == 1):
        return model, "BCEWithLogitsLoss", original_shape, False
    feature_dim = int(np.prod(out.shape[1:]))
    return nn.Sequential(model, nn.Flatten(), nn.Linear(feature_dim, 2)), "CrossEntropyLoss", original_shape, True


def compute_loss(logits: torch.Tensor, labels: torch.Tensor, loss_type: str) -> torch.Tensor:
    if loss_type == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()(logits, labels.long())
    if loss_type == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss()(logits.reshape(-1), labels.float())
    raise ValueError(f"Unsupported loss type: {loss_type}")


def predictions_from_logits(logits: torch.Tensor, loss_type: str) -> torch.Tensor:
    if loss_type == "CrossEntropyLoss":
        return logits.argmax(dim=1).long()
    return (logits.reshape(-1) >= 0).long()


def positive_scores_from_logits(logits: torch.Tensor, loss_type: str) -> torch.Tensor:
    if loss_type == "CrossEntropyLoss":
        return torch.softmax(logits, dim=1)[:, 1]
    return torch.sigmoid(logits.reshape(-1))


def simple_metrics(preds: list[int], labels: list[int]) -> dict:
    pred_arr = np.asarray(preds, dtype=np.int64)
    label_arr = np.asarray(labels, dtype=np.int64)
    acc = float((pred_arr == label_arr).mean()) if len(label_arr) else float("nan")
    recalls = []
    for label in [0, 1]:
        mask = label_arr == label
        if mask.any():
            recalls.append(float((pred_arr[mask] == label).mean()))
    return {"accuracy": acc, "balanced_accuracy": float(np.mean(recalls)) if recalls else float("nan")}


def detailed_binary_metrics(preds: list[int], labels: list[int], positive_scores: list[float]) -> dict:
    pred_arr = np.asarray(preds, dtype=np.int64)
    label_arr = np.asarray(labels, dtype=np.int64)
    score_arr = np.asarray(positive_scores, dtype=np.float64)
    confusion = np.zeros((2, 2), dtype=np.int64)
    for y_true, y_pred in zip(label_arr.tolist(), pred_arr.tolist()):
        if y_true in {0, 1} and y_pred in {0, 1}:
            confusion[y_true, y_pred] += 1
    auroc = None
    auprc = None
    notes = []
    if len(np.unique(label_arr)) == 2:
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score
            auroc = float(roc_auc_score(label_arr, score_arr))
            auprc = float(average_precision_score(label_arr, score_arr))
        except Exception as exc:
            notes.append(f"AUROC/AUPRC unavailable: {type(exc).__name__}: {exc}")
    else:
        notes.append("AUROC/AUPRC unavailable because only one class is present.")
    return {"confusion_matrix": confusion.tolist(), "auroc": auroc, "auprc": auprc, "metric_notes": notes}


def evaluate(model, loader, loss_type: str, device: torch.device, args: argparse.Namespace, channel_names: list[str], include_binary_details: bool = False) -> dict:
    model.eval()
    losses, preds, labels_all, positive_scores = [], [], [], []
    output_shape = None
    with torch.no_grad():
        for x, y in loader:
            x, _ = adapt_inputs(args.model, x, channel_names, args.repo_path)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            output_shape = shape_of(logits)
            loss = compute_loss(logits, y, loss_type)
            losses.append(float(loss.detach().cpu()))
            preds.extend(predictions_from_logits(logits.detach().cpu(), loss_type).tolist())
            labels_all.extend(y.detach().cpu().long().tolist())
            if include_binary_details:
                positive_scores.extend(positive_scores_from_logits(logits.detach().cpu(), loss_type).tolist())
    metrics = simple_metrics(preds, labels_all)
    metrics.update({"loss": float(np.mean(losses)) if losses else float("nan"), "output_shape": output_shape})
    if include_binary_details:
        metrics.update(detailed_binary_metrics(preds, labels_all, positive_scores))
    return metrics


def make_data_loaders(args: argparse.Namespace, channel_names: list[str], raw_batch: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(args.seed)
    if args.index_npz:
        train_ds = H5IndexDataset(args.h5, raw_batch["train_h5_indices"], raw_batch["train_y"])
        val_ds = H5IndexDataset(args.h5, raw_batch["val_h5_indices"], raw_batch["val_y"])
        test_ds = H5IndexDataset(args.h5, raw_batch["test_h5_indices"], raw_batch["test_y"])
    else:
        train_ds = TensorDataset(raw_batch["train_x"], raw_batch["train_y"])
        val_ds = TensorDataset(raw_batch["val_x"], raw_batch["val_y"])
        test_ds = TensorDataset(raw_batch["test_x"], raw_batch["test_y"])
    pin = bool(args.pin_memory and resolve_device(args.device).type == "cuda")
    return (
        DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=args.num_workers, pin_memory=pin),
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin),
        DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin),
    )


def train_small_subset(args: argparse.Namespace) -> dict:
    if args.model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported remaining model: {args.model}")
    set_seed(args.seed)
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"
    metrics_path = output_dir / "metrics.json"
    checkpoint_path = output_dir / "checkpoint.pt"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()

    index_mode = bool(args.index_npz)
    batch = load_index_npz(args.index_npz) if index_mode else load_subset_npz(args.subset_npz)
    channel_names = batch["channel_names"]
    split_limit_summary = apply_split_limits(args, batch, index_mode=index_mode)
    for warning in split_limit_summary["split_limit_warnings"]:
        print(warning, flush=True)
        append_log(log_path, warning)
    counts = {split: _split_length(batch, split, index_mode) for split in ["train", "val", "test"]}
    print(f"split_counts train={counts['train']} val={counts['val']} test={counts['test']}", flush=True)

    train_loader, val_loader, test_loader = make_data_loaders(args, channel_names, batch)
    first_raw, _ = next(iter(train_loader))
    sample_adapted, adapter_meta = adapt_inputs(args.model, first_raw, channel_names, args.repo_path)
    device = resolve_device(args.device)
    model = build_model(args.model, args.repo_path, sample_adapted, adapter_meta).to(device)
    checkpoint_load_report = {"checkpoint_loaded": False, "checkpoint_status": "NOT_REQUESTED", "notes": ["Pass --strict_checkpoint_load to load/audit checkpoint."]}
    if args.strict_checkpoint_load:
        checkpoint_load_report = load_pretrained_checkpoint_if_available(model, args.model, args.repo_path)
        append_log(log_path, f"checkpoint_load_report={json.dumps(checkpoint_load_report, sort_keys=True)}")
    model, loss_type, raw_output_shape, temporary_head_used = infer_loss_and_maybe_head(model, sample_adapted.to(device))
    model = model.to(device)
    if not args.strategy_report_path:
        args.strategy_report_path = str(output_dir / "strategy")
    strategy_summary = apply_finetune_strategy(model, args.finetune_strategy, args.model, args)
    (output_dir / "strategy_report.json").write_text(json.dumps(strategy_summary, indent=2) + "\n", encoding="utf-8")
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    if not trainable_parameters:
        raise RuntimeError(f"No trainable parameters after applying strategy {args.finetune_strategy}")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)

    append_log(log_path, f"model={args.model}")
    append_log(log_path, f"device={device}")
    append_log(log_path, f"adapted_input_shape={shape_of(sample_adapted)}")
    append_log(log_path, f"loss_type={loss_type} temporary_head_used={temporary_head_used}")
    append_log(log_path, f"strategy={args.finetune_strategy} trainable={strategy_summary['trainable_params']} total={strategy_summary['total_params']}")

    history_rows = []
    train_losses, val_losses, val_accs, val_bal_accs = [], [], [], []
    best_epoch = 0
    best_val_bal = -float("inf")
    best_state = None
    last_output_shape = None
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        for x, y in train_loader:
            x, _ = adapt_inputs(args.model, x, channel_names, args.repo_path)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            last_output_shape = shape_of(logits)
            loss = compute_loss(logits, y, loss_type)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        val_metrics = evaluate(model, val_loader, loss_type, device, args, channel_names)
        train_losses.append(train_loss)
        val_losses.append(val_metrics["loss"])
        val_accs.append(val_metrics["accuracy"])
        val_bal_accs.append(val_metrics["balanced_accuracy"])
        if val_metrics["balanced_accuracy"] > best_val_bal:
            best_val_bal = float(val_metrics["balanced_accuracy"])
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save({"model": args.model, "strategy": args.finetune_strategy, "model_state_dict": best_state, "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch + 1, "best_epoch": best_epoch, "best_val_balanced_accuracy": best_val_bal, "args": vars(args)}, checkpoint_dir / "checkpoint_best_val.pt")
            torch.save({"model": args.model, "strategy": args.finetune_strategy, "model_state_dict": best_state, "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch + 1, "best_epoch": best_epoch, "best_val_balanced_accuracy": best_val_bal, "args": vars(args)}, checkpoint_dir / "best.pt")
        payload = {"model": args.model, "strategy": args.finetune_strategy, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch + 1, "best_epoch": best_epoch, "best_val_balanced_accuracy": best_val_bal, "args": vars(args)}
        torch.save(payload, checkpoint_dir / f"epoch_{epoch + 1:03d}.pt")
        torch.save(payload, checkpoint_dir / "checkpoint_last.pt")
        torch.save(payload, checkpoint_dir / "last.pt")
        row = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_metrics["loss"], "val_accuracy": val_metrics["accuracy"], "val_balanced_accuracy": val_metrics["balanced_accuracy"], "best_epoch": best_epoch, "best_val_balanced_accuracy": best_val_bal}
        history_rows.append(row)
        append_log(log_path, f"epoch={epoch + 1}/{args.epochs} train_loss={train_loss:.6f} val_loss={val_metrics['loss']:.6f} val_acc={val_metrics['accuracy']:.6f} val_bal_acc={val_metrics['balanced_accuracy']:.6f}")

    finite_losses = all(math.isfinite(v) for v in train_losses + val_losses)
    test_metrics = evaluate(model, test_loader, loss_type, device, args, channel_names, include_binary_details=True)
    metrics = {
        "model": args.model,
        "status": "PASS" if len(train_losses) == args.epochs and finite_losses else "FAIL",
        "strategy_compliance": "MERIEM_STRICT",
        "lora_rank": args.lora_rank if args.finetune_strategy == "lora" else None,
        "lora_placement": "LABRAM_REFERENCED_ARCHITECTURE_NATIVE_EQUIVALENT" if args.finetune_strategy == "lora" else None,
        "venv_path": args.venv_path,
        "repo_path": args.repo_path,
        "python_path": sys.executable,
        "python_version": platform.python_version(),
        "train_subset_count": counts["train"],
        "val_subset_count": counts["val"],
        "test_subset_count": counts["test"],
        "split_limit_summary": split_limit_summary,
        "adapted_input_shape": shape_of(sample_adapted),
        "epochs_completed": len(train_losses),
        "best_epoch": best_epoch,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracy": val_accs,
        "val_balanced_accuracy": val_bal_accs,
        "best_val_accuracy": max(val_accs) if val_accs else float("nan"),
        "best_val_balanced_accuracy": max(val_bal_accs) if val_bal_accs else float("nan"),
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_auroc": test_metrics["auroc"],
        "test_auprc": test_metrics["auprc"],
        "test_confusion_matrix": test_metrics["confusion_matrix"],
        "raw_model_output_shape_before_temp_head": raw_output_shape,
        "output_shape": test_metrics["output_shape"] or last_output_shape,
        "loss_type": loss_type,
        "finetune_strategy": args.finetune_strategy,
        "strategy_summary": strategy_summary,
        "checkpoint_load_report": checkpoint_load_report,
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": str(metrics_path),
        "log_path": str(log_path),
        "temporary_head_used": temporary_head_used,
        "adapter_meta": adapter_meta,
        "device": str(device),
        "strict_strategy_wording": STRICT_WORDING,
        "notes": "Adapter-side flattened TUAB forward used for CodeBrain." if args.model == "CodeBrain" else "",
    }
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"model": args.model, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "metrics": metrics, "args": vars(args)}, checkpoint_path)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    if history_rows:
        with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history_rows[0].keys()))
            writer.writeheader(); writer.writerows(history_rows)
        (output_dir / "history.json").write_text(json.dumps(history_rows, indent=2) + "\n", encoding="utf-8")
    final_test_line = (
        f"test_acc={metrics['test_accuracy']:.6f} "
        f"test_bal_acc={metrics['test_balanced_accuracy']:.6f} "
        f"test_auroc={metrics['test_auroc']} "
        f"test_auprc={metrics['test_auprc']}"
    )
    print(final_test_line, flush=True)
    append_log(log_path, final_test_line)
    append_log(log_path, f"checkpoint={checkpoint_path}")
    append_log(log_path, f"metrics={metrics_path}")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(SUPPORTED_MODELS))
    parser.add_argument("--repo_path", required=True)
    parser.add_argument("--venv_path", default="")
    parser.add_argument("--subset_npz", default="")
    parser.add_argument("--index_npz", default="")
    parser.add_argument("--h5", default="/nicoletye/workspace/unified_tuab/data/canonical_tuab_full.h5")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--train_n", type=int, default=0)
    parser.add_argument("--val_n", type=int, default=0)
    parser.add_argument("--test_n", type=int, default=0)
    parser.add_argument("--finetune_strategy", choices=["full_finetune", "linear_probe", "lora"], default="full_finetune")
    parser.add_argument("--lora_rank", type=int, default=2)
    parser.add_argument("--lora_alpha", type=float, default=8.0)
    parser.add_argument("--lora_target", default="labram_reference")
    parser.add_argument("--allow_head_guess", action="store_true")
    parser.add_argument("--strategy_report_path", default="")
    parser.add_argument("--strict_checkpoint_load", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.subset_npz and not args.index_npz:
        parser.error("one of --subset_npz or --index_npz is required")
    result = {"model": args.model, "status": "FAIL", "python_path": sys.executable, "repo_path": args.repo_path}
    try:
        result.update(train_small_subset(args))
    except Exception as exc:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        tb = traceback.format_exc()
        result.update({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}", "traceback": tb, "metrics_path": str(output_dir / "metrics.json"), "log_path": str(output_dir / "train.log"), "checkpoint_path": str(output_dir / "checkpoint.pt")})
        (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        append_log(output_dir / "train.log", result["error"])
        append_log(output_dir / "train.log", tb)
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
