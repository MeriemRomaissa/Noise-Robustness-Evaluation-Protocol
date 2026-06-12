#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# EEGNET_TUAB_V2.py — LaBraM-aligned EEGNet training for TUAB
#
# Changes vs. V1 (fairness fixes for comparison with LaBraM):
#   1. No subsetting by default (subset_train/val/test=0) — matches LaBraM full-dataset training
#   2. No class weighting by default (use_class_weights=0) — LaBraM uses plain BCEWithLogitsLoss
#   3. No early stopping by default (early_stop_patience=9999) — LaBraM trains all epochs
#   4. Metrics aligned to LaBraM TUAB log keys: pr_auc + roc_auc instead of cohen_kappa + f1

import os
import sys
import json
import math
import pickle
import random
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
)
from tqdm import tqdm
from braindecode.models import EEGNet


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lr(epoch: int, warmup_epochs: int, total_epochs: int, base_lr: float, min_lr: float = 1e-8) -> float:
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer, lr: float):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def ensure_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_tuab_label_cache(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        cache = json.load(f)

    if cache.get("dataset") != "TUAB":
        raise ValueError(f"Expected TUAB cache, got: {cache.get('dataset')}")

    return cache


class TUABPKLDataset(Dataset):
    X_KEYS = ["X", "x", "signal", "data", "eeg"]
    Y_KEYS = ["y", "label", "Y", "target"]

    def __init__(self, root: str, files: List[str], label_offset: int = 0, labels: List[int] = None):
        self.root = root
        self.files = files
        self.label_offset = label_offset
        self.labels = labels

    def __len__(self):
        return len(self.files)

    def _find_key(self, sample: Dict, candidates: List[str], kind: str):
        for k in candidates:
            if k in sample:
                return k
        raise KeyError(f"Cannot find {kind} key. Available keys: {list(sample.keys())}")

    def _read_pickle(self, index: int):
        path = os.path.join(self.root, self.files[index])
        with open(path, "rb") as f:
            sample = pickle.load(f)
        return sample, path

    def _extract_label_from_sample(self, sample: Dict) -> int:
        y_key = self._find_key(sample, self.Y_KEYS, "y")
        y = sample[y_key]

        if isinstance(y, np.ndarray):
            if y.ndim == 0:
                return int(y.item()) - self.label_offset
            return int(y.reshape(-1)[0]) - self.label_offset

        if hasattr(y, "item"):
            try:
                return int(y.item()) - self.label_offset
            except Exception:
                pass

        if hasattr(y, "__len__") and not isinstance(y, (str, bytes)) and len(y) > 0:
            return int(y[0]) - self.label_offset

        return int(y) - self.label_offset

    def __getitem__(self, index):
        sample, path = self._read_pickle(index)
        x_key = self._find_key(sample, self.X_KEYS, "X")
        # LaBraM alignment: normalize EEG amplitude by /100.0 (engine_for_finetuning.py:64)
        X = ensure_numpy(sample[x_key]).astype(np.float32) / 100.0

        if self.labels is not None:
            y = int(self.labels[index])
        else:
            y = self._extract_label_from_sample(sample)

        if X.ndim == 3:
            if X.shape[0] == 1:
                X = X[0]
            elif X.shape[-1] == 1:
                X = X[..., 0]
            else:
                raise ValueError(f"Unsupported 3D shape {X.shape} in {path}")
        elif X.ndim != 2:
            raise ValueError(f"Expected 2D/3D X, got {X.shape} in {path}")

        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def prepare_tuab_full_dataset(data_root, label_cache_json, label_offset=0):
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")

    for d in [train_dir, val_dir, test_dir]:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Missing directory: {d}")

    train_files = sorted([f for f in os.listdir(train_dir) if f.endswith(".pkl")])
    val_files = sorted([f for f in os.listdir(val_dir) if f.endswith(".pkl")])
    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".pkl")])

    cache = load_tuab_label_cache(label_cache_json)

    cache_root = str(Path(cache["data_root"]).resolve())
    data_root_resolved = str(Path(data_root).resolve())
    if cache_root != data_root_resolved:
        print(f"WARNING: cache data_root = {cache_root}")
        print(f"WARNING: current data_root = {data_root_resolved}")
        print("WARNING: continuing, but please make sure they match the same TUAB dataset.")

    train_labels = cache["splits"]["train"]["labels"]
    val_labels = cache["splits"]["val"]["labels"]
    test_labels = cache["splits"]["test"]["labels"]

    if len(train_files) != len(train_labels):
        raise ValueError("Train file count does not match cached train labels.")
    if len(val_files) != len(val_labels):
        raise ValueError("Val file count does not match cached val labels.")
    if len(test_files) != len(test_labels):
        raise ValueError("Test file count does not match cached test labels.")

    train_ds = TUABPKLDataset(train_dir, train_files, label_offset=label_offset, labels=train_labels)
    val_ds = TUABPKLDataset(val_dir, val_files, label_offset=label_offset, labels=val_labels)
    test_ds = TUABPKLDataset(test_dir, test_files, label_offset=label_offset, labels=test_labels)
    return train_ds, val_ds, test_ds


def make_binary_class_aware_subset(dataset, labels, target_size, seed):
    """Return (Subset, selected_indices, subset_labels). target_size<=0 returns full dataset."""
    labels = np.asarray(labels, dtype=np.int64)
    rng = random.Random(seed)

    idx0 = np.where(labels == 0)[0].tolist()
    idx1 = np.where(labels == 1)[0].tolist()

    rng.shuffle(idx0)
    rng.shuffle(idx1)

    if target_size <= 0:
        selected = list(range(len(labels)))
        rng.shuffle(selected)
        subset = Subset(dataset, selected)
        subset_labels = labels[selected]
        return subset, selected, subset_labels

    half = target_size // 2
    take0 = min(half, len(idx0))
    take1 = min(half, len(idx1))

    selected = idx0[:take0] + idx1[:take1]

    remaining_needed = target_size - len(selected)
    remaining_pool = idx0[take0:] + idx1[take1:]
    rng.shuffle(remaining_pool)
    selected.extend(remaining_pool[:remaining_needed])

    rng.shuffle(selected)

    subset = Subset(dataset, selected)
    subset_labels = labels[selected]
    return subset, selected, subset_labels


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Returns loss, accuracy, balanced_accuracy, pr_auc, roc_auc.
    pr_auc and roc_auc use class-1 softmax probability as the positive score,
    matching LaBraM's TUAB binary metric set (val_pr_auc, val_roc_auc).
    """
    model.eval()
    total_loss = 0.0
    all_logits, all_preds, all_labels = [], [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)

        all_logits.append(logits.cpu())
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    logits_all = torch.cat(all_logits, dim=0)

    # Softmax probability for the positive class (class 1)
    proba_pos = torch.softmax(logits_all, dim=1)[:, 1].numpy()

    return {
        "loss": float(total_loss / len(labels)),
        "accuracy": float((preds == labels).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "pr_auc": float(average_precision_score(labels, proba_pos)),
        "roc_auc": float(roc_auc_score(labels, proba_pos)),
    }


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for X, y in tqdm(loader, desc="Training", leave=False):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()

        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)

        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    return (
        float(total_loss / len(labels)),
        float((preds == labels).mean()),
        float(balanced_accuracy_score(labels, preds)),
    )


def main():
    parser = argparse.ArgumentParser(
        description="TUAB EEGNet training — LaBraM-aligned (V2, fair comparison)"
    )

    parser.add_argument("--data_root", type=str, default=r"E:\processed")
    parser.add_argument("--label_cache_json", type=str, default=r"D:\EEGNet\tuab_label_cache.json")
    parser.add_argument("--output_dir", type=str, default=r"D:\EEGNet\checkpoints\TUAB\EEGNET\v2_labram_aligned\seed_42")

    # FIX 1: default=0 → use full dataset, matching LaBraM's prepare_TUAB_dataset (no subsetting)
    parser.add_argument("--subset_train", type=int, default=0)
    parser.add_argument("--subset_val", type=int, default=0)
    parser.add_argument("--subset_test", type=int, default=0)
    parser.add_argument("--label_offset", type=int, default=0)

    parser.add_argument("--n_chans", type=int, default=23)
    parser.add_argument("--n_times", type=int, default=2000)
    parser.add_argument("--n_outputs", type=int, default=2)

    # EEGNet architecture — original published values (Lawhern et al. 2018), not tuned
    parser.add_argument("--F1", type=int, default=8)
    parser.add_argument("--D", type=int, default=2)
    parser.add_argument("--F2", type=int, default=16)
    parser.add_argument("--kernel_length", type=int, default=64)
    parser.add_argument("--depthwise_kernel_length", type=int, default=16)
    parser.add_argument("--pool1_kernel_size", type=int, default=4)
    parser.add_argument("--pool2_kernel_size", type=int, default=8)
    parser.add_argument("--drop_prob", type=float, default=0.25)
    parser.add_argument("--conv_spatial_max_norm", type=float, default=1.0)
    parser.add_argument("--norm_rate", type=float, default=0.25)
    parser.add_argument("--final_layer_with_constraint", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--pin_memory", type=int, default=0)
    parser.add_argument("--save_ckpt_freq", type=int, default=1)

    # FIX 2: default=0 — LaBraM uses plain CrossEntropyLoss/BCE with no class weights
    parser.add_argument("--use_class_weights", type=int, default=0)

    # FIX 3: default=9999 — LaBraM has no early stopping; set absurdly high to disable
    parser.add_argument("--early_stop_patience", type=int, default=9999)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    seed_everything(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    subset_desc = (
        f"train={args.subset_train or 'ALL'}, val={args.subset_val or 'ALL'}, test={args.subset_test or 'ALL'}"
    )
    print("=" * 90)
    print("TUAB EEGNet Training — V2 (LaBraM-aligned, fair comparison)")
    print(f"Device           : {args.device}")
    print(f"Data root        : {args.data_root}")
    print(f"Label cache      : {args.label_cache_json}")
    print(f"Output dir       : {args.output_dir}")
    print(f"Dataset sizes    : {subset_desc}")
    print(f"Class weights    : {'ON' if args.use_class_weights else 'OFF (LaBraM-aligned)'}")
    print(f"Early stopping   : patience={args.early_stop_patience} ({'disabled' if args.early_stop_patience >= 9999 else 'active'})")
    print(f"EEG normalisation: /100.0  (matches LaBraM engine_for_finetuning.py:64)")
    print(f"Metrics          : accuracy, balanced_accuracy, pr_auc, roc_auc  (matches LaBraM TUAB log keys)")
    print(f"Loader           : num_workers={args.num_workers}, pin_memory={args.pin_memory}")
    print("=" * 90)

    train_full, val_full, test_full = prepare_tuab_full_dataset(
        data_root=args.data_root,
        label_cache_json=args.label_cache_json,
        label_offset=args.label_offset,
    )

    train_ds, _, train_labels = make_binary_class_aware_subset(
        train_full, train_full.labels, args.subset_train, args.seed
    )
    val_ds, _, val_labels = make_binary_class_aware_subset(
        val_full, val_full.labels, args.subset_val, args.seed + 1
    )
    test_ds, _, test_labels = make_binary_class_aware_subset(
        test_full, test_full.labels, args.subset_test, args.seed + 2
    )

    print(f"Train samples : {len(train_labels):,}  class counts: {np.bincount(train_labels, minlength=2).tolist()}")
    print(f"Val   samples : {len(val_labels):,}  class counts: {np.bincount(val_labels, minlength=2).tolist()}")
    print(f"Test  samples : {len(test_labels):,}  class counts: {np.bincount(test_labels, minlength=2).tolist()}")

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": bool(args.pin_memory),
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    model = EEGNet(
        n_chans=args.n_chans,
        n_outputs=args.n_outputs,
        n_times=args.n_times,
        F1=args.F1,
        D=args.D,
        F2=args.F2,
        kernel_length=args.kernel_length,
        depthwise_kernel_length=args.depthwise_kernel_length,
        pool1_kernel_size=args.pool1_kernel_size,
        pool2_kernel_size=args.pool2_kernel_size,
        conv_spatial_max_norm=args.conv_spatial_max_norm,
        drop_prob=args.drop_prob,
        final_layer_with_constraint=bool(args.final_layer_with_constraint),
        norm_rate=args.norm_rate,
    ).to(args.device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}  (LaBraM-base: 5,820,137)")

    # FIX 2: class weights OFF by default — enable with --use_class_weights 1
    if args.use_class_weights:
        counts = np.bincount(train_labels, minlength=args.n_outputs).astype(np.float32)
        weights = counts.sum() / np.maximum(counts, 1.0)
        weights = weights / weights.sum() * len(weights)
        class_weights = torch.tensor(weights, dtype=torch.float32, device=args.device)
        print(f"Class counts : {counts.astype(int)}")
        print(f"Class weights: {class_weights.cpu().numpy().round(4)}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    log_path = out_dir / "log.txt"
    best_val_bal = -1.0
    best_val_acc = -1.0
    no_improve_count = 0

    print("-" * 110)
    print(
        f"{'Epoch':>5} | {'LR':^10} | {'Train Loss':^11} | {'Train Acc':^10} | {'Train Bal':^10} | "
        f"{'Val Acc':^9} | {'Val Bal':^9} | {'Val ROC':^9} | {'Test Acc':^9} | {'Test Bal':^9} | {'Test ROC':^9}"
    )
    print("-" * 110)

    for epoch in range(args.epochs):
        lr = get_lr(epoch, args.warmup_epochs, args.epochs, args.lr, min_lr=args.min_lr)
        set_lr(optimizer, lr)

        train_loss, train_acc, train_bal = train_one_epoch(
            model, train_loader, criterion, optimizer, args.device, grad_clip=args.grad_clip
        )
        val_metrics = evaluate(model, val_loader, criterion, args.device)
        test_metrics = evaluate(model, test_loader, criterion, args.device)

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), out_dir / "best_model_by_val_accuracy.pt")

        improved_bal = val_metrics["balanced_accuracy"] > (best_val_bal + args.early_stop_min_delta)
        if improved_bal:
            best_val_bal = val_metrics["balanced_accuracy"]
            no_improve_count = 0
            torch.save(model.state_dict(), out_dir / "best_model_by_val_balanced_accuracy.pt")
        else:
            no_improve_count += 1

        if (epoch + 1) % args.save_ckpt_freq == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                    "best_val_bal": best_val_bal,
                    "args": vars(args),
                },
                out_dir / f"checkpoint_epoch_{epoch}.pt",
            )

        # FIX 4: log keys match LaBraM TUAB log structure
        #   added:   train_min_lr, val_pr_auc, val_roc_auc, test_pr_auc, test_roc_auc
        #   removed: val_cohen_kappa, val_f1_weighted, test_cohen_kappa, test_f1_weighted,
        #            early_stop_no_improve_count, best_val_balanced_accuracy_so_far
        log_stats = {
            "epoch": epoch,
            "train_lr": round(lr, 10),
            "train_min_lr": round(args.min_lr, 10),
            "train_loss": round(train_loss, 6),
            "train_class_acc": round(train_acc, 6),
            "train_balanced_accuracy": round(train_bal, 6),
            "train_weight_decay": args.weight_decay,
            "val_loss": round(val_metrics["loss"], 6),
            "val_accuracy": round(val_metrics["accuracy"], 6),
            "val_balanced_accuracy": round(val_metrics["balanced_accuracy"], 6),
            "val_pr_auc": round(val_metrics["pr_auc"], 6),
            "val_roc_auc": round(val_metrics["roc_auc"], 6),
            "test_loss": round(test_metrics["loss"], 6),
            "test_accuracy": round(test_metrics["accuracy"], 6),
            "test_balanced_accuracy": round(test_metrics["balanced_accuracy"], 6),
            "test_pr_auc": round(test_metrics["pr_auc"], 6),
            "test_roc_auc": round(test_metrics["roc_auc"], 6),
            "n_parameters": n_params,
        }

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")

        print(
            f"{epoch:5d} | {lr:.2e} | {train_loss:11.4f} | {train_acc:10.4f} | {train_bal:10.4f} | "
            f"{val_metrics['accuracy']:9.4f} | {val_metrics['balanced_accuracy']:9.4f} | {val_metrics['roc_auc']:9.4f} | "
            f"{test_metrics['accuracy']:9.4f} | {test_metrics['balanced_accuracy']:9.4f} | {test_metrics['roc_auc']:9.4f}"
        )

        # FIX 3: early stopping is effectively disabled when patience=9999
        if no_improve_count >= args.early_stop_patience:
            print("=" * 90)
            print(f"Early stopping triggered at epoch {epoch}.")
            print(
                f"No val balanced_accuracy improvement > {args.early_stop_min_delta} "
                f"for {args.early_stop_patience} consecutive epochs."
            )
            print("=" * 90)
            break

    print("=" * 90)
    print("Done.")
    print(f"Best val accuracy         : {best_val_acc:.4f}")
    print(f"Best val balanced accuracy: {best_val_bal:.4f}")
    print(f"Saved to                  : {out_dir}")
    print("=" * 90)


if __name__ == "__main__":
    main()
