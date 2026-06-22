#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-model small-subset training validation worker for EEG-FM TUAB adapters."""

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

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eegfm_tiny_train_worker import (
    adapt_inputs,
    build_model,
    compute_loss,
    infer_loss_and_maybe_head,
    load_pretrained_checkpoint_if_available,
    load_tiny_npz,
    shape_of,
)
from finetune_strategy_utils import apply_finetune_strategy


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
            h5py = import_h5py()
            self._h5 = h5py.File(self.h5_path, "r")
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
        "split_counts": json.loads(str(data["split_counts_json"][0])),
        "tiny_counts": json.loads(str(data["tiny_counts_json"][0])),
    }


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
        f.flush()


def _split_length(batch: dict, split: str, index_mode: bool) -> int:
    if index_mode:
        return int(len(batch[f"{split}_h5_indices"]))
    return int(len(batch[f"{split}_x"]))


def _subset_array(value, selected: np.ndarray):
    if torch.is_tensor(value):
        return value[torch.as_tensor(selected, dtype=torch.long)]
    return value[selected]


def _limit_one_split(
    batch: dict,
    split: str,
    requested_n: int,
    seed: int,
    index_mode: bool,
) -> tuple[int, int, str | None]:
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
    limits = {
        "train": int(args.train_n or 0),
        "val": int(args.val_n or 0),
        "test": int(args.test_n or 0),
    }
    seed_offsets = {"train": 0, "val": 1009, "test": 2003}
    summary = {
        "requested_train_n": limits["train"],
        "requested_val_n": limits["val"],
        "requested_test_n": limits["test"],
        "original_train_count": None,
        "original_val_count": None,
        "original_test_count": None,
        "actual_train_count": None,
        "actual_val_count": None,
        "actual_test_count": None,
        "split_limit_warnings": [],
    }
    for split in ["train", "val", "test"]:
        original, used, warning = _limit_one_split(
            batch,
            split,
            limits[split],
            seed=int(args.seed) + seed_offsets[split],
            index_mode=index_mode,
        )
        summary[f"original_{split}_count"] = original
        summary[f"actual_{split}_count"] = used
        if warning:
            summary["split_limit_warnings"].append(warning)
    return summary


def predictions_from_logits(logits: torch.Tensor, loss_type: str) -> torch.Tensor:
    if loss_type == "CrossEntropyLoss":
        return logits.argmax(dim=1).long()
    if loss_type == "BCEWithLogitsLoss":
        return (logits.reshape(-1) >= 0).long()
    raise ValueError(f"Unsupported loss type: {loss_type}")


def positive_scores_from_logits(logits: torch.Tensor, loss_type: str) -> torch.Tensor:
    if loss_type == "CrossEntropyLoss":
        return torch.softmax(logits, dim=1)[:, 1]
    if loss_type == "BCEWithLogitsLoss":
        return torch.sigmoid(logits.reshape(-1))
    raise ValueError(f"Unsupported loss type: {loss_type}")


def simple_metrics(preds: list[int], labels: list[int]) -> dict:
    pred_arr = np.asarray(preds, dtype=np.int64)
    label_arr = np.asarray(labels, dtype=np.int64)
    acc = float((pred_arr == label_arr).mean()) if len(label_arr) else float("nan")
    recalls = []
    for label in [0, 1]:
        mask = label_arr == label
        if mask.any():
            recalls.append(float((pred_arr[mask] == label).mean()))
    bal_acc = float(np.mean(recalls)) if recalls else float("nan")
    return {"accuracy": acc, "balanced_accuracy": bal_acc}


def detailed_binary_metrics(preds: list[int], labels: list[int], positive_scores: list[float]) -> dict:
    pred_arr = np.asarray(preds, dtype=np.int64)
    label_arr = np.asarray(labels, dtype=np.int64)
    score_arr = np.asarray(positive_scores, dtype=np.float64)
    confusion = np.zeros((2, 2), dtype=np.int64)
    for y_true, y_pred in zip(label_arr.tolist(), pred_arr.tolist()):
        if y_true in {0, 1} and y_pred in {0, 1}:
            confusion[y_true, y_pred] += 1

    notes = []
    unique_labels = np.unique(label_arr)
    auroc = None
    auprc = None
    if len(unique_labels) == 2:
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score

            auroc = float(roc_auc_score(label_arr, score_arr))
            auprc = float(average_precision_score(label_arr, score_arr))
        except Exception as exc:
            notes.append(f"AUROC/AUPRC set to null because sklearn metric computation failed: {type(exc).__name__}: {exc}.")
    else:
        present = ",".join(str(int(x)) for x in unique_labels.tolist()) if len(unique_labels) else "none"
        notes.append(f"AUROC/AUPRC set to null because test labels contain only one class ({present}).")

    return {
        "confusion_matrix": confusion.astype(int).tolist(),
        "auroc": auroc,
        "auprc": auprc,
        "metric_notes": notes,
    }


def evaluate(
    model,
    loader,
    loss_type: str,
    device: torch.device,
    args: argparse.Namespace,
    channel_names: list[str],
    include_binary_details: bool = False,
) -> dict:
    model.eval()
    losses = []
    preds = []
    labels_all = []
    positive_scores = []
    output_shape = None
    with torch.no_grad():
        for x, y in loader:
            x, _ = adapt_inputs(args.model, x, channel_names, args.repo_path)
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            output_shape = shape_of(logits)
            loss = compute_loss(logits, y, loss_type)
            losses.append(float(loss.detach().cpu()))
            preds.extend(predictions_from_logits(logits.detach().cpu(), loss_type).tolist())
            if include_binary_details:
                positive_scores.extend(positive_scores_from_logits(logits.detach().cpu(), loss_type).tolist())
            labels_all.extend(y.detach().cpu().long().tolist())
    metrics = simple_metrics(preds, labels_all)
    metrics.update(
        {
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "output_shape": output_shape,
        }
    )
    if include_binary_details:
        metrics.update(detailed_binary_metrics(preds, labels_all, positive_scores))
    return metrics


def make_data_loaders(args: argparse.Namespace, channel_names: list[str], raw_batch: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    if args.index_npz:
        train_ds = H5IndexDataset(args.h5, raw_batch["train_h5_indices"], raw_batch["train_y"])
        val_ds = H5IndexDataset(args.h5, raw_batch["val_h5_indices"], raw_batch["val_y"])
        test_ds = H5IndexDataset(args.h5, raw_batch["test_h5_indices"], raw_batch["test_y"])
    else:
        train_ds = TensorDataset(raw_batch["train_x"], raw_batch["train_y"])
        val_ds = TensorDataset(raw_batch["val_x"], raw_batch["val_y"])
        test_ds = TensorDataset(raw_batch["test_x"], raw_batch["test_y"])
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and resolve_device(args.device).type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and resolve_device(args.device).type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and resolve_device(args.device).type == "cuda",
    )
    return train_loader, val_loader, test_loader


def train_small_subset(args: argparse.Namespace) -> dict:
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
    batch = load_index_npz(args.index_npz) if index_mode else load_tiny_npz(args.subset_npz)
    channel_names = batch["channel_names"]
    split_limit_summary = apply_split_limits(args, batch, index_mode=index_mode)
    for warning in split_limit_summary["split_limit_warnings"]:
        print(warning, flush=True)
        append_log(log_path, warning)
    if index_mode:
        train_count = int(len(batch["train_h5_indices"]))
        val_count = int(len(batch["val_h5_indices"]))
        test_count = int(len(batch["test_h5_indices"]))
    else:
        train_count = int(len(batch["train_x"]))
        val_count = int(len(batch["val_x"]))
        test_count = int(len(batch["test_x"]))
    count_message = f"split_counts train={train_count} val={val_count} test={test_count}"
    print(count_message, flush=True)

    train_loader, val_loader, test_loader = make_data_loaders(args, channel_names, batch)
    first_raw, _ = next(iter(train_loader))
    sample_adapted, adapter_meta = adapt_inputs(args.model, first_raw, channel_names, args.repo_path)

    device = resolve_device(args.device)
    model = build_model(args.model, args.repo_path, sample_adapted, adapter_meta).to(device)
    checkpoint_load_report = {
        "checkpoint_loaded": False,
        "checkpoint_status": "NOT_REQUESTED",
        "notes": ["Pass --strict_checkpoint_load to audit/load model checkpoint before strategy application."],
    }
    if args.strict_checkpoint_load:
        checkpoint_load_report = load_pretrained_checkpoint_if_available(
            model,
            args.model,
            args.repo_path,
            args.pretrained_checkpoint,
        )
        append_log(log_path, f"checkpoint_load_report={json.dumps(checkpoint_load_report, sort_keys=True)}")
    model, loss_type, raw_output_shape, temporary_head_used = infer_loss_and_maybe_head(model, sample_adapted.to(device))
    model = model.to(device)

    if args.freeze_backbone and args.finetune_strategy == "full_finetune":
        args.finetune_strategy = "linear_probe"
    if not args.strategy_report_path:
        args.strategy_report_path = str(output_dir / "strategy")
    strategy_summary = apply_finetune_strategy(model, args.finetune_strategy, args.model, args)
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    if not trainable_parameters:
        raise RuntimeError(f"No trainable parameters after applying strategy {args.finetune_strategy}")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    train_losses = []
    val_losses = []
    val_accs = []
    val_bal_accs = []
    history_rows = []
    epochs_completed = 0
    last_output_shape = None
    best_epoch = 0
    best_val_bal_acc_for_checkpoint = -float("inf")
    best_state = None

    append_log(log_path, f"model={args.model}")
    append_log(log_path, f"device={device}")
    append_log(log_path, f"train={train_count} val={val_count} test={test_count}")
    append_log(log_path, f"adapted_input_shape={shape_of(sample_adapted)}")
    append_log(log_path, f"loss_type={loss_type} temporary_head_used={temporary_head_used}")
    append_log(
        log_path,
        (
            f"finetune_strategy={args.finetune_strategy} "
            f"trainable_params={strategy_summary['trainable_params']} "
            f"total_params={strategy_summary['total_params']}"
        ),
    )

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        for x, y in train_loader:
            x, _ = adapt_inputs(args.model, x, channel_names, args.repo_path)
            x = x.to(device)
            y = y.to(device)
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
        epochs_completed += 1
        if val_metrics["balanced_accuracy"] > best_val_bal_acc_for_checkpoint:
            best_val_bal_acc_for_checkpoint = float(val_metrics["balanced_accuracy"])
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "model": args.model,
                    "strategy": args.finetune_strategy,
                    "model_state_dict": best_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "best_epoch": best_epoch,
                    "best_val_balanced_accuracy": best_val_bal_acc_for_checkpoint,
                    "args": vars(args),
                },
                checkpoint_dir / "checkpoint_best_val.pt",
            )
            torch.save(
                {
                    "model": args.model,
                    "strategy": args.finetune_strategy,
                    "model_state_dict": best_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "best_epoch": best_epoch,
                    "best_val_balanced_accuracy": best_val_bal_acc_for_checkpoint,
                    "args": vars(args),
                },
                checkpoint_dir / "best.pt",
            )
        epoch_payload = {
            "model": args.model,
            "strategy": args.finetune_strategy,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch + 1,
            "best_epoch": best_epoch,
            "best_val_balanced_accuracy": best_val_bal_acc_for_checkpoint,
            "args": vars(args),
        }
        torch.save(epoch_payload, checkpoint_dir / f"epoch_{epoch + 1:03d}.pt")
        torch.save(epoch_payload, checkpoint_dir / "checkpoint_last.pt")
        torch.save(epoch_payload, checkpoint_dir / "last.pt")
        history_rows.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "best_epoch": best_epoch,
                "best_val_balanced_accuracy": best_val_bal_acc_for_checkpoint,
            }
        )

        append_log(
            log_path,
            (
                f"epoch={epoch + 1}/{args.epochs} train_loss={train_loss:.6f} "
                f"val_loss={val_metrics['loss']:.6f} val_acc={val_metrics['accuracy']:.6f} "
                f"val_bal_acc={val_metrics['balanced_accuracy']:.6f}"
            ),
        )

    finite_losses = all(math.isfinite(v) for v in train_losses + val_losses)
    best_val_acc = max(val_accs) if val_accs else float("nan")
    best_val_bal_acc = max(val_bal_accs) if val_bal_accs else float("nan")
    test_metrics = evaluate(model, test_loader, loss_type, device, args, channel_names, include_binary_details=True)

    metrics = {
        "model": args.model,
        "status": "PASS" if epochs_completed == args.epochs and finite_losses else "FAIL",
        "venv_path": args.venv_path,
        "repo_path": args.repo_path,
        "python_path": sys.executable,
        "python_version": platform.python_version(),
        "train_subset_count": train_count,
        "val_subset_count": val_count,
        "test_subset_count": test_count,
        "requested_train_n": args.train_n,
        "requested_val_n": args.val_n,
        "requested_test_n": args.test_n,
        "split_limit_summary": split_limit_summary,
        "adapted_input_shape": shape_of(sample_adapted),
        "epochs_completed": epochs_completed,
        "best_epoch": best_epoch,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracy": val_accs,
        "val_balanced_accuracy": val_bal_accs,
        "best_val_accuracy": best_val_acc,
        "best_val_balanced_accuracy": best_val_bal_acc,
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
        "notes": "",
    }
    notes = []
    if temporary_head_used:
        notes.append("Adapter-side temporary linear classification head used for feature-shaped output.")
    if args.model == "CodeBrain":
        notes.append("Adapter-side flattened TUAB forward used; original broken TUAB forward path is not used.")
    notes.extend(test_metrics.get("metric_notes", []))
    metrics["notes"] = " ".join(notes)

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(
        {
            "model": args.model,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "args": vars(args),
        },
        checkpoint_path,
    )
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    history_path = output_dir / "history.csv"
    if history_rows:
        with history_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history_rows[0].keys()))
            writer.writeheader()
            writer.writerows(history_rows)
        (output_dir / "history.json").write_text(json.dumps(history_rows, indent=2) + "\n", encoding="utf-8")
    append_log(log_path, f"checkpoint={checkpoint_path}")
    append_log(log_path, f"checkpoint_last={checkpoint_dir / 'last.pt'}")
    append_log(log_path, f"checkpoint_best={checkpoint_dir / 'best.pt'}")
    append_log(log_path, f"metrics={metrics_path}")
    append_log(log_path, f"history={history_path}")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
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
    parser.add_argument("--lora_target", default="auto")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--allow_head_guess", action="store_true")
    parser.add_argument("--strategy_report_path", default="")
    parser.add_argument("--strict_checkpoint_load", action="store_true")
    parser.add_argument("--pretrained_checkpoint", default="")
    args = parser.parse_args()
    if not args.subset_npz and not args.index_npz:
        parser.error("one of --subset_npz or --index_npz is required")
    for name in ["train_n", "val_n", "test_n"]:
        if getattr(args, name) < 0:
            parser.error(f"--{name} must be >= 0")

    result = {
        "model": args.model,
        "venv_path": args.venv_path,
        "repo_path": args.repo_path,
        "status": "FAIL",
        "python_path": sys.executable,
    }
    try:
        result.update(train_small_subset(args))
    except Exception as exc:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        tb = traceback.format_exc()
        result.update(
            {
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": tb,
                "metrics_path": str(output_dir / "metrics.json"),
                "log_path": str(output_dir / "train.log"),
                "checkpoint_path": str(output_dir / "checkpoint.pt"),
            }
        )
        (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        append_log(output_dir / "train.log", result["error"])
        append_log(output_dir / "train.log", tb)
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
