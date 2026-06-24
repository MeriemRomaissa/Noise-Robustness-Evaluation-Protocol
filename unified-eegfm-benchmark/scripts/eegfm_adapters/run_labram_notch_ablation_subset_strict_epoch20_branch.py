#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run one LaBraM strict-recipe epoch-20 notch-ablation subset branch."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_labram_full_unified_h5_epoch10 as base  # noqa: E402
import run_labram_unified_h5_strict_original_engine as strict  # noqa: E402


DEFAULT_H5 = PROJECT_ROOT / "data" / "canonical_tuab_full.h5"
DEFAULT_EXACT_SPLIT = PROJECT_ROOT / "reports" / "labram_exact_original_processed_split" / "canonical_h5_labram_exact_original_processed_split_index.csv"
DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "labram_notch_ablation_subset_v1" / "labram_notch_ablation_exact_subset_manifest.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "labram_notch_ablation_subset_v1_strict_epoch20_2gpu"
DEFAULT_PROCESSED_ROOT = Path("/nicoletye/datasets/TUAB/processed")
DEFAULT_REPO = Path("/nicoletye/workspace/Noise Robustness Evaluation Protocol/EEG-FM/Labram")
DEFAULT_FINETUNE = DEFAULT_REPO / "checkpoints" / "labram-base.pth"
DEFAULT_UNIFIED_50HZ_NPZ = PROJECT_ROOT / "outputs" / "labram_notch_ablation_subset_v1" / "unified_50hz" / "unified_50hz_subset.npz"
BRANCHES = ["original_pkl_50hz", "unified_50hz", "unified_60hz"]
SPLITS = ["train", "val", "test"]


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_branch_status(output_root: str, branch: str, status: str, extra: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {
        "branch": branch,
        "status": status,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        payload.update(extra)
    save_json(Path(output_root) / branch / "branch_status.json", payload)


def load_manifest(path: Path) -> dict[str, list[dict[str, Any]]]:
    rows = {split: [] for split in SPLITS}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            split = row["canonical_split"]
            if split not in rows:
                continue
            item = dict(row)
            item["h5_index"] = int(item["h5_index"])
            item["label"] = int(item["label"])
            item["window_idx"] = int(item["window_idx"])
            rows[split].append(item)
    return rows


def manifest_counts(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, int]]:
    out = {}
    for split, rows in rows_by_split.items():
        counts = Counter(int(row["label"]) for row in rows)
        out[split] = {"n": len(rows), "normal_0": counts.get(0, 0), "abnormal_1": counts.get(1, 0)}
    return out


class OriginalPklDataset(Dataset):
    def __init__(self, processed_root: Path, rows: list[dict[str, Any]]):
        self.processed_root = processed_root
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        filename = f"{row['recording_id']}_{row['window_idx']}.pkl"
        path = self.processed_root / row["canonical_split"] / filename
        if not path.exists():
            raise FileNotFoundError(f"missing original PKL: {path}")
        with path.open("rb") as f:
            payload = pickle.load(f)
        x = np.asarray(payload["X"], dtype=np.float32)
        y = int(payload.get("y", row["label"]))
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long), torch.tensor(int(row["h5_index"]), dtype=torch.long)


class UnifiedH5ManifestDataset(Dataset):
    def __init__(self, h5_path: Path, rows: list[dict[str, Any]]):
        self.h5_path = str(h5_path)
        self.rows = rows
        self._h5 = None

    def __len__(self) -> int:
        return len(self.rows)

    def _file(self):
        if self._h5 is None:
            h5py = base.import_h5py()
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        f = self._file()
        x = f["eeg"][int(row["h5_index"])].astype(np.float32)
        y = int(row["label"])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long), torch.tensor(int(row["h5_index"]), dtype=torch.long)

    def __del__(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass


class UnifiedNpzDataset(Dataset):
    def __init__(self, npz_path: Path, split: str):
        data = np.load(npz_path, allow_pickle=True)
        self.x = data[f"{split}_x"].astype(np.float32)
        self.y = data[f"{split}_y"].astype(np.int64)
        index_key = f"{split}_h5_indices"
        self.h5_indices = data[index_key].astype(np.int64) if index_key in data else np.arange(len(self.y), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.x[idx]), torch.tensor(int(self.y[idx]), dtype=torch.long), torch.tensor(int(self.h5_indices[idx]), dtype=torch.long)


def make_branch_datasets(args: argparse.Namespace, rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Dataset]:
    if args.branch == "original_pkl_50hz":
        return {split: OriginalPklDataset(Path(args.processed_root), rows_by_split[split]) for split in SPLITS}
    if args.branch == "unified_60hz":
        return {split: UnifiedH5ManifestDataset(Path(args.h5), rows_by_split[split]) for split in SPLITS}
    if args.branch == "unified_50hz":
        npz_path = Path(args.unified_50hz_npz)
        if not npz_path.exists():
            raise FileNotFoundError(
                f"unified_50hz subset data is not built yet: {npz_path}. "
                "Build/review it before running ablation."
            )
        return {split: UnifiedNpzDataset(npz_path, split) for split in SPLITS}
    raise ValueError(f"Unsupported branch: {args.branch}")


def make_loader(args: argparse.Namespace, dataset: Dataset, shuffle: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory and base.resolve_device(args.device).type == "cuda"),
        persistent_workers=False,
    )


def branch_recipe_diagnostics(args: argparse.Namespace, counts: dict[str, Any], checkpoint_report: dict[str, Any], opt_sched: dict[str, Any]) -> dict[str, Any]:
    diagnostics = strict.build_recipe_diagnostics(args, counts, checkpoint_report, opt_sched)
    intentional_epoch_override = args.epochs == 20
    strict_subset = (
        diagnostics["warmup_active"]
        and diagnostics["lr_schedule_active"]
        and diagnostics["layer_decay_active"]
        and diagnostics["checkpoint_loaded"]
        and args.model == "labram_base_patch200_200"
        and args.batch_size == 64
        and args.lr == 5e-4
        and args.weight_decay == 0.05
        and args.layer_decay == 0.65
        and args.drop_path == 0.1
        and args.disable_qkv_bias
        and args.disable_rel_pos_bias
        and args.abs_pos_emb
        and args.seed == 0
        and intentional_epoch_override
    )
    diagnostics["recipe_classification"] = "STRICT_MATCH" if strict_subset else diagnostics["recipe_classification"]
    diagnostics["intentional_epoch_override"] = intentional_epoch_override
    diagnostics["epoch_override_note"] = "Subset validation uses epochs=20 instead of original 50 by design."
    diagnostics["branch"] = args.branch
    return diagnostics


def evaluate(model, loader, loss_type: str, device: torch.device, args: argparse.Namespace, channel_names: list[str], save_predictions: Path | None = None) -> dict[str, Any]:
    return base.evaluate(model, loader, loss_type, device, args, channel_names, max_batches=args.max_eval_batches, save_predictions=save_predictions)


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    base.set_seed(args.seed)
    output_dir = Path(args.output_root) / args.branch
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "diagnostics").mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    rows_by_split = load_manifest(Path(args.subset_manifest))
    counts = manifest_counts(rows_by_split)
    save_json(output_dir / "diagnostics" / "split_subset_counts.json", counts)

    if args.dry_check_only:
        data_ok = True
        reason = ""
        try:
            datasets = make_branch_datasets(args, rows_by_split)
            for split in SPLITS:
                if len(datasets[split]) == 0:
                    raise ValueError(f"{split} dataset is empty")
            _ = datasets["train"][0]
        except Exception as exc:
            data_ok = False
            reason = f"{type(exc).__name__}: {exc}"
        payload = {"status": "DRY_CHECK_PASS" if data_ok else "DRY_CHECK_FAIL", "branch": args.branch, "data_available": data_ok, "reason": reason, "split_subset_counts": counts}
        save_json(output_dir / "diagnostics" / "branch_metadata.json", payload)
        write_branch_status(args.output_root, args.branch, payload["status"], payload)
        print(json.dumps(payload, indent=2))
        return payload

    datasets = make_branch_datasets(args, rows_by_split)
    h5_meta = base.h5_metadata(args.h5)
    channel_names = h5_meta["channel_names"]
    device = base.resolve_device(args.device)
    model = strict.StrictLaBraMWrapper(args)
    checkpoint_report = strict.load_pretrained(model.model, args.finetune)
    opt_sched = strict.setup_original_optimizer_and_schedules(args, model.model, counts["train"]["n"])
    diagnostics = branch_recipe_diagnostics(args, counts, checkpoint_report, opt_sched)
    if diagnostics["recipe_classification"] != "STRICT_MATCH":
        raise RuntimeError(f"Strict branch recipe classification failed: {diagnostics['recipe_classification']}")
    save_json(output_dir / "diagnostics" / "recipe_equivalence.json", diagnostics)
    save_json(output_dir / "diagnostics" / "checkpoint_load_report.json", checkpoint_report)
    save_json(
        output_dir / "diagnostics" / "branch_metadata.json",
        {
            "branch": args.branch,
            "source": args.branch,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "subset_manifest": args.subset_manifest,
            "output_dir": str(output_dir),
        },
    )
    write_branch_status(
        args.output_root,
        args.branch,
        "RUNNING",
        {
            "output_dir": str(output_dir),
            "subset_manifest": args.subset_manifest,
            "epochs": args.epochs,
        },
    )
    config = vars(args).copy()
    config["split_subset_counts"] = counts
    config["recipe_classification"] = diagnostics["recipe_classification"]
    save_json(output_dir / "config.json", config)

    model = model.to(device)
    optimizer = opt_sched["optimizer"]
    train_loader = make_loader(args, datasets["train"], shuffle=True)
    val_loader = make_loader(args, datasets["val"], shuffle=False)
    test_loader = make_loader(args, datasets["test"], shuffle=False)
    lr_values = opt_sched["lr_schedule_values"]
    wd_values = opt_sched["wd_schedule_values"]
    steps_per_epoch = opt_sched["num_training_steps_per_epoch"]
    loss_type = "BCEWithLogitsLoss"
    log_path = output_dir / "train.log"
    base.emit_event(log_path, f"START_BRANCH branch={args.branch}")
    best_val_bal = -float("inf")
    best_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        optimizer.zero_grad(set_to_none=True)
        for data_iter_step, (x, y, _) in enumerate(train_loader):
            step = data_iter_step // args.update_freq
            if step >= steps_per_epoch:
                continue
            it = (epoch - 1) * steps_per_epoch + step
            strict.apply_step_schedule(optimizer, lr_values, wd_values, it)
            x, _ = base.adapt_inputs("LaBraM", x, channel_names, args.repo_path)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = base.compute_loss(logits, y, loss_type) / args.update_freq
            loss.backward()
            if (data_iter_step + 1) % args.update_freq == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()) * args.update_freq)
        val = evaluate(model, val_loader, loss_type, device, args, channel_names)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "val_balanced_accuracy": val["balanced_accuracy"],
            "val_auroc": val["auroc"],
            "val_auprc": val["auprc"],
            "epoch_time_seconds": "",
        }
        history.append(row)
        base.save_epoch_csv(output_dir / "epoch_metrics.csv", history)
        payload = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch, "best_val_balanced_accuracy": best_val_bal}
        torch.save(payload, output_dir / "checkpoints" / f"epoch_{epoch:03d}.pt")
        torch.save(payload, output_dir / "checkpoints" / "checkpoint_last.pt")
        if row["val_balanced_accuracy"] > best_val_bal:
            best_val_bal = row["val_balanced_accuracy"]
            best_epoch = epoch
            torch.save(payload, output_dir / "checkpoints" / "checkpoint_best_val.pt")
        base.append_log(log_path, json.dumps(row))

    test = evaluate(model, test_loader, loss_type, device, args, channel_names, save_predictions=output_dir / "predictions_test.csv")
    metrics = {
        "status": "PASS",
        "branch": args.branch,
        "recipe_classification": diagnostics["recipe_classification"],
        "epochs_completed": args.epochs,
        "best_epoch": best_epoch,
        "best_val_balanced_accuracy": best_val_bal,
        "test_loss": test["loss"],
        "test_accuracy": test["accuracy"],
        "test_balanced_accuracy": test["balanced_accuracy"],
        "test_auroc": test["auroc"],
        "test_auprc": test["auprc"],
        "test_f1": test["f1"],
        "test_sensitivity": test["sensitivity"],
        "test_specificity": test["specificity"],
        "test_confusion_matrix": test["confusion_matrix"],
        "checkpoint_last": str(output_dir / "checkpoints" / "checkpoint_last.pt"),
        "checkpoint_best_val": str(output_dir / "checkpoints" / "checkpoint_best_val.pt"),
    }
    save_json(output_dir / "metrics.json", metrics)
    write_branch_status(args.output_root, args.branch, "PASS", {"metrics_path": str(output_dir / "metrics.json"), "epochs_completed": args.epochs})
    base.emit_event(log_path, "EXIT_CODE=0")
    print(json.dumps(metrics, indent=2))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=BRANCHES, required=True)
    parser.add_argument("--subset_manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--split_csv", default=str(DEFAULT_EXACT_SPLIT))
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--h5", default=str(DEFAULT_H5))
    parser.add_argument("--processed_root", default=str(DEFAULT_PROCESSED_ROOT))
    parser.add_argument("--unified_50hz_npz", default=str(DEFAULT_UNIFIED_50HZ_NPZ))
    parser.add_argument("--repo_path", default=str(DEFAULT_REPO))
    parser.add_argument("--finetune", default=str(DEFAULT_FINETUNE))
    parser.add_argument("--model", default="labram_base_patch200_200")
    parser.add_argument("--dataset", default="TUAB")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--weight_decay_end", type=float, default=None)
    parser.add_argument("--update_freq", type=int, default=1)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_steps", type=int, default=-1)
    parser.add_argument("--layer_decay", type=float, default=0.65)
    parser.add_argument("--drop_path", type=float, default=0.1)
    parser.add_argument("--save_ckpt_freq", type=int, default=1)
    parser.add_argument("--disable_rel_pos_bias", action="store_true", default=True)
    parser.add_argument("--abs_pos_emb", action="store_true", default=True)
    parser.add_argument("--disable_qkv_bias", action="store_true", default=True)
    parser.add_argument("--disable_weight_decay_on_rel_pos_bias", action="store_true", default=False)
    parser.add_argument("--opt", default="adamw")
    parser.add_argument("--opt_eps", type=float, default=1e-8)
    parser.add_argument("--opt_betas", type=float, nargs="+", default=None)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--save_every_epoch", action="store_true", default=True)
    parser.add_argument("--no_mid_epoch_checkpoints", action="store_true", default=True)
    parser.add_argument("--dry_check_only", action="store_true")
    parser.add_argument("--max_eval_batches", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run(args)
    except Exception as exc:
        output_dir = Path(args.output_root) / args.branch
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {"status": "FAIL", "branch": args.branch, "error": f"{type(exc).__name__}: {exc}"}
        save_json(output_dir / "metrics.json", payload)
        write_branch_status(args.output_root, args.branch, "FAIL", {"error": payload["error"], "metrics_path": str(output_dir / "metrics.json")})
        print(json.dumps(payload, indent=2))
        return 1
    return 0 if result.get("status") in {"PASS", "DRY_CHECK_PASS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
