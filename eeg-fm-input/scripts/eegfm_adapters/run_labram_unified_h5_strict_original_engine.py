#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict original-recipe LaBraM runner for unified H5 exact split.

This runner keeps the unified H5 dataset path, but uses original LaBraM
fine-tuning mechanics for model construction, checkpoint load, parameter
grouping, layer-wise LR decay, and step-level cosine LR/WD schedules.
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
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_labram_full_unified_h5_epoch10 as base  # noqa: E402


PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_H5 = PROJECT_ROOT / "data" / "canonical_tuab_full.h5"
DEFAULT_SPLIT = PROJECT_ROOT / "reports" / "labram_exact_original_processed_split" / "canonical_h5_labram_exact_original_processed_split_index.csv"
DEFAULT_REPO = Path("/nicoletye/workspace/Noise Robustness Evaluation Protocol/EEG-FM/Labram")
DEFAULT_FINETUNE = DEFAULT_REPO / "checkpoints" / "labram-base.pth"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "labram_full_unified_h5_strict_original_engine_v1"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "labram_strict_recipe_equivalence_audit"

ORIGINAL_EXPECTED = {
    "model": "labram_base_patch200_200",
    "weight_decay": 0.05,
    "batch_size": 64,
    "lr": 5e-4,
    "update_freq": 1,
    "warmup_epochs": 5,
    "epochs": 50,
    "layer_decay": 0.65,
    "drop_path": 0.1,
    "save_ckpt_freq": 5,
    "disable_rel_pos_bias": True,
    "abs_pos_emb": True,
    "dataset": "TUAB",
    "disable_qkv_bias": True,
    "seed": 0,
}


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(payload: Any) -> tuple[dict[str, torch.Tensor], str]:
    def is_state_dict(value: Any) -> bool:
        return isinstance(value, dict) and value and all(hasattr(v, "shape") for v in value.values())

    if is_state_dict(payload):
        return dict(payload), "raw_state_dict"
    if isinstance(payload, dict):
        for key in ["model", "state_dict", "module", "student"]:
            if key in payload and is_state_dict(payload[key]):
                return dict(payload[key]), f"checkpoint[{key!r}]"
    raise ValueError("No tensor state_dict found in checkpoint")


def strip_prefixes(key: str) -> str:
    for prefix in ["module.", "model.", "student."]:
        if key.startswith(prefix):
            key = key[len(prefix) :]
    return key


class StrictLaBraMWrapper(torch.nn.Module):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        sys.path.insert(0, args.repo_path)
        import modeling_finetune

        factory = getattr(modeling_finetune, args.model)
        self.model = factory(
            pretrained=False,
            num_classes=1,
            EEG_size=2000,
            drop_path_rate=float(args.drop_path),
            init_values=0.1,
            qkv_bias=not bool(args.disable_qkv_bias),
            use_abs_pos_emb=bool(args.abs_pos_emb),
            use_rel_pos_bias=not bool(args.disable_rel_pos_bias),
            use_mean_pooling=True,
            init_scale=0.001,
        )

    def forward(self, x):
        input_chans = torch.arange(0, x.shape[1] + 1, device=x.device)
        return self.model(x, input_chans=input_chans)


def load_pretrained(model: torch.nn.Module, checkpoint: str) -> dict[str, Any]:
    path = Path(checkpoint)
    if not path.exists():
        return {"checkpoint_loaded": False, "checkpoint_path": str(path), "error": "checkpoint not found"}
    payload = torch_load(path)
    state, source = extract_state_dict(payload)
    state = {strip_prefixes(str(k)): v for k, v in state.items()}
    model_state = model.state_dict()
    compatible = {}
    skipped_shape = []
    unexpected = []
    for key, value in state.items():
        if key not in model_state:
            unexpected.append(key)
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            skipped_shape.append({"key": key, "checkpoint_shape": list(value.shape), "model_shape": list(model_state[key].shape)})
            continue
        compatible[key] = value
    load_result = model.load_state_dict(compatible, strict=False)
    return {
        "checkpoint_loaded": bool(compatible),
        "checkpoint_path": str(path),
        "checkpoint_source": source,
        "loaded_key_count": len(compatible),
        "model_key_count": len(model_state),
        "missing_key_count": len(load_result.missing_keys),
        "unexpected_key_count": len(unexpected),
        "shape_mismatch_count": len(skipped_shape),
        "loaded_ratio": len(compatible) / max(1, len(model_state)),
        "sample_missing_keys": list(load_result.missing_keys)[:20],
        "sample_unexpected_keys": unexpected[:20],
        "sample_shape_mismatches": skipped_shape[:20],
    }


def setup_original_optimizer_and_schedules(args: argparse.Namespace, model: torch.nn.Module, train_count: int) -> dict[str, Any]:
    sys.path.insert(0, args.repo_path)
    import utils
    from optim_factory import LayerDecayValueAssigner, create_optimizer

    num_training_steps_per_epoch = train_count // (args.batch_size * args.update_freq)
    if num_training_steps_per_epoch <= 0:
        raise ValueError(f"num_training_steps_per_epoch={num_training_steps_per_epoch}; check train_count/batch_size/update_freq")
    num_layers = model.get_num_layers()
    assigner = None
    assigner_values = []
    if args.layer_decay < 1.0:
        assigner_values = [args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)]
        assigner = LayerDecayValueAssigner(assigner_values)

    skip_weight_decay_list = model.no_weight_decay()
    if args.disable_weight_decay_on_rel_pos_bias:
        for i in range(num_layers):
            skip_weight_decay_list.add(f"blocks.{i}.attn.relative_position_bias_table")

    optimizer = create_optimizer(
        args,
        model,
        skip_list=skip_weight_decay_list,
        get_num_layer=assigner.get_layer_id if assigner is not None else None,
        get_layer_scale=assigner.get_scale if assigner is not None else None,
    )
    lr_schedule_values = utils.cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
        warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    return {
        "optimizer": optimizer,
        "lr_schedule_values": lr_schedule_values,
        "wd_schedule_values": wd_schedule_values,
        "num_training_steps_per_epoch": num_training_steps_per_epoch,
        "num_layers": num_layers,
        "assigner_values": assigner_values,
        "skip_weight_decay_list": sorted(skip_weight_decay_list),
    }


def optimizer_group_diagnostics(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    lr_scales = [float(group.get("lr_scale", 1.0)) for group in optimizer.param_groups]
    return {
        "optimizer_param_group_count": len(optimizer.param_groups),
        "distinct_lr_scale_count": len(set(round(x, 12) for x in lr_scales)),
        "min_lr_scale": min(lr_scales) if lr_scales else None,
        "max_lr_scale": max(lr_scales) if lr_scales else None,
        "example_parameter_groups": [
            {
                "index": idx,
                "lr": group.get("lr"),
                "weight_decay": group.get("weight_decay"),
                "lr_scale": group.get("lr_scale", 1.0),
                "param_count": sum(p.numel() for p in group.get("params", [])),
            }
            for idx, group in enumerate(optimizer.param_groups[:10])
        ],
    }


def build_recipe_diagnostics(
    args: argparse.Namespace,
    split_counts: dict[str, Any],
    checkpoint_report: dict[str, Any],
    opt_sched: dict[str, Any],
) -> dict[str, Any]:
    lr_schedule = opt_sched["lr_schedule_values"]
    warmup_steps = args.warmup_steps if args.warmup_steps > 0 else args.warmup_epochs * opt_sched["num_training_steps_per_epoch"]
    lr_after_warmup_index = max(0, min(len(lr_schedule) - 1, warmup_steps - 1))
    lr_changes = bool(len(set(np.round(lr_schedule[: min(len(lr_schedule), 1000)], 12))) > 1 or not np.isclose(lr_schedule[0], lr_schedule[-1]))
    group_diag = optimizer_group_diagnostics(opt_sched["optimizer"])
    exact_split_used = Path(args.split_csv).resolve() == DEFAULT_SPLIT.resolve() and split_counts["train"]["n"] == 295751
    checkpoint_loaded = bool(checkpoint_report.get("checkpoint_loaded")) and checkpoint_report.get("loaded_ratio", 0) >= 0.80
    warmup_active = args.warmup_epochs > 0 and len(lr_schedule) > warmup_steps > 0 and float(lr_schedule[0]) < float(lr_schedule[lr_after_warmup_index])
    lr_schedule_active = lr_changes and float(lr_schedule[-1]) < float(max(lr_schedule))
    layer_decay_active = group_diag["distinct_lr_scale_count"] > 1 and group_diag["min_lr_scale"] < group_diag["max_lr_scale"]
    original_settings_used = (
        args.model == ORIGINAL_EXPECTED["model"]
        and args.batch_size == ORIGINAL_EXPECTED["batch_size"]
        and args.lr == ORIGINAL_EXPECTED["lr"]
        and args.weight_decay == ORIGINAL_EXPECTED["weight_decay"]
        and args.epochs == ORIGINAL_EXPECTED["epochs"]
        and args.layer_decay == ORIGINAL_EXPECTED["layer_decay"]
        and args.drop_path == ORIGINAL_EXPECTED["drop_path"]
        and args.disable_qkv_bias == ORIGINAL_EXPECTED["disable_qkv_bias"]
        and args.disable_rel_pos_bias == ORIGINAL_EXPECTED["disable_rel_pos_bias"]
        and args.abs_pos_emb == ORIGINAL_EXPECTED["abs_pos_emb"]
        and args.seed == ORIGINAL_EXPECTED["seed"]
    )
    if warmup_active and lr_schedule_active and layer_decay_active and checkpoint_loaded and exact_split_used and original_settings_used:
        classification = "STRICT_MATCH"
    elif checkpoint_loaded and exact_split_used:
        classification = "NEAR_MATCH"
    else:
        classification = "NOT_MATCH"

    diagnostics = {
        "warmup_active": warmup_active,
        "lr_schedule_active": lr_schedule_active,
        "first_lr": float(lr_schedule[0]),
        "lr_after_warmup": float(lr_schedule[lr_after_warmup_index]),
        "max_lr": float(max(lr_schedule)),
        "final_lr": float(lr_schedule[-1]),
        "lr_changes_over_steps": lr_changes,
        "warmup_steps": int(warmup_steps),
        "total_schedule_steps": int(len(lr_schedule)),
        "layer_decay_active": layer_decay_active,
        **group_diag,
        "checkpoint_loaded": checkpoint_loaded,
        "checkpoint_report": checkpoint_report,
        "exact_split_used": exact_split_used,
        "split_counts": split_counts,
        "drop_path_value": args.drop_path,
        "disable_qkv_bias": args.disable_qkv_bias,
        "disable_rel_pos_bias": args.disable_rel_pos_bias,
        "abs_pos_emb": args.abs_pos_emb,
        "num_layers": opt_sched["num_layers"],
        "assigner_values": opt_sched["assigner_values"],
        "original_optimizer_utilities_used": True,
        "custom_unified_h5_loop_used": True,
        "recipe_classification": classification,
    }
    return diagnostics


def write_report(report_dir: Path, diagnostics: dict[str, Any]) -> None:
    save_json(report_dir / "strict_recipe_equivalence.json", diagnostics)
    lines = [
        "# LaBraM Strict Recipe Equivalence Audit",
        "",
        f"- Recipe classification: **{diagnostics['recipe_classification']}**",
        f"- Warmup active: {diagnostics['warmup_active']}",
        f"- LR schedule active: {diagnostics['lr_schedule_active']}",
        f"- First LR: {diagnostics['first_lr']}",
        f"- LR after warmup: {diagnostics['lr_after_warmup']}",
        f"- Max LR: {diagnostics['max_lr']}",
        f"- Final LR: {diagnostics['final_lr']}",
        f"- Layer-wise LR decay active: {diagnostics['layer_decay_active']}",
        f"- Optimizer param groups: {diagnostics['optimizer_param_group_count']}",
        f"- Distinct lr_scale values: {diagnostics['distinct_lr_scale_count']}",
        f"- Min/max lr_scale: {diagnostics['min_lr_scale']} / {diagnostics['max_lr_scale']}",
        f"- Checkpoint loaded: {diagnostics['checkpoint_loaded']}",
        f"- Exact split used: {diagnostics['exact_split_used']}",
        f"- Drop path: {diagnostics['drop_path_value']}",
        f"- disable_qkv_bias: {diagnostics['disable_qkv_bias']}",
        f"- disable_rel_pos_bias: {diagnostics['disable_rel_pos_bias']}",
        f"- abs_pos_emb: {diagnostics['abs_pos_emb']}",
        "",
        "This audit was generated without heavy training, H5 rebuild, or H5 modification.",
    ]
    (report_dir / "strict_recipe_equivalence_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def apply_step_schedule(optimizer: torch.optim.Optimizer, lr_values: np.ndarray, wd_values: np.ndarray, step_index: int) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr_values[step_index]) * float(group.get("lr_scale", 1.0))
        if wd_values is not None and group.get("weight_decay", 0) > 0:
            group["weight_decay"] = float(wd_values[step_index])


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    base.set_seed(args.seed)
    output_root = Path(args.output_root)
    report_dir = Path(args.report_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "diagnostics").mkdir(parents=True, exist_ok=True)
    log_path = output_root / "train.log"

    split_data = base.load_split_index(args.split_csv)
    counts = base.split_counts(split_data)
    h5_meta = base.h5_metadata(args.h5)
    channel_names = h5_meta["channel_names"]
    model_wrapper = StrictLaBraMWrapper(args)
    checkpoint_report = load_pretrained(model_wrapper.model, args.finetune)
    opt_sched = setup_original_optimizer_and_schedules(args, model_wrapper.model, counts["train"]["n"])
    diagnostics = build_recipe_diagnostics(args, counts, checkpoint_report, opt_sched)
    write_report(report_dir, diagnostics)
    save_json(output_root / "diagnostics" / "strict_recipe_equivalence.json", diagnostics)
    save_json(output_root / "diagnostics" / "checkpoint_load_report.json", checkpoint_report)
    save_json(output_root / "diagnostics" / "split_counts.json", counts)

    config = vars(args).copy()
    config.update(
        {
            "python_path": sys.executable,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "h5_metadata": h5_meta,
            "recipe_classification": diagnostics["recipe_classification"],
        }
    )
    save_json(output_root / "config.json", config)
    if args.dry_run:
        payload = {"status": "DRY_RUN", **diagnostics}
        save_json(output_root / "metrics.json", payload)
        print(json.dumps(payload, indent=2))
        return payload

    device = base.resolve_device(args.device)
    model_wrapper = model_wrapper.to(device)
    optimizer = opt_sched["optimizer"]
    train_loader = base.make_loader(args, split_data["train"], shuffle=True)
    val_loader = base.make_loader(args, split_data["val"], shuffle=False)
    test_loader = base.make_loader(args, split_data["test"], shuffle=False)
    loss_type = "BCEWithLogitsLoss"
    lr_values = opt_sched["lr_schedule_values"]
    wd_values = opt_sched["wd_schedule_values"]
    steps_per_epoch = opt_sched["num_training_steps_per_epoch"]
    best_val_bal = -float("inf")
    best_epoch = 0
    epoch_rows = []
    base.emit_event(log_path, "START_TRAINING_STRICT_ORIGINAL_ENGINE")
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model_wrapper.train()
        losses = []
        last_batch_idx = 0
        optimizer.zero_grad(set_to_none=True)
        for data_iter_step, (x, y, _) in enumerate(train_loader):
            step = data_iter_step // args.update_freq
            if step >= steps_per_epoch:
                continue
            if args.max_train_batches_per_epoch and data_iter_step >= args.max_train_batches_per_epoch:
                break
            it = (epoch - 1) * steps_per_epoch + step
            apply_step_schedule(optimizer, lr_values, wd_values, it)
            x, _ = base.adapt_inputs("LaBraM", x, channel_names, args.repo_path)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model_wrapper(x)
            loss = base.compute_loss(logits, y, loss_type) / args.update_freq
            loss.backward()
            if (data_iter_step + 1) % args.update_freq == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            losses.append(float(loss.detach().cpu()) * args.update_freq)
            last_batch_idx = data_iter_step + 1
        val = base.evaluate(model_wrapper, val_loader, loss_type, device, args, channel_names, max_batches=args.max_val_batches)
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
        epoch_rows.append(row)
        base.save_epoch_csv(output_root / "epoch_metrics.csv", epoch_rows)
        if row["val_balanced_accuracy"] > best_val_bal:
            best_val_bal = row["val_balanced_accuracy"]
            best_epoch = epoch
            torch.save({"model_state_dict": model_wrapper.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch}, output_root / "checkpoint_best_val.pt")
        if epoch % args.save_ckpt_freq == 0 or epoch == args.epochs:
            torch.save({"model_state_dict": model_wrapper.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch}, output_root / f"checkpoint_epoch_{epoch:03d}.pt")
        torch.save({"model_state_dict": model_wrapper.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch}, output_root / "checkpoint_last.pt")
        base.append_log(log_path, json.dumps(row))

    test = base.evaluate(model_wrapper, test_loader, loss_type, device, args, channel_names, max_batches=args.max_test_batches, save_predictions=output_root / "predictions_test.csv")
    metrics = {
        "status": "PASS",
        "recipe_classification": diagnostics["recipe_classification"],
        "epochs_completed": epoch_rows[-1]["epoch"] if epoch_rows else 0,
        "best_epoch": best_epoch,
        "best_val_balanced_accuracy": best_val_bal,
        "test_loss": test["loss"],
        "test_accuracy": test["accuracy"],
        "test_balanced_accuracy": test["balanced_accuracy"],
        "test_auroc": test["auroc"],
        "test_auprc": test["auprc"],
        "test_confusion_matrix": test["confusion_matrix"],
        "checkpoint_loaded": diagnostics["checkpoint_loaded"],
        "exact_split_used": diagnostics["exact_split_used"],
    }
    save_json(output_root / "metrics.json", metrics)
    base.emit_event(log_path, "EXIT_CODE=0")
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", default=str(DEFAULT_H5))
    parser.add_argument("--split_csv", default=str(DEFAULT_SPLIT))
    parser.add_argument("--repo_path", default=str(DEFAULT_REPO))
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report_dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--model", default="labram_base_patch200_200")
    parser.add_argument("--finetune", default=str(DEFAULT_FINETUNE))
    parser.add_argument("--dataset", default="TUAB")
    parser.add_argument("--epochs", type=int, default=50)
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
    parser.add_argument("--save_ckpt_freq", type=int, default=5)
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
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--max_train_batches_per_epoch", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)
    parser.add_argument("--max_test_batches", type=int, default=0)
    parser.add_argument("--diagnostic_samples_per_split", type=int, default=0)
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as exc:
        output_root = Path(args.output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        tb = traceback.format_exc()
        payload = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}", "traceback": tb}
        save_json(output_root / "metrics.json", payload)
        print(json.dumps(payload, indent=2))
        return 1
    return 0 if result.get("status") in {"PASS", "DRY_RUN"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
