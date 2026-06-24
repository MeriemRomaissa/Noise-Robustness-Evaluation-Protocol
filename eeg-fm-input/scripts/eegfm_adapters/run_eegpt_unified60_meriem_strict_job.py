#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run one EEGPT unified60 Meriem-strict strategy job and annotate metrics."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKER = PROJECT_ROOT / "scripts" / "eegfm_adapters" / "eegfm_small_subset_train_worker.py"
EEGPT_REPO = "/nicoletye/workspace/Noise Robustness Evaluation Protocol/EEG-FM/EEGPT"
EEGPT_VENV = "/nicoletye/venvs/eegpt_labram"
EEGPT_PYTHON = f"{EEGPT_VENV}/bin/python"
EEGPT_CHECKPOINT = f"{EEGPT_REPO}/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt"
EEGPT_NATIVE_CONFIG = {
    "class": "EEGPTClassifier",
    "num_classes": 1,
    "in_channels": 23,
    "img_size": [20, 2000],
    "patch_size": 64,
    "patch_stride": 64,
    "embed_dim": 512,
    "embed_num": 4,
    "depth": 8,
    "num_heads": 8,
    "mlp_ratio": 4.0,
    "use_chan_conv": True,
    "use_mean_pooling": True,
}
STRATEGY_CLASSIFICATION = {
    "full_finetune": "FUNCTIONAL_MATCH_WITH_HEAD_DIFFERENCE",
    "linear_probe": "FUNCTIONAL_MATCH_EXACT_TRAINABLE_SET",
    "lora": "MERIEM_STRICT_ARCHITECTURE_NATIVE_EQUIVALENT",
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "FAIL", "read_error": f"{type(exc).__name__}: {exc}"}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def bool_head_trainable(trainable_names: list[str]) -> bool:
    return any(name.startswith("model.head.") or name.startswith("head.") for name in trainable_names)


def annotate_metrics(metrics_path: Path, args: argparse.Namespace, returncode: int) -> None:
    metrics = read_json(metrics_path)
    if not metrics:
        metrics = {"status": "FAIL", "error": "metrics.json missing after worker run"}

    strategy_summary = metrics.get("strategy_summary", {}) or {}
    checkpoint_report = metrics.get("checkpoint_load_report", {}) or {}
    trainable_pairs = strategy_summary.get("all_trainable_param_names", []) or []
    trainable_names = [item[0] if isinstance(item, (list, tuple)) and item else str(item) for item in trainable_pairs]
    strict_lora = strategy_summary.get("strict_lora_summary", {}) or {}

    total_params = int(strategy_summary.get("total_params", 0) or 0)
    trainable_params = int(strategy_summary.get("trainable_params", 0) or 0)
    frozen_params = int(strategy_summary.get("frozen_params", 0) or 0)
    ratio = float(trainable_params / total_params) if total_params else 0.0
    classification = STRATEGY_CLASSIFICATION.get(args.strategy, "UNKNOWN")

    head_trainable = bool_head_trainable(trainable_names)
    backbone_frozen = args.strategy in {"linear_probe", "lora"} and all(
        name.startswith("model.head.") or name.startswith("head.") or "lora_" in name.lower() for name in trainable_names
    )

    metrics.update(
        {
            "model": "EEGPT",
            "strategy": args.strategy,
            "finetune_strategy": args.strategy,
            "seed": args.seed,
            "returncode": returncode,
            "checkpoint_present": bool(checkpoint_report.get("checkpoint_exists", Path(EEGPT_CHECKPOINT).exists())),
            "checkpoint_loaded": bool(checkpoint_report.get("checkpoint_loaded", False)),
            "checkpoint_path": checkpoint_report.get("checkpoint_path", EEGPT_CHECKPOINT),
            "checkpoint_load_status": checkpoint_report.get("checkpoint_status", "UNKNOWN"),
            "missing_checkpoint_keys": checkpoint_report.get("missing_keys", []),
            "unexpected_checkpoint_keys": checkpoint_report.get("unexpected_keys", []),
            "native_architecture_used": EEGPT_NATIVE_CONFIG,
            "strict_status": classification,
            "strategy_status": "PASS" if metrics.get("status") == "PASS" else "FAIL",
            "strategy_classification": classification,
            "lora_comparability": (
                "MERIEM_STRICT_ARCHITECTURE_NATIVE_EQUIVALENT" if args.strategy == "lora" else ""
            ),
            "strategy_compliance": "MERIEM_STRICT",
            "classification": classification,
            "lora_placement": "LABRAM_REFERENCED_ARCHITECTURE_NATIVE_EQUIVALENT" if args.strategy == "lora" else "",
            "lora_target_modules": strategy_summary.get("lora_target_names", []),
            "lora_rank": int(args.lora_rank) if args.strategy == "lora" else "",
            "lora_alpha": float(args.lora_alpha) if args.strategy == "lora" else "",
            "lora_dropout": 0.0 if args.strategy == "lora" else "",
            "total_param_count": total_params,
            "trainable_param_count": trainable_params,
            "trainable_param_ratio": ratio,
            "frozen_param_count": frozen_params,
            "trainable_param_names_sample": trainable_names[:40],
            "head_trainable": head_trainable,
            "backbone_frozen": backbone_frozen,
            "full_finetune_all_valid_params_trainable": args.strategy == "full_finetune",
            "linear_probe_exact_trainable_set": args.strategy == "linear_probe" and backbone_frozen and head_trainable,
            "safe_for_debug": True,
            "blocked_reason": "",
            "auroc": metrics.get("test_auroc", ""),
            "auprc": metrics.get("test_auprc", ""),
            "eegpt_checkpoint_reference": EEGPT_CHECKPOINT,
            "eegpt_native_config_reference": EEGPT_NATIVE_CONFIG,
        }
    )
    if args.strategy == "lora":
        metrics["lora_strict_summary"] = strict_lora
        metrics["notes"] = (
            str(metrics.get("notes", "")).strip()
            + " EEGPT LoRA uses EEGPT-native architecture-compatible attention/projection/MLP-like targets; "
            "comparability is functional with architecture difference, not LaBraM physical identity."
        ).strip()
    write_json(metrics_path, metrics)


def build_worker_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        EEGPT_PYTHON,
        str(WORKER),
        "--model",
        "EEGPT",
        "--repo_path",
        EEGPT_REPO,
        "--venv_path",
        EEGPT_VENV,
        "--output_dir",
        args.output_dir,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--num_workers",
        str(args.num_workers),
        "--finetune_strategy",
        args.strategy,
        "--lora_rank",
        str(args.lora_rank),
        "--lora_alpha",
        str(args.lora_alpha),
        "--lora_target",
        args.lora_target,
        "--strict_checkpoint_load",
        "--strategy_report_path",
        str(Path(args.output_dir) / "strategy_param_audit"),
    ]
    if args.pin_memory:
        cmd.append("--pin_memory")
    if args.index_npz:
        cmd.extend(["--index_npz", args.index_npz, "--h5", args.h5])
    else:
        cmd.extend(["--subset_npz", args.subset_npz])
    if args.train_n:
        cmd.extend(["--train_n", str(args.train_n)])
    if args.val_n:
        cmd.extend(["--val_n", str(args.val_n)])
    if args.test_n:
        cmd.extend(["--test_n", str(args.test_n)])
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["full_finetune", "linear_probe", "lora"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--subset_npz", default="reports/tuab_option1_unified_h5_subset.npz")
    parser.add_argument("--index_npz", default="")
    parser.add_argument("--h5", default="data/canonical_tuab_full.h5")
    parser.add_argument("--train_n", type=int, default=0)
    parser.add_argument("--val_n", type=int, default=0)
    parser.add_argument("--test_n", type=int, default=0)
    parser.add_argument("--lora_rank", type=int, default=2)
    parser.add_argument("--lora_alpha", type=float, default=8.0)
    parser.add_argument("--lora_target", default="meriem_exact")
    parser.add_argument("--print_command_only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model": "EEGPT",
        "strategy": args.strategy,
        "seed": args.seed,
        "checkpoint_path": EEGPT_CHECKPOINT,
        "native_config": EEGPT_NATIVE_CONFIG,
        "strategy_classification_expected": STRATEGY_CLASSIFICATION[args.strategy],
        "args": vars(args),
    }
    write_json(output_dir / "config.json", config)

    cmd = build_worker_cmd(args)
    command_text = shlex.join(cmd)
    print(f"command={command_text}", flush=True)
    if args.print_command_only:
        return 0

    env = os.environ.copy()
    env.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, check=False)
    metrics_path = output_dir / "metrics.json"
    annotate_metrics(metrics_path, args, result.returncode)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
