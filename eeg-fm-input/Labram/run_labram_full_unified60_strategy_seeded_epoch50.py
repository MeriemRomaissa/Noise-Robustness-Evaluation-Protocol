#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Seeded LaBraM unified60 epoch-50 strategy wrapper.

Real mode delegates to the confirmed strict seed-0 reference runners and only
changes seed/output/report paths. Debug mode uses the existing subset worker to
exercise launcher mechanics without running full-dataset training.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_H5 = PROJECT_ROOT / "data" / "canonical_tuab_full.h5"
DEFAULT_SPLIT = PROJECT_ROOT / "reports" / "labram_exact_original_processed_split" / "canonical_h5_labram_exact_original_processed_split_index.csv"
DEFAULT_SUBSET_NPZ = PROJECT_ROOT / "reports" / "tuab_option1_unified_h5_subset.npz"
DEFAULT_REPO = Path("/nicoletye/workspace/Noise Robustness Evaluation Protocol/EEG-FM/Labram")
DEFAULT_FINETUNE = DEFAULT_REPO / "checkpoints" / "labram-base.pth"
PYTHON = "/nicoletye/venvs/eegpt_labram/bin/python"


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def strategy_dir_name(strategy: str) -> str:
    return "lora_meriem_exact" if strategy == "lora_meriem_exact" else strategy


def real_strategy_arg(strategy: str) -> str:
    if strategy == "linear_probe":
        return "linear_probe"
    if strategy == "lora_meriem_exact":
        return "lora"
    raise ValueError(f"No strategy arg for {strategy}")


def strict_common_args(args: argparse.Namespace) -> list[str]:
    return [
        "--h5",
        str(args.h5),
        "--split_csv",
        str(args.split_csv),
        "--repo_path",
        str(args.repo_path),
        "--output_root",
        str(args.output_root),
        "--report_dir",
        str(args.report_dir),
        "--finetune",
        str(args.finetune),
        "--model",
        "labram_base_patch200_200",
        "--dataset",
        "TUAB",
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        "5e-4",
        "--min_lr",
        "1e-6",
        "--weight_decay",
        "0.05",
        "--update_freq",
        "1",
        "--warmup_epochs",
        "5",
        "--layer_decay",
        "0.65",
        "--drop_path",
        "0.1",
        "--save_ckpt_freq",
        "1",
        "--disable_rel_pos_bias",
        "--abs_pos_emb",
        "--disable_qkv_bias",
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--num_workers",
        str(args.num_workers),
        "--pin_memory",
        "--allow_seed_variant_strict",
    ]


def build_real_command(args: argparse.Namespace) -> list[str]:
    if args.strategy == "full_finetune":
        script = SCRIPT_DIR / "run_labram_full_unified60_strict_epoch50.py"
        return [PYTHON, str(script), *strict_common_args(args)]
    script = SCRIPT_DIR / "run_labram_full_unified60_strategy_strict_epoch50.py"
    cmd = [PYTHON, str(script), "--strategy", real_strategy_arg(args.strategy), *strict_common_args(args)]
    if args.strategy == "lora_meriem_exact":
        cmd.extend(["--lora_rank", "2", "--lora_alpha", "8", "--lora_target", "meriem_exact"])
    return cmd


def build_debug_command(args: argparse.Namespace) -> list[str]:
    worker = SCRIPT_DIR / "eegfm_small_subset_train_worker.py"
    strategy = "lora" if args.strategy == "lora_meriem_exact" else args.strategy
    cmd = [
        PYTHON,
        str(worker),
        "--model",
        "LaBraM",
        "--repo_path",
        str(args.repo_path),
        "--venv_path",
        "/nicoletye/venvs/eegpt_labram",
        "--subset_npz",
        str(args.subset_npz),
        "--output_dir",
        str(args.output_root),
        "--epochs",
        str(args.debug_epochs),
        "--train_n",
        str(args.train_n),
        "--val_n",
        str(args.val_n),
        "--test_n",
        str(args.test_n),
        "--batch_size",
        str(args.debug_batch_size),
        "--num_workers",
        str(args.debug_num_workers),
        "--lr",
        "5e-4",
        "--weight_decay",
        "0.05",
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--finetune_strategy",
        strategy,
        "--lora_rank",
        "2",
        "--lora_alpha",
        "8",
        "--lora_target",
        "meriem_exact",
        "--strategy_report_path",
        str(Path(args.output_root) / "strategy"),
    ]
    return cmd


def write_debug_metadata(args: argparse.Namespace, returncode: int) -> None:
    out = Path(args.output_root)
    metrics_path = out / "metrics.json"
    metrics = load_json(metrics_path)
    metrics.update(
        {
            "debug_only": True,
            "full_dataset": False,
            "do_not_report_as_final": True,
            "reference_behavior_note": "Debug mode uses eegfm_small_subset_train_worker.py to exercise launcher mechanics only.",
            "strategy_requested": args.strategy,
            "seed": args.seed,
            "returncode": returncode,
        }
    )
    save_json(metrics_path, metrics)
    config = vars(args).copy()
    config.update(
        {
            "debug_only": True,
            "full_dataset": False,
            "do_not_report_as_final": True,
            "command_mode": "debug_subset_worker",
        }
    )
    save_json(out / "config.json", config)
    strategy_src = out / "strategy" / "trainable_params.json"
    if strategy_src.exists():
        strategy_payload = load_json(strategy_src)
        save_json(out / "strategy_param_audit.json", strategy_payload)


def run(args: argparse.Namespace) -> int:
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    cmd = build_debug_command(args) if args.debug else build_real_command(args)
    command_text = shlex.join(cmd)
    print(f"command={command_text}", flush=True)
    if args.print_command_only:
        return 0
    proc = subprocess.run(cmd, text=True, check=False)
    if args.debug:
        write_debug_metadata(args, proc.returncode)
    return int(proc.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--strategy", choices=["full_finetune", "linear_probe", "lora_meriem_exact"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--report_dir", required=True)
    parser.add_argument("--h5", default=str(DEFAULT_H5))
    parser.add_argument("--split_csv", default=str(DEFAULT_SPLIT))
    parser.add_argument("--subset_npz", default=str(DEFAULT_SUBSET_NPZ))
    parser.add_argument("--repo_path", default=str(DEFAULT_REPO))
    parser.add_argument("--finetune", default=str(DEFAULT_FINETUNE))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_epochs", type=int, default=1)
    parser.add_argument("--train_n", type=int, default=512)
    parser.add_argument("--val_n", type=int, default=128)
    parser.add_argument("--test_n", type=int, default=128)
    parser.add_argument("--debug_batch_size", type=int, default=4)
    parser.add_argument("--debug_num_workers", type=int, default=0)
    parser.add_argument("--print_command_only", action="store_true")
    return parser.parse_args()


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
