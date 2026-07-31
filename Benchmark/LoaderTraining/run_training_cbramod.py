#!/usr/bin/env python3
"""Connector for EEG-FM/CBraMod/finetune_main.py.

CBraMod has one original fine-tuning path. The Benchmark connector loads YAML
as defaults through the author script and only appends explicit CLI overrides.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "Benchmark" / "Config" / "cbramod.yaml"
CBRAMOD_SCRIPT = REPO_ROOT / "EEG-FM" / "CBraMod" / "finetune_main.py"


def build_cmd(args: argparse.Namespace) -> list[str]:
    """Build a command list; every option and value is a separate token."""
    cmd = [
        sys.executable,
        str(CBRAMOD_SCRIPT),
        "--config",
        str(args.config),
    ]

    # These optional values intentionally override YAML defaults.
    append_if_set(cmd, "--datasets_dir", args.original_data)
    append_if_set(cmd, "--data_source", args.data_source)
    append_if_set(cmd, "--h5_file", args.h5_file)
    append_if_set(cmd, "--split_index", args.split_index)
    append_if_set(cmd, "--tuab_mode", args.tuab_mode)
    append_if_set(cmd, "--train_samples", args.train_samples)
    append_if_set(cmd, "--validation_samples", args.validation_samples)
    append_if_set(cmd, "--test_samples", args.test_samples)
    append_if_set(cmd, "--foundation_dir", args.checkpoint)
    append_if_set(cmd, "--model_dir", args.output_dir)
    append_if_set(cmd, "--batch_size", args.batch_size)
    append_if_set(cmd, "--num_workers", args.num_workers)
    append_if_set(cmd, "--epochs", args.epochs)
    append_if_set(cmd, "--seed", args.seed)
    append_if_set(cmd, "--lr", args.lr)
    append_if_set(cmd, "--weight_decay", args.weight_decay)
    append_if_set(cmd, "--cuda", args.cuda)

    if args.extra_args:
        extra_args = args.extra_args[1:] if args.extra_args[0] == "--" else args.extra_args
        cmd.extend(extra_args)
    return cmd


def append_if_set(cmd: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CBraMod through the Benchmark YAML connector.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    #parser.add_argument("--dry_run", action="store_true", help="Print the command without launching training.")

    parser.add_argument("--original_data", default=None, help="Override YAML paths.original_data.")
    parser.add_argument("--data_source", default=None, choices=["original", "tuab_unified60"], help="Override YAML data.source.")
    parser.add_argument("--h5_file", default=None, help="Override YAML paths.h5_file.")
    parser.add_argument("--split_index", default=None, help="Override YAML paths.split_index.")
    parser.add_argument("--tuab_mode", default=None, choices=["subset_tuab", "full_tuab"], help="Override YAML study_case.tuab_mode.")
    parser.add_argument("--train_samples", default=None, type=int, help="Override unified H5 train row cap.")
    parser.add_argument("--validation_samples", default=None, type=int, help="Override unified H5 validation row cap.")
    parser.add_argument("--test_samples", default=None, type=int, help="Override unified H5 test row cap.")
    parser.add_argument("--checkpoint", default=None, help="Override YAML paths.checkpoint.")
    parser.add_argument("--output_dir", default=None, help="Override YAML paths.output.")
    parser.add_argument("--batch_size", default=None, type=int)
    parser.add_argument("--num_workers", default=None, type=int)
    parser.add_argument("--epochs", default=None, type=int)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--lr", default=None, type=float)
    parser.add_argument("--weight_decay", default=None, type=float)
    parser.add_argument("--cuda", default=None, type=int)
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional native CBraMod args after '--', for example: -- --classifier all_patch_reps",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cmd = build_cmd(args)
    print("COMMAND:", shlex.join(cmd), flush=True)
    if args.dry_run:
        return 0
    return subprocess.run(cmd, cwd=CBRAMOD_SCRIPT.parent).returncode


if __name__ == "__main__":
    raise SystemExit(main())
