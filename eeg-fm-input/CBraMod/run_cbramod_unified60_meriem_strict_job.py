#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CBraMod isolated remaining-model Meriem-strict debug job wrapper."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path('/nicoletye/workspace/unified_tuab')
WORKER = PROJECT_ROOT / 'scripts/eegfm_adapters/eegfm_small_subset_train_worker_remaining_models.py'
MODEL = 'CBraMod'
REPO = '/nicoletye/workspace/Noise Robustness Evaluation Protocol/EEG-FM/CBraMod'
VENV = '/nicoletye/venvs/cbramod_csbrain'


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--strategy', required=True, choices=['full_finetune', 'linear_probe', 'lora'])
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--index_npz', required=True)
    p.add_argument('--h5', default='/nicoletye/workspace/unified_tuab/data/canonical_tuab_full.h5')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--train_n', type=int, default=512)
    p.add_argument('--val_n', type=int, default=128)
    p.add_argument('--test_n', type=int, default=128)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--device', default='auto')
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--pin_memory', action='store_true')
    p.add_argument('--dry_run', action='store_true')
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(WORKER),
        '--model', MODEL,
        '--repo_path', REPO,
        '--venv_path', VENV,
        '--index_npz', args.index_npz,
        '--h5', args.h5,
        '--output_dir', str(output_dir),
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr),
        '--weight_decay', str(args.weight_decay),
        '--seed', str(args.seed),
        '--device', args.device,
        '--num_workers', str(args.num_workers),
        '--train_n', str(args.train_n),
        '--val_n', str(args.val_n),
        '--test_n', str(args.test_n),
        '--finetune_strategy', args.strategy,
        '--lora_rank', '2',
        '--lora_alpha', '8.0',
        '--lora_target', 'labram_reference',
        '--strict_checkpoint_load',
    ]
    if args.pin_memory:
        cmd.append('--pin_memory')
    command_payload = {'model': MODEL, 'strategy': args.strategy, 'seed': args.seed, 'cmd': cmd}
    (output_dir / 'command.json').write_text(json.dumps(command_payload, indent=2) + '\n', encoding='utf-8')
    if args.dry_run:
        print(json.dumps({'status': 'DRY_RUN', **command_payload}, indent=2))
        return 0
    rc = subprocess.call(cmd)
    metrics_path = output_dir / 'metrics.json'
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
            metrics['strategy_compliance'] = 'MERIEM_STRICT'
            if args.strategy == 'lora':
                metrics['lora_rank'] = 2
                metrics['lora_placement'] = 'LABRAM_REFERENCED_ARCHITECTURE_NATIVE_EQUIVALENT'
            metrics['remaining_model_isolated_worker'] = str(WORKER)
            metrics_path.write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
        except Exception:
            pass
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
