#!/usr/bin/env python3
"""
Multi-Seed Stability Study Runner
Runs fine-tuning with different seeds to evaluate model stability.

    Usage:
    python run_multiseed_stability.py --strategy freeze_backbone --seeds 42 123
    python run_multiseed_stability.py --strategy original --seeds 42 123 --epochs 25
    python run_multiseed_stability.py --strategy original --seeds 0 --dataset TUEV
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_training(seed, strategy, epochs=25, base_dir="checkpoints", log_dir="log", freeze_early_n=None, lora_rank=None, dataset="TUAB", lr="6e-5"):
    """Run fine-tuning for a single seed."""
    
    # Create strategy-specific directories with hyperparameter suffixes (new structure)
    dir_suffix = ""
    if strategy == "freeze_early_layers" and freeze_early_n is not None:
        dir_suffix = f"_n{freeze_early_n}"
    elif strategy == "lora" and lora_rank is not None:
        dir_suffix = f"_r{lora_rank}"
    dir_suffix += f"_lr{lr}"

    # New structure: checkpoints/DATASET/Labram/strategy[_nN|_rN]/seed_N
    checkpoint_path = Path(base_dir) / dataset.upper() / "Labram" / f"{strategy}{dir_suffix}" / f"seed_{seed}"
    log_path = Path(log_dir) / dataset.upper() / "Labram" / f"{strategy}{dir_suffix}" / f"seed_{seed}"

    checkpoint_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Build the training command - use python directly without distributed launcher
    cmd = [
        "python",
        "run_class_finetuning.py",
        f"--output_dir={checkpoint_path}",
        f"--log_dir={log_path}",
        "--model=labram_base_patch200_200",
        "--finetune=./checkpoints/labram-base.pth",
        "--weight_decay=0.05",
        "--batch_size=64",
        f"--lr={lr}",
        "--update_freq=1",
        "--warmup_epochs=5",
        f"--epochs={epochs}",
        "--layer_decay=0.65",
        "--drop_path=0.1",
        "--save_ckpt_freq=5",
        "--disable_rel_pos_bias",
        "--abs_pos_emb",
        f"--dataset={dataset}",
        "--disable_qkv_bias",
        f"--seed={seed}",
        f"--finetune_strategy={strategy}",
    ]
    
    # Add strategy-specific parameters
    if strategy == "freeze_early_layers" and freeze_early_n is not None:
        cmd.append(f"--freeze_early_n={freeze_early_n}")
    elif strategy == "lora" and lora_rank is not None:
        cmd.append(f"--lora_rank={lora_rank}")
    
    print(f"\n{'='*80}")
    print(f"Running: Strategy={strategy}, Seed={seed}, Epochs={epochs}, Dataset={dataset}, LR={lr}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Log: {log_path}")
    print(f"{'='*80}\n")
    print(" ".join(cmd))
    print()
    
    result = subprocess.run(cmd, cwd="/home/meriem-ubuntu/Projects/Foundation-models/Labram")
    
    if result.returncode != 0:
        print(f"ERROR: Training failed for seed {seed} with strategy {strategy}")
        return False
    
    print(f"\nSUCCESS: Training completed for seed {seed} with strategy {strategy}\n")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-seed fine-tuning experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single freeze_early_n value
  python run_multiseed_stability.py --strategy freeze_early_layers --seeds 42 123 --freeze_early_n 6
  
  # Multiple freeze_early_n values (for ablation study)
  python run_multiseed_stability.py --strategy freeze_early_layers --seeds 42 123 --freeze_early_ns 3 6 9
  
  # LoRA with different ranks
  python run_multiseed_stability.py --strategy lora --seeds 42 123 --lora_rank 8
  
  # LoRA rank ablation study
  python run_multiseed_stability.py --strategy lora --seeds 42 123 --lora_ranks 4 8 16
  
    # Basic strategies
    python run_multiseed_stability.py --strategy freeze_backbone --seeds 42 123
    python run_multiseed_stability.py --strategy original --seeds 42 123 --epochs 25
    python run_multiseed_stability.py --strategy original --seeds 0 --dataset TUEV
        """
    )
    
    parser.add_argument(
        "--strategy",
        type=str,
        default="freeze_backbone",
        choices=["original", "freeze_backbone", "freeze_early_layers", "aggressive_reg", "lora"],
        help="Fine-tuning strategy to use"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123],
        help="List of seeds to run (default: 42 123)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of epochs (default: 25)"
    )
    parser.add_argument(
        "--freeze_early_n",
        type=int,
        default=None,
        help="Number of early layers to freeze (for freeze_early_layers strategy)"
    )
    parser.add_argument(
        "--freeze_early_ns",
        type=int,
        nargs="+",
        default=None,
        help="Multiple freeze_early_n values for ablation study. Overrides --freeze_early_n"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=None,
        help="LoRA rank (for lora strategy). If not specified, uses hardcoded defaults in finetune_strategies.py"
    )
    parser.add_argument(
        "--lora_ranks",
        type=int,
        nargs="+",
        default=None,
        help="Multiple LoRA ranks for ablation study. Overrides --lora_rank"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="checkpoints",
        help="Base directory for checkpoints"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="log",
        help="Base directory for logs"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="TUAB",
        choices=["TUAB", "TUEV"],
        help="Dataset to use (TUAB or TUEV)"
    )
    parser.add_argument(
        "--lr",
        type=str,
        default="6e-5",
        help="Learning rate (default: 6e-5)"
    )
    parser.add_argument(
        "--lrs",
        type=str,
        nargs="+",
        default=None,
        help="Multiple learning rates for ablation study (e.g. 1e-5 2e-5 3e-5). Overrides --lr"
    )
    
    args = parser.parse_args()
    
    # Determine which values to test
    test_values = [None]  # Default: run once with strategy defaults
    test_label = ""
    
    if args.strategy == "freeze_early_layers":
        if args.freeze_early_ns is not None:
            test_values = args.freeze_early_ns
            test_label = "freeze_early_n values"
        elif args.freeze_early_n is not None:
            test_values = [args.freeze_early_n]
            test_label = "freeze_early_n"
    elif args.strategy == "lora":
        if args.lora_ranks is not None:
            test_values = args.lora_ranks
            test_label = "LoRA ranks"
        elif args.lora_rank is not None:
            test_values = [args.lora_rank]
            test_label = "LoRA rank"
        # else: test_values stays [None], no suffix added to directory

    # Determine learning rates to test
    lr_values = args.lrs if args.lrs is not None else [args.lr]
    
    print(f"\n{'='*80}")
    print(f"Multi-Seed Stability Study")
    print(f"Strategy: {args.strategy}")
    print(f"Dataset: {args.dataset}")
    print(f"Seeds: {args.seeds}")
    print(f"Epochs: {args.epochs}")
    if test_label:
        print(f"{test_label}: {test_values}")
    print(f"Learning rate(s): {lr_values}")
    print(f"{'='*80}\n")
    
    all_results = {}
    
    for lr in lr_values:
     for test_value in test_values:
        print(f"\n{'='*80}")
        if args.strategy == "freeze_early_layers" and test_value is not None:
            print(f"Testing freeze_early_n = {test_value}")
        elif args.strategy == "lora" and test_value is not None:
            print(f"Testing LoRA rank = {test_value}")
        print(f"{'='*80}\n")
        
        failed_seeds = []
        successful_seeds = []
        
        for seed in args.seeds:
            if args.strategy == "freeze_early_layers":
                success = run_training(
                    seed=seed,
                    strategy=args.strategy,
                    epochs=args.epochs,
                    base_dir=args.base_dir,
                    log_dir=args.log_dir,
                    freeze_early_n=test_value,
                    dataset=args.dataset,
                    lr=lr
                )
            elif args.strategy == "lora":
                success = run_training(
                    seed=seed,
                    strategy=args.strategy,
                    epochs=args.epochs,
                    base_dir=args.base_dir,
                    log_dir=args.log_dir,
                    lora_rank=test_value,
                    dataset=args.dataset,
                    lr=lr
                )
            else:
                success = run_training(
                    seed=seed,
                    strategy=args.strategy,
                    epochs=args.epochs,
                    base_dir=args.base_dir,
                    log_dir=args.log_dir,
                    dataset=args.dataset,
                    lr=lr
                )
            
            if success:
                successful_seeds.append(seed)
            else:
                failed_seeds.append(seed)
        
        # Store results for this test value
        key = f"{test_value if test_value is not None else 'default'}_lr{lr}"
        all_results[key] = {
            "successful": successful_seeds,
            "failed": failed_seeds
        }

        # Print summary for this iteration
        print(f"\n{'='*80}")
        print("ITERATION SUMMARY")
        print(f"{'='*80}")
        if args.strategy == "freeze_early_layers" and test_value is not None:
            print(f"freeze_early_n: {test_value}")
        elif args.strategy == "lora" and test_value is not None:
            print(f"LoRA rank: {test_value}")
        print(f"LR: {lr}")
        print(f"Strategy: {args.strategy}")
        print(f"Successful seeds: {successful_seeds}")
        if failed_seeds:
            print(f"Failed seeds: {failed_seeds}")
        print(f"{'='*80}\n")
    
    # Print final summary
    print(f"\n{'='*80}")
    print("FINAL MULTI-SEED TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"Strategy: {args.strategy}")
    for key, results in all_results.items():
        if key != "default":
            if args.strategy == "freeze_early_layers":
                print(f"\n  freeze_early_n = {key}:")
            elif args.strategy == "lora":
                print(f"\n  LoRA rank = {key}:")
        else:
            print(f"\n  (Default configuration):")
        print(f"    Successful seeds: {results['successful']}")
        if results['failed']:
            print(f"    Failed seeds: {results['failed']}")
    print(f"{'='*80}\n")
    print(f"Successful seeds: {successful_seeds}")
    if failed_seeds:
        print(f"Failed seeds: {failed_seeds}")
    print(f"{'='*80}\n")
    
    return 0 if len(failed_seeds) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
