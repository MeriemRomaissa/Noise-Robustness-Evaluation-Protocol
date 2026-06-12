#!/usr/bin/env python3
"""
Multi-Seed Stability Analysis Script
Analyzes fine-tuning results across multiple seeds and computes stability metrics.

Metrics Computed:
1. Test Accuracy: Mean ± Std across seeds
2. Gen_gap_last: train_ba[last_epoch] - test_ba[last_epoch]
3. Gen_gap_best: train_ba[best_val_epoch] - test_ba[best_val_epoch]  ← both from same epoch
4. TSS (Training Saturation Speed): first epoch reaching threshold% of max val accuracy
5. RCP (Relative Convergence Point): best_val_epoch / total_epochs  ∈ (0, 1]
6. Model Capacity: Number of trainable parameters
7. TLCR (Training Loss Convergence Ratio): train_loss[epoch_0] / train_loss[last_epoch]

Usage:
    python analyze_multiseed_stability.py --strategy freeze_backbone --dataset TUAB --model LABRAM
    python analyze_multiseed_stability.py --strategy original --seeds 0 42 123 --dataset TUEV --model LABRAM
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from openpyxl import Workbook, load_workbook


# ---------------------------------------------------------------------------
# Log discovery
# ---------------------------------------------------------------------------

def find_log_files(
    strategy: str,
    seeds: List[int],
    log_dir: str = "log",
    ckpt_dir: str = "checkpoints",
    freeze_early_n: Optional[int] = None,
    lora_rank: Optional[int] = None,
    dataset: str = "TUAB",
    model: str = "LABRAM",
    run_name: Optional[str] = None,
) -> Dict[int, Path]:
    """Find log directories for each seed.

    Supports directory structure: checkpoints/DATASET/MODEL/strategy/seed_N/
    """
    log_files = {}

    dir_suffix = ""
    if strategy == "freeze_early_layers" and freeze_early_n is not None:
        dir_suffix = f"_n{freeze_early_n}"
    elif strategy == "lora" and lora_rank is not None:
        dir_suffix = f"_r{lora_rank}"

    dataset_upper = dataset.upper()
    model_dir = "Labram" if model.upper() == "LABRAM" else model.upper()
    run_label = run_name if run_name else f"{strategy}{dir_suffix}"
    ckpt_base_path = Path(ckpt_dir) / dataset_upper / model_dir / run_label
    base_log_txt = ckpt_base_path / "log.txt"

    for seed in seeds:
        seed_dir = ckpt_base_path / f"seed_{seed}"

        if (seed_dir / "log.txt").exists():
            log_files[seed] = seed_dir
            continue

        if base_log_txt.exists():
            log_files[seed] = ckpt_base_path
            continue

        # Fallback: legacy log directory layout
        log_base_path = Path(log_dir) / f"finetune_{dataset.lower()}_{strategy}{dir_suffix}"
        legacy_seed_dir = log_base_path / f"seed_{seed}"

        summary_files = list(legacy_seed_dir.glob("**/summary_stats.json")) if legacy_seed_dir.exists() else []
        if summary_files:
            log_files[seed] = summary_files[0]
        elif legacy_seed_dir.exists() and list(legacy_seed_dir.glob("checkpoint-*.pth")):
            log_files[seed] = legacy_seed_dir

    return log_files


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def extract_metrics_from_checkpoint(log_dir: Path, main_performance: str = "normal") -> Dict:
    """Parse log.txt (one JSON object per line) and return per-epoch histories
    plus final-epoch scalars.

    Histories collected (one entry per logged epoch):
        val_acc_history, val_balanced_acc_history
        train_acc_history, train_balanced_acc_history
        test_acc_history, test_balanced_acc_history   ← needed for gen_gap_best

    Final-epoch scalars (from the last line):
        train_loss, val_loss,
        train_class_acc, train_balanced_accuracy,
        val_accuracy, val_balanced_accuracy,
        test_accuracy, test_balanced_accuracy,
        n_parameters
    """
    metrics = {
        # final-epoch scalars
        "train_loss": None,
        "val_loss": None,
        "train_class_acc": None,
        "train_balanced_accuracy": None,
        "val_accuracy": None,
        "val_balanced_accuracy": None,
        "test_accuracy": None,
        "test_balanced_accuracy": None,
        "n_parameters": None,
        # per-epoch histories
        "val_acc_history": [],
        "val_balanced_acc_history": [],
        "train_acc_history": [],
        "train_balanced_acc_history": [],
        "test_acc_history": [],           # NEW — required for gen_gap_best
        "test_balanced_acc_history": [],  # NEW — required for gen_gap_best
        "train_loss_history": [],         # required for TLCR
    }

    if not log_dir.exists():
        print(f"  WARNING: Log directory not found: {log_dir}")
        return metrics

    json_log = log_dir / "log.txt"
    if json_log.exists():
        try:
            with open(json_log, "r") as f:
                lines = [l.strip() for l in f if l.strip()]

            for line in lines:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if metrics["n_parameters"] is None and "n_parameters" in data:
                    metrics["n_parameters"] = data["n_parameters"]

                # validation histories
                metrics["val_acc_history"].append(data.get("val_accuracy", np.nan))
                metrics["val_balanced_acc_history"].append(data.get("val_balanced_accuracy", np.nan))

                # training histories
                metrics["train_acc_history"].append(data.get("train_class_acc", np.nan))
                metrics["train_balanced_acc_history"].append(data.get("train_balanced_accuracy", np.nan))

                # test histories — critical for gen_gap_best correctness
                metrics["test_acc_history"].append(data.get("test_accuracy", np.nan))
                metrics["test_balanced_acc_history"].append(data.get("test_balanced_accuracy", np.nan))

                # train loss history — required for TLCR
                metrics["train_loss_history"].append(data.get("train_loss", np.nan))

            # final-epoch scalars from last line
            if lines:
                try:
                    data = json.loads(lines[-1])
                    metrics["train_loss"] = data.get("train_loss")
                    metrics["val_loss"] = data.get("val_loss")
                    metrics["train_class_acc"] = data.get("train_class_acc")
                    metrics["train_balanced_accuracy"] = data.get("train_balanced_accuracy")
                    metrics["val_accuracy"] = data.get("val_accuracy")
                    metrics["val_balanced_accuracy"] = data.get("val_balanced_accuracy")
                    metrics["test_accuracy"] = data.get("test_accuracy")
                    metrics["test_balanced_accuracy"] = data.get("test_balanced_accuracy")
                except json.JSONDecodeError:
                    pass

            return metrics

        except Exception as e:
            print(f"  WARNING: Could not parse JSON log: {e}")

    # Fallback: plain-text log files (best-effort scalar extraction only,
    # no per-epoch history available in this format)
    log_txt_files = sorted(log_dir.glob("*.txt"))
    if not log_txt_files:
        print(f"  WARNING: No log files found in {log_dir}")
        return metrics

    for log_file in reversed(log_txt_files):
        with open(log_file, "r") as f:
            lines = f.read().split("\n")

        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                def _extract(key):
                    if key + ":" in line:
                        return float(line.split(key + ":")[1].strip().split()[0])
                    return None

                if metrics["train_loss"] is None:
                    metrics["train_loss"] = _extract("train_loss")
                if metrics["val_loss"] is None:
                    metrics["val_loss"] = _extract("val_loss")
                if metrics["train_class_acc"] is None:
                    v = _extract("train_class_acc")
                    metrics["train_class_acc"] = v / 100.0 if v is not None else None
                if metrics["train_balanced_accuracy"] is None:
                    v = _extract("train_balanced_accuracy")
                    metrics["train_balanced_accuracy"] = v / 100.0 if v is not None else None
                if metrics["val_accuracy"] is None:
                    v = _extract("val_accuracy")
                    metrics["val_accuracy"] = v / 100.0 if v is not None else None
                if metrics["val_balanced_accuracy"] is None:
                    v = _extract("val_balanced_accuracy")
                    metrics["val_balanced_accuracy"] = v / 100.0 if v is not None else None
                if metrics["test_accuracy"] is None:
                    v = _extract("test_accuracy")
                    metrics["test_accuracy"] = v / 100.0 if v is not None else None
                if metrics["test_balanced_accuracy"] is None:
                    v = _extract("test_balanced_accuracy")
                    metrics["test_balanced_accuracy"] = v / 100.0 if v is not None else None
            except (IndexError, ValueError):
                continue

        if any(v is not None for v in metrics.values()):
            break

    return metrics


# ---------------------------------------------------------------------------
# Per-seed summary (single source of truth for all derived metrics)
# ---------------------------------------------------------------------------

def compute_seed_summary(metrics: Dict, tss_threshold: float, main_performance: str) -> Dict:
    """Derive all scalar metrics for one seed from its history.

    gen_gap_last : train_ba[last_epoch]  - test_ba[last_epoch]
    gen_gap_best : train_ba[best_val_epoch] - test_ba[best_val_epoch]
                   ← both quantities taken from the SAME epoch (the fix)
    tss          : first epoch (1-based) where val_ba >= tss_threshold * max(val_ba)
    rcp          : best_val_epoch / total_epochs
    """
    summary = {
        "capacity": metrics.get("n_parameters"),
        "test_acc": None,
        "test_acc_last": None,
        "test_acc_best": None,
        "gen_gap_last": None,
        "gen_gap_best": None,
        "tss": None,
        "best_epoch": None,
        "rcp": None,
        "tlcr": None,
    }

    if main_performance == "balanced":
        summary["test_acc"] = metrics.get("test_balanced_accuracy")  # kept for backward compat
        summary["test_acc_last"] = metrics.get("test_balanced_accuracy")
        # last-epoch scalars for gen_gap_last
        train_last = metrics.get("train_balanced_accuracy")
        test_last  = metrics.get("test_balanced_accuracy")
        # per-epoch histories
        val_history   = metrics.get("val_balanced_acc_history") or []
        train_history = metrics.get("train_balanced_acc_history") or []
        test_history  = metrics.get("test_balanced_acc_history") or []
    else:
        summary["test_acc"] = metrics.get("test_accuracy")  # kept for backward compat
        summary["test_acc_last"] = metrics.get("test_accuracy")
        train_last = metrics.get("train_class_acc")
        test_last  = metrics.get("test_accuracy")
        val_history   = metrics.get("val_acc_history") or []
        train_history = metrics.get("train_acc_history") or []
        test_history  = metrics.get("test_acc_history") or []

    # gen_gap_last: both values are final-epoch scalars — no history needed
    if train_last is not None and test_last is not None:
        summary["gen_gap_last"] = float(train_last) - float(test_last)

    if not val_history:
        return summary

    val_arr   = np.array(val_history,   dtype=float)
    train_arr = np.array(train_history, dtype=float) if train_history else None
    test_arr  = np.array(test_history,  dtype=float) if test_history  else None

    total_epochs    = len(val_arr)
    best_epoch_idx  = int(np.nanargmax(val_arr))   # 0-based
    best_epoch      = best_epoch_idx + 1            # 1-based for reporting
    max_val         = float(np.nanmax(val_arr))

    # TSS
    tss_level = tss_threshold * max_val
    crossing  = np.where(val_arr >= tss_level)[0]
    if len(crossing) > 0:
        summary["tss"] = int(crossing[0]) + 1  # 1-based

    # best_epoch and RCP
        summary["best_epoch"] = best_epoch
        summary["rcp"] = best_epoch / total_epochs    # gen_gap_best — BOTH train and test taken from best_val_epoch
    if (train_arr is not None
            and test_arr is not None
            and len(train_arr) > best_epoch_idx
            and len(test_arr)  > best_epoch_idx):
        train_at_best = float(train_arr[best_epoch_idx])
        test_at_best  = float(test_arr[best_epoch_idx])
        if not (np.isnan(train_at_best) or np.isnan(test_at_best)):
            summary["gen_gap_best"] = train_at_best - test_at_best
            summary["test_acc_best"] = test_at_best

    # TLCR: train_loss[epoch_0] / train_loss[last_epoch]
    loss_history = metrics.get("train_loss_history") or []
    if len(loss_history) >= 2:
        loss_epoch0 = float(loss_history[0])
        loss_last   = float(loss_history[-1])
        if not (np.isnan(loss_epoch0) or np.isnan(loss_last)) and loss_last != 0:
            summary["tlcr"] = loss_epoch0 / loss_last

    return summary


# ---------------------------------------------------------------------------
# Aggregate metrics across seeds (printing + return)
# ---------------------------------------------------------------------------

def compute_stability_metrics(
    all_metrics: Dict[int, Dict],
    tss_threshold: float,
    main_performance: str,
) -> Dict:
    """Print per-seed details and aggregate stability metrics across seeds."""

    test_accuracies        = []
    gen_gaps_last          = []
    gen_gaps_best          = []
    tss_values             = []
    rcp_values             = []
    tlcr_values            = []
    n_parameters           = None

    print("\n" + "=" * 80)
    print("PER-SEED METRICS")
    print("=" * 80)

    for seed in sorted(all_metrics.keys()):
        metrics = all_metrics[seed]
        print(f"\nSeed {seed}:")

        if n_parameters is None and metrics.get("n_parameters") is not None:
            n_parameters = metrics["n_parameters"]

        # Print raw values
        if main_performance == "balanced":
            print(f"  Train Loss:              {metrics.get('train_loss')}")
            print(f"  Val Loss:                {metrics.get('val_loss')}")
            tba = metrics.get("train_balanced_accuracy")
            print(f"  Train Balanced Accuracy: {tba if tba is not None else 'N/A'}")
            print(f"  Val Balanced Accuracy:   {metrics.get('val_balanced_accuracy')}")
            print(f"  Test Balanced Accuracy:  {metrics.get('test_balanced_accuracy')}")
        else:
            print(f"  Train Loss:      {metrics.get('train_loss')}")
            print(f"  Val Loss:        {metrics.get('val_loss')}")
            print(f"  Train Accuracy:  {metrics.get('train_class_acc')}")
            print(f"  Val Accuracy:    {metrics.get('val_accuracy')}")
            print(f"  Test Accuracy:   {metrics.get('test_accuracy')}")

        # Derive all metrics from a single consistent function
        summary = compute_seed_summary(metrics, tss_threshold, main_performance)

        if summary["test_acc"] is not None:
            test_accuracies.append(summary["test_acc"])

        print(f"  test_acc_last:   {summary['test_acc_last']:.4f}" if summary['test_acc_last'] is not None else "  test_acc_last:   N/A")
        print(f"  test_acc_best:   {summary['test_acc_best']:.4f}" if summary['test_acc_best'] is not None else "  test_acc_best:   N/A")

        if summary["gen_gap_last"] is not None:
            gen_gaps_last.append(summary["gen_gap_last"])
            print(f"  Gen_gap_last:    {summary['gen_gap_last']:.4f}")
        else:
            print(f"  Gen_gap_last:    N/A")

        if summary["gen_gap_best"] is not None:
            gen_gaps_best.append(summary["gen_gap_best"])
            print(f"  Gen_gap_best:    {summary['gen_gap_best']:.4f}")
        else:
            print(f"  Gen_gap_best:    N/A (test history missing — check log)")

        if summary["tss"] is not None:
            tss_values.append(summary["tss"])
            print(f"  TSS:             Epoch {summary['tss']}")

        if summary["rcp"] is not None:
            rcp_values.append(summary["rcp"])
            print(f"  RCP:             {summary['rcp']:.4f}  (best epoch / total epochs)")

        if summary["tlcr"] is not None:
            tlcr_values.append(summary["tlcr"])
            print(f"  TLCR:            {summary['tlcr']:.4f}  (train_loss[epoch_0] / train_loss[last_epoch])")
        else:
            print(f"  TLCR:            N/A")

    # Aggregate
    print("\n" + "=" * 80)
    print("AGGREGATE STABILITY METRICS (Across All Seeds)")
    print("=" * 80)

    results = {}
    test_accs_last = [s["test_acc_last"] for s in [compute_seed_summary(all_metrics[sd], tss_threshold, main_performance) for sd in sorted(all_metrics)] if s["test_acc_last"] is not None]
    test_accs_best = [s["test_acc_best"] for s in [compute_seed_summary(all_metrics[sd], tss_threshold, main_performance) for sd in sorted(all_metrics)] if s["test_acc_best"] is not None]

    if test_accs_last:
        mean_last = np.mean(test_accs_last)
        std_last  = np.std(test_accs_last)
        results["test_acc_last_mean"] = mean_last
        results["test_acc_last_std"]  = std_last
        label = "Test Balanced Accuracy (last epoch)" if main_performance == "balanced" else "Test Accuracy (last epoch)"
        print(f"\n1. {label}")
        print(f"   Mean ± Std : {mean_last:.4f} ± {std_last:.4f}")
        print(f"   Values     : {[f'{x:.4f}' for x in test_accs_last]}")

    if test_accs_best:
        mean_best = np.mean(test_accs_best)
        std_best  = np.std(test_accs_best)
        results["test_acc_best_mean"] = mean_best
        results["test_acc_best_std"]  = std_best
        label = "Test Balanced Accuracy (best val epoch)" if main_performance == "balanced" else "Test Accuracy (best val epoch)"
        print(f"\n2. {label}")
        print(f"   Mean ± Std : {mean_best:.4f} ± {std_best:.4f}")
        print(f"   Values     : {[f'{x:.4f}' for x in test_accs_best]}")

    if test_accuracies:
        mean_acc = np.mean(test_accuracies)
        std_acc  = np.std(test_accuracies)
        results["test_accuracy_mean"] = mean_acc
        results["test_accuracy_std"]  = std_acc
        label = "Test Balanced Accuracy" if main_performance == "balanced" else "Test Accuracy"
        print(f"\n3. {label} (legacy aggregate)")
        print(f"   Mean ± Std : {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"   Values     : {[f'{x:.4f}' for x in test_accuracies]}")

    if gen_gaps_last:
        mean_gl = np.mean(gen_gaps_last)
        results["gen_gap_last_mean"] = mean_gl
        label = "Gen_gap_last  (train_ba[last] − test_ba[last])"
        print(f"\n2. {label}")
        print(f"   Mean   : {mean_gl:.4f}")
        print(f"   Values : {[f'{x:.4f}' for x in gen_gaps_last]}")

    if gen_gaps_best:
        mean_gb = np.mean(gen_gaps_best)
        results["gen_gap_best_mean"] = mean_gb
        label = "Gen_gap_best  (train_ba[best_val_epoch] − test_ba[best_val_epoch])"
        print(f"\n3. {label}")
        print(f"   Mean   : {mean_gb:.4f}")
        print(f"   Values : {[f'{x:.4f}' for x in gen_gaps_best]}")

    if tss_values:
        mean_tss = np.mean(tss_values)
        results["tss_mean"] = mean_tss
        print(f"\n4. Training Saturation Speed (TSS)  [threshold = {tss_threshold:.0%} of max val]")
        print(f"   Mean epoch : {mean_tss:.1f}")
        print(f"   Values     : {[str(int(x)) for x in tss_values]}")

    if rcp_values:
        mean_rcp = np.mean(rcp_values)
        results["rcp_mean"] = mean_rcp
        print(f"\n5. Relative Convergence Point (RCP = best_val_epoch / total_epochs)")
        print(f"   Mean   : {mean_rcp:.4f}")
        print(f"   Values : {[f'{x:.4f}' for x in rcp_values]}")
        print(f"   → Overfitting window: last {(1 - mean_rcp)*100:.0f}% of training wasted on average")

    if tlcr_values:
        mean_tlcr = np.mean(tlcr_values)
        results["tlcr_mean"] = mean_tlcr
        print(f"\n6. Training Loss Convergence Ratio (TLCR = train_loss[epoch_0] / train_loss[last_epoch])")
        print(f"   Mean   : {mean_tlcr:.4f}")
        print(f"   Values : {[f'{x:.4f}' for x in tlcr_values]}")
        print(f"   → Higher TLCR = greater relative loss reduction (stronger convergence)")

    if n_parameters is not None:
        results["n_parameters"] = n_parameters
        print(f"\n6. Model Capacity")
        print(f"   Trainable parameters: {n_parameters:,}")

    print("\n" + "=" * 80)
    return results


# ---------------------------------------------------------------------------
# Excel output
# ---------------------------------------------------------------------------

def build_strategy_label(strategy: str, freeze_early_n: Optional[int], lora_rank: Optional[int]) -> str:
    if strategy == "freeze_early_layers" and freeze_early_n is not None:
        return f"{strategy}_n{freeze_early_n}"
    if strategy == "lora" and lora_rank is not None:
        return f"{strategy}_r{lora_rank}"
    return strategy


def get_log_start_time(log_path: Path) -> str:
    candidate = (log_path / "log.txt") if log_path.is_dir() else log_path
    if candidate.exists():
        return datetime.fromtimestamp(os.path.getctime(candidate)).strftime("%Y-%m-%d %H:%M:%S")
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_rows_to_excel(rows: List[Dict], excel_path: Path) -> None:
    excel_path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "Time", "Model", "Dataset", "Strategy", "Seed",
        "Capacity", "test_acc_last", "test_acc_best", "Test Acc",
        "Gen_gap_last", "Gen_gap_best",
        "TSS", "Best_epoch", "RCP", "TLCR",
        "tss_threshold", "main_performance",
    ]

    if excel_path.exists():
        wb = load_workbook(excel_path)
        ws = wb.active
    else:
        wb = Workbook()
        ws = wb.active
        ws.title = "Results"
        ws.append(headers)

    for row in rows:
        ws.append([
            row.get("Time"),
            row.get("Model"),
            row.get("Dataset"),
            row.get("Strategy"),
            row.get("Seed"),
            row.get("Capacity"),
            row.get("test_acc_last"),
            row.get("test_acc_best"),
            row.get("Test Acc"),
            row.get("Gen_gap_last"),
            row.get("Gen_gap_best"),
            row.get("TSS"),
            row.get("Best_epoch"),
            row.get("RCP"),
            row.get("TLCR"),
            row.get("tss_threshold"),
            row.get("main_performance"),
        ])

    wb.save(excel_path)
    print(f"\nResults appended to: {excel_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze multi-seed fine-tuning stability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_multiseed_stability.py --strategy freeze_backbone --dataset TUAB --model LABRAM
  python analyze_multiseed_stability.py --strategy original --seeds 0 42 123 --dataset TUEV --model LABRAM
        """,
    )
    parser.add_argument("--strategy", type=str, default="freeze_backbone",
                        choices=["original", "freeze_backbone", "freeze_early_layers", "aggressive_reg", "lora"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 123])
    parser.add_argument("--dataset", type=str, default="TUAB", choices=["TUAB", "TUEV"])
    parser.add_argument("--model", type=str, default="LABRAM", choices=["LABRAM", "EEGNET"])
    parser.add_argument("--log_dir", type=str, default="log")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--freeze_early_n", type=int, default=None)
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--tss_threshold", type=float, default=0.95)
    parser.add_argument("--main_performance", type=str, default="normal", choices=["normal", "balanced"])
    parser.add_argument("--excel_path", type=str, default="checkpoints/stability_results.xlsx")
    parser.add_argument("--run_name", type=str, default=None)

    args = parser.parse_args()

    print(f"\nMulti-Seed Stability Analysis")
    print(f"Strategy : {args.strategy}")
    print(f"Seeds    : {args.seeds}")
    print(f"Dataset  : {args.dataset}")
    print(f"Model    : {args.model}")
    if args.freeze_early_n is not None:
        print(f"Freeze N : {args.freeze_early_n}")
    if args.lora_rank is not None:
        print(f"LoRA rank: {args.lora_rank}")

    log_files = find_log_files(
        args.strategy, args.seeds, args.log_dir, args.ckpt_dir,
        args.freeze_early_n, args.lora_rank,
        dataset=args.dataset, model=args.model, run_name=args.run_name,
    )

    if not log_files:
        print(f"ERROR: No log files found for strategy '{args.strategy}'")
        return 1

    print(f"Found logs for seeds: {sorted(log_files.keys())}")

    all_metrics: Dict[int, Dict] = {}
    for seed in args.seeds:
        if seed not in log_files:
            print(f"\nWARNING: No log found for seed {seed}, skipping...")
            continue
        print(f"\nExtracting metrics for seed {seed} from {log_files[seed]} ...")
        m = extract_metrics_from_checkpoint(log_files[seed], args.main_performance)
        all_metrics[seed] = m
        if not any(v is not None for v in m.values()):
            print(f"  WARNING: Could not extract any metrics from {log_files[seed]}")

    if not all_metrics:
        print("\nERROR: Could not extract metrics from any seed")
        return 1

    compute_stability_metrics(all_metrics, args.tss_threshold, args.main_performance)

    strategy_label = build_strategy_label(args.strategy, args.freeze_early_n, args.lora_rank)
    excel_rows = []
    for seed, metrics in sorted(all_metrics.items()):
        summary = compute_seed_summary(metrics, args.tss_threshold, args.main_performance)
        excel_rows.append({
            "Time":             get_log_start_time(log_files[seed]),
            "Model":            args.model,
            "Dataset":          args.dataset,
            "Strategy":         strategy_label,
            "Seed":             seed,
            "Capacity":         summary["capacity"],
            "test_acc_last":    summary["test_acc_last"],
            "test_acc_best":    summary["test_acc_best"],
            "Test Acc":         summary["test_acc"],
            "Gen_gap_last":     summary["gen_gap_last"],
            "Gen_gap_best":     summary["gen_gap_best"],
            "TSS":              summary["tss"],
            "Best_epoch":       summary["best_epoch"],
            "RCP":              summary["rcp"],
            "TLCR":             summary["tlcr"],
            "tss_threshold":    args.tss_threshold,
            "main_performance": args.main_performance,
        })

    append_rows_to_excel(excel_rows, Path(args.excel_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())