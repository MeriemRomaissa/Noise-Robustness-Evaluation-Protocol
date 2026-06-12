#!/usr/bin/env python3
"""
EEGNet Multi-Seed Stability Analysis Script
Analyzes EEGNet fine-tuning results across multiple seeds and computes stability metrics.

Adapted from LaBraM's analyze_multiseed_stability.py to handle EEGNet log formats.

Metrics Computed:
1. Test Accuracy: Mean ± Std across seeds
2. Gen_gap_last: train_ba[last_epoch] - test_ba[last_epoch]
3. Gen_gap_best: train_ba[best_val_epoch] - test_ba[best_val_epoch]  ← both from same epoch
4. TSS (Training Saturation Speed): first epoch reaching threshold% of max val accuracy
5. RCP (Relative Convergence Point): best_val_epoch / total_epochs  ∈ (0, 1]
6. Model Capacity: Number of trainable parameters
7. TLCR (Training Loss Convergence Ratio): train_loss[epoch_0] / train_loss[last_epoch]

Directory structure:
    checkpoints/<DATASET>/EEGNET/<run_name>/seed_<N>/log.txt

Supports two EEGNet log formats:
  Format A (TUAB v3, original TUEV) — LaBraM-compatible keys:
      train_class_acc, val_accuracy, val_balanced_accuracy, test_accuracy, test_balanced_accuracy, n_parameters
  Format B (TUEV v7 and similar) — shortened keys, comment headers:
      train_acc, train_bal, val_acc, val_bal, test_acc, test_bal  (no n_parameters — read from config.json)

Usage:
    python analyze_multiseed_stability.py --run_name original_epochs50 --dataset TUAB --seeds 0 42 123 256 512
    python analyze_multiseed_stability.py --run_name larger_model_v7 --dataset TUEV --seeds 42
    python analyze_multiseed_stability.py --run_name original --dataset TUEV --seeds 0 42 123 256 512
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
# Key-name mapping:  normalise both log formats to a common schema
# ---------------------------------------------------------------------------

# Format A keys → common keys  (identity — already LaBraM-compatible)
# Format B keys → common keys  (short names → full names)
FORMAT_B_MAP = {
    "train_acc":  "train_class_acc",
    "train_bal":  "train_balanced_accuracy",
    "val_acc":    "val_accuracy",
    "val_bal":    "val_balanced_accuracy",
    "test_acc":   "test_accuracy",
    "test_bal":   "test_balanced_accuracy",
    "val_kappa":  "val_cohen_kappa",
    "val_f1":     "val_f1_weighted",
    "test_kappa": "test_cohen_kappa",
}


def normalise_keys(data: dict) -> dict:
    """Map shortened Format-B keys to canonical Format-A names.
    Passes Format-A keys through untouched.
    """
    out = {}
    for k, v in data.items():
        out[FORMAT_B_MAP.get(k, k)] = v
    return out


# ---------------------------------------------------------------------------
# Log discovery
# ---------------------------------------------------------------------------

def find_log_files(
    run_name: str,
    seeds: List[int],
    ckpt_dir: str,
    dataset: str,
) -> Dict[int, Path]:
    """Find log directories for each seed.

    Looks for: <ckpt_dir>/<DATASET>/EEGNET/<run_name>/seed_<N>/log.txt
    """
    log_files = {}
    dataset_upper = dataset.upper()
    ckpt_base_path = Path(ckpt_dir) / dataset_upper / "EEGNET" / run_name

    if not ckpt_base_path.exists():
        print(f"WARNING: Base path does not exist: {ckpt_base_path}")
        return log_files

    for seed in seeds:
        seed_dir = ckpt_base_path / f"seed_{seed}"
        if (seed_dir / "log.txt").exists():
            log_files[seed] = seed_dir
        else:
            print(f"  WARNING: No log.txt found at {seed_dir}")

    return log_files


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def extract_metrics_from_checkpoint(log_dir: Path) -> Dict:
    """Parse log.txt (one JSON object per line) and return per-epoch histories
    plus final-epoch scalars.

    Handles both Format A and Format B logs (normalises keys internally).
    For Format B, reads n_parameters from config.json if available.

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
        "test_acc_history": [],
        "test_balanced_acc_history": [],
        "train_loss_history": [],         # required for TLCR
    }

    if not log_dir.exists():
        print(f"  WARNING: Log directory not found: {log_dir}")
        return metrics

    # Try to get n_parameters from config.json (Format B logs don't have it)
    config_json = log_dir / "config.json"
    if config_json.exists():
        try:
            with open(config_json, "r") as f:
                config = json.load(f)
            # config.json doesn't store n_parameters directly, but we can
            # try to extract it if present
            if "n_parameters" in config:
                metrics["n_parameters"] = config["n_parameters"]
        except Exception:
            pass

    json_log = log_dir / "log.txt"
    if not json_log.exists():
        print(f"  WARNING: No log.txt found in {log_dir}")
        return metrics

    try:
        with open(json_log, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        for line in lines:
            # Skip comment lines (Format B: "# V7 Training Log ...")
            if line.startswith("#"):
                continue

            try:
                data = normalise_keys(json.loads(line))
            except json.JSONDecodeError:
                continue

            # n_parameters (Format A has it in each line)
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

        # final-epoch scalars from last non-comment JSON line
        for line in reversed(lines):
            if line.startswith("#"):
                continue
            try:
                data = normalise_keys(json.loads(line))
                metrics["train_loss"] = data.get("train_loss")
                metrics["val_loss"] = data.get("val_loss")
                metrics["train_class_acc"] = data.get("train_class_acc")
                metrics["train_balanced_accuracy"] = data.get("train_balanced_accuracy")
                metrics["val_accuracy"] = data.get("val_accuracy")
                metrics["val_balanced_accuracy"] = data.get("val_balanced_accuracy")
                metrics["test_accuracy"] = data.get("test_accuracy")
                metrics["test_balanced_accuracy"] = data.get("test_balanced_accuracy")
                break
            except json.JSONDecodeError:
                continue

    except Exception as e:
        print(f"  WARNING: Could not parse log: {e}")

    return metrics


# ---------------------------------------------------------------------------
# Per-seed summary (single source of truth for all derived metrics)
# ---------------------------------------------------------------------------

def compute_seed_summary(metrics: Dict, tss_threshold: float, main_performance: str) -> Dict:
    """Derive all scalar metrics for one seed from its history.

    gen_gap_last : train_ba[last_epoch]  - test_ba[last_epoch]
    gen_gap_best : train_ba[best_val_epoch] - test_ba[best_val_epoch]
                   ← both quantities taken from the SAME epoch
    tss          : first epoch (1-based) where val_ba >= tss_threshold * max(val_ba)
    rcp          : best_val_epoch / total_epochs
    """
    summary = {
        "capacity": metrics.get("n_parameters"),
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
        summary["test_acc_last"] = metrics.get("test_balanced_accuracy")
        # last-epoch scalars for gen_gap_last
        train_last = metrics.get("train_balanced_accuracy")
        test_last  = metrics.get("test_balanced_accuracy")
        # per-epoch histories
        val_history   = metrics.get("val_balanced_acc_history") or []
        train_history = metrics.get("train_balanced_acc_history") or []
        test_history  = metrics.get("test_balanced_acc_history") or []
    else:
        summary["test_acc_last"] = metrics.get("test_accuracy")
        train_last = metrics.get("train_class_acc")
        test_last  = metrics.get("test_accuracy")
        val_history   = metrics.get("val_acc_history") or []
        train_history = metrics.get("train_acc_history") or []
        test_history  = metrics.get("test_acc_history") or []

    # gen_gap_last: both values are final-epoch scalars
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
    summary["rcp"] = best_epoch / total_epochs

    # gen_gap_best — BOTH train and test taken from best_val_epoch
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

    test_accuracies_last   = []
    test_accuracies_best   = []
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

        if summary["test_acc_last"] is not None:
            test_accuracies_last.append(summary["test_acc_last"])

        if summary["test_acc_best"] is not None:
            test_accuracies_best.append(summary["test_acc_best"])

        if summary["test_acc_last"] is not None:
            print(f"  test_acc_last:   {summary['test_acc_last']:.4f}")
        else:
            print("  test_acc_last:   N/A")

        if summary["test_acc_best"] is not None:
            print(f"  test_acc_best:   {summary['test_acc_best']:.4f}")
        else:
            print("  test_acc_best:   N/A (history missing — check log)")

        if summary["gen_gap_last"] is not None:
            gen_gaps_last.append(summary["gen_gap_last"])
            print(f"  Gen_gap_last:    {summary['gen_gap_last']:.4f}")
        else:
            print(f"  Gen_gap_last:    N/A")

        if summary["gen_gap_best"] is not None:
            gen_gaps_best.append(summary["gen_gap_best"])
            print(f"  Gen_gap_best:    {summary['gen_gap_best']:.4f}")
        else:
            print(f"  Gen_gap_best:    N/A (history missing — check log)")

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

    if test_accuracies_last:
        mean_acc = np.mean(test_accuracies_last)
        std_acc  = np.std(test_accuracies_last)
        results["test_acc_last_mean"] = mean_acc
        results["test_acc_last_std"]  = std_acc
        label = "Test Balanced Accuracy" if main_performance == "balanced" else "Test Accuracy"
        print(f"\n1. {label} (last epoch)")
        print(f"   Mean ± Std : {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"   Values     : {[f'{x:.4f}' for x in test_accuracies_last]}")

    if test_accuracies_best:
        mean_acc_best = np.mean(test_accuracies_best)
        std_acc_best  = np.std(test_accuracies_best)
        results["test_acc_best_mean"] = mean_acc_best
        results["test_acc_best_std"]  = std_acc_best
        label = "Test Balanced Accuracy" if main_performance == "balanced" else "Test Accuracy"
        print(f"\n2. {label} (at best val epoch)")
        print(f"   Mean ± Std : {mean_acc_best:.4f} ± {std_acc_best:.4f}")
        print(f"   Values     : {[f'{x:.4f}' for x in test_accuracies_best]}")

    if gen_gaps_last:
        mean_gl = np.mean(gen_gaps_last)
        results["gen_gap_last_mean"] = mean_gl
        label = "Gen_gap_last  (train_ba[last] − test_ba[last])"
        print(f"\n3. {label}")
        print(f"   Mean   : {mean_gl:.4f}")
        print(f"   Values : {[f'{x:.4f}' for x in gen_gaps_last]}")

    if gen_gaps_best:
        mean_gb = np.mean(gen_gaps_best)
        results["gen_gap_best_mean"] = mean_gb
        label = "Gen_gap_best  (train_ba[best_val_epoch] − test_ba[best_val_epoch])"
        print(f"\n4. {label}")
        print(f"   Mean   : {mean_gb:.4f}")
        print(f"   Values : {[f'{x:.4f}' for x in gen_gaps_best]}")

    if tss_values:
        mean_tss = np.mean(tss_values)
        results["tss_mean"] = mean_tss
        print(f"\n5. Training Saturation Speed (TSS)  [threshold = {tss_threshold:.0%} of max val]")
        print(f"   Mean epoch : {mean_tss:.1f}")
        print(f"   Values     : {[str(int(x)) for x in tss_values]}")

    if rcp_values:
        mean_rcp = np.mean(rcp_values)
        results["rcp_mean"] = mean_rcp
        print(f"\n6. Relative Convergence Point (RCP = best_val_epoch / total_epochs)")
        print(f"   Mean   : {mean_rcp:.4f}")
        print(f"   Values : {[f'{x:.4f}' for x in rcp_values]}")
        print(f"   → Overfitting window: last {(1 - mean_rcp)*100:.0f}% of training wasted on average")

    if tlcr_values:
        mean_tlcr = np.mean(tlcr_values)
        results["tlcr_mean"] = mean_tlcr
        print(f"\n7. Training Loss Convergence Ratio (TLCR = train_loss[epoch_0] / train_loss[last_epoch])")
        print(f"   Mean   : {mean_tlcr:.4f}")
        print(f"   Values : {[f'{x:.4f}' for x in tlcr_values]}")
        print(f"   → Higher TLCR = greater relative loss reduction (stronger convergence)")

    if n_parameters is not None:
        results["n_parameters"] = n_parameters
        print(f"\n7. Model Capacity")
        print(f"   Trainable parameters: {n_parameters:,}")

    print("\n" + "=" * 80)
    return results


# ---------------------------------------------------------------------------
# Excel output
# ---------------------------------------------------------------------------

def get_log_start_time(log_path: Path) -> str:
    candidate = (log_path / "log.txt") if log_path.is_dir() else log_path
    if candidate.exists():
        return datetime.fromtimestamp(os.path.getctime(candidate)).strftime("%Y-%m-%d %H:%M:%S")
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_rows_to_excel(rows: List[Dict], excel_path: Path) -> None:
    excel_path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "Time", "Model", "Dataset", "Strategy", "Seed",
        "Capacity", "test_acc_last", "test_acc_best",
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
        description="Analyze EEGNet multi-seed fine-tuning stability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_multiseed_stability.py --run_name original_epochs50 --dataset TUAB --seeds 0 42 123 256 512
  python analyze_multiseed_stability.py --run_name larger_model_v7 --dataset TUEV --seeds 42
  python analyze_multiseed_stability.py --run_name original --dataset TUEV --seeds 0 42 123 256 512
        """,
    )
    parser.add_argument("--run_name", type=str, required=True,
                        help="Subfolder name under checkpoints/<DATASET>/EEGNET/ "
                             "(e.g., original, original_epochs50, larger_model_v7)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 123, 256, 512])
    parser.add_argument("--dataset", type=str, default="TUAB", choices=["TUAB", "TUEV"])
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints",
                        help="Base checkpoint directory (default: checkpoints)")
    parser.add_argument("--tss_threshold", type=float, default=0.95)
    parser.add_argument("--main_performance", type=str, default="normal",
                        choices=["normal", "balanced"],
                        help="'normal' uses val_accuracy / test_accuracy; "
                             "'balanced' uses val_balanced_accuracy / test_balanced_accuracy")
    parser.add_argument("--excel_path", type=str, default=None,
                        help="Path to Excel output file (default: checkpoints/eegnet_stability_results.xlsx)")

    args = parser.parse_args()

    if args.excel_path is None:
        args.excel_path = os.path.join(args.ckpt_dir, "eegnet_stability_results.xlsx")

    print(f"\nEEGNet Multi-Seed Stability Analysis")
    print(f"Run name : {args.run_name}")
    print(f"Seeds    : {args.seeds}")
    print(f"Dataset  : {args.dataset}")
    print(f"Ckpt dir : {args.ckpt_dir}")

    log_files = find_log_files(
        args.run_name, args.seeds, args.ckpt_dir, args.dataset,
    )

    if not log_files:
        print(f"ERROR: No log files found for run_name '{args.run_name}' under "
              f"{args.ckpt_dir}/{args.dataset}/EEGNET/{args.run_name}/")
        return 1

    print(f"Found logs for seeds: {sorted(log_files.keys())}")

    all_metrics: Dict[int, Dict] = {}
    for seed in args.seeds:
        if seed not in log_files:
            print(f"\nWARNING: No log found for seed {seed}, skipping...")
            continue
        print(f"\nExtracting metrics for seed {seed} from {log_files[seed]} ...")
        m = extract_metrics_from_checkpoint(log_files[seed])
        all_metrics[seed] = m
        if not any(v is not None for k, v in m.items() if not k.endswith("_history")):
            print(f"  WARNING: Could not extract any metrics from {log_files[seed]}")

    if not all_metrics:
        print("\nERROR: Could not extract metrics from any seed")
        return 1

    compute_stability_metrics(all_metrics, args.tss_threshold, args.main_performance)

    excel_rows = []
    for seed, metrics in sorted(all_metrics.items()):
        summary = compute_seed_summary(metrics, args.tss_threshold, args.main_performance)
        excel_rows.append({
            "Time":             get_log_start_time(log_files[seed]),
            "Model":            "EEGNET",
            "Dataset":          args.dataset,
            "Strategy":         args.run_name,
            "Seed":             seed,
            "Capacity":         summary["capacity"],
            "test_acc_last":    summary["test_acc_last"],
            "test_acc_best":    summary["test_acc_best"],
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
