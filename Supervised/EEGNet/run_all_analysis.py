#!/usr/bin/env python3
"""
Batch analysis: runs analyze_multiseed_stability functions on all EEGNET versions
and writes everything into a single Excel file.

Bypasses the hardcoded EEGNET/ subfolder expected by the original script's
find_log_files() — seed dirs are specified directly here instead.
"""

from analyze_multiseed_stability import (
    extract_metrics_from_checkpoint,
    compute_seed_summary,
    append_rows_to_excel,
    get_log_start_time,
)

from pathlib import Path

BASE  = Path("/home/meriem-ubuntu/Projects/Foundation-models/EEGNET_Paper_v2/checkpoints")
EXCEL = BASE / "eegnet_all_results.xlsx"
TSS   = 0.95

# (dataset, strategy_label, base_path, seeds_list, main_performance)
# main_performance: "normal" for TUAB (accuracy), "balanced" for TUEV (balanced accuracy)
RUNS = [
    # ── TUAB ──────────────────────────────────────────────────────────────────
    ("TUAB", "EEGNET_Paper_V2",
     BASE / "TUAB/EEGNET_Paper_V2",
     [0, 42, 123, 256, 512],
     "normal"),

    # ── TUEV — ablation (single seed 42 only) ─────────────────────────────────
    ("TUEV", "EEGNET_Paper_V2__no_balance",
     BASE / "TUEV/EEGNET_Paper_V2",
     [42], "balanced"),

    ("TUEV", "EEGNET_TUEV_V3_weighted__bad_weights",
     BASE / "TUEV/EEGNET_TUEV_V3_weighted",
     [42], "balanced"),

    ("TUEV", "EEGNET_TUEV_V3_weighted2__sqrt_weights_only",
     BASE / "TUEV/EEGNET_TUEV_V3_weighted2",
     [42], "balanced"),

    ("TUEV", "EEGNET_Paper_V3_old__focal+sampler+weights",
     BASE / "TUEV/EEGNET_Paper_V3_old",
     [42], "balanced"),

    ("TUEV", "EEGNET_Paper_V4__sampler+weights_no_focal",
     BASE / "TUEV/EEGNET_Paper_V4",
     [42], "balanced"),

    ("TUEV", "EEGNET_Paper_V5__focal+weights_no_sampler",
     BASE / "TUEV/EEGNET_Paper_V5",
     [42], "balanced"),

    ("TUEV", "EEGNET_Paper_V6__sampler_only",
     BASE / "TUEV/EEGNET_Paper_V6",
     [42], "balanced"),

    # ── TUEV V3 — main result, all 5 seeds ────────────────────────────────────
    ("TUEV", "EEGNET_Paper_V3__focal+sampler+weights",
     BASE / "TUEV/EEGNET_Paper_V3",
     [0, 42, 123, 256, 512], "balanced"),
]

print("=" * 70)
print("EEGNET Batch Analysis — all versions → single Excel")
print(f"Output: {EXCEL}")
print("=" * 70)

excel_rows = []

for dataset, strategy, base_path, seeds, main_perf in RUNS:
    base_path = Path(base_path)
    for seed in seeds:
        seed_dir = base_path / f"seed_{seed}"
        log_file = seed_dir / "log.txt"

        if not log_file.exists():
            print(f"\nSKIP  {dataset}/{strategy}/seed_{seed}  — no log.txt")
            continue

        print(f"\nProcessing  {dataset} | {strategy} | seed {seed}")
        metrics = extract_metrics_from_checkpoint(seed_dir)
        summary = compute_seed_summary(metrics, TSS, main_perf)

        metric_label = "test_bal_acc" if main_perf == "balanced" else "test_acc"
        print(f"  {metric_label}_last : {summary['test_acc_last']}")
        print(f"  {metric_label}_best : {summary['test_acc_best']}")
        print(f"  gen_gap_best        : {summary['gen_gap_best']}")
        print(f"  best_epoch          : {summary['best_epoch']}")
        print(f"  RCP                 : {summary['rcp']}")
        print(f"  TLCR                : {summary['tlcr']}")

        excel_rows.append({
            "Time":             get_log_start_time(seed_dir),
            "Model":            "EEGNET",
            "Dataset":          dataset,
            "Strategy":         strategy,
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
            "tss_threshold":    TSS,
            "main_performance": main_perf,
        })

print("\n" + "=" * 70)
append_rows_to_excel(excel_rows, EXCEL)
print(f"Total rows written: {len(excel_rows)}")
print("=" * 70)
