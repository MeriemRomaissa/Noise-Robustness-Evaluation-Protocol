#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


DEFAULT_H5 = "/nicoletye/workspace/unified_tuab/data/canonical_tuab_full.h5"
DEFAULT_REPORT_DIR = "/nicoletye/workspace/unified_tuab/reports"

# LaBraM processed full TUAB reference counts.
# We use this only as a reference for train/val ratio, not to discard H5 windows.
LABRAM_COUNTS = {
    "train": {"n": 295751, "normal_0": 145918, "abnormal_1": 149833},
    "val":   {"n": 76387,  "normal_0": 37791,  "abnormal_1": 38596},
    "test":  {"n": 36694,  "normal_0": 19907,  "abnormal_1": 16787},
}

INVALID_GROUP_VALUES = {"", "none", "nan", "null", "na", "n/a", "unknown"}


def decode_array(arr):
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8", errors="ignore"))
        elif hasattr(x, "decode"):
            try:
                out.append(x.decode("utf-8", errors="ignore"))
            except Exception:
                out.append(str(x))
        else:
            out.append(str(x))
    return out


def load_h5_metadata(h5_path):
    with h5py.File(h5_path, "r") as f:
        n = f["eeg"].shape[0]
        meta = f["metadata"]

        df = pd.DataFrame({
            "h5_index": np.arange(n, dtype=np.int64),
            "label": np.asarray(meta["label"][:], dtype=np.int64),
            "h5_original_split": decode_array(meta["split"][:]),
            "source_path": decode_array(meta["source_path"][:]),
            "sample_id": decode_array(meta["sample_id"][:]),
            "recording_id": decode_array(meta["recording_id"][:]),
            "subject_id": decode_array(meta["subject_id"][:]),
            "start_time": np.asarray(meta["start_time"][:]),
            "end_time": np.asarray(meta["end_time"][:]),
        })

        if "raw_hash" in meta:
            df["raw_hash"] = decode_array(meta["raw_hash"][:])
        else:
            df["raw_hash"] = ""

    return df


def choose_group_key(df, requested):
    if requested != "auto":
        return requested

    # Prefer subject-level splitting if available and meaningful.
    subject = df["subject_id"].astype(str).str.strip()
    n_subject = subject[subject != ""].nunique()

    recording = df["recording_id"].astype(str).str.strip()
    n_recording = recording[recording != ""].nunique()

    if n_subject > 1:
        return "subject_id"
    return "recording_id"


def _valid_group_value(value):
    return str(value).strip().lower() not in INVALID_GROUP_VALUES


def add_group_columns(df, preferred_group_key):
    """
    Add per-row group_key/group_id columns.
    Prefer the requested key, falling back to recording_id when the preferred
    value is empty or invalid.
    """
    preferred = df[preferred_group_key].astype(str)
    fallback = df["recording_id"].astype(str)
    use_preferred = preferred.map(_valid_group_value)

    df["group_key"] = np.where(use_preferred, preferred_group_key, "recording_id")
    df["group_id"] = np.where(use_preferred, preferred.str.strip(), fallback.str.strip())
    return df


def split_counts(df, split_col="canonical_split"):
    rows = []
    for split in ["train", "val", "test"]:
        sub = df[df[split_col] == split]
        labels = sub["label"].to_numpy()
        rows.append({
            "split": split,
            "n": int(len(sub)),
            "normal_0": int((labels == 0).sum()),
            "abnormal_1": int((labels == 1).sum()),
        })
    return rows


def score_counts(c0, c1, target0, target1):
    total = c0 + c1
    target_total = target0 + target1

    # Relative error in both class counts and total count.
    e0 = abs(c0 - target0) / max(target0, 1)
    e1 = abs(c1 - target1) / max(target1, 1)
    et = abs(total - target_total) / max(target_total, 1)

    return e0 + e1 + 0.5 * et


def select_validation_groups(group_df, target0, target1, seed=42, n_iter=500):
    """
    Randomized greedy search.
    Select whole subject/recording groups for validation while matching class-count targets.
    """
    records = group_df.to_dict("records")
    best_selected = set()
    best_score = float("inf")
    best_counts = (0, 0)

    rng_master = random.Random(seed)

    for it in range(n_iter):
        rng = random.Random(rng_master.randint(0, 10**12))
        order = records[:]
        rng.shuffle(order)

        selected = set()
        c0, c1 = 0, 0
        current_score = score_counts(c0, c1, target0, target1)

        for g in order:
            new0 = c0 + int(g["normal_0"])
            new1 = c1 + int(g["abnormal_1"])
            new_score = score_counts(new0, new1, target0, target1)

            # Add group if it improves target matching.
            if new_score < current_score:
                selected.add(g["group_id"])
                c0, c1 = new0, new1
                current_score = new_score

        if current_score < best_score:
            best_score = current_score
            best_selected = selected
            best_counts = (c0, c1)

    return best_selected, best_counts, best_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=str, default=DEFAULT_H5)
    parser.add_argument("--report_dir", type=str, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--group_key", type=str, default="auto",
                        choices=["auto", "subject_id", "recording_id"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_iter", type=int, default=500)
    args = parser.parse_args()

    h5_path = Path(args.h5)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    out_csv = report_dir / "canonical_h5_labram_referenced_max_coverage_split_index.csv"
    out_json = report_dir / "canonical_h5_labram_referenced_max_coverage_split_index.json"
    out_summary = report_dir / "canonical_h5_labram_referenced_max_coverage_split_summary.txt"

    print("=" * 100)
    print("Build Canonical H5 LaBraM-Referenced Max-Coverage Split Index")
    print("=" * 100)
    print(f"H5 path    : {h5_path}")
    print(f"Report dir : {report_dir}")

    df = load_h5_metadata(h5_path)

    group_key = choose_group_key(df, args.group_key)
    print(f"Group key  : {group_key}")

    # Build split from existing H5 train/test.
    # Max coverage rule:
    # - original H5 test remains test
    # - original H5 train is split into train + validation
    df["canonical_split"] = df["h5_original_split"].copy()

    train_pool_mask = df["h5_original_split"] == "train"
    test_pool_mask = df["h5_original_split"] == "test"

    train_pool = df[train_pool_mask].copy()
    test_pool = df[test_pool_mask].copy()

    # LaBraM reference train/val ratio within non-test data.
    labram_train_n = LABRAM_COUNTS["train"]["n"]
    labram_val_n = LABRAM_COUNTS["val"]["n"]
    labram_val_fraction = labram_val_n / (labram_train_n + labram_val_n)

    train_labels = train_pool["label"].to_numpy()
    train0 = int((train_labels == 0).sum())
    train1 = int((train_labels == 1).sum())

    target_val0 = int(round(train0 * labram_val_fraction))
    target_val1 = int(round(train1 * labram_val_fraction))
    target_val_total = target_val0 + target_val1

    print("")
    print("Reference")
    print("-" * 100)
    print(f"LaBraM train n: {labram_train_n}")
    print(f"LaBraM val n  : {labram_val_n}")
    print(f"LaBraM val fraction within train+val: {labram_val_fraction:.6f}")

    print("")
    print("Canonical H5 original split")
    print("-" * 100)
    print(f"Original train pool n: {len(train_pool)}")
    print(f"Original test pool n : {len(test_pool)}")
    print(f"Original train labels: normal_0={train0}, abnormal_1={train1}")
    print(f"Target val labels    : normal_0={target_val0}, abnormal_1={target_val1}, total={target_val_total}")

    # Prepare group counts from original train pool only.
    train_pool = add_group_columns(train_pool, group_key)

    group_df = (
        train_pool
        .groupby("group_id", sort=False)
        .agg(
            n=("h5_index", "count"),
            normal_0=("label", lambda x: int((np.asarray(x) == 0).sum())),
            abnormal_1=("label", lambda x: int((np.asarray(x) == 1).sum())),
        )
        .reset_index()
    )

    print("")
    print("Validation group selection")
    print("-" * 100)
    print(f"Candidate groups from original train: {len(group_df)}")
    selected_val_groups, best_counts, best_score = select_validation_groups(
        group_df,
        target0=target_val0,
        target1=target_val1,
        seed=args.seed,
        n_iter=args.n_iter,
    )

    print(f"Selected val groups: {len(selected_val_groups)}")
    print(f"Selected val labels: normal_0={best_counts[0]}, abnormal_1={best_counts[1]}, total={sum(best_counts)}")
    print(f"Target match score : {best_score:.6f}")

    # Assign validation split.
    val_h5_indices = train_pool[train_pool["group_id"].isin(selected_val_groups)]["h5_index"].to_numpy()
    df.loc[df["h5_index"].isin(val_h5_indices), "canonical_split"] = "val"

    # Add group_id to final df using the same fallback rule.
    df = add_group_columns(df, group_key)

    # Leakage checks.
    train_groups = set(df.loc[df["canonical_split"] == "train", "group_id"])
    val_groups = set(df.loc[df["canonical_split"] == "val", "group_id"])
    test_groups = set(df.loc[df["canonical_split"] == "test", "group_id"])
    train_val_leakage_groups = sorted(train_groups & val_groups)

    # This stricter diagnostic is reported separately because original test is
    # intentionally preserved as test and is not reallocated by this script.
    group_split_n = df.groupby("group_id")["canonical_split"].nunique()
    all_split_overlap_groups = group_split_n[group_split_n > 1].index.tolist()
    original_test_preserved = bool((df.loc[test_pool_mask, "canonical_split"] == "test").all())

    # Counts.
    counts_rows = split_counts(df, split_col="canonical_split")

    print("")
    print("Final canonical split counts")
    print("-" * 100)
    for row in counts_rows:
        print(row)

    print("")
    print("Leakage check")
    print("-" * 100)
    print(f"Train/val overlapping groups: {len(train_val_leakage_groups)}")
    print(f"Original H5 test preserved as test: {original_test_preserved}")
    print(f"Groups appearing in more than one final split, including test diagnostic: {len(all_split_overlap_groups)}")
    if train_val_leakage_groups:
        print("WARNING: Some group IDs appear in both train and val. Showing first 20:")
        print(train_val_leakage_groups[:20])

    # Save split index.
    keep_cols = [
        "h5_index",
        "canonical_split",
        "h5_original_split",
        "label",
        "group_key",
        "group_id",
        "subject_id",
        "recording_id",
        "sample_id",
        "start_time",
        "end_time",
        "source_path",
        "raw_hash",
    ]
    df[keep_cols].to_csv(out_csv, index=False)

    summary = {
        "h5_path": str(h5_path),
        "total_h5_windows": int(len(df)),
        "group_key": group_key,
        "seed": args.seed,
        "n_iter": args.n_iter,
        "labram_reference_counts": LABRAM_COUNTS,
        "labram_val_fraction_within_train_val": labram_val_fraction,
        "target_val_counts_from_canonical_train": {
            "normal_0": target_val0,
            "abnormal_1": target_val1,
            "n": target_val_total,
        },
        "selected_val_counts": {
            "normal_0": int(best_counts[0]),
            "abnormal_1": int(best_counts[1]),
            "n": int(sum(best_counts)),
        },
        "final_split_counts": counts_rows,
        "leakage_check": {
            "train_val_no_overlap": len(train_val_leakage_groups) == 0,
            "train_val_leakage_group_count": int(len(train_val_leakage_groups)),
            "train_val_leakage_groups_first_50": train_val_leakage_groups[:50],
            "original_test_preserved": original_test_preserved,
            "all_split_overlap_group_count_including_test_diagnostic": int(len(all_split_overlap_groups)),
            "all_split_overlap_groups_first_50": all_split_overlap_groups[:50],
        },
        "output_paths": {
            "csv": str(out_csv),
            "json": str(out_json),
            "txt_summary": str(out_summary),
        },
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = []
    lines.append("Canonical H5 LaBraM-Referenced Max-Coverage Split Summary")
    lines.append("=" * 100)
    lines.append(f"H5 path: {h5_path}")
    lines.append(f"Total H5 windows: {len(df)}")
    lines.append(f"Group key: {group_key}")
    lines.append("")
    lines.append("Design rule:")
    lines.append("- Use LaBraM train/val/test as reference policy.")
    lines.append("- Maximize canonical H5 usage.")
    lines.append("- Keep original canonical H5 test split as test.")
    lines.append("- Split original canonical H5 train into train/val using LaBraM train/val ratio.")
    lines.append("- Select whole subject/recording groups for validation.")
    lines.append("")
    lines.append("LaBraM reference counts:")
    lines.append(json.dumps(LABRAM_COUNTS, indent=2))
    lines.append("")
    lines.append(f"LaBraM val fraction within train+val: {labram_val_fraction:.6f}")
    lines.append("")
    lines.append("Target validation counts from canonical H5 train pool:")
    lines.append(f"normal_0={target_val0}, abnormal_1={target_val1}, total={target_val_total}")
    lines.append("")
    lines.append("Selected validation counts:")
    lines.append(f"normal_0={best_counts[0]}, abnormal_1={best_counts[1]}, total={sum(best_counts)}")
    lines.append(f"Target match score: {best_score:.6f}")
    lines.append("")
    lines.append("Final split counts:")
    for row in counts_rows:
        lines.append(str(row))
    lines.append("")
    lines.append("Leakage check:")
    lines.append(f"Train/val overlapping groups: {len(train_val_leakage_groups)}")
    lines.append(f"Train/val leakage check passed: {len(train_val_leakage_groups) == 0}")
    lines.append(f"Original H5 test preserved as test: {original_test_preserved}")
    lines.append(
        "Groups appearing in more than one final split, including test diagnostic: "
        f"{len(all_split_overlap_groups)}"
    )
    if train_val_leakage_groups:
        lines.append("WARNING first 50 train/val leakage groups:")
        lines.append(str(train_val_leakage_groups[:50]))
    lines.append("")
    lines.append("Output files:")
    lines.append(str(out_csv))
    lines.append(str(out_json))
    lines.append(str(out_summary))

    out_summary.write_text("\n".join(lines), encoding="utf-8")

    print("")
    print("Saved:")
    print(out_csv)
    print(out_json)
    print(out_summary)
    print("")
    print("Done.")


if __name__ == "__main__":
    main()
