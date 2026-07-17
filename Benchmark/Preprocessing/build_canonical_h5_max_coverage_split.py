#!/usr/bin/env python3
"""Stage 2 of the unified TUAB preprocessing pipeline.

Read metadata from the canonical H5, preserve its official test allocation,
and divide the original training pool into train and validation sets. Whole
subjects are preferred as groups; recording groups are used when subject
metadata is unavailable. The LaBraM train-to-validation ratio defines the
target size, while every canonical H5 window remains assigned to one split.

Outputs retain the contract consumed by Loader/loader_common.py:

* canonical_h5_labram_referenced_max_coverage_split_index.csv
* canonical_h5_labram_referenced_max_coverage_split_index.json
* canonical_h5_labram_referenced_max_coverage_split_summary.txt

The CSV always includes ``h5_index``, ``canonical_split``, and ``label``.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

try:
    import h5py
except ModuleNotFoundError:
    h5py = None

import numpy as np
import pandas as pd


LABRAM_COUNTS = {
    "train": {"n": 295751, "normal_0": 145918, "abnormal_1": 149833},
    "val": {"n": 76387, "normal_0": 37791, "abnormal_1": 38596},
    "test": {"n": 36694, "normal_0": 19907, "abnormal_1": 16787},
}

INVALID_GROUP_VALUES = {"", "none", "nan", "null", "na", "n/a", "unknown"}
OUTPUT_STEM = "canonical_h5_labram_referenced_max_coverage_split"
INDEX_COLUMNS = [
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


@dataclass(frozen=True)
class ValidationSearch:
    """Hold class targets and reproducible group-search controls."""

    normal_target: int
    abnormal_target: int
    seed: int = 42
    iterations: int = 500


@dataclass(frozen=True)
class ValidationTargets:
    """Record how the LaBraM reference ratio is applied to this train pool."""

    validation_fraction: float
    normal_count: int
    abnormal_count: int


@dataclass(frozen=True)
class LeakageReport:
    """Record group overlap separately for train/val and the preserved test set."""

    train_validation: list[str]
    train_test: list[str]
    validation_test: list[str]
    all_split_overlap: list[str]
    original_test_preserved: bool


@dataclass(frozen=True)
class SplitOutputs:
    """Name the three stable artifacts produced by the split stage."""

    csv: Path
    json: Path
    text: Path


@dataclass(frozen=True)
class SplitReport:
    """Collect the evidence shared by JSON and human-readable reports."""

    h5_path: Path
    frame: pd.DataFrame
    group_key: str
    search: ValidationSearch
    targets: ValidationTargets
    selected_counts: tuple[int, int]
    best_score: float
    split_counts: list[dict]
    leakage: LeakageReport
    fail_on_test_group_overlap: bool
    outputs: SplitOutputs


def main() -> None:
    """Read arguments, build one canonical split, verify it, and save reports."""
    args = parse_arguments()
    require_split_dependencies()
    validate_split_request(args)
    build_canonical_split(args)


def build_canonical_split(args: argparse.Namespace) -> None:
    """Execute the group-aware split policy from H5 metadata to final artifacts."""
    outputs = build_output_paths(args.report_dir)
    frame = load_h5_metadata(args.h5)
    group_key = choose_group_key(frame, args.group_key)

    frame["canonical_split"] = frame["h5_original_split"].copy()
    train_pool = frame[frame["h5_original_split"] == "train"].copy()
    test_mask = frame["h5_original_split"] == "test"
    targets = calculate_validation_targets(train_pool)

    grouped_train = add_group_columns(train_pool, group_key)
    candidate_groups = summarize_candidate_groups(grouped_train)
    search = ValidationSearch(
        normal_target=targets.normal_count,
        abnormal_target=targets.abnormal_count,
        seed=args.seed,
        iterations=args.n_iter,
    )
    selected_groups, selected_counts, best_score = select_validation_groups(
        candidate_groups,
        search,
    )

    frame = assign_validation_groups(frame, grouped_train, selected_groups)
    frame = add_group_columns(frame, group_key)
    leakage = inspect_group_leakage(frame, test_mask)
    enforce_leakage_policy(leakage, args.fail_on_test_group_overlap)
    verify_final_assignments(frame, leakage)

    report = SplitReport(
        h5_path=args.h5,
        frame=frame,
        group_key=group_key,
        search=search,
        targets=targets,
        selected_counts=selected_counts,
        best_score=best_score,
        split_counts=count_labels_by_split(frame),
        leakage=leakage,
        fail_on_test_group_overlap=args.fail_on_test_group_overlap,
        outputs=outputs,
    )
    save_split_artifacts(report)
    print_split_result(report, len(candidate_groups), len(selected_groups))


def parse_arguments() -> argparse.Namespace:
    """Expose paths and reproducible search controls without hiding defaults."""
    parser = argparse.ArgumentParser(
        description="Build the canonical group-aware train/val/test split index."
    )
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--report_dir", type=Path, required=True)
    parser.add_argument(
        "--group_key",
        default="auto",
        choices=["auto", "subject_id", "recording_id"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_iter", type=int, default=500)
    parser.add_argument(
        "--fail_on_test_group_overlap",
        action="store_true",
        help=(
            "Stop if an official test group also appears in train or validation. "
            "Without this flag, preserve the official test split and report overlap."
        ),
    )
    return parser.parse_args()


def require_split_dependencies() -> None:
    """Fail before reading data when H5 support is unavailable."""
    if h5py is None:
        raise RuntimeError("Install h5py before building the canonical split index")


def validate_split_request(args: argparse.Namespace) -> None:
    """Reject missing inputs and non-positive search effort before computation."""
    if not args.h5.is_file():
        raise FileNotFoundError(f"Canonical H5 file not found: {args.h5}")
    if args.n_iter < 1:
        raise ValueError("--n_iter must be at least 1")
    args.report_dir.mkdir(parents=True, exist_ok=True)


def build_output_paths(report_dir: Path) -> SplitOutputs:
    """Preserve the artifact filenames expected by the benchmark documentation."""
    return SplitOutputs(
        csv=report_dir / f"{OUTPUT_STEM}_index.csv",
        json=report_dir / f"{OUTPUT_STEM}_index.json",
        text=report_dir / f"{OUTPUT_STEM}_summary.txt",
    )


def load_h5_metadata(h5_path: Path) -> pd.DataFrame:
    """Read only the provenance fields required for split construction."""
    with h5py.File(h5_path, "r") as h5_file:
        if "eeg" not in h5_file or "metadata" not in h5_file:
            raise KeyError(f"{h5_path} must contain 'eeg' and 'metadata'")
        metadata = h5_file["metadata"]
        required = {
            "label", "split", "source_path", "sample_id", "recording_id",
            "subject_id", "start_time", "end_time",
        }
        missing = required - set(metadata.keys())
        if missing:
            raise KeyError(f"{h5_path} metadata is missing {sorted(missing)}")

        number_of_windows = h5_file["eeg"].shape[0]
        frame = pd.DataFrame({
            "h5_index": np.arange(number_of_windows, dtype=np.int64),
            "label": np.asarray(metadata["label"][:], dtype=np.int64),
            "h5_original_split": decode_array(metadata["split"][:]),
            "source_path": decode_array(metadata["source_path"][:]),
            "sample_id": decode_array(metadata["sample_id"][:]),
            "recording_id": decode_array(metadata["recording_id"][:]),
            "subject_id": decode_array(metadata["subject_id"][:]),
            "start_time": np.asarray(metadata["start_time"][:]),
            "end_time": np.asarray(metadata["end_time"][:]),
        })
        frame["raw_hash"] = (
            decode_array(metadata["raw_hash"][:])
            if "raw_hash" in metadata
            else ""
        )
    validate_h5_metadata(frame, h5_path)
    return frame


def decode_array(values) -> list[str]:
    """Decode H5 byte strings without changing row order."""
    decoded = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8", errors="ignore"))
        elif hasattr(value, "decode"):
            try:
                decoded.append(value.decode("utf-8", errors="ignore"))
            except Exception:
                decoded.append(str(value))
        else:
            decoded.append(str(value))
    return decoded


def validate_h5_metadata(frame: pd.DataFrame, h5_path: Path) -> None:
    """Require binary labels and the official train/test source allocation."""
    labels = set(frame["label"].unique())
    if not labels <= {0, 1}:
        raise ValueError(f"{h5_path} contains non-binary labels: {sorted(labels)}")
    splits = set(frame["h5_original_split"].astype(str))
    missing = {"train", "test"} - splits
    if missing:
        raise ValueError(f"{h5_path} is missing original splits: {sorted(missing)}")


def choose_group_key(frame: pd.DataFrame, requested: str) -> str:
    """Prefer subject groups when meaningful, otherwise use recordings."""
    if requested != "auto":
        return requested
    subjects = frame["subject_id"].astype(str).str.strip()
    valid_subjects = subjects.map(is_valid_group_value)
    return "subject_id" if subjects[valid_subjects].nunique() > 1 else "recording_id"


def is_valid_group_value(value) -> bool:
    """Reject empty or placeholder identifiers before leakage checks."""
    return str(value).strip().lower() not in INVALID_GROUP_VALUES


def add_group_columns(frame: pd.DataFrame, preferred_key: str) -> pd.DataFrame:
    """Assign one leakage group per row, falling back to recording identity."""
    frame = frame.copy()
    preferred = frame[preferred_key].astype(str)
    fallback = frame["recording_id"].astype(str)
    use_preferred = preferred.map(is_valid_group_value)
    frame["group_key"] = np.where(use_preferred, preferred_key, "recording_id")
    frame["group_id"] = np.where(
        use_preferred,
        preferred.str.strip(),
        fallback.str.strip(),
    )
    if not frame["group_id"].map(is_valid_group_value).all():
        raise ValueError("Some rows have neither a valid subject nor recording ID")
    return frame


def calculate_validation_targets(train_pool: pd.DataFrame) -> ValidationTargets:
    """Apply LaBraM's validation fraction separately to both TUAB classes."""
    if train_pool.empty:
        raise ValueError("Canonical H5 original train pool is empty")
    reference_train = LABRAM_COUNTS["train"]["n"]
    reference_validation = LABRAM_COUNTS["val"]["n"]
    validation_fraction = reference_validation / (
        reference_train + reference_validation
    )
    labels = train_pool["label"].to_numpy()
    normal = int((labels == 0).sum())
    abnormal = int((labels == 1).sum())
    return ValidationTargets(
        validation_fraction=validation_fraction,
        normal_count=int(round(normal * validation_fraction)),
        abnormal_count=int(round(abnormal * validation_fraction)),
    )


def summarize_candidate_groups(train_pool: pd.DataFrame) -> pd.DataFrame:
    """Count normal and abnormal training windows within each indivisible group."""
    return (
        train_pool.groupby("group_id", sort=False)
        .agg(
            n=("h5_index", "count"),
            normal_0=("label", lambda values: int((np.asarray(values) == 0).sum())),
            abnormal_1=("label", lambda values: int((np.asarray(values) == 1).sum())),
        )
        .reset_index()
    )


def select_validation_groups(
    group_frame: pd.DataFrame,
    search: ValidationSearch,
) -> tuple[set[str], tuple[int, int], float]:
    """Search reproducibly for whole groups that best match class targets."""
    records = group_frame.to_dict("records")
    best_selected: set[str] = set()
    best_counts = (0, 0)
    best_score = float("inf")
    master_random = random.Random(search.seed)

    for _ in range(search.iterations):
        iteration_random = random.Random(master_random.randint(0, 10**12))
        order = records[:]
        iteration_random.shuffle(order)
        selected: set[str] = set()
        normal_count = 0
        abnormal_count = 0
        current_score = score_validation_target_match(0, 0, search)

        for group in order:
            candidate_normal = normal_count + int(group["normal_0"])
            candidate_abnormal = abnormal_count + int(group["abnormal_1"])
            candidate_score = score_validation_target_match(
                candidate_normal,
                candidate_abnormal,
                search,
            )
            if candidate_score < current_score:
                selected.add(group["group_id"])
                normal_count = candidate_normal
                abnormal_count = candidate_abnormal
                current_score = candidate_score

        if current_score < best_score:
            best_selected = selected
            best_counts = (normal_count, abnormal_count)
            best_score = current_score

    return best_selected, best_counts, best_score


def score_validation_target_match(
    normal_count: int,
    abnormal_count: int,
    search: ValidationSearch,
) -> float:
    """Combine relative class-count and total-count errors into one search score."""
    total = normal_count + abnormal_count
    target_total = search.normal_target + search.abnormal_target
    normal_error = abs(normal_count - search.normal_target) / max(search.normal_target, 1)
    abnormal_error = abs(abnormal_count - search.abnormal_target) / max(search.abnormal_target, 1)
    total_error = abs(total - target_total) / max(target_total, 1)
    return normal_error + abnormal_error + 0.5 * total_error


def assign_validation_groups(
    frame: pd.DataFrame,
    grouped_train: pd.DataFrame,
    selected_groups: set[str],
) -> pd.DataFrame:
    """Move all windows from selected training groups into validation."""
    validation_indices = grouped_train.loc[
        grouped_train["group_id"].isin(selected_groups),
        "h5_index",
    ].to_numpy()
    frame = frame.copy()
    frame.loc[
        frame["h5_index"].isin(validation_indices),
        "canonical_split",
    ] = "val"
    return frame


def inspect_group_leakage(
    frame: pd.DataFrame,
    original_test_mask: pd.Series,
) -> LeakageReport:
    """Report train/val leakage separately from official-test overlap."""
    groups = {
        split: set(frame.loc[frame["canonical_split"] == split, "group_id"])
        for split in ("train", "val", "test")
    }
    group_split_counts = frame.groupby("group_id")["canonical_split"].nunique()
    all_overlap = sorted(group_split_counts[group_split_counts > 1].index.tolist())
    return LeakageReport(
        train_validation=sorted(groups["train"] & groups["val"]),
        train_test=sorted(groups["train"] & groups["test"]),
        validation_test=sorted(groups["val"] & groups["test"]),
        all_split_overlap=all_overlap,
        original_test_preserved=bool(
            (frame.loc[original_test_mask, "canonical_split"] == "test").all()
        ),
    )


def enforce_leakage_policy(
    leakage: LeakageReport,
    fail_on_test_group_overlap: bool,
) -> None:
    """Always reject train/val leakage and optionally reject official-test overlap."""
    if leakage.train_validation:
        raise RuntimeError(
            "Train/validation group leakage detected: "
            + ", ".join(leakage.train_validation[:20])
        )
    test_overlap = leakage.train_test + leakage.validation_test
    if fail_on_test_group_overlap and test_overlap:
        raise RuntimeError(
            "Official test groups overlap train or validation: "
            + ", ".join(sorted(set(test_overlap))[:20])
        )


def verify_final_assignments(frame: pd.DataFrame, leakage: LeakageReport) -> None:
    """Require one valid assignment per H5 row and preservation of official test."""
    unknown = set(frame["canonical_split"]) - {"train", "val", "test"}
    if unknown:
        raise ValueError(f"Unknown canonical split labels: {sorted(unknown)}")
    if frame["h5_index"].duplicated().any():
        raise ValueError("Duplicate h5_index values found in the final split index")
    if not leakage.original_test_preserved:
        raise RuntimeError("The official H5 test allocation was not preserved")


def count_labels_by_split(frame: pd.DataFrame) -> list[dict]:
    """Count total, normal, and abnormal windows in each final split."""
    rows = []
    for split in ("train", "val", "test"):
        labels = frame.loc[frame["canonical_split"] == split, "label"].to_numpy()
        rows.append({
            "split": split,
            "n": int(len(labels)),
            "normal_0": int((labels == 0).sum()),
            "abnormal_1": int((labels == 1).sum()),
        })
    return rows


def save_split_artifacts(report: SplitReport) -> None:
    """Write the loader index and two consistent reproducibility reports."""
    report.frame[INDEX_COLUMNS].to_csv(report.outputs.csv, index=False)
    summary = build_split_summary(report)
    report.outputs.json.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    report.outputs.text.write_text(
        format_text_report(summary, report.best_score),
        encoding="utf-8",
    )


def build_split_summary(report: SplitReport) -> dict:
    """Build one evidence record used by both JSON and text reports."""
    leakage = report.leakage
    return {
        "h5_path": str(report.h5_path),
        "total_h5_windows": int(len(report.frame)),
        "group_key": report.group_key,
        "seed": report.search.seed,
        "n_iter": report.search.iterations,
        "labram_reference_counts": LABRAM_COUNTS,
        "labram_val_fraction_within_train_val": report.targets.validation_fraction,
        "target_val_counts_from_canonical_train": {
            "normal_0": report.targets.normal_count,
            "abnormal_1": report.targets.abnormal_count,
            "n": report.targets.normal_count + report.targets.abnormal_count,
        },
        "selected_val_counts": {
            "normal_0": int(report.selected_counts[0]),
            "abnormal_1": int(report.selected_counts[1]),
            "n": int(sum(report.selected_counts)),
        },
        "final_split_counts": report.split_counts,
        "leakage_check": {
            "train_val_no_overlap": not leakage.train_validation,
            "train_val_leakage_group_count": len(leakage.train_validation),
            "train_val_leakage_groups_first_50": leakage.train_validation[:50],
            "original_test_preserved": leakage.original_test_preserved,
            "train_test_overlap_group_count": len(leakage.train_test),
            "train_test_overlap_groups_first_50": leakage.train_test[:50],
            "validation_test_overlap_group_count": len(leakage.validation_test),
            "validation_test_overlap_groups_first_50": leakage.validation_test[:50],
            "all_split_overlap_group_count_including_test_diagnostic": len(
                leakage.all_split_overlap
            ),
            "all_split_overlap_groups_first_50": leakage.all_split_overlap[:50],
            "fail_on_test_group_overlap": report.fail_on_test_group_overlap,
        },
        "output_paths": {
            "csv": str(report.outputs.csv),
            "json": str(report.outputs.json),
            "txt_summary": str(report.outputs.text),
        },
    }


def format_text_report(summary: dict, best_score: float) -> str:
    """Render the JSON evidence as a concise human-readable report."""
    target = summary["target_val_counts_from_canonical_train"]
    selected = summary["selected_val_counts"]
    leakage = summary["leakage_check"]
    lines = [
        "Canonical H5 LaBraM-Referenced Max-Coverage Split Summary",
        "=" * 80,
        f"H5 path: {summary['h5_path']}",
        f"Total H5 windows: {summary['total_h5_windows']}",
        f"Group key: {summary['group_key']}",
        f"Search seed / iterations: {summary['seed']} / {summary['n_iter']}",
        "",
        "Design policy:",
        "- Preserve the official canonical H5 test allocation.",
        "- Split the official training pool into train and validation.",
        "- Use the LaBraM train/validation ratio as the validation target.",
        "- Assign whole subject or recording groups.",
        "- Keep every canonical H5 window assigned.",
        "",
        "Target validation counts:",
        f"normal_0={target['normal_0']}, abnormal_1={target['abnormal_1']}, n={target['n']}",
        "Selected validation counts:",
        f"normal_0={selected['normal_0']}, abnormal_1={selected['abnormal_1']}, n={selected['n']}",
        f"Target-match score: {best_score:.6f}",
        "",
        "Final split counts:",
        *(str(row) for row in summary["final_split_counts"]),
        "",
        "Leakage audit:",
        f"Train/validation overlap: {leakage['train_val_leakage_group_count']}",
        f"Train/test overlap: {leakage['train_test_overlap_group_count']}",
        f"Validation/test overlap: {leakage['validation_test_overlap_group_count']}",
        f"Official test preserved: {leakage['original_test_preserved']}",
        f"Fail-on-test-overlap requested: {leakage['fail_on_test_group_overlap']}",
        "",
        "Output files:",
        *summary["output_paths"].values(),
    ]
    return "\n".join(lines) + "\n"


def print_split_result(
    report: SplitReport,
    candidate_group_count: int,
    selected_group_count: int,
) -> None:
    """Print the scientific decisions and final output locations."""
    print("Build Canonical H5 LaBraM-Referenced Max-Coverage Split Index")
    print(f"H5 path: {report.h5_path}")
    print(f"Group key: {report.group_key}")
    print(f"Candidate groups: {candidate_group_count}")
    print(f"Selected validation groups: {selected_group_count}")
    print(f"Selected validation labels: {report.selected_counts}")
    print(f"Target-match score: {report.best_score:.6f}")
    for row in report.split_counts:
        print(row)
    print(f"Train/validation overlap: {len(report.leakage.train_validation)}")
    print(f"Train/test overlap: {len(report.leakage.train_test)}")
    print(f"Validation/test overlap: {len(report.leakage.validation_test)}")
    print("Saved:")
    print(report.outputs.csv)
    print(report.outputs.json)
    print(report.outputs.text)


if __name__ == "__main__":
    main()
