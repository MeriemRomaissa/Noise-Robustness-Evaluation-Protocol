#!/usr/bin/env python3
"""Registry for the Option 1 EEG-FM original-vs-unified comparison.

This module is deliberately declarative: it records paths, known blockers, and
manual command templates. It does not run training, preprocessing, or eval.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any


ROOT = Path("/nicoletye/workspace/unified_tuab")
SCRIPT_DIR = ROOT / "scripts" / "eegfm_adapters"
REPORT_DIR = ROOT / "reports"
OUTPUT_ROOT = ROOT / "outputs" / "eegfm_option1_comparison"
LOG_ROOT = ROOT / "logs" / "eegfm_option1_comparison"
REPO_ROOT = Path("/nicoletye/workspace/Noise Robustness Evaluation Protocol/EEG-FM")

OPTION1_SUBSET_NPZ = REPORT_DIR / "tuab_option1_unified_h5_subset.npz"
RECORDING_MANIFEST = REPORT_DIR / "tuab_original_vs_unified_matched_recording_manifest.csv"
RAW_EDF_SUBSET = ROOT / "data" / "tuab_option1_raw_edf_subset"
DEFAULT_EPOCHS = 2
DEFAULT_BATCH_SIZE = 8
DEFAULT_SEED = 42

MODELS = ["LaBraM", "EEGPT", "BIOT", "CBraMod", "EEGMamba", "CSBrain", "CodeBrain"]

REPO_PATHS = {
    "LaBraM": REPO_ROOT / "Labram",
    "EEGPT": REPO_ROOT / "EEGPT",
    "BIOT": REPO_ROOT / "Biot",
    "CBraMod": REPO_ROOT / "CBraMod",
    "EEGMamba": REPO_ROOT / "EEGMamba",
    "CSBrain": REPO_ROOT / "CSBrain",
    "CodeBrain": REPO_ROOT / "Codebrain",
}

VENV_PATHS = {
    "LaBraM": Path("/nicoletye/venvs/eegpt_labram"),
    "EEGPT": Path("/nicoletye/venvs/eegpt_labram"),
    "BIOT": Path("/nicoletye/venvs/biot_env"),
    "CBraMod": Path("/nicoletye/venvs/cbramod_csbrain"),
    "EEGMamba": Path("/nicoletye/venvs/eegmamba_env"),
    "CSBrain": Path("/nicoletye/venvs/cbramod_csbrain"),
    "CodeBrain": Path("/nicoletye/venvs/codebrain_env"),
}


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def python_path(model: str) -> Path:
    return VENV_PATHS[model] / "bin" / "python"


def option1_output_dir(model: str, branch: str) -> Path:
    return OUTPUT_ROOT / model / branch


def option1_log_dir(model: str, branch: str) -> Path:
    return LOG_ROOT / model / branch


def option1_artifacts(model: str, branch: str) -> dict[str, str]:
    out_dir = option1_output_dir(model, branch)
    log_dir = option1_log_dir(model, branch)
    return {
        "output_dir": str(out_dir),
        "log_dir": str(log_dir),
        "metrics_path": str(out_dir / "metrics.json"),
        "checkpoint_path": str(out_dir / "checkpoint.pt"),
        "log_path": str(log_dir / "train.log"),
    }


def labram_original_artifacts() -> dict[str, str]:
    return {
        "output_dir": str(ROOT / "outputs" / "labram_option1_original_subset"),
        "log_dir": str(ROOT / "logs" / "labram_option1_original_subset"),
        "metrics_path": str(REPORT_DIR / "labram_option1_original_train_smoke_metrics.json"),
        "checkpoint_path": str(ROOT / "outputs" / "labram_option1_original_subset" / "checkpoint_smoke.pt"),
        "log_path": str(ROOT / "logs" / "labram_option1_original_subset" / "train.log"),
    }


def labram_unified_artifacts() -> dict[str, str]:
    return {
        "output_dir": str(ROOT / "outputs" / "labram_option1_unified_matched_subset" / "LaBraM"),
        "log_dir": str(ROOT / "outputs" / "labram_option1_unified_matched_subset" / "LaBraM"),
        "metrics_path": str(ROOT / "outputs" / "labram_option1_unified_matched_subset" / "LaBraM" / "metrics.json"),
        "checkpoint_path": str(ROOT / "outputs" / "labram_option1_unified_matched_subset" / "LaBraM" / "checkpoint.pt"),
        "log_path": str(ROOT / "outputs" / "labram_option1_unified_matched_subset" / "LaBraM" / "train.log"),
        "test_eval_metrics_path": str(REPORT_DIR / "labram_option1_unified_test_eval_metrics.json"),
        "test_eval_report_path": str(REPORT_DIR / "labram_option1_unified_test_eval_report.md"),
    }


def eegpt_original_processed_root() -> Path:
    return ROOT / "data" / "eegpt_option1_original_processed"


def biot_original_processed_root() -> Path:
    return ROOT / "data" / "biot_option1_original_processed"


def cbramod_original_processed_root() -> Path:
    return ROOT / "data" / "cbramod_option1_original_processed"


def csbrain_original_processed_root() -> Path:
    return ROOT / "data" / "csbrain_option1_original_processed"


def codebrain_original_processed_root() -> Path:
    return ROOT / "data" / "codebrain_option1_original_processed"


def unified_training_command(model: str) -> str:
    if model == "LaBraM":
        return "# LaBraM unified branch is already DONE; rerun intentionally suppressed."
    if model == "EEGMamba":
        return "# EEGMamba is BLOCKED_WITH_REASON; no command generated."
    env = "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2"
    cmd = [
        q(python_path(model)),
        q(SCRIPT_DIR / "eegfm_small_subset_train_worker.py"),
        "--model",
        q(model),
        "--repo_path",
        q(REPO_PATHS[model]),
        "--venv_path",
        q(VENV_PATHS[model]),
        "--subset_npz",
        q(OPTION1_SUBSET_NPZ),
        "--output_dir",
        q(option1_output_dir(model, "unified")),
        "--epochs",
        str(DEFAULT_EPOCHS),
        "--batch_size",
        str(DEFAULT_BATCH_SIZE),
        "--lr",
        "0.0001",
        "--weight_decay",
        "0.0005",
        "--seed",
        str(DEFAULT_SEED),
        "--device",
        "auto",
    ]
    log_path = option1_log_dir(model, "unified") / "terminal.log"
    mkdirs = f"mkdir -p {q(option1_output_dir(model, 'unified'))} {q(option1_log_dir(model, 'unified'))}"
    return f"cd {q(ROOT)} && {mkdirs} && {env} {' '.join(cmd)} 2>&1 | tee {q(log_path)}"


def eegpt_original_preprocessing_command(limit_recordings_per_split_label: int | None, dry_run: bool) -> str:
    output_root = (
        ROOT / "data" / "eegpt_option1_original_processed_smoke"
        if limit_recordings_per_split_label
        else eegpt_original_processed_root()
    )
    log_dir = option1_log_dir("EEGPT", "original")
    if dry_run:
        log_name = "preprocess_dry_run_terminal.log"
        mode_note = "# EEGPT original preprocessing dry-run: validates only, no data processing."
    elif limit_recordings_per_split_label:
        log_name = "preprocess_smoke_terminal.log"
        mode_note = "# EEGPT original preprocessing smoke: processes only a tiny Option 1 subset."
    else:
        log_name = "preprocess_full_option1_subset_terminal.log"
        mode_note = "# EEGPT full Option 1 subset original preprocessing: subset-only, not full TUAB."
    cmd = [
        q(python_path("EEGPT")),
        q(SCRIPT_DIR / "eegpt_option1_make_tuab_wrapper.py"),
        "--raw_subset_root",
        q(RAW_EDF_SUBSET),
        "--recording_manifest",
        q(RECORDING_MANIFEST),
        "--output_root",
        q(output_root),
    ]
    if dry_run:
        cmd.append("--dry_run")
    else:
        cmd.append("--force")
    if limit_recordings_per_split_label:
        cmd.extend(["--limit_recordings_per_split_label", str(limit_recordings_per_split_label)])
    mkdirs = f"mkdir -p {q(output_root)} {q(log_dir)}"
    return f"{mode_note}\ncd {q(ROOT)} && {mkdirs} && {' '.join(cmd)} 2>&1 | tee {q(log_dir / log_name)}"


def eegpt_original_training_command() -> str:
    output_dir = option1_output_dir("EEGPT", "original")
    log_dir = option1_log_dir("EEGPT", "original")
    cmd = [
        q(python_path("EEGPT")),
        q(SCRIPT_DIR / "eegpt_option1_train_wrapper.py"),
        "--eegpt_repo",
        q(REPO_PATHS["EEGPT"]),
        "--processed_root",
        q(eegpt_original_processed_root()),
        "--output_dir",
        q(output_dir),
        "--log_dir",
        q(log_dir),
        "--seed",
        str(DEFAULT_SEED),
        "--epochs",
        str(DEFAULT_EPOCHS),
        "--batch_size",
        str(DEFAULT_BATCH_SIZE),
        "--device",
        "auto",
    ]
    mkdirs = f"mkdir -p {q(output_dir)} {q(log_dir)}"
    return f"cd {q(ROOT)} && {mkdirs} && {' '.join(cmd)} 2>&1 | tee {q(log_dir / 'terminal.log')}"


def biot_original_preprocessing_command(limit_recordings_per_split_label: int | None, dry_run: bool) -> str:
    output_root = (
        ROOT / "data" / "biot_option1_original_processed_smoke"
        if limit_recordings_per_split_label
        else biot_original_processed_root()
    )
    log_dir = option1_log_dir("BIOT", "original")
    if dry_run:
        log_name = "preprocess_dry_run_terminal.log"
        mode_note = "# BIOT original preprocessing dry-run: validates only, no data processing."
    elif limit_recordings_per_split_label:
        log_name = "preprocess_smoke_terminal.log"
        mode_note = "# BIOT original preprocessing smoke: processes only a tiny Option 1 subset."
    else:
        log_name = "preprocess_full_option1_subset_terminal.log"
        mode_note = "# BIOT full Option 1 subset original preprocessing: subset-only, not full TUAB."
    cmd = [
        q(python_path("BIOT")),
        q(SCRIPT_DIR / "biot_option1_make_tuab_wrapper.py"),
        "--raw_subset_root",
        q(RAW_EDF_SUBSET),
        "--recording_manifest",
        q(RECORDING_MANIFEST),
        "--output_root",
        q(output_root),
    ]
    if dry_run:
        cmd.append("--dry_run")
    else:
        cmd.append("--force")
    if limit_recordings_per_split_label:
        cmd.extend(["--limit_recordings_per_split_label", str(limit_recordings_per_split_label)])
    mkdirs = f"mkdir -p {q(output_root)} {q(log_dir)}"
    return f"{mode_note}\ncd {q(ROOT)} && {mkdirs} && {' '.join(cmd)} 2>&1 | tee {q(log_dir / log_name)}"


def biot_original_training_command() -> str:
    output_dir = option1_output_dir("BIOT", "original")
    log_dir = option1_log_dir("BIOT", "original")
    cmd = [
        q(python_path("BIOT")),
        q(SCRIPT_DIR / "biot_option1_train_wrapper.py"),
        "--biot_repo",
        q(REPO_PATHS["BIOT"]),
        "--processed_root",
        q(biot_original_processed_root()),
        "--output_dir",
        q(output_dir),
        "--log_dir",
        q(log_dir),
        "--seed",
        str(DEFAULT_SEED),
        "--epochs",
        str(DEFAULT_EPOCHS),
        "--batch_size",
        str(DEFAULT_BATCH_SIZE),
        "--device",
        "auto",
    ]
    mkdirs = f"mkdir -p {q(output_dir)} {q(log_dir)}"
    env = "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2"
    return f"cd {q(ROOT)} && {mkdirs} && {env} {' '.join(cmd)} 2>&1 | tee {q(log_dir / 'terminal.log')}"


def cbramod_original_preprocessing_command(limit_recordings_per_split_label: int | None, dry_run: bool) -> str:
    output_root = (
        ROOT / "data" / "cbramod_option1_original_processed_smoke"
        if limit_recordings_per_split_label
        else cbramod_original_processed_root()
    )
    log_dir = option1_log_dir("CBraMod", "original")
    if dry_run:
        log_name = "preprocess_dry_run_terminal.log"
        mode_note = "# CBraMod original preprocessing dry-run: validates only, no data processing."
    elif limit_recordings_per_split_label:
        log_name = "preprocess_smoke_terminal.log"
        mode_note = "# CBraMod original preprocessing smoke: processes only a tiny Option 1 subset."
    else:
        log_name = "preprocess_full_option1_subset_terminal.log"
        mode_note = "# CBraMod full Option 1 subset original preprocessing: subset-only, not full TUAB."
    cmd = [
        q(python_path("CBraMod")),
        q(SCRIPT_DIR / "cbramod_option1_make_tuab_wrapper.py"),
        "--raw_subset_root",
        q(RAW_EDF_SUBSET),
        "--recording_manifest",
        q(RECORDING_MANIFEST),
        "--output_root",
        q(output_root),
    ]
    if dry_run:
        cmd.append("--dry_run")
    else:
        cmd.append("--force")
    if limit_recordings_per_split_label:
        cmd.extend(["--limit_recordings_per_split_label", str(limit_recordings_per_split_label)])
    mkdirs = f"mkdir -p {q(output_root)} {q(log_dir)}"
    return f"{mode_note}\ncd {q(ROOT)} && {mkdirs} && {' '.join(cmd)} 2>&1 | tee {q(log_dir / log_name)}"


def cbramod_original_training_command() -> str:
    output_dir = option1_output_dir("CBraMod", "original")
    log_dir = option1_log_dir("CBraMod", "original")
    cmd = [
        q(python_path("CBraMod")),
        q(SCRIPT_DIR / "cbramod_option1_train_wrapper.py"),
        "--cbramod_repo",
        q(REPO_PATHS["CBraMod"]),
        "--processed_root",
        q(cbramod_original_processed_root()),
        "--output_dir",
        q(output_dir),
        "--log_dir",
        q(log_dir),
        "--seed",
        str(DEFAULT_SEED),
        "--epochs",
        str(DEFAULT_EPOCHS),
        "--batch_size",
        str(DEFAULT_BATCH_SIZE),
        "--device",
        "auto",
    ]
    mkdirs = f"mkdir -p {q(output_dir)} {q(log_dir)}"
    env = "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2"
    return f"cd {q(ROOT)} && {mkdirs} && {env} {' '.join(cmd)} 2>&1 | tee {q(log_dir / 'terminal.log')}"


def csbrain_original_preprocessing_command(limit_recordings_per_split_label: int | None, dry_run: bool) -> str:
    output_root = (
        ROOT / "data" / "csbrain_option1_original_processed_smoke"
        if limit_recordings_per_split_label
        else csbrain_original_processed_root()
    )
    log_dir = option1_log_dir("CSBrain", "original")
    if dry_run:
        log_name = "preprocess_dry_run_terminal.log"
        mode_note = "# CSBrain original preprocessing dry-run: validates only, no data processing."
    elif limit_recordings_per_split_label:
        log_name = "preprocess_smoke_terminal.log"
        mode_note = "# CSBrain original preprocessing smoke: processes only a tiny Option 1 subset."
    else:
        log_name = "preprocess_full_option1_subset_terminal.log"
        mode_note = "# CSBrain full Option 1 subset original preprocessing: subset-only, not full TUAB."
    cmd = [
        q(python_path("CSBrain")),
        q(SCRIPT_DIR / "csbrain_option1_make_tuab_wrapper.py"),
        "--raw_subset_root",
        q(RAW_EDF_SUBSET),
        "--recording_manifest",
        q(RECORDING_MANIFEST),
        "--output_root",
        q(output_root),
    ]
    if dry_run:
        cmd.append("--dry_run")
    else:
        cmd.append("--force")
    if limit_recordings_per_split_label:
        cmd.extend(["--limit_recordings_per_split_label", str(limit_recordings_per_split_label)])
    mkdirs = f"mkdir -p {q(output_root)} {q(log_dir)}"
    return f"{mode_note}\ncd {q(ROOT)} && {mkdirs} && {' '.join(cmd)} 2>&1 | tee {q(log_dir / log_name)}"


def csbrain_original_training_command() -> str:
    output_dir = option1_output_dir("CSBrain", "original")
    log_dir = option1_log_dir("CSBrain", "original")
    cmd = [
        q(python_path("CSBrain")),
        q(SCRIPT_DIR / "csbrain_option1_train_wrapper.py"),
        "--csbrain_repo",
        q(REPO_PATHS["CSBrain"]),
        "--processed_root",
        q(csbrain_original_processed_root()),
        "--output_dir",
        q(output_dir),
        "--log_dir",
        q(log_dir),
        "--seed",
        str(DEFAULT_SEED),
        "--epochs",
        str(DEFAULT_EPOCHS),
        "--batch_size",
        str(DEFAULT_BATCH_SIZE),
        "--device",
        "auto",
    ]
    mkdirs = f"mkdir -p {q(output_dir)} {q(log_dir)}"
    env = "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2"
    return f"cd {q(ROOT)} && {mkdirs} && {env} {' '.join(cmd)} 2>&1 | tee {q(log_dir / 'terminal.log')}"


def codebrain_original_preprocessing_command(limit_recordings_per_split_label: int | None, dry_run: bool) -> str:
    output_root = (
        ROOT / "data" / "codebrain_option1_original_processed_smoke"
        if limit_recordings_per_split_label
        else codebrain_original_processed_root()
    )
    log_dir = option1_log_dir("CodeBrain", "original")
    if dry_run:
        log_name = "preprocess_dry_run_terminal.log"
        mode_note = "# CodeBrain original preprocessing dry-run: validates only, no data processing."
    elif limit_recordings_per_split_label:
        log_name = "preprocess_smoke_terminal.log"
        mode_note = "# CodeBrain original preprocessing smoke: processes only a tiny Option 1 subset."
    else:
        log_name = "preprocess_full_option1_subset_terminal.log"
        mode_note = "# CodeBrain full Option 1 subset original preprocessing: subset-only, not full TUAB."
    cmd = [
        q(python_path("CodeBrain")),
        q(SCRIPT_DIR / "codebrain_option1_make_tuab_wrapper.py"),
        "--raw_subset_root",
        q(RAW_EDF_SUBSET),
        "--recording_manifest",
        q(RECORDING_MANIFEST),
        "--output_root",
        q(output_root),
    ]
    if dry_run:
        cmd.append("--dry_run")
    else:
        cmd.append("--force")
    if limit_recordings_per_split_label:
        cmd.extend(["--limit_recordings_per_split_label", str(limit_recordings_per_split_label)])
    mkdirs = f"mkdir -p {q(output_root)} {q(log_dir)}"
    return f"{mode_note}\ncd {q(ROOT)} && {mkdirs} && {' '.join(cmd)} 2>&1 | tee {q(log_dir / log_name)}"


def codebrain_original_training_command() -> str:
    output_dir = option1_output_dir("CodeBrain", "original")
    log_dir = option1_log_dir("CodeBrain", "original")
    cmd = [
        q(python_path("CodeBrain")),
        q(SCRIPT_DIR / "codebrain_option1_train_wrapper.py"),
        "--codebrain_repo",
        q(REPO_PATHS["CodeBrain"]),
        "--processed_root",
        q(codebrain_original_processed_root()),
        "--output_dir",
        q(output_dir),
        "--log_dir",
        q(log_dir),
        "--seed",
        str(DEFAULT_SEED),
        "--epochs",
        str(DEFAULT_EPOCHS),
        "--batch_size",
        str(DEFAULT_BATCH_SIZE),
        "--device",
        "auto",
    ]
    mkdirs = f"mkdir -p {q(output_dir)} {q(log_dir)}"
    env = "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2"
    return f"cd {q(ROOT)} && {mkdirs} && {env} {' '.join(cmd)} 2>&1 | tee {q(log_dir / 'terminal.log')}"


def labram_test_eval_command() -> str:
    return (
        f"cd {q(ROOT)} && {q(python_path('LaBraM'))} "
        f"{q(SCRIPT_DIR / 'evaluate_labram_unified_option1_test.py')}"
    )


def comparison_command(model: str) -> str:
    if model == "LaBraM":
        return (
            f"cd {q(ROOT)} && {q(python_path('LaBraM'))} "
            f"{q(SCRIPT_DIR / 'generate_labram_option1_comparison.py')}"
        )
    if model == "EEGPT":
        return (
            f"cd {q(ROOT)} && {q(python_path('EEGPT'))} "
            f"{q(SCRIPT_DIR / 'generate_eegpt_option1_comparison.py')}"
        )
    if model == "BIOT":
        return (
            f"cd {q(ROOT)} && {q(python_path('BIOT'))} "
            f"{q(SCRIPT_DIR / 'generate_biot_option1_comparison.py')}"
        )
    if model == "CBraMod":
        return (
            f"cd {q(ROOT)} && {q(python_path('CBraMod'))} "
            f"{q(SCRIPT_DIR / 'generate_cbramod_option1_comparison.py')}"
        )
    if model == "CSBrain":
        return (
            f"cd {q(ROOT)} && {q(python_path('CSBrain'))} "
            f"{q(SCRIPT_DIR / 'generate_csbrain_option1_comparison.py')}"
        )
    if model == "CodeBrain":
        return (
            f"cd {q(ROOT)} && {q(python_path('CodeBrain'))} "
            f"{q(SCRIPT_DIR / 'generate_codebrain_option1_comparison.py')}"
        )
    return "# Comparison command is pending until both original and unified branch metrics exist."


def branch_template(model: str, branch: str) -> str:
    if branch == "unified":
        return unified_training_command(model)
    if branch == "test_eval":
        if model == "LaBraM":
            return labram_test_eval_command()
        return "# Test-eval command pending; add after the model unified branch is trained."
    if branch == "compare":
        return comparison_command(model)
    if branch == "original":
        if model == "LaBraM":
            return "# LaBraM original branch is already DONE; rerun intentionally suppressed."
        if model == "EEGPT":
            return eegpt_original_training_command()
        if model == "BIOT":
            return biot_original_training_command()
        if model == "CBraMod":
            return cbramod_original_training_command()
        if model == "CSBrain":
            return csbrain_original_training_command()
        if model == "CodeBrain":
            return codebrain_original_training_command()
        return "# Original raw-EDF branch wrapper is not available yet for this model."
    raise ValueError(f"unsupported branch: {branch}")


def known_blocker(model: str) -> str:
    if model == "EEGMamba":
        return "Known unresolved dependency conflicts; skip detailed debugging for now."
    return ""


def base_registry() -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        original_artifacts = labram_original_artifacts() if model == "LaBraM" else option1_artifacts(model, "original")
        unified_artifacts = labram_unified_artifacts() if model == "LaBraM" else option1_artifacts(model, "unified")
        registry[model] = {
            "model": model,
            "repo_path": str(REPO_PATHS[model]),
            "venv_path": str(VENV_PATHS[model]),
            "python_path": str(python_path(model)),
            "known_blocker": known_blocker(model),
            "branches": {
                "original": {
                    "artifacts": original_artifacts,
                    "command": branch_template(model, "original"),
                    "commands": {
                        "preprocessing_dry_run": (
                            eegpt_original_preprocessing_command(None, True)
                            if model == "EEGPT"
                            else biot_original_preprocessing_command(None, True)
                            if model == "BIOT"
                            else cbramod_original_preprocessing_command(None, True)
                            if model == "CBraMod"
                            else csbrain_original_preprocessing_command(None, True)
                            if model == "CSBrain"
                            else codebrain_original_preprocessing_command(None, True)
                            if model == "CodeBrain"
                            else ""
                        ),
                        "preprocessing_smoke": (
                            eegpt_original_preprocessing_command(1, False)
                            if model == "EEGPT"
                            else biot_original_preprocessing_command(1, False)
                            if model == "BIOT"
                            else cbramod_original_preprocessing_command(1, False)
                            if model == "CBraMod"
                            else csbrain_original_preprocessing_command(1, False)
                            if model == "CSBrain"
                            else codebrain_original_preprocessing_command(1, False)
                            if model == "CodeBrain"
                            else ""
                        ),
                        "preprocessing_full_option1_subset": (
                            eegpt_original_preprocessing_command(None, False)
                            if model == "EEGPT"
                            else biot_original_preprocessing_command(None, False)
                            if model == "BIOT"
                            else cbramod_original_preprocessing_command(None, False)
                            if model == "CBraMod"
                            else csbrain_original_preprocessing_command(None, False)
                            if model == "CSBrain"
                            else codebrain_original_preprocessing_command(None, False)
                            if model == "CodeBrain"
                            else ""
                        ),
                        "train": (
                            eegpt_original_training_command()
                            if model == "EEGPT"
                            else biot_original_training_command()
                            if model == "BIOT"
                            else cbramod_original_training_command()
                            if model == "CBraMod"
                            else csbrain_original_training_command()
                            if model == "CSBrain"
                            else codebrain_original_training_command()
                            if model == "CodeBrain"
                            else branch_template(model, "original")
                        ),
                    },
                    "notes": (
                        "LaBraM is complete. EEGPT, BIOT, CBraMod, CSBrain, and CodeBrain have non-invasive "
                        "raw-EDF wrappers for full Option 1 subset original preprocessing. CSBrain/CodeBrain "
                        "wrappers are format-compatible because no TUAB raw EDF maker was found in those repos."
                    ),
                },
                "unified": {
                    "artifacts": unified_artifacts,
                    "command": branch_template(model, "unified"),
                    "notes": "Uses frozen matched H5 NPZ and adapter worker.",
                },
                "test_eval": {
                    "artifacts": {
                        "metrics_path": str(REPORT_DIR / f"{model.lower()}_option1_unified_test_eval_metrics.json"),
                        "report_path": str(REPORT_DIR / f"{model.lower()}_option1_unified_test_eval_report.md"),
                    },
                    "command": branch_template(model, "test_eval"),
                    "notes": "LaBraM evaluator exists; others pending until unified branches are trained.",
                },
                "compare": {
                    "artifacts": {
                        "json_path": str(REPORT_DIR / f"{model.lower()}_option1_original_vs_unified_comparison.json"),
                        "csv_path": str(REPORT_DIR / f"{model.lower()}_option1_original_vs_unified_comparison.csv"),
                        "md_path": str(REPORT_DIR / f"{model.lower()}_option1_original_vs_unified_comparison.md"),
                    },
                    "command": branch_template(model, "compare"),
                    "commands": {
                        "compare": branch_template(model, "compare"),
                    },
                    "notes": "LaBraM comparison generator exists; generic collector handles cross-model table.",
                },
            },
            "notes": "",
        }
    return registry


def load_inventory() -> dict[str, dict[str, Any]]:
    path = REPORT_DIR / "eegfm_venv_repo_inventory.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return {}
    return {str(row.get("model")): row for row in data if isinstance(row, dict)}


def get_registry() -> dict[str, dict[str, Any]]:
    registry = base_registry()
    inventory = load_inventory()
    for model, inv in inventory.items():
        if model not in registry:
            continue
        registry[model]["inventory_status"] = inv.get("status", "")
        registry[model]["inventory_import_test"] = inv.get("import_test", "")
        registry[model]["inventory_notes"] = inv.get("notes", "")
    return registry


def ensure_output_dirs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print(json.dumps(get_registry(), indent=2))
