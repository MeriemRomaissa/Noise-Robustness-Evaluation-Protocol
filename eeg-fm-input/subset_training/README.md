# All-6 EEG-FM Unified60 Subset Benchmark

## Purpose

This development-platform benchmark runs all 6 EEG foundation models under one sequential 1-GPU launcher. It verifies that every model follows the same unified60 subset benchmark protocol before or alongside full task-platform runs.

## Models

- LaBraM
- BIOT
- EEGPT
- CBraMod
- CSBrain
- CodeBrain

## Dataset

- Source H5: `/nicoletye/workspace/unified_tuab/data/canonical_tuab_full.h5`
- Unified preprocessing: 60 Hz unified TUAB H5
- Split source: `/nicoletye/workspace/unified_tuab/reports/labram_exact_original_processed_split/canonical_h5_labram_exact_original_processed_split_index.csv`
- Fixed subset index: `reports/all6_unified60_subset_epoch15_5seed_1gpu_dev_v1/all6_fixed_subset_seed42_index.npz`

Subset:

- train: 8192 windows
- validation: 2048 windows
- test: 2048 windows
- subset seed: 42

All jobs consume the fixed index NPZ and pass `train_n=0`, `val_n=0`, and `test_n=0`, so training seeds affect model randomness only, not subset membership.

## Training Setup

- epochs: 15
- batch size: 64
- GPU: 1 GPU only
- execution: sequential
- total jobs: 90
- models: 6
- strategies: 3
- seeds: 0, 42, 123, 256, 512

## 3-GPU Development Platform Parallel Run

This package now also includes a 3-GPU parallel launcher for the AI Station development platform:

- intended platform: 3 GPU / 63 CPU / 4-hour limit
- launcher: `scripts/eegfm_adapters/run_all6_unified60_subset_epoch15_5seed_3gpu_dev_parallel.sh`
- monitor: `scripts/eegfm_adapters/monitor_all6_unified60_subset_epoch15_5seed_3gpu_dev.sh`
- default `GPU_LIST=0,1,2`
- default `MAX_PARALLEL=3`
- same 90-job subset benchmark as the 1-GPU sequential launcher
- same fixed subset index and Meriem-strict/Meriem-mostly-strict mappings
- same output root for resume: `outputs/all6_unified60_subset_epoch15_5seed_1gpu_dev_v1/`

The 4-hour development window may not finish all 90 jobs. Resume with `FRESH=0` or by omitting `FRESH`; completed jobs with `metrics.json` status `PASS` are skipped. Use `FRESH=1` only for an intentional clean restart of the dev subset output root. Full benchmark outputs are untouched.

## Fine-Tuning Strategies

All 6 EEG-FMs are configured to match Meriem's strategy design as closely as possible.

### Full Fine-Tuning

- pretrained checkpoint is loaded
- backbone and classifier/head are trainable
- trainable/frozen audit is saved

### Linear Probe

- pretrained checkpoint is loaded
- backbone is frozen
- classifier/head is trainable
- trainable/frozen audit is saved

### LoRA

- pretrained checkpoint is loaded
- base backbone is frozen
- LoRA rank=2 is inserted
- LoRA parameters and classifier/head are trainable
- trainable/frozen audit and LoRA target list are saved

## Fairness Alignment With Meriem/LaBraM

- LaBraM is the reference implementation for Meriem-exact LoRA placement.
- LaBraM dev jobs explicitly pass `/nicoletye/workspace/Noise Robustness Evaluation Protocol/EEG-FM/Labram/checkpoints/labram-base.pth`.
- BIOT, EEGPT, CBraMod, and CSBrain use architecture-native equivalent mappings to preserve the same PEFT semantics.
- CodeBrain uses architecture-native partial LoRA mapping because its SSSM-style architecture does not expose the same LaBraM-style q/k/v + MLP structure.
- CodeBrain LoRA must be reported as `MERIEM_STRICT_ARCHITECTURE_NATIVE_EQUIVALENT_PARTIAL`, not exact LaBraM placement.

## Metrics

- validation metrics are logged every epoch
- best checkpoint is selected by best validation balanced accuracy
- test metrics are evaluated once at the end
- final test metrics are stored in `metrics.json`

Required final metrics:

- `test_accuracy`
- `test_balanced_accuracy`
- `test_auroc`
- `test_auprc`
- `best_val_balanced_accuracy`
- `best_epoch`

## Outputs

Output root:

`outputs/all6_unified60_subset_epoch15_5seed_1gpu_dev_v1/`

Report root:

`reports/all6_unified60_subset_epoch15_5seed_1gpu_dev_v1/`

Expected job files:

- `metrics.json`
- `history.csv`
- `history.json`
- `strategy_report.json` or strategy audit JSON
- per-job `terminal.log`

Aggregate files:

- `summary.md`
- `summary.csv`
- `summary.json`
- `summary_by_model_strategy.csv`
- `script_reference_audit.md`
- `RUNBOOK.md`
- `PATCH_NOTES.md`
- `SCRIPT_CHANGELOG.md`
- `all6_dev_package_manifest.json`
- `script_manifest.sha256`
- `script_snapshot/`

## Important Caveat

This is a subset development benchmark, not the final full-dataset result. Full benchmark outputs remain in separate model-specific epoch-50 roots.

Codex patched and dry-tested this package on 2026-06-22. No real training, H5 rebuild, raw EDF modification, checkpoint modification, or full benchmark output modification was performed during the patch task.
