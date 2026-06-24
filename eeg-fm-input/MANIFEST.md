# Package Manifest

This package preserves the original repo-relative script layout so launcher-internal paths such as `scripts/eegfm_adapters/...` work as written.

This package contains source files only.

## Top-Level Documentation

```text
README.md
DATA_TO_PROVIDE_SEPARATELY.md
ENVIRONMENT.md
MANIFEST.md
VERIFY_PACKAGE.md
requirements.txt
```

## Unified Preprocessing Scripts

```text
scripts/build_canonical_tuab.py
scripts/build_canonical_h5_max_coverage_split.py
```

## EEG-FM Adapter, Option 1, Ablation, And Benchmark Scripts

All adapter-related scripts are under the original path:

```text
scripts/eegfm_adapters/
```

Included files:

```text
scripts/eegfm_adapters/audit_labram_original_vs_unified_full_for_meriem.py
scripts/eegfm_adapters/biot_option1_make_tuab_wrapper.py
scripts/eegfm_adapters/build_all6_unified60_subset_epoch15_fixed_index.py
scripts/eegfm_adapters/build_labram_exact_subset_manifest.py
scripts/eegfm_adapters/build_labram_unified_50hz_subset_from_raw_exact_manifest.py
scripts/eegfm_adapters/build_option1_preprocessing_matrix.py
scripts/eegfm_adapters/canonical_h5_subset.py
scripts/eegfm_adapters/cbramod_option1_make_tuab_wrapper.py
scripts/eegfm_adapters/codebrain_option1_make_tuab_wrapper.py
scripts/eegfm_adapters/create_tuab_option1_raw_edf_symlink_subset.py
scripts/eegfm_adapters/csbrain_option1_make_tuab_wrapper.py
scripts/eegfm_adapters/eegfm_small_subset_train_worker.py
scripts/eegfm_adapters/eegfm_small_subset_train_worker_remaining_models.py
scripts/eegfm_adapters/eegfm_tiny_train_worker.py
scripts/eegfm_adapters/eegpt_option1_make_tuab_wrapper.py
scripts/eegfm_adapters/finetune_strategy_utils.py
scripts/eegfm_adapters/finetune_strategy_utils_remaining_models.py
scripts/eegfm_adapters/labram_option1_make_tuab_wrapper.py
scripts/eegfm_adapters/monitor_all6_unified60_subset_epoch15_5seed_3gpu_dev.sh
scripts/eegfm_adapters/monitor_labram_notch_ablation_subset_strict_epoch20_2gpu.sh
scripts/eegfm_adapters/option1_eegfm_registry.py
scripts/eegfm_adapters/run_all6_unified60_subset_epoch15_5seed_3gpu_dev_parallel.sh
scripts/eegfm_adapters/run_biot_unified60_meriem_strict_job.py
scripts/eegfm_adapters/run_cbramod_unified60_meriem_strict_job.py
scripts/eegfm_adapters/run_codebrain_unified60_meriem_strict_job.py
scripts/eegfm_adapters/run_csbrain_unified60_meriem_strict_job.py
scripts/eegfm_adapters/run_eegpt_unified60_meriem_strict_job.py
scripts/eegfm_adapters/run_labram_notch_ablation_subset_strict_epoch20_2gpu_task.sh
scripts/eegfm_adapters/run_labram_notch_ablation_subset_strict_epoch20_branch.py
scripts/eegfm_adapters/run_labram_unified_h5_strict_original_engine.py
scripts/eegfm_adapters/summarize_all6_unified60_subset_epoch15_5seed_1gpu_dev.py
scripts/eegfm_adapters/summarize_labram_notch_ablation_subset_strict_epoch20_2gpu.py
scripts/eegfm_adapters/validate_tuab_option1_subset_artifacts.py
```

## Local Helper Dependency Check

The package includes the small local helpers required by launcher and worker imports:

```text
eegfm_small_subset_train_worker.py -> eegfm_tiny_train_worker.py, finetune_strategy_utils.py
eegfm_small_subset_train_worker_remaining_models.py -> canonical_h5_subset.py, finetune_strategy_utils_remaining_models.py
eegfm_tiny_train_worker.py -> canonical_h5_subset.py
run_all6_unified60_subset_epoch15_5seed_3gpu_dev_parallel.sh -> scripts/eegfm_adapters/build_all6_unified60_subset_epoch15_fixed_index.py and model job wrappers
run_labram_notch_ablation_subset_strict_epoch20_2gpu_task.sh -> scripts/eegfm_adapters/run_labram_unified_h5_strict_original_engine.py and ablation helpers
```

## Excluded Files

The package intentionally excludes:

```text
*.edf
*.h5
*.npz
*.pkl
*.pt
*.pth
*.ckpt
```
