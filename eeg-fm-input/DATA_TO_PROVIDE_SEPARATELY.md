# Data To Provide Separately

This package is scripts and documentation only. The following files or folders must be provided separately on the machine where the workflow is run.

## Track A: Unified Preprocessing Smoke Test

Provide the tiny 8-EDF raw subset folder from USB:

```text
Raw TUAB subset folder/
  eval/abnormal/01_tcp_ar/aaaaabdo_s003_t000.edf
  eval/abnormal/01_tcp_ar/aaaaabsk_s007_t000.edf
  eval/normal/01_tcp_ar/aaaaaayx_s002_t000.edf
  eval/normal/01_tcp_ar/aaaaacad_s003_t000.edf
  train/abnormal/01_tcp_ar/aaaaaaaq_s004_t000.edf
  train/abnormal/01_tcp_ar/aaaaaaaq_s005_t001.edf
  train/normal/01_tcp_ar/aaaaaaav_s004_t000.edf
  train/normal/01_tcp_ar/aaaaaabn_s005_t000.edf
```

The smoke-test output should be a debug H5, for example:

```text
data/canonical_tuab_debug_balanced_8files_rebuilt.h5
```

Do not name the smoke-test output `canonical_tuab_full.h5`.

## Track B: Option 1 Original-vs-Unified

Provide the reviewed Option 1 raw EDF subset if running original-vs-unified diagnostics:

```text
data/tuab_option1_raw_edf_subset/
```

This folder contains symlinks or raw EDF files arranged by canonical split and label. It is not the direct input for the 90-job all-6 unified benchmark.

## Track C: All-6 Unified H5 Benchmark

Provide the frozen full canonical H5:

```text
data/canonical_tuab_full.h5
```

Provide the fixed subset index:

```text
reports/all6_unified60_subset_epoch15_5seed_1gpu_dev_v1/all6_fixed_subset_seed42_index.npz
```

The fixed subset NPZ stores indices only. The EEG data are read from `canonical_tuab_full.h5`.

## Model Repositories And Checkpoints

The benchmark scripts expect the EEG-FM repositories and checkpoints to exist at the AI Station paths used during development, for example:

```text
/nicoletye/workspace/Noise Robustness Evaluation Protocol/EEG-FM/Labram
/nicoletye/workspace/Noise Robustness Evaluation Protocol/EEG-FM/EEGPT
/nicoletye/workspace/Noise Robustness Evaluation Protocol/EEG-FM/Biot
/nicoletye/workspace/Noise Robustness Evaluation Protocol/EEG-FM/CBraMod
/nicoletye/workspace/Noise Robustness Evaluation Protocol/EEG-FM/CSBrain
/nicoletye/workspace/Noise Robustness Evaluation Protocol/EEG-FM/Codebrain
```

Large files such as EDF, H5, NPZ, PKL, PT, PTH, and CKPT files are intentionally not included in this package.
