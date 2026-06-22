# EEG-FM Unified TUAB Benchmark Input Package

This folder contains scripts, documentation, runbooks, audit files, and small split/index artifacts for the EEG-FM unified TUAB benchmark. It is intended to be GitHub-safe.

Large data/model files are intentionally excluded from this export. Do not commit full datasets, raw EDF files, H5 files, pretrained checkpoints, or model output checkpoints to GitHub.

## Required Large Files on AI Station

Dataset:

```text
/nicoletye/workspace/unified_tuab/data/canonical_tuab_full.h5
```

LaBraM checkpoint:

```text
/nicoletye/workspace/Noise Robustness Evaluation Protocol/EEG-FM/Labram/checkpoints/labram-base.pth
```

BIOT checkpoint:

```text
/nicoletye/workspace/Noise Robustness Evaluation Protocol/EEG-FM/Biot/pretrained-models/EEG-PREST-16-channels.ckpt
```

EEGPT checkpoint:

```text
/nicoletye/workspace/Noise Robustness Evaluation Protocol/EEG-FM/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt
```

CBraMod checkpoint:

```text
/nicoletye/workspace/Noise Robustness Evaluation Protocol/EEG-FM/CBraMod/pretrained_weights/pretrained_weights.pth
```

CSBrain checkpoint:

```text
/nicoletye/workspace/Noise Robustness Evaluation Protocol/EEG-FM/CSBrain/downloaded_weights/pth/CSBrain.pth
```

CodeBrain checkpoint:

```text
/nicoletye/workspace/Noise Robustness Evaluation Protocol/EEG-FM/Codebrain/Checkpoints/CodeBrain.pth
```

## Benchmark Setup

- Models: LaBraM, BIOT, EEGPT, CBraMod, CSBrain, CodeBrain
- Strategies: `full_finetune`, `linear_probe`, `lora`
- Seeds: `0`, `42`, `123`, `256`, `512`
- Subset benchmark epochs: `15`
- Fixed subset size: train `8192`, val `2048`, test `2048`
- Fixed subset seed: `42`
- Full benchmark epoch-50 scripts are also included where available.

## Meriem-Strict Strategy Mapping

- LaBraM: Meriem-exact reference.
- BIOT, EEGPT, CBraMod, CSBrain: architecture-native equivalent mapping.
- CodeBrain: `MERIEM_STRICT_ARCHITECTURE_NATIVE_EQUIVALENT_PARTIAL`.

## Folder Contents

- `subset_training/`: subset benchmark README/runbook, script audit, fixed subset index, exact split CSV, and subset launcher/monitor.
- `Labram/`: LaBraM full unified60 strategy launch and summary scripts.
- `BIOT/`: BIOT Meriem-strict job wrapper.
- `EEGPT/`: EEGPT Meriem-strict job wrapper.
- `Labram_BIOT_EEGPT_shared/`: shared worker and strategy utilities used by LaBraM, BIOT, and EEGPT.
- `CBraMod/`, `CSBrain/`, `CodeBrain/`: remaining-model Meriem-strict job wrappers.
- `CBraMod_CSBrain_CodeBrain_shared/`: isolated worker and strategy utilities for CBraMod, CSBrain, and CodeBrain.

See `MANIFEST.json` for per-file original source paths, sizes, and SHA-256 hashes.
