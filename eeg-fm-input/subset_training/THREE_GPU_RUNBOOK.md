# 3-GPU All-6 Unified60 Subset Development Runbook

## Purpose

This runbook is for the AI Station development platform with 3 GPUs, 63 CPU, and a 4-hour runtime limit. It runs the same all-6 EEG-FM unified60 subset benchmark as the sequential package, but schedules up to three jobs in parallel.

## Platform Configuration

- GPUs: `GPU_LIST=0,1,2`
- Max parallel jobs: `MAX_PARALLEL=3`
- CPU defaults: `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `NUM_WORKERS=8`
- Runtime expectation: 4 hours may not complete all 90 jobs; resume is expected.

## Benchmark Setup

- Models: LaBraM, BIOT, EEGPT, CBraMod, CSBrain, CodeBrain
- Strategies: full fine-tune, linear probe, LoRA
- Seeds: 0, 42, 123, 256, 512
- Total jobs: 90
- Epochs: 15
- Batch size: 64
- Fixed subset: train 8192, validation 2048, test 2048
- Fixed subset seed: 42

Every job consumes:

`reports/all6_unified60_subset_epoch15_5seed_1gpu_dev_v1/all6_fixed_subset_seed42_index.npz`

and passes `--train_n 0 --val_n 0 --test_n 0`, so model seeds do not change subset membership.

## Why 4 Hours May Not Finish

Four wall-clock hours on 3 GPUs gives about 12 GPU-hours. Finishing all 90 jobs within that time requires an average job time under 8 minutes. If jobs run slower, let the task stop and resume later with `FRESH=0`.

## Full Dry Run

```bash
cd /nicoletye/workspace/unified_tuab && \
DRY_RUN=1 FRESH=1 GPU_LIST=0,1,2 MAX_PARALLEL=3 EPOCHS=15 BATCH_SIZE=64 NUM_WORKERS=8 \
bash scripts/eegfm_adapters/run_all6_unified60_subset_epoch15_5seed_3gpu_dev_parallel.sh
```

Expected:

- `planned_jobs=90`
- `gpu_list=0,1,2`
- `max_parallel=3`
- no real training

## 3-Job Real Debug

Run this before the full 90-job task:

```bash
cd /nicoletye/workspace/unified_tuab && \
MODEL_FILTER=labram,biot,eegpt STRATEGY_FILTER=full_finetune SEED_FILTER=0 JOB_LIMIT=3 \
FRESH=1 GPU_LIST=0,1,2 MAX_PARALLEL=3 EPOCHS=1 BATCH_SIZE=64 NUM_WORKERS=8 \
bash scripts/eegfm_adapters/run_all6_unified60_subset_epoch15_5seed_3gpu_dev_parallel.sh
```

## Real 3-GPU Launch

```bash
cd /nicoletye/workspace/unified_tuab && \
GPU_LIST=0,1,2 MAX_PARALLEL=3 EPOCHS=15 BATCH_SIZE=64 NUM_WORKERS=8 \
bash scripts/eegfm_adapters/run_all6_unified60_subset_epoch15_5seed_3gpu_dev_parallel.sh
```

## Monitor

```bash
cd /nicoletye/workspace/unified_tuab && \
bash scripts/eegfm_adapters/monitor_all6_unified60_subset_epoch15_5seed_3gpu_dev.sh
```

## Resume

Use `FRESH=0` or omit `FRESH`; completed jobs with `metrics.json` status `PASS` are skipped.

```bash
cd /nicoletye/workspace/unified_tuab && \
FRESH=0 GPU_LIST=0,1,2 MAX_PARALLEL=3 EPOCHS=15 BATCH_SIZE=64 NUM_WORKERS=8 \
bash scripts/eegfm_adapters/run_all6_unified60_subset_epoch15_5seed_3gpu_dev_parallel.sh
```

Do not use `FRESH=1` for resume. `FRESH=1` deletes only the dev subset output root, but it is still a clean restart.

## Summarize

```bash
cd /nicoletye/workspace/unified_tuab && \
/nicoletye/venvs/eegnet/bin/python scripts/eegfm_adapters/summarize_all6_unified60_subset_epoch15_5seed_1gpu_dev.py
```

## Meriem-Strict Notes

- LaBraM explicitly uses `labram-base.pth`.
- LaBraM LoRA target is `meriem_exact`, rank 2, alpha 8.
- BIOT, EEGPT, CBraMod, and CSBrain use Meriem-strict or architecture-native equivalent mappings.
- CodeBrain LoRA remains `MERIEM_STRICT_ARCHITECTURE_NATIVE_EQUIVALENT_PARTIAL`; do not report it as exact LaBraM placement.

## Safety Notes

- This is a subset development benchmark, not a full-dataset task.
- Full benchmark output roots are not touched.
- Raw EDFs, canonical H5 data, preprocessing scripts, and checkpoint files are not modified.
