# Meriem Reproduction Package

This package is a source-only reproduction bundle for the unified TUAB preprocessing and EEG-FM benchmark workflow. It contains scripts and documentation only. It does not include raw EDF files, H5 files, NPZ data, checkpoints, processed PKL files, or training outputs.

The workflow has three separate tracks. They should not be mixed.

## Track A: Unified Preprocessing Smoke Test

Purpose: test the unified preprocessing script on the tiny 8-EDF raw subset from USB. This is only a smoke test. It should produce a debug H5, not the final full dataset.

Input provided separately:

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

Main scripts:

```text
scripts/build_canonical_tuab.py
scripts/build_canonical_h5_max_coverage_split.py
```

Preprocessing implemented by `build_canonical_tuab.py`:

- Raw TUAB EDF input.
- Canonical 23-channel referential order:
  `FP1 FP2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T3 T4 T5 T6 A1 A2 FZ CZ PZ T1 T2`
- Missing canonical channels are zero-padded.
- Bandpass filter: `0.1-75 Hz`.
- Notch filter: `60 Hz`.
- Resampling: `200 Hz`.
- Windowing: `10 s`, `2000` samples, no overlap.
- Units: microvolts.
- Labels: `normal -> 0`, `abnormal -> 1`.
- Original split mapping: `edf/train -> train`, `edf/eval -> test`.

Smoke-test command template:

```bash
cd /path/to/unified_tuab

python scripts/build_canonical_tuab.py \
  --edf_root "/path/to/Raw TUAB subset folder" \
  --output data/canonical_tuab_debug_balanced_8files_rebuilt.h5 \
  --log_dir logs/build_debug \
  --report_dir reports/build_debug \
  --run_name debug_balanced_8files \
  --overwrite
```

The expected reference debug output on AI Station was:

```text
canonical_tuab_debug_balanced_8files.h5
/eeg shape = (1040, 23, 2000)
dtype = float32
split includes train and test
label includes 0 and 1
```

Full TUAB build reference from AI Station:

```text
EDF files scanned: 2993
EDF files processed: 2993
Skipped EDF files: 0
Total windows stored: 409455
H5 shape: (409455, 23, 2000)
dtype: float32
```

Do not rebuild the full H5 unless that is explicitly intended.

## Track B: Option 1 Original-vs-Unified Matched Subset

Purpose: compare model-original raw/processed pipelines against the unified H5 adapter pipeline using matched EDF recording identity where possible.

This track uses the Option 1 raw EDF symlink subset from USB or from a reviewed manifest. It is optional diagnostic evidence. It is not the direct input to the 90-job all-6 benchmark.

Main scripts:

```text
scripts/eegfm_adapters/create_tuab_option1_raw_edf_symlink_subset.py
scripts/eegfm_adapters/validate_tuab_option1_subset_artifacts.py
scripts/eegfm_adapters/*_option1_make_tuab_wrapper.py
scripts/eegfm_adapters/build_option1_preprocessing_matrix.py
scripts/eegfm_adapters/audit_labram_original_vs_unified_full_for_meriem.py
```

LaBraM notch-ablation subset scripts:

```text
scripts/eegfm_adapters/build_labram_exact_subset_manifest.py
scripts/eegfm_adapters/build_labram_unified_50hz_subset_from_raw_exact_manifest.py
scripts/eegfm_adapters/run_labram_notch_ablation_subset_strict_epoch20_2gpu_task.sh
scripts/eegfm_adapters/monitor_labram_notch_ablation_subset_strict_epoch20_2gpu.sh
scripts/eegfm_adapters/summarize_labram_notch_ablation_subset_strict_epoch20_2gpu.py
```

Important distinction:

- Track B raw EDF subset is for original-vs-unified diagnostic comparisons.
- Track B does not replace the full canonical H5 used by Track C.

## Track C: All-6 Unified H5 Benchmark

Purpose: run the 90-job EEG-FM benchmark on unified TUAB preprocessing.

The all-6 benchmark uses:

```text
canonical_tuab_full.h5
all6_fixed_subset_seed42_index.npz
```

The fixed subset NPZ stores train/validation/test indices only. The actual EEG data are read from `canonical_tuab_full.h5`.

Models:

```text
LaBraM
BIOT
EEGPT
CBraMod
CSBrain
CodeBrain
```

Strategies:

```text
full_finetune
linear_probe
lora
```

Settings:

```text
Seeds: 0, 42, 123, 256, 512
Epochs: 15
Train windows: 8192
Validation windows: 2048
Test windows: 2048
Fixed subset seed: 42
Batch size: 64
Development GPUs: 0,1,2
Max parallel jobs: 3
Workers: 8
```

Main command:

```bash
cd /nicoletye/workspace/unified_tuab && \
mkdir -p task_logs /nicoletye/workspace/tmp && \
TMPDIR=/nicoletye/workspace/tmp \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 \
FRESH=1 \
GPU_LIST=0,1,2 \
MAX_PARALLEL=3 \
EPOCHS=15 \
BATCH_SIZE=64 \
NUM_WORKERS=8 \
bash scripts/eegfm_adapters/run_all6_unified60_subset_epoch15_5seed_3gpu_dev_parallel.sh \
2>&1 | tee task_logs/all6_unified60_subset_epoch15_5seed_3gpu_dev_$(date +%Y%m%d_%H%M%S).log
```

Resume command:

```bash
cd /nicoletye/workspace/unified_tuab && \
mkdir -p task_logs /nicoletye/workspace/tmp && \
TMPDIR=/nicoletye/workspace/tmp \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 \
FRESH=0 \
GPU_LIST=0,1,2 \
MAX_PARALLEL=3 \
EPOCHS=15 \
BATCH_SIZE=64 \
NUM_WORKERS=8 \
bash scripts/eegfm_adapters/run_all6_unified60_subset_epoch15_5seed_3gpu_dev_parallel.sh \
2>&1 | tee task_logs/all6_unified60_subset_epoch15_5seed_3gpu_dev_resume_$(date +%Y%m%d_%H%M%S).log
```

Monitor:

```bash
cd /nicoletye/workspace/unified_tuab && \
bash scripts/eegfm_adapters/monitor_all6_unified60_subset_epoch15_5seed_3gpu_dev.sh
```

Summarize:

```bash
cd /nicoletye/workspace/unified_tuab && \
/nicoletye/venvs/eegnet/bin/python scripts/eegfm_adapters/summarize_all6_unified60_subset_epoch15_5seed_1gpu_dev.py
```

## Environment

See `ENVIRONMENT.md` and `requirements.txt`.

On AI Station, the inspected Python environments used Python 3.10.15 and Torch `2.3.0a0+6ddf5cf85e.nv24.04` with CUDA `12.4` available. Do not blindly reinstall PyTorch from `requirements.txt`; use the platform-compatible PyTorch/CUDA wheel or the existing AI Station venvs.

## What This Package Does Not Contain

This package intentionally excludes:

- raw EDF files
- full or debug H5 files
- NPZ EEG data or index files
- checkpoints
- processed PKL data
- experiment outputs

See `DATA_TO_PROVIDE_SEPARATELY.md`.
