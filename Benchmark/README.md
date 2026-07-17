# EEG Foundation Model Benchmark

One reproducible TUAB fine-tuning workflow for six EEG foundation models:
LaBraM, EEGPT, BIOT, CBraMod, CSBrain, and CodeBrain.

```text
Preprocessing -> Loader -> Config -> StudyCase -> Training
```

Every experiment choice lives in `Config/*.yaml`. The Python files define the
fixed scientific and execution rules and should not need local editing.

## Repository structure

```text
Benchmark/
├── Preprocessing/   Build the canonical EEG windows and the split index
├── Loader/          Convert canonical windows into model-ready tensors
├── Config/          Reader-editable paths and experiment choices
├── StudyCase/        Controlled channel-ablation transformations
└── Training/         Model construction, checkpoint loading, and training
```

## Canonical dataset

`build_canonical_tuab.py` reads raw TUAB EDF recordings and writes one shared
H5 dataset. Every window follows the same assumptions:

- Montage: 23-channel referential
- Sampling rate: 200 Hz, window: 10 s (2,000 samples), shape `[23,2000]`
- Data type: `float32`, units: microvolts
- Filtering: 0.1-75 Hz band-pass, 60 Hz notch
- No window overlap

The H5 file stays model-neutral. Bipolar reconstruction, scaling, and
patching happen later, per model, inside `Loader/`.

## Model input specifications

| Model | Input montage | Loader operation | Model-ready shape | Scaling |
|---|---|---|---|---|
| LaBraM | 23-channel referential | Form ten 1-second patches | `[B,23,10,200]` | None |
| EEGPT | 23-channel referential | Validate, keep as-is | `[B,23,2000]` | None |
| BIOT | 16-channel bipolar | Reconstruct bipolar | `[B,16,2000]` | Per-channel q95 absolute normalization |
| CBraMod | 16-channel bipolar | Reconstruct bipolar, patch | `[B,16,10,200]` | Divide by 100 |
| CSBrain | 16-channel bipolar | Reconstruct bipolar, patch | `[B,16,10,200]` | Configurable, default divide by 100 |
| CodeBrain | 16-channel bipolar | Reconstruct bipolar, patch | `[B,16,10,200]` | Divide by 100 |

Original per-sample PKLs for BIOT, CBraMod, CSBrain, and CodeBrain are
already bipolar; their PKL loaders scale and patch only, without
reconstructing the montage a second time.

## Installation

```bash
pip install numpy pandas h5py mne tqdm pyyaml scikit-learn torch
```

Each YAML file also needs `paths.model_repo` pointing at the matching author
repository. Install that repository's own dependencies separately.

## Step 1: Build the canonical H5 file

```bash
python Preprocessing/build_canonical_tuab.py \
  --edf_root /path/to/TUAB/edf \
  --output /path/to/canonical_tuab.h5 \
  --log_dir /path/to/logs \
  --report_dir /path/to/reports
```

Use `--dry_run` to inspect source files without writing H5 data.
`--overwrite` or `--resume` is required to touch an existing output.

## Step 2: Build the canonical split index

```bash
python Preprocessing/build_canonical_h5_max_coverage_split.py \
  --h5 /path/to/canonical_tuab.h5 \
  --report_dir /path/to/reports \
  --seed 42
```

Assigns every H5 row to train, validation, or test while keeping subject or
recording groups together, so no group leaks across splits.

## Step 3: Edit one model configuration

Edit the matching file in `Config/`, for example `Config/labram.yaml`.
Readers may change:

- author repository, checkpoint, dataset, split-index, and output paths
- `unified60` vs. original-PKL data source, and how many samples to use
- batch size, workers, training shuffle, `drop_last`, pinned memory
- seeds, epochs, optimizer, learning rate, weight decay, scheduler
- fine-tuning strategy: full fine-tune, linear probe, or LoRA
- validation selection metric and classification threshold
- documented model-specific choices (e.g. CSBrain's `scale_mode`)
- LaBraM/EEGPT channel-study mode

A pretrained checkpoint is mandatory. Training stops if it's missing, or if
no compatible backbone keys load.

## Step 4: Run training

```bash
python Training/train_labram.py    --config Config/labram.yaml
python Training/train_eegpt.py     --config Config/eegpt.yaml
python Training/train_biot.py      --config Config/biot.yaml
python Training/train_cbramod.py   --config Config/cbramod.yaml
python Training/train_csbrain.py   --config Config/csbrain.yaml
python Training/train_codebrain.py --config Config/codebrain.yaml
```

Run one at a time unless the machine has enough independent GPU/CPU capacity
for concurrent runs.

## Channel study case

LaBraM and EEGPT configs expose:

```yaml
study_case:
  channel_mode: 23channels   # or 16channels_zeropadded
```

`23channels` keeps all 23 real signals. `16channels_zeropadded` keeps the 16
selected signals and zeroes the other seven channel positions. Both choices
preserve the model's normal tensor shape, so this isolates signal
availability from any architecture or input-shape change.

## Fine-tuning strategies

```yaml
fine_tuning:
  strategy: full_finetune   # or linear_probe, lora
  lora: {rank: 8, alpha: 16, dropout: 0.0}
```

`full_finetune` updates the pretrained backbone and task head.
`linear_probe` freezes the backbone and trains only the task head.
`lora` freezes ordinary backbone weights and trains inserted low-rank
updates on safe linear layers; `lora` settings are read only when selected.
Target layers are found and audited automatically, so no model-internal
layer names need to be entered by hand.

## What each run reports

Every seed follows the same sequence: set all seeds, load and audit the
pretrained checkpoint, train with the selected strategy, evaluate every
epoch on validation, keep the best validation epoch, reload it, then
evaluate the test split exactly once. Test data never influences model or
epoch selection. Reported metrics: balanced accuracy, AUROC, AUPRC.

Each `outputs/<model>/seed_<n>/` directory holds `config.json`,
`checkpoint_load.json`, `fine_tuning.json`, `history.json`, `best_model.pt`,
and `result.json`. The model's output directory also holds
`all_seed_results.json` and `summary.json` (mean/std across seeds).

## Loader settings that affect execution, not results

`num_workers` and `pin_memory` change how fast data loads, not what values
it contains. `drop_last_train` discards the final incomplete training batch
— it's a batching setting, not a noise filter. Validation and test are never
shuffled and never drop a batch.

## Reproducibility checklist

Keep, per reported experiment: the exact YAML used; the preprocessing and
split-index outputs; the author-repo and benchmark commit; the checkpoint
filename and its loading audit; every seed's results; the software
environment and GPU model. `config.json`, the checkpoint audit, training
history, and summary files in each output directory are the run-level
record. Dataset and checkpoint files stay external — do not commit them.
