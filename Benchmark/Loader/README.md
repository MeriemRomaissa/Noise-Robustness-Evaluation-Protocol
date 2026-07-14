# TUAB EEG-FM Loader Guide

## Purpose

These seven files provide one consistent loading interface for LaBraM, EEGPT, BIOT,
CBraMod, CSBrain, and CodeBrain. The shared module handles file access and batching.
Each model module contains only the decisions that make that model's input different.

The loaders do not train a model and do not repeat upstream EEG preprocessing. They
assume that unified60 already contains 23 referential channels, 2,000 samples per
window, and a sampling rate of 200 Hz.

## Reader's path through the code

Read any model file from top to bottom:

1. `get_loader_settings()` states the scientific input assumptions.
2. `build_unified60_loader()` is the entry point for shared H5 data.
3. `build_original_loader()` is the entry point for original PKL data.
4. `transform_unified60_input()` shows the unified60-to-model recipe.
5. `transform_original_input()` appears when original PKL data is already bipolar.
6. Small model-specific helpers explain normalization, scaling, or validation.

Read `loader_common.py` when you want to understand how samples are located, opened,
converted to tensors, or combined into batches.

## How the calls connect

For unified60 data:

```text
model.build_unified60_loader(config, split)
  -> loader_common.build_unified60_loader(..., model_transform)
  -> Unified60Dataset reads split rows and H5 channel metadata
  -> __getitem__ reads one [23,2000] window
  -> model_transform prepares that window for the selected model
  -> make_dataloader groups prepared windows into batches
```

For original data:

```text
model.build_original_loader(config, split)
  -> loader_common.build_original_loader(..., original_transform)
  -> OriginalPklDataset finds sorted <root>/<split>/*.pkl files
  -> __getitem__ reads one trusted {'X', 'y'} sample
  -> original_transform validates and prepares X
  -> make_dataloader groups prepared windows into batches
```

The transform function is passed into the shared Dataset rather than hard-coded there.
This keeps file loading identical while allowing each model to preserve its own model specific decisions.

## Model-specific decisions

| Model | Unified60 input | Original PKL | Batch shape |
|---|---|---|---|
| LaBraM | Validate 23 referential channels | Same | `[B,23,10,200]` |
| EEGPT | Validate 23 referential channels | Same | `[B,23,2000]` |
| BIOT | Reconstruct 16 bipolar channels; q95 normalize | q95 normalize existing bipolar input | `[B,16,2000]` |
| CBraMod | Reconstruct bipolar; divide by 100; reshape to patches | Divide existing bipolar input by 100; reshape to patches | `[B,16,10,200]` |
| CSBrain | Reconstruct bipolar; selected scale; reshape to patches | Apply selected scale to existing bipolar input; reshape to patches | `[B,16,10,200]` |
| CodeBrain | Reconstruct bipolar; divide by 100; reshape to patches | Divide existing bipolar input by 100; reshape to patches | `[B,16,10,200]` |

Original PKL data for the four bipolar models is already bipolar. It must not be
reconstructed a second time. That is why those files have separate unified60 and
original transform functions.

## Basic use

Place `loader_common.py` beside all six model files. Import the required model loader:

```python
from labram_loader import build_unified60_loader, validate_loader_batch

config = {
    "h5_path": "/path/to/unified60.h5",
    "split_index_path": "/path/to/split_index.csv",
    "batch_size": 64,
    "num_workers": 12,
    "pin_memory": True,
}

train_loader = build_unified60_loader(config, "train")
first_batch = next(iter(train_loader))
validate_loader_batch(first_batch)
```

To use original files:

```python
from biot_loader import build_original_loader

config = {
    "original_data_path": "/path/to/biot_pkls",
    "batch_size": 64,
    "num_workers": 12,
    "pin_memory": True,
}

test_loader = build_original_loader(config, "test")
```

## Choices available to a reproducing reader

All models accept the same DataLoader options:

| Config key | Default | Meaning |
|---|---:|---|
| `batch_size` | `64` | Samples grouped into one batch |
| `num_workers` | `0` | Background loading processes |
| `shuffle` | train only | Override sample shuffling deliberately |
| `drop_last` | `False` | Keep the final incomplete batch by default |
| `pin_memory` | `False` | Enable pinned CPU memory for GPU transfer |

Unified60 additionally requires:

- `h5_path`
- `split_index_path`
- optional `channel_names` when H5 metadata does not provide them

Original loading requires:

- `original_data_path`, containing `<root>/train`, `<root>/val`, and `<root>/test`

Model-specific choices:

- LaBraM: `patched=True` by default; set `False` for `[B,23,2000]` output.
- CSBrain: `scale_mode="div100"` by default; alternatives are `mul1000` and
  `mul10000` for the documented ablations.

## Reproducibility responsibilities outside these files

The loaders sort original PKL filenames and preserve validation/test order. Training
shuffling uses PyTorch's random-number generator. The run script must set the global
random seeds before building or iterating over loaders if identical ordering is needed.

The run script must also record:

- model loader and source type;
- split-index file version;
- batch size, workers, shuffling, `drop_last`, and `pin_memory`;
- LaBraM patch choice or CSBrain scale mode;
- upstream channel order, montage, sampling rate, units, and preprocessing version.

`pin_memory` changes transfer performance, not EEG values. `drop_last=True` discards
the final incomplete batch; it does not remove noisy EEG samples. Validation and test
should normally use `drop_last=False` so every sample contributes to evaluation.

## Why the shared module exists

The previous six files repeated the same H5 lifecycle, CSV parsing, channel cleaning,
bipolar reconstruction, PKL loading, and DataLoader policy.
