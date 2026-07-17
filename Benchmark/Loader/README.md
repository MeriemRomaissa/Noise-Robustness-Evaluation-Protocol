# Loader design

A loader's only job is turning one stored EEG window into the montage, scaling,
and shape one model expects. It doesn't build models, load checkpoints, pick a
fine-tuning strategy, or run an optimizer — that's `Training/`.

## How data flows through here

```text
Preprocessing (canonical H5 + split-index CSV)
        ↓
Config/<model>.yaml            picks the source and split
        ↓
loader_<model>.py              declares what this model needs
        ↓
loader_common.py               reads files, builds tensors, batches them
        ↓
(optional) StudyCase transform
        ↓
model
```

The dependency only goes one way: every `loader_<model>.py` uses
`loader_common.py`, never the reverse. That's what keeps file reading and
batching reusable while each model's own scientific choices stay local to its
own file.

## Who owns what

`loader_common.py` handles everything that has to behave identically across
all six models — path/split validation, split-index parsing, H5 and PKL
reading, channel-name decoding, the shared bipolar-reconstruction and
patch-reshaping math, and building the actual `DataLoader`.

Each `loader_<model>.py` owns exactly one thing: that model's input contract.
Referential or bipolar. Scaled or not. Patched into `[.,10,200]` or left as
`[.,2000]`. Nothing else lives there — no checkpoint logic, no metrics, no
channel-ablation logic (that's StudyCase's job, wired in from Training).

## Two entry points, always

Every model loader exposes the same pair:

```python
build_unified60_loader(config, split)
build_original_loader(config, split)
```

`training_common.py` picks one based on `data.source` in the model's YAML.
Nothing outside `Loader/` should need to call anything lower-level than these
two.

For unified60, one call does: validate config and split → read indices,
labels, and channel names → pull one `[23,2000]` referential window → run it
through the model's transform → convert to tensors → batch.

Original PKLs skip the reconstruction step for BIOT, CBraMod, CSBrain, and
CodeBrain — those files are already stored as 16-channel bipolar, so loading
them again through bipolar reconstruction would be wrong, not just redundant.

## Why every transform takes the same three arguments

The shared datasets call every model's transform through one signature:

```python
transform(window, channel_names, config)
```

`window` is one EEG window or batch. `channel_names` is only needed when a
model has to reconstruct bipolar channels from the referential source.
`config` carries model-specific data choices, like BIOT's normalization
epsilon or CSBrain's scale mode. LaBraM and EEGPT don't use the last two
arguments at all, but they keep the same signature anyway so the shared
Dataset classes can call all six transforms the same way without knowing
which model they're talking to.

## What each model actually does to a window

Every unified60 window starts the same: 10 seconds, 200 Hz, 23 referential
channels, `[B,23,2000]`.

| Model | What changes | Batch shape out |
|---|---|---|
| LaBraM | nothing but shape — regrouped into ten 200-sample patches | `[B,23,10,200]` |
| EEGPT | nothing at all | `[B,23,2000]` |
| BIOT | reconstruct 16 bipolar channels, divide by each channel's 95th-percentile amplitude | `[B,16,2000]` |
| CBraMod | reconstruct 16 bipolar channels, divide by 100, patch | `[B,16,10,200]` |
| CSBrain | reconstruct 16 bipolar channels, apply the configured scale, patch | `[B,16,10,200]` |
| CodeBrain | reconstruct 16 bipolar channels, divide by 100, patch | `[B,16,10,200]` |

Patching only regroups samples — it never touches amplitude values.

## Channel ablation isn't a loader concern

LaBraM and EEGPT loaders always build the normal 23-position input. Whether
an experiment actually uses `23channels` or `16channels_zeropadded` is
decided afterward, by `training_common.py` calling into the model's
StudyCase module. That's deliberate: both conditions read the same stored
windows, the same split indices, the same batching, and end at the same
model-ready shape. The only thing that changes is which channel positions
carry real signal.

## Batching knobs

Set in the YAML, not in code:

```yaml
loader:
  batch_size: 64
  num_workers: 8
  shuffle_train: true
  drop_last_train: false
  pin_memory: true
```

Train can shuffle and drop its last partial batch; validation and test never
do either. `num_workers`/`pin_memory` only affect load speed, not what values
end up in a batch.

## One sanity check worth knowing about

```python
validate_loader_batch(batch)
```

Every model loader has one. It just confirms a finished batch is the shape
that model expects — mostly useful when writing tests or debugging a new
preprocessing run, not something training calls itself.
