# EEG Foundation Model Benchmark

One reproducible TUAB fine-tuning workflow for six EEG foundation models:
LaBraM, EEGPT, BIOT, CBraMod, CSBrain, and CodeBrain.

```text
Preprocessing -> Loader -> Config -> StudyCase -> Training -> Evaluation
```

The benchmark keeps two kinds of settings separate:

- editable benchmark controls: the values a benchmark run may change
- locked model recipes: author/model-specific details that should not be tuned
  across models

## Repository Structure

```text
Benchmark/
├── Preprocessing/   Build the canonical EEG windows and split index
├── Loader/          Convert stored windows into model-ready tensors
├── Config/          Reproduction controls plus locked recipe records
├── StudyCase/        Controlled channel-count case-study transforms
├── Training/         Model construction, checkpoint loading, and training
└── Evaluation/       Output files and post-training result readers
```

## Canonical Dataset

`Preprocessing/build_canonical_tuab.py` reads raw TUAB EDF recordings and
writes one shared H5 dataset. Every window follows the same assumptions:

- montage: 23-channel referential
- sampling rate: 200 Hz, window: 10 s, shape `[23,2000]`
- data type: `float32`, unit: microvolts
- filtering: 0.1-75 Hz band-pass, 60 Hz notch
- no window overlap

The H5 stays model-neutral. Bipolar reconstruction, scaling, and patching
happen later in `Loader/`.

## What Can Be Changed For A Benchmark Run

Only these normal controls should change across comparable runs:

| Parameter | Meaning |
|---|---|
| seeds | e.g. `[0, 42, 123, 256, 512]` |
| train/val/test data | same files and same split indices |
| epochs | fixed budget such as 15 or 30 |
| batch size | default 64 unless the run plan says otherwise |
| evaluation metrics | balanced accuracy, ROC-AUC, PR-AUC, accuracy |
| classification threshold | default 0.5 |
| selection metric | validation balanced accuracy unless explicitly stated |

The YAML sections for those controls are top-level `paths` for dataset/output
files, `data`, `loader.batch_size`, `training.seeds`, `training.epochs`, and
`fine_tuning.strategy`, and `evaluation`.

The same controls are available from the command line. `--model-repo` and
`--checkpoint` are path overrides for machines where the `EEG-FM` repos or
checkpoints live somewhere else; they are not scientific tuning choices.

```bash
python Benchmark/Training/train_labram.py \
  --model-repo /path/to/EEG-FM/Labram \
  --checkpoint /path/to/labram-base.pth \
  --h5-file /path/to/canonical_tuab_full.h5 \
  --split-index /path/to/split_index.csv \
  --output /path/to/output/labram \
  --tuab-mode subset_tuab \
  --seeds 0 42 123 256 512 \
  --epochs 15 \
  --batch-size 64 \
  --evaluation-metrics balanced_accuracy roc_auc pr_auc accuracy \
  --classification-threshold 0.5 \
  --selection-metric balanced_accuracy
```

Use `--dry-run` to print the resolved config without loading data, checkpoints,
or models.

## Fine-Tuning Strategy Flow

```text
Config fine_tuning.strategy
→ Training/training_common.py reads the resolved config
→ StudyCase/Finetuning/finetuning_strategies.py applies the parameter policy
→ optimizer receives trainable parameters only
```

Each config exposes the same three choices:

```yaml
fine_tuning:
  strategy: full_finetune  # choices: [full_finetune, freeze_backbone, lora]
```

`full_finetune` trains all floating-point or complex parameters.
`freeze_backbone` freezes the backbone and trains only the classifier/task
head, which is the standard linear-probing setup. `lora` freezes ordinary
backbone weights and trains the LoRA targets defined in
`StudyCase/Finetuning/finetuning_strategies.py` plus the task head. The
default target mode uses the model-specific attention and MLP policies in that
file.

## TUAB Dataset Study Case

Every config has:

```yaml
study_case:
  tuab_mode: subset_tuab  # options: [subset_tuab, full_tuab]
```

`subset_tuab` uses the configured H5 file and split index with fixed caps:
8,192 train, 2,048 validation, and 2,048 test rows. `full_tuab` uses all rows
from the configured split index with no sample caps. Neither mode re-splits
data or changes train/validation/test membership.

The YAML gives the default, and `--tuab-mode` overrides it:

```bash
python Benchmark/Training/train_labram.py --config Benchmark/Config/labram.yaml --tuab-mode subset_tuab
python Benchmark/Training/train_labram.py --config Benchmark/Config/labram.yaml --tuab-mode full_tuab
```

## What Stays Model-Specific

Do not tune these as normal benchmark arguments:

- learning rate
- weight decay
- warmup epochs, warmup LR, or minimum LR
- drop path
- gradient clipping
- optimizer type or betas
- scheduler details
- model architecture parameters
- preprocessing/scaling assumptions
- checkpoint key mapping rules
- author repository import structure

These values live under `fixed_recipe` in each YAML, or directly inside the
model-specific `Training/train_<model>.py` file only when they are wrapper
rules needed to import author classes or map author checkpoint namespaces.
Those wrapper rules are audited bridge code, not user-editable recipe settings.

`fixed_recipe.paths.model_repo` and `fixed_recipe.paths.checkpoint` are model
dependency paths. They are merged into the runtime config and may still be
overridden with `--model-repo` and `--checkpoint` when a machine stores them in
a different place.

## Model Input Specifications

| Model | Input montage | Loader operation | Model-ready shape | Scaling |
|---|---|---|---|---|
| LaBraM | 23-channel referential | Form ten 1-second patches | `[B,23,10,200]` | None |
| EEGPT | 23-channel referential | Validate, keep as-is | `[B,23,2000]` | None |
| BIOT | 16-channel bipolar | Reconstruct bipolar | `[B,16,2000]` | q95 absolute normalization |
| CBraMod | 16-channel bipolar | Reconstruct bipolar, patch | `[B,16,10,200]` | Divide by 100 |
| CSBrain | 16-channel bipolar | Reconstruct bipolar, patch | `[B,16,10,200]` | Locked recipe scale mode |
| CodeBrain | 16-channel bipolar | Reconstruct bipolar, patch | `[B,16,10,200]` | Divide by 100 |

Original PKLs for the bipolar models are already bipolar; their original-data
loaders scale and patch only.

## Running A Model

```bash
python Benchmark/Training/train_labram.py    --config Benchmark/Config/labram.yaml
python Benchmark/Training/train_eegpt.py     --config Benchmark/Config/eegpt.yaml
python Benchmark/Training/train_biot.py      --config Benchmark/Config/biot.yaml
python Benchmark/Training/train_cbramod.py   --config Benchmark/Config/cbramod.yaml
python Benchmark/Training/train_csbrain.py   --config Benchmark/Config/csbrain.yaml
python Benchmark/Training/train_codebrain.py --config Benchmark/Config/codebrain.yaml
```

A pretrained checkpoint is mandatory. Training stops if the checkpoint is
missing or if no compatible backbone keys load.

## Reported Artifacts

Each seed follows the same sequence: set seeds, load and audit the pretrained
checkpoint, train, evaluate validation and test every epoch for compatible
logging, keep the best validation checkpoint, reload it, and evaluate the test
split once for the final result.

Every `outputs/<model>/seed_<n>/` directory holds `config.json`,
`checkpoint_load.json`, `fine_tuning.json`, `history.json`, `best_model.pt`,
and `result.json`. The model output directory also holds
`all_seed_results.json` and `summary.json`.

For compatibility with LaBraM-style analysis, each seed directory also holds
`log.txt`, `checkpoint.pth`, `checkpoint-<epoch>.pth`, and
`checkpoint-best.pth`. Test metrics in `log.txt` are for analysis only; model
selection still uses validation metrics only.

`Benchmark/Evaluation/analyze_log_txt.py` can read those `log.txt` files and
write summary JSON/CSV files after training.
