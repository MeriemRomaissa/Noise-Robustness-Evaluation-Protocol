# Training

`Training/` owns the shared benchmark protocol and the thin model-specific
wrappers needed to build each author model.

## Dependency Structure

```text
Training/train_<model>.py
    ├── reads Config/<model>.yaml
    ├── imports Training/training_common.py
    ├── imports the author model repository from resolved paths.model_repo
    ├── calls Loader/loader_<model>.py through training_common.py
    └── calls StudyCase/Channel only when the locked recipe selects it
```

The model files own construction, author import paths, and checkpoint key
mapping. `training_common.py` owns seeds, loaders, checkpoint auditing,
train/validation/test flow, metric calculation, and best-checkpoint selection.

## User Controls And Path Overrides

The command line exposes normal benchmark controls plus local path overrides:

```text
--model-repo
--checkpoint
--h5-file
--split-index
--original-data
--output
--data-source
--train-samples
--validation-samples
--test-samples
--tuab-mode
--seeds
--epochs
--batch-size
--fine-tuning-strategy
--evaluation-metrics
--classification-threshold
--selection-metric
--dry-run
```

Optimizer, scheduler, LR, weight decay, warmup, clipping, drop path, scaling,
architecture, checkpoint mapping, and LoRA rank/alpha/layers/target placement
are locked implementation details. They are not ordinary benchmark controls.
Fine-tuning strategy is intentionally user-facing in `fine_tuning.strategy`;
LoRA internals live in `StudyCase/Finetuning/finetuning_strategies.py`.
`--model-repo` and `--checkpoint` exist only as machine-path overrides after
`fixed_recipe.paths` is merged.

`--tuab-mode` selects the TUAB dataset study case. `subset_tuab` keeps the
configured split index and caps train/validation/test to 8192/2048/2048.
`full_tuab` keeps the configured split index and uses all rows. Neither mode
creates a new split.

## Flow Per Seed

```text
set random seed
choose CPU or GPU
create seed output folder
save resolved config
build train/validation/test loaders
build optional study-case transform
build model from author repository
load pretrained checkpoint
apply fine_tuning.strategy through StudyCase/Finetuning/finetuning_strategies.py
build optimizer over trainable parameters only
train for each epoch
write Evaluation/log.txt and checkpoint files
select best validation checkpoint
reload best checkpoint
evaluate test once
save seed result
```

After all seeds finish, `summary.json` reports mean/std across seeds for the
configured metrics.

## Fine-Tuning Strategy Flow

```text
terminal CLI overrides, if provided
→ Config/<model>.yaml fine_tuning.strategy
→ training_common.py resolves config and fixed_recipe paths
→ StudyCase/Finetuning/finetuning_strategies.py reads the strategy
→ strategy records the linked EEG-FM author source files
→ model-specific parameter policy is applied before optimizer creation
→ optimizer receives trainable parameters only
```

Supported names are exactly `full_finetune`, `freeze_backbone`, and `lora`.
`freeze_backbone` is the standard linear-probing behavior: freeze the backbone
and train only the classifier/task head. `lora` dispatches through six
explicit functions: `lora_labram`, `lora_eegpt`, `lora_biot`,
`lora_cbramod`, `lora_csbrain`, and `lora_codebrain`.

The Strategy layer does not import or execute author training scripts. It
records the configured `EEG-FM/<model>` source files as provenance and applies
the Benchmark parameter policy to the already-constructed author model.

## Model-Specific Default Flow

```text
EEG-FM original repo parser/defaults
→ Config/<model>.yaml fixed_recipe
→ train_<model>.py reads fixed_recipe.model/training/paths
→ model, optimizer, scheduler, and checkpoint loading are initialized
```

Author architecture, classifier, dropout, checkpoint, optimizer, scheduler,
and scaling values are mirrored under `fixed_recipe` when direct parser import
would execute author CLI code or assume author-local paths.

Users should edit only the top-level YAML controls and the approved CLI
overrides above. `fixed_recipe` records locked EEG-FM dependency values; the
Benchmark reads and merges it, but it is not a normal tuning surface.

Shared bridge helpers in `training_common.py` keep repeated dependency logic
in one place:

- `add_repo_to_import_path` resolves the configured `EEG-FM/<model>` folder.
- `namespace_from_config` builds the small argparse-like objects expected by
  author constructors.
- `load_prefixed_checkpoint` loads the configured checkpoint from `EEG-FM` and
  applies the audited internal namespace bridge needed for author checkpoint
  keys. These prefix rules are not user-facing config knobs.

## Output Compatibility

`Benchmark/Evaluation/output_format.py` writes the author-compatible output
files that later analysis needs:

- `log.txt`: one JSON object per epoch
- `checkpoint.pth`: latest epoch
- `checkpoint-<epoch>.pth`: each epoch
- `checkpoint-best.pth`: best validation epoch

The epoch log includes train, validation, and test metrics so LaBraM-style
stability analysis can read it. Test metrics are never used for checkpoint
selection; the selected checkpoint is still chosen by the configured
validation metric.
