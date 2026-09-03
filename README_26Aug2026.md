history record of what I changed to this repo this past month_26Aug2026:

# New folder
Newly added folder: /Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output

no code change in scripts, just got output from training finetuning, output paths:
1. /Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output/EEG-FM/Labram/debug_output
2. /Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output/EEG-FM/EEGPT/debug_output
3. /Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output/EEG-FM/CSBrain/debug_output
4. /Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output/EEG-FM/Codebrain/debug_output
5. //Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output/EEG-FM/CBraMod
6. /Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output/EEG-FM/Biot

Newly added document named 'terminal command' under /Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output, it only has 6 terminal command used to run training finetuning and get output. 

How I got the training done: the idea was not changing any exisiting codes in /Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output/EEG-FM, but used external connector runner scripts to connect TUAB.pkl subset to the traiing scripts.

EEGMamba do not have output because AIstation environment conflict. 

Note: needs debugging, will further discuss what to debug.

# New lines of code added
To connect from /Noise-Robustness-Evaluation-Protocol/Benchmark to finetuning scripts in /Noise-Robustness-Evaluation-Protocol/EEG-FM,  a few lines of new codes were added to training scripts in /Noise-Robustness-Evaluation-Protocol/EEG-FM.

You will know which codes were newly added with #newly added codes.

# Deleted files
Old README.md from /Noise-Robustness-Evaluation-Protocol/Benchmark was deleted to avoid confusion.

/Noise-Robustness-Evaluation-Protocol-demo/Benchmark/Evaluation was deleted to avoid confusion, there's no use for it if we will have a unified ouput folder such as /Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output

# yet another new folder produced
Noise-Robustness-Evaluation-Protocol/Benchmark/Outputs/save_finetune_checkpoints was produced from loading TUAB H5 subsets and train finetuning from /Noise-Robustness-Evaluation-Protocol/EEG-FM. The outputs were not unified, but it proves that our benchmark loader scripts work, and it works well with training scripts from /Noise-Robustness-Evaluation-Protocol/EEG-FM.

/Noise-Robustness-Evaluation-Protocol/Benchmark/Outputs/save_finetune_checkpoints/README_long.md is a document for log metrics, checkpoint saving and file structure. README_short is just the list of log metrics.

# PDF file added
/Noise-Robustness-Evaluation-Protocol/TABLE.pdf: This pdf shows a table of default training hyperparameters for each model, manually checked from training scripts from /Noise-Robustness-Evaluation-Protocol/EEG-FM.

/Noise-Robustness-Evaluation-Protocol/output diff.pdf: This pdf shows the diffrences between checkpoints and logs output from /Noise-Robustness-Evaluation-Protocol/Benchmark vs /Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output.

# what mainly changed in this repo from last time
1. /Noise-Robustness-Evaluation-Protocol/Benchmark/Config: I checked though all .yaml files again, so there are code change (just a bit) after our last discussion.
2. I completely rewrote the code in /Noise-Robustness-Evaluation-Protocol/Benchmark/LoaderTraining. Only 6 scripts, the only function is the connector loads YAML as
defaults through the author script and only appends explicit CLI overrides. You can ignore /Noise-Robustness-Evaluation-Protocol/Benchmark/LoaderTraining/Tests, it was left in the folder to remind me about labram 

# This folder will not be used but important to keep
/Noise-Robustness-Evaluation-Protocol/Benchmark/LoaderTraining/Tests
There is a README.md to explain the reason why the script in this folder exist.

# Notes for later discussion:
1. Please look through /Noise-Robustness-Evaluation-Protocol/Benchmark/LoaderTraining. I selected and added in the default params from /Noise-Robustness-Evaluation-Protocol/EEG-FM training scripts. The training runs also work. But I might still miss something. 
2. /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output was created after running a short real training with 2 epochs. I inspected the outputs, the details need to be discussed face-to-face. 
3. I think the names of folders and scripts from  /Noise-Robustness-Evaluation-Protocol/Benchmark are not well-written. We shoulld discuss and rename them to avoid confusion. 

**Dependency Link**

```text
Raw TUAB EDF
  ↓
Benchmark/Preprocessing/build_canonical_tuab.py
  ↓
canonical_tuab_full.h5
  ↓
Benchmark/Preprocessing/build_canonical_h5_max_coverage_split.py
  ↓
canonical split_index.csv
  ↓
Benchmark/Config/{model}.yaml
  ↓
Benchmark/LoaderTraining/run_training_{model}.py
  ↓
EEG-FM/{model}/original finetuning script
  ↓
Benchmark/DataLoader/loader_{model}.py + loader_common.py
  ↓
model training
  ↓
output_dir/log.txt + checkpoints
```

**Step 1: Build Unified H5**

```bash
cd /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol

python Benchmark/Preprocessing/build_canonical_tuab.py \
  --edf_root /path/to/TUAB/edf \
  --output /path/to/canonical_tuab_full.h5 \
  --report_dir /path/to/preprocessing_reports \
  --overwrite
```

This creates the unified TUAB H5:

```text
/eeg -> [N, 23, 2000]
/metadata/label
/metadata/split
/metadata/subject_id
/metadata/recording_id
/metadata/source_path
```

Preprocessing logic:

```text
EDF
→ read with MNE
→ match 23 canonical TUAB channels
→ bandpass / notch / resample
→ convert to microvolts
→ window to 10 s at 200 Hz
→ write H5
```

**Step 2: Build Split Index**

```bash
python Benchmark/Preprocessing/build_canonical_h5_max_coverage_split.py \
  --h5 /path/to/canonical_tuab_full.h5 \
  --report_dir /path/to/split_reports
```

This creates:

```text
canonical_h5_labram_referenced_max_coverage_split_index.csv
canonical_h5_labram_referenced_max_coverage_split_index.json
canonical_h5_labram_referenced_max_coverage_split_summary.txt
```

The CSV is what training uses:

```text
h5_index
canonical_split
h5_original_split
label
recording_id
source_path
```

**Step 3: Fill Model YAML**

Example:

```yaml
paths:
  h5_file: /path/to/canonical_tuab_full.h5
  split_index: /path/to/canonical_h5_labram_referenced_max_coverage_split_index.csv
  original_data: /path/to/original/model_pkls
  checkpoint: /path/to/EEG-FM/model/checkpoint
  output: outputs/model_name

data:
  source: tuab_unified60

study_case:
  tuab_mode: subset_tuab

loader:
  batch_size: 64

training:
  seeds: [0, 42, 123, 256, 512]
  epochs: 15
```

Config files:

```text
Benchmark/Config/labram.yaml
Benchmark/Config/eegpt.yaml
Benchmark/Config/biot.yaml
Benchmark/Config/cbramod.yaml
Benchmark/Config/csbrain.yaml
Benchmark/Config/codebrain.yaml
```

**Step 4: Training Entry Points**

Benchmark connector scripts:

```text
Benchmark/LoaderTraining/run_training_labram.py
Benchmark/LoaderTraining/run_training_eegpt.py
Benchmark/LoaderTraining/run_training_biot.py
Benchmark/LoaderTraining/run_training_cbramod.py
Benchmark/LoaderTraining/run_training_csbrain.py
Benchmark/LoaderTraining/run_training_codebrain.py
```

These call the original EEG-FM scripts:

```text
LaBraM    -> EEG-FM/Labram/run_class_finetuning.py
EEGPT     -> EEG-FM/EEGPT/downstream_tueg/run_class_finetuning_EEGPT_change.py
BIOT      -> EEG-FM/Biot/run_binary_supervised.py
CBraMod   -> EEG-FM/CBraMod/finetune_main.py
CSBrain   -> EEG-FM/CSBrain/finetune_main.py
CodeBrain -> EEG-FM/Codebrain/Downstream/finetune_main.py
```

**Step 5: Loader Dependency**

When `data.source: tuab_unified60`:

```text
EEG-FM training script
  ↓
Benchmark/DataLoader/loader_{model}.py
  ↓
Benchmark/DataLoader/loader_common.py
  ↓
H5 rows selected by split_index.csv
```

Model input shapes:

```text
LaBraM    [B, 23, 10, 200]
EEGPT     [B, 23, 2000]
BIOT      [B, 16, 2000]
CBraMod   [B, 16, 10, 200]
CSBrain   [B, 16, 10, 200]
CodeBrain [B, 16, 10, 200]
```

**Step 6: Example Finetuning Commands**

LaBraM:

```bash
python Benchmark/LoaderTraining/run_training_labram.py \
  --config Benchmark/Config/labram.yaml \
  --data_source tuab_unified60 \
  --h5_file /path/to/canonical_tuab_full.h5 \
  --split_index /path/to/split_index.csv \
  --output_dir Benchmark/Outputs/save_finetune_checkpoints/labram/checkpoints \
  --epochs 15 \
  --seed 42 \
  --device cuda:0
```

EEGPT:

```bash
python Benchmark/LoaderTraining/run_training_eegpt.py \
  --config Benchmark/Config/eegpt.yaml \
  --data_source tuab_unified60 \
  --h5_file /path/to/canonical_tuab_full.h5 \
  --split_index /path/to/split_index.csv \
  --output_dir Benchmark/Outputs/save_finetune_checkpoints/eegpt/checkpoints \
  --epochs 15 \
  --seed 42 \
  --device cuda:0
```

CBraMod:

```bash
python Benchmark/LoaderTraining/run_training_cbramod.py \
  --config Benchmark/Config/cbramod.yaml \
  --data_source tuab_unified60 \
  --h5_file /path/to/canonical_tuab_full.h5 \
  --split_index /path/to/split_index.csv \
  --output_dir Benchmark/Outputs/save_finetune_checkpoints/cbramod/checkpoints \
  --epochs 15 \
  --seed 42 \
  --cuda 0
```

Same pattern for BIOT, CSBrain, CodeBrain.

**Output Step**

Outputs are written by the EEG-FM training scripts, not by preprocessing:

```text
output_dir/log.txt
output_dir/checkpoint.pth
output_dir/checkpoint-best.pth
```

For CodeBrain, native output may append `TUAB/`:

```text
output_dir/TUAB/log.txt
output_dir/TUAB/checkpoint.pth
```

**Important Current Caveat**

Some `Benchmark/LoaderTraining/run_training_*.py` scripts currently still reference `args.dry_run`, but the `--dry_run` parser line is commented out in several files. So normal training commands are the intended path, but dry-run mode may need a small cleanup before use.

========================
I checked the two LaBraM folders.

The meaningful source differences are only:

- `EEG-FM/Labram/run_class_finetuning.py`
- `EEG-FM/Labram/engine_for_finetuning.py`

Folder-only differences:

- Main repo has:
  - `EEG-FM/Labram/checkpoints/`
  - `EEG-FM/Labram/log/`
- Debug copy has:
  - `debug-unified-logging-output/EEG-FM/Labram/debug_output/`

Everything else, excluding generated/cache/output folders, has the same file set.

In `run_class_finetuning.py`, the main repo version has extra Benchmark integration that the debug copy does not have:

- `--config` YAML loading
- Benchmark path/config parsing
- `data_source=tuab_unified60`
- H5 file + split index support
- subset/full TUAB sample controls
- channel study-case support
- Benchmark loader import path setup
- output/log/checkpoint handling tied to Benchmark config
- extra CLI fields for LoRA/evaluation/report metrics

In `engine_for_finetuning.py`, the main repo version adds configurable classification threshold handling:

- `train_one_epoch(..., classification_threshold=0.5)`
- `evaluate(..., classification_threshold=0.5)`
- binary predictions use the passed threshold instead of hardcoded `0.5`
- metric calls pass `classification_threshold`

So the short version is:

- `/Noise-Robustness-Evaluation-Protocol/EEG-FM/Labram` is the Benchmark-connected version.
- `/Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output/EEG-FM/Labram` is the debug unified-logging-output version and does not include the newer Benchmark H5/config connector logic.
==================================================

Yes. I used `git diff --no-index` to compare the two pairs directly.

Command form used:

```bash
git diff --no-index \
  /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output/EEG-FM/Labram/engine_for_finetuning.py \
  /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/EEG-FM/Labram/engine_for_finetuning.py
```

```bash
git diff --no-index \
  /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output/EEG-FM/Labram/run_class_finetuning.py \
  /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/EEG-FM/Labram/run_class_finetuning.py
```

Diff direction: `debug-unified-logging-output` → main `EEG-FM/Labram`, so `+` lines are what exist in the main Benchmark-connected LaBraM script.

Results:

```text
engine_for_finetuning.py | 18 ++++++++++++------
run_class_finetuning.py  | 315 +++++++++++++++++++--
```

Main differences:

`engine_for_finetuning.py`

- Adds `classification_threshold=0.5` argument to:
  - `train_one_epoch(...)`
  - `evaluate(...)`
- Replaces hardcoded binary threshold `0.5` with `classification_threshold`.
- Passes `classification_threshold` into `utils.get_metrics(...)`.

`run_class_finetuning.py`

- Adds Benchmark YAML support via `--config`.
- Adds config-to-argparse default mapping.
- Adds Benchmark unified H5 support:
  - `--data_source tuab_unified60`
  - `--h5_file`
  - `--split_index`
  - `--train_samples`
  - `--validation_samples`
  - `--test_samples`
- Adds study-case controls:
  - `--tuab_mode`
  - `--channel_mode`
- Adds Benchmark loader import path handling.
- Adds unified H5 dataset construction using `Benchmark/DataLoader/loader_common.py`.
- Adds channel case transform via `Labram_23ch_vs_16ch`.
- Adds classification-threshold forwarding into train/val/test evaluation.
- Changes best-checkpoint selection logic in the visible diff area back to accuracy-based selection, instead of the debug branch’s ROC-AUC binary selection.

So yes: `git diff --no-index` confirms the main LaBraM scripts have the Benchmark connector/H5/config changes, while the `debug-unified-logging-output` copy is missing those Benchmark integration changes.

# New folder
Newly added folder: /Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output

no code change in scripts, just got output from training finetuning, output paths:
1. /Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output/EEG-FM/Labram/debug_output
2. /Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output/EEG-FM/EEGPT/debug_output
3. /Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output/EEG-FM/CSBrain/debug_output
4. /Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output/EEG-FM/Codebrain/debug_output
5. //Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output/EEG-FM/CBraMod
6. /Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output/EEG-FM/Biot

Newly added document named 'terminal command' under /Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output, it only has 6 terminal command used to run training finetuning and get output. 

How I got the training done: the idea was not changing any exisiting codes in /Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output/EEG-FM, but used external connector runner scripts to connect TUAB.pkl subset to the traiing scripts.

EEGMamba do not have output because AIstation environment conflict. 

Note: needs debugging, will further discuss what to debug.

# New lines of code added
To connect from /Noise-Robustness-Evaluation-Protocol/Benchmark to finetuning scripts in /Noise-Robustness-Evaluation-Protocol/EEG-FM,  a few lines of new codes were added to training scripts in /Noise-Robustness-Evaluation-Protocol/EEG-FM.

You will know which codes were newly added with #newly added codes.

# Deleted files
Old README.md from /Noise-Robustness-Evaluation-Protocol/Benchmark was deleted to avoid confusion.

/Noise-Robustness-Evaluation-Protocol-demo/Benchmark/Evaluation was deleted to avoid confusion, there's no use for it if we will have a unified ouput folder such as /Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output

# yet another new folder produced
Noise-Robustness-Evaluation-Protocol/Benchmark/Outputs/save_finetune_checkpoints was produced from loading TUAB H5 subsets and train finetuning from /Noise-Robustness-Evaluation-Protocol/EEG-FM. The outputs were not unified, but it proves that our benchmark loader scripts work, and it works well with training scripts from /Noise-Robustness-Evaluation-Protocol/EEG-FM.

/Noise-Robustness-Evaluation-Protocol/Benchmark/Outputs/save_finetune_checkpoints/README_long.md is a document for log metrics, checkpoint saving and file structure. README_short is just the list of log metrics.

# PDF file added
/Noise-Robustness-Evaluation-Protocol/TABLE.pdf: This pdf shows a table of default training hyperparameters for each model, manually checked from training scripts from /Noise-Robustness-Evaluation-Protocol/EEG-FM.

/Noise-Robustness-Evaluation-Protocol/output diff.pdf: This pdf shows the diffrences between checkpoints and logs output from /Noise-Robustness-Evaluation-Protocol/Benchmark vs /Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output.

# what mainly changed in this repo from last time
1. /Noise-Robustness-Evaluation-Protocol/Benchmark/Config: I checked though all .yaml files again, so there are code change (just a bit) after our last discussion.
2. I completely rewrote the code in /Noise-Robustness-Evaluation-Protocol/Benchmark/LoaderTraining. Only 6 scripts, the only function is the connector loads YAML as
defaults through the author script and only appends explicit CLI overrides. You can ignore /Noise-Robustness-Evaluation-Protocol/Benchmark/LoaderTraining/Tests, it was left in the folder to remind me about labram 

# This folder will not be used but important to keep
/Noise-Robustness-Evaluation-Protocol/Benchmark/LoaderTraining/Tests
There is a README.md to explain the reason why the script in this folder exist.

# Notes for later discussion:
1. Please look through /Noise-Robustness-Evaluation-Protocol/Benchmark/LoaderTraining. I selected and added in the default params from /Noise-Robustness-Evaluation-Protocol/EEG-FM training scripts. The training runs also work. But I might still miss something. 
2. /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output was created after running a short real training with 2 epochs. I inspected the outputs, the details need to be discussed face-to-face. 
3. I think the names of folders and scripts from  /Noise-Robustness-Evaluation-Protocol/Benchmark are not well-written. We shoulld discuss and rename them to avoid confusion. 

**Dependency Link**

```text
Raw TUAB EDF
  ↓
Benchmark/Preprocessing/build_canonical_tuab.py
  ↓
canonical_tuab_full.h5
  ↓
Benchmark/Preprocessing/build_canonical_h5_max_coverage_split.py
  ↓
canonical split_index.csv
  ↓
Benchmark/Config/{model}.yaml
  ↓
Benchmark/LoaderTraining/run_training_{model}.py
  ↓
EEG-FM/{model}/original finetuning script
  ↓
Benchmark/DataLoader/loader_{model}.py + loader_common.py
  ↓
model training
  ↓
output_dir/log.txt + checkpoints
```

**Step 1: Build Unified H5**

```bash
cd /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol

python Benchmark/Preprocessing/build_canonical_tuab.py \
  --edf_root /path/to/TUAB/edf \
  --output /path/to/canonical_tuab_full.h5 \
  --report_dir /path/to/preprocessing_reports \
  --overwrite
```

This creates the unified TUAB H5:

```text
/eeg -> [N, 23, 2000]
/metadata/label
/metadata/split
/metadata/subject_id
/metadata/recording_id
/metadata/source_path
```

Preprocessing logic:

```text
EDF
→ read with MNE
→ match 23 canonical TUAB channels
→ bandpass / notch / resample
→ convert to microvolts
→ window to 10 s at 200 Hz
→ write H5
```

**Step 2: Build Split Index**

```bash
python Benchmark/Preprocessing/build_canonical_h5_max_coverage_split.py \
  --h5 /path/to/canonical_tuab_full.h5 \
  --report_dir /path/to/split_reports
```

This creates:

```text
canonical_h5_labram_referenced_max_coverage_split_index.csv
canonical_h5_labram_referenced_max_coverage_split_index.json
canonical_h5_labram_referenced_max_coverage_split_summary.txt
```

The CSV is what training uses:

```text
h5_index
canonical_split
h5_original_split
label
recording_id
source_path
```

**Step 3: Fill Model YAML**

Example:

```yaml
paths:
  h5_file: /path/to/canonical_tuab_full.h5
  split_index: /path/to/canonical_h5_labram_referenced_max_coverage_split_index.csv
  original_data: /path/to/original/model_pkls
  checkpoint: /path/to/EEG-FM/model/checkpoint
  output: outputs/model_name

data:
  source: tuab_unified60

study_case:
  tuab_mode: subset_tuab

loader:
  batch_size: 64

training:
  seeds: [0, 42, 123, 256, 512]
  epochs: 15
```

Config files:

```text
Benchmark/Config/labram.yaml
Benchmark/Config/eegpt.yaml
Benchmark/Config/biot.yaml
Benchmark/Config/cbramod.yaml
Benchmark/Config/csbrain.yaml
Benchmark/Config/codebrain.yaml
```

**Step 4: Training Entry Points**

Benchmark connector scripts:

```text
Benchmark/LoaderTraining/run_training_labram.py
Benchmark/LoaderTraining/run_training_eegpt.py
Benchmark/LoaderTraining/run_training_biot.py
Benchmark/LoaderTraining/run_training_cbramod.py
Benchmark/LoaderTraining/run_training_csbrain.py
Benchmark/LoaderTraining/run_training_codebrain.py
```

These call the original EEG-FM scripts:

```text
LaBraM    -> EEG-FM/Labram/run_class_finetuning.py
EEGPT     -> EEG-FM/EEGPT/downstream_tueg/run_class_finetuning_EEGPT_change.py
BIOT      -> EEG-FM/Biot/run_binary_supervised.py
CBraMod   -> EEG-FM/CBraMod/finetune_main.py
CSBrain   -> EEG-FM/CSBrain/finetune_main.py
CodeBrain -> EEG-FM/Codebrain/Downstream/finetune_main.py
```

**Step 5: Loader Dependency**

When `data.source: tuab_unified60`:

```text
EEG-FM training script
  ↓
Benchmark/DataLoader/loader_{model}.py
  ↓
Benchmark/DataLoader/loader_common.py
  ↓
H5 rows selected by split_index.csv
```

Model input shapes:

```text
LaBraM    [B, 23, 10, 200]
EEGPT     [B, 23, 2000]
BIOT      [B, 16, 2000]
CBraMod   [B, 16, 10, 200]
CSBrain   [B, 16, 10, 200]
CodeBrain [B, 16, 10, 200]
```

**Step 6: Example Finetuning Commands**

LaBraM:

```bash
python Benchmark/LoaderTraining/run_training_labram.py \
  --config Benchmark/Config/labram.yaml \
  --data_source tuab_unified60 \
  --h5_file /path/to/canonical_tuab_full.h5 \
  --split_index /path/to/split_index.csv \
  --output_dir Benchmark/Outputs/save_finetune_checkpoints/labram/checkpoints \
  --epochs 15 \
  --seed 42 \
  --device cuda:0
```

EEGPT:

```bash
python Benchmark/LoaderTraining/run_training_eegpt.py \
  --config Benchmark/Config/eegpt.yaml \
  --data_source tuab_unified60 \
  --h5_file /path/to/canonical_tuab_full.h5 \
  --split_index /path/to/split_index.csv \
  --output_dir Benchmark/Outputs/save_finetune_checkpoints/eegpt/checkpoints \
  --epochs 15 \
  --seed 42 \
  --device cuda:0
```

CBraMod:

```bash
python Benchmark/LoaderTraining/run_training_cbramod.py \
  --config Benchmark/Config/cbramod.yaml \
  --data_source tuab_unified60 \
  --h5_file /path/to/canonical_tuab_full.h5 \
  --split_index /path/to/split_index.csv \
  --output_dir Benchmark/Outputs/save_finetune_checkpoints/cbramod/checkpoints \
  --epochs 15 \
  --seed 42 \
  --cuda 0
```

Same pattern for BIOT, CSBrain, CodeBrain.

**Output Step**

Outputs are written by the EEG-FM training scripts, not by preprocessing:

```text
output_dir/log.txt
output_dir/checkpoint.pth
output_dir/checkpoint-best.pth
```

For CodeBrain, native output may append `TUAB/`:

```text
output_dir/TUAB/log.txt
output_dir/TUAB/checkpoint.pth
```

**Important Current Caveat**

Some `Benchmark/LoaderTraining/run_training_*.py` scripts currently still reference `args.dry_run`, but the `--dry_run` parser line is commented out in several files. So normal training commands are the intended path, but dry-run mode may need a small cleanup before use.

========================
I checked the two LaBraM folders.

The meaningful source differences are only:

- `EEG-FM/Labram/run_class_finetuning.py`
- `EEG-FM/Labram/engine_for_finetuning.py`

Folder-only differences:

- Main repo has:
  - `EEG-FM/Labram/checkpoints/`
  - `EEG-FM/Labram/log/`
- Debug copy has:
  - `debug-unified-logging-output/EEG-FM/Labram/debug_output/`

Everything else, excluding generated/cache/output folders, has the same file set.

In `run_class_finetuning.py`, the main repo version has extra Benchmark integration that the debug copy does not have:

- `--config` YAML loading
- Benchmark path/config parsing
- `data_source=tuab_unified60`
- H5 file + split index support
- subset/full TUAB sample controls
- channel study-case support
- Benchmark loader import path setup
- output/log/checkpoint handling tied to Benchmark config
- extra CLI fields for LoRA/evaluation/report metrics

In `engine_for_finetuning.py`, the main repo version adds configurable classification threshold handling:

- `train_one_epoch(..., classification_threshold=0.5)`
- `evaluate(..., classification_threshold=0.5)`
- binary predictions use the passed threshold instead of hardcoded `0.5`
- metric calls pass `classification_threshold`

So the short version is:

- `/Noise-Robustness-Evaluation-Protocol/EEG-FM/Labram` is the Benchmark-connected version.
- `/Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output/EEG-FM/Labram` is the debug unified-logging-output version and does not include the newer Benchmark H5/config connector logic.
==================================================

Yes. I used `git diff --no-index` to compare the two pairs directly.

Command form used:

```bash
git diff --no-index \
  /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output/EEG-FM/Labram/engine_for_finetuning.py \
  /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/EEG-FM/Labram/engine_for_finetuning.py
```

```bash
git diff --no-index \
  /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output/EEG-FM/Labram/run_class_finetuning.py \
  /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/EEG-FM/Labram/run_class_finetuning.py
```

Diff direction: `debug-unified-logging-output` → main `EEG-FM/Labram`, so `+` lines are what exist in the main Benchmark-connected LaBraM script.

Results:

```text
engine_for_finetuning.py | 18 ++++++++++++------
run_class_finetuning.py  | 315 +++++++++++++++++++--
```

Main differences:

`engine_for_finetuning.py`

- Adds `classification_threshold=0.5` argument to:
  - `train_one_epoch(...)`
  - `evaluate(...)`
- Replaces hardcoded binary threshold `0.5` with `classification_threshold`.
- Passes `classification_threshold` into `utils.get_metrics(...)`.

`run_class_finetuning.py`

- Adds Benchmark YAML support via `--config`.
- Adds config-to-argparse default mapping.
- Adds Benchmark unified H5 support:
  - `--data_source tuab_unified60`
  - `--h5_file`
  - `--split_index`
  - `--train_samples`
  - `--validation_samples`
  - `--test_samples`
- Adds study-case controls:
  - `--tuab_mode`
  - `--channel_mode`
- Adds Benchmark loader import path handling.
- Adds unified H5 dataset construction using `Benchmark/DataLoader/loader_common.py`.
- Adds channel case transform via `Labram_23ch_vs_16ch`.
- Adds classification-threshold forwarding into train/val/test evaluation.
- Changes best-checkpoint selection logic in the visible diff area back to accuracy-based selection, instead of the debug branch’s ROC-AUC binary selection.

So yes: `git diff --no-index` confirms the main LaBraM scripts have the Benchmark connector/H5/config changes, while the `debug-unified-logging-output` copy is missing those Benchmark integration changes.

# rename folder name
Rename /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/Benchmark/Loader to /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/Benchmark/DataLoader

Updated the Benchmark connector references from Loader to DataLoader in:
EEG-FM/Labram/run_class_finetuning.py;
EEG-FM/EEGPT/downstream_tueg/run_class_finetuning_EEGPT_change.py;
EEG-FM/Biot/run_binary_supervised.py;
EEG-FM/CBraMod/finetune_main.py;
EEG-FM/CSBrain/finetune_main.py;
EEG-FM/Codebrain/Downstream/finetune_main.py;
Benchmark/Preprocessing/build_canonical_h5_max_coverage_split.py;
Benchmark/DataLoader/README.md

# fix log.txt inconsistency issue
Added missing 'train_balanced_accuracy' metric to CBraMod, CSBrain, CodeBrain and BIOT.

scripts modified:
debug-unified-logging-output/EEG-FM/CBraMod/finetune_trainer.py;
debug-unified-logging-output/EEG-FM/Codebrain/Downstream/finetune_trainer.py;
debug-unified-logging-output/EEG-FM/CSBrain/finetune_trainer.py;

train_min_le and train_loss_scale are LaBraM and EEGPT specific metrics so the other EEG-FM will have log schema fields:
"train_min_lr": null;
"train_loss_scale": null

# fix repeated-epoch logging issue
At each run start, log.txt is reset once, then each epoch appends one fresh JSON line, preventing duplicated epoch logs across reruns.

Files changed:
debug-unified-logging-output/EEG-FM/Labram/run_class_finetuning.py;
debug-unified-logging-output/EEG-FM/EEGPT/downstream_tueg/run_class_finetuning_EEGPT_change.py;
debug-unified-logging-output/EEG-FM/EEGPT/downstream_tueg/run_class_finetuning_EEGPT_change_tuev.py;
debug-unified-logging-output/EEG-FM/Biot/run_binary_supervised.py;
debug-unified-logging-output/EEG-FM/Biot/run_multiclass_supervised.py;

# checked
From /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output, scaler in checkpoint keys is LaBraM/EEGPT training-engine-specific. 

There is no need to add scaler to other EEG-FMs, so no code change needed.

# checked
Keep the parse_args() blocks from /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/Benchmark/LoaderTraining

These connector scripts needs this block to let users optionally override YAML values from the terminal.

The "default=None" values are intentional. 
They mean: If user does not type a terminal override,
do not block YAML or native training-script defaults.

append_if_set() only forwards real user-provided overrides.

Notes:
There are three layers:
Layer 1: Native training script default
EEG-FM/Labram/run_class_finetuning.py
example: --epochs default=30

Layer 2: YAML config
Benchmark/Config/labram.yaml
example: training.epochs: 15

Layer 3: Terminal override
python run_training_labram.py --epochs 2

Priority is:
terminal override > YAML config > native training script default

Examples:
If user types terminal override:
python run_training_labram.py --epochs 2
then final epochs = 2.

If user does not type terminal override, but YAML has:
training:
  epochs: 15
then final epochs = 15.

If user does not type terminal override, and YAML does not contain epochs, then final epochs comes from the native LaBraM script:
parser.add_argument("--epochs", default=30)
then final epochs = 30.

Summary: default=None in the runner is what allows YAML/native defaults to still work. If the runner used default=2, then it would always pass --epochs 2 and would accidentally override YAML every time.

# merge training scripts
From /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output/EEG-FM and /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/EEG-FM

main target folder: /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/EEG-FM

Two conflicts found and resolved.

Conflict 1: CodeBrain folder layout resolved 
CodeBrain: debug_output/CodeBrain/TUAB/log.txt to debug_output/CodeBrain/log.txt
Others:    debug_output/<Model>/log.txt

Conflict 2: EEG-FM/Biot/run_binary_supervised.py imports resolved
Final import set keeps all imports from EEG-FM/Biot/run_binary_supervised.py
import json;
import sys;
from pathlib import Path

The merge preserved:
Benchmark/H5 loader connector logic from the main EEG-FM;
Unified output/logging/checkpoint behavior from debug-unified-logging-output;
Consistent log.txt schema

Note:
schema = same metric columns inside log.txt
layout = same folder/file path pattern on disk

# removed 
/nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output/EEG-FM

/nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/output diff.pdf

# new folder added
/nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/Benchmark/Inference

this folder has 6 inference scripts:
Benchmark/Inference/run_inference_labram.py;
Benchmark/Inference/run_inference_eegpt.py;
Benchmark/Inference/run_inference_biot.py;
Benchmark/Inference/run_inference_cbramod.py;
Benchmark/Inference/run_inference_csbrain.py;
Benchmark/Inference/run_inference_codebrain.py

scripts include Full finetuning, LORA, EEGNet, will later include ATCNet and ShallowFBCSPNet.

# new folder and file added
/nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/Benchmark/Plotting/plot_nmt_ood_results.py

this script makes plots.
Reads saved nmt_ood_results.json files from Benchmark/Inference.

output:
nmt_ood_accuracy.png/pdf;
nmt_ood_balanced_accuracy.png/pdf;
nmt_ood_roc_auc.png/pdf;
nmt_ood_pr_auc.png/pdf;
clean_vs_nmt_ood_accuracy.png/pdf;
clean_vs_nmt_ood_balanced_accuracy.png/pdf;
clean_vs_nmt_ood_roc_auc.png/pdf;
clean_vs_nmt_ood_pr_auc.png/pdf;
nmt_ood_relative_robustness.png/pdf;
nmt_ood_accuracy_drop.png/pdf;
nmt_ood_plot_records.csv;
nmt_ood_plot_records.json

plot command:
cd /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol

python Benchmark/Plotting/plot_nmt_ood_results.py \
  --input_glob "Benchmark/Outputs/**/nmt_ood_results.json" \
  --output_dir Benchmark/Outputs/nmt_ood_plots


# new files created for NMT dataset
/nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/Benchmark/Preprocessing/preprocess_nmt_to_labram_ood.py

/nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/Benchmark/DataLoader/loader_nmt_ood.py

loader_nmt_ood.py is the shared NMT OOD data connector between NMT preprocessing and the six model-specific inference scripts.


# test run NMT subset
All six inference run succeeded.

output results are saved under /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/Benchmark/Outputs/nmt_ood_inference_smoke

Each model folder now has:
log.txt;
metrics.json;
nmt_ood_results.json

Each log.txt has 2 JSON lines:
1. <Model> Full FT on NMT_OOD
2. EEGNet on NMT_OOD

example terminal command:
extract the EEGNet block into the standalone JSON format expected by the inference scripts:
cd /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol && \
mkdir -p Benchmark/Outputs/nmt_ood_baselines && \
python - <<'PY'
import json
from pathlib import Path

src = Path("/nicoletye/workspace/Labram_with_finetuning_adaptors/Labram with finetuning adaptors/Overfitting_Evaluation/Results/NMT_OOD/nmt_ood_results.json")
dst = Path("Benchmark/Outputs/nmt_ood_baselines/nmt_eegnet_results.json")

data = json.loads(src.read_text())
eegnet_nmt = data["eegnet"]["nmt_ood"]
dst.write_text(json.dumps(eegnet_nmt, indent=2) + "\n")
print(dst.resolve())
print(json.dumps(eegnet_nmt, indent=2))
PY

BIOT inference test run:
cd /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol && \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
PYTHONUNBUFFERED=1 \
/nicoletye/venvs/biot_env/bin/python Benchmark/Inference/run_inference_biot.py \
  --checkpoint Benchmark/Outputs/save_finetune_checkpoints/biot/log/TUAB-BIOT-0.001-8-200-200-100/checkpoints/epoch=1-step=16.ckpt \
  --eegnet_json Benchmark/Outputs/nmt_ood_baselines/nmt_eegnet_results.json \
  --nmt_dir /nicoletye/workspace/nmt_scalp_eeg_labram_ood/test \
  --output_dir Benchmark/Outputs/nmt_ood_inference_smoke/biot \
  --max_samples 32 \
  --batch_size 8 \
  --device cuda

Reuse EEGNet saved metrics from existing 
/nicoletye/workspace/Labram_with_finetuning_adaptors/Labram with finetuning adaptors/Overfitting_Evaluation/Results/NMT_OOD/nmt_ood_results.json as the baseline results, then compare the EEG-FM inference results.

# commit and pushed to github
/nicoletye/workspace/Noise-Robustness-Evaluation-Protocol



