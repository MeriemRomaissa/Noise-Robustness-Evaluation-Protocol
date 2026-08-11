What I changed to this repo this past week:

# rename folder name
Rename /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/Benchmark/Loader to /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/Benchmark/DataLoader

Updated the Benchmark connector references from Loader to DataLoader in:
EEG-FM/Labram/run_class_finetuning.py
EEG-FM/EEGPT/downstream_tueg/run_class_finetuning_EEGPT_change.py
EEG-FM/Biot/run_binary_supervised.py
EEG-FM/CBraMod/finetune_main.py
EEG-FM/CSBrain/finetune_main.py
EEG-FM/Codebrain/Downstream/finetune_main.py
Benchmark/Preprocessing/build_canonical_h5_max_coverage_split.py
Benchmark/DataLoader/README.md

# fix log.txt inconsistency issue
Added missing 'train_balanced_accuracy' metric to CBraMod, CSBrain, CodeBrain and BIOT.

scripts modified:
debug-unified-logging-output/EEG-FM/CBraMod/finetune_trainer.py
debug-unified-logging-output/EEG-FM/Codebrain/Downstream/finetune_trainer.py
debug-unified-logging-output/EEG-FM/CSBrain/finetune_trainer.py

train_min_le and train_loss_scale are LaBraM and EEGPT specific metrics so the other EEG-FM will have log schema fields:
"train_min_lr": null
"train_loss_scale": null

# fix repeated-epoch logging issue
At each run start, log.txt is reset once, then each epoch appends one fresh JSON line, preventing duplicated epoch logs across reruns.

Files changed:
debug-unified-logging-output/EEG-FM/Labram/run_class_finetuning.py
debug-unified-logging-output/EEG-FM/EEGPT/downstream_tueg/run_class_finetuning_EEGPT_change.py
debug-unified-logging-output/EEG-FM/EEGPT/downstream_tueg/run_class_finetuning_EEGPT_change_tuev.py
debug-unified-logging-output/EEG-FM/Biot/run_binary_supervised.py
debug-unified-logging-output/EEG-FM/Biot/run_multiclass_supervised.py

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
import json
import sys
from pathlib import Path

The merge preserved:
-Benchmark/H5 loader connector logic from the main EEG-FM
-Unified output/logging/checkpoint behavior from debug-unified-logging-output
-Consistent log.txt schema

Note:
schema = same metric columns inside log.txt
layout = same folder/file path pattern on disk

# removed 
/nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output/EEG-FM

/nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/output diff.pdf

# commit and pushed to github
/nicoletye/workspace/Noise-Robustness-Evaluation-Protocol
