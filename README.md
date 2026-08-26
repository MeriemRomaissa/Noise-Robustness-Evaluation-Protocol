What I changed to this repo this past week:

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

# added LORA folder
/nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/Benchmark/StudyCase/Finetuning/finetuning_strategies.py 

removed /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/Benchmark/choose_StudyCase/Finetuning.py

Added six model-specific LoRA helpers in finetuning_strategies.py: lora_labram, lora_eegpt, lora_biot, lora_cbramod, lora_csbrain, lora_codebrain.

Kept LoRA architecture-safe: no original EEG-FM model architecture rewrite, only parameter-efficient wrapping/parametrization.

Kept LoRA defaults centralized: rank 2, alpha 8.0, layers all, init scale 0.01.

# will add lora to inference later
To evaluate LoRA for all six EEG-FMs later, the inference scripts will import apply_lora_strategy(...), apply it before loading LoRA checkpoint weights, then run inference. Should be no issue here.

