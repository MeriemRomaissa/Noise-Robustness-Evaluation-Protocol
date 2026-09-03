history record of what I changed to this repo this month:

# Checked YAML scripts
Lora and freeze backbone options were already added to /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/Benchmark/Config

# modified LORA folder
/nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/Benchmark/FinetuningStrategy/

removed /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/Benchmark/choose_StudyCase/Finetuning.py

Added six model-specific LoRA helper scripts: lora_labram.py, lora_eegpt.py, lora_biot.py, lora_cbramod.py, lora_csbrain.py, lora_codebrain.py. Added _lora_common.py.

Kept LoRA architecture-safe: no original EEG-FM model architecture rewrite

Kept LoRA defaults: Temporal Conv2d LoRA remains at rank 4, while attention/MLP LoRA remains at rank 2, alpha 8.0, layers all, init scale 0.01.

# standardized output
The current code now fulfills the standardized output contract for all three strategies:
-full_finetune
-freeze_backbone
-lora

saved standardized output under /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/Benchmark/Outputs

Each output folder contains:
<model>/
├── log.txt
├── checkpoint.pth
├── checkpoint-best.pth
└── strategy/             # LoRA only
    ├── lora_targets.json
    ├── trainable_params.json
    └── trainable_params.txt

All six log records:
- Are valid JSON.
- Have exactly the same 20 metric fields.
- Represent epochs 0 and 1.
- Use null only for train_min_lr and train_loss_scale, as intended.
- Have no missing or unexpected fields.

All checkpoints contain the standardized keys:
args, epoch, model, optimizer

# modified inference scripts
added lora and freeze backbone to inference scripts

for example, to evaluate LoRA for all six EEG-FMs, the inference scripts will import apply_lora_strategy(...), apply it before loading LoRA checkpoint weights, then run inference. Same for freeze backbone.

build original model
→ apply checkpoint strategy
→ create LoRA structure or freeze backbone
→ load checkpoint state_dict
→ run NMT OOD inference
→ compare against full fine-tuning and EEGNet

# updated environment
Previous environment was incompatible to run cbramod, csbrain and codebrain.

umap-learn was downgraded from 0.5.12 to 0.5.5

Installed combination that works:
h5py=3.16.0
umap-learn=0.5.5
scikit-learn=1.2.0
numpy=1.24.4
scipy=1.12.0
CUDA=True

# confirmed: rank =2 for attention/MLP and rank =4 for temporal Conv2d in lora
in /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/Benchmark/FinetuningStrategy/_lora_common.py:

Now, kept centralized LoRA defaults: rank 2 for attention/MLP adapters, rank 4 for temporal Conv2d adapters, alpha 8.0, all eligible layers, and initialization scale 0.01. LoRA remains architecture-safe: original EEG-FM model definitions are unchanged, with adapters applied through runtime wrapping or weight parametrization.

Labram and CSBrain has additional default setting  rank=4 for temporal Conv2d.

| Model | Attention | MLP/feed-forward | Current LoRA targets |
|---|---|---|---|
| LaBraM | Yes, `qkv` attention | Yes, `fc1/fc2` | Attention + MLP + temporal Conv2d |
| EEGPT | Yes, `qkv` attention | Yes, `fc1/fc2` | Attention + MLP |
| BIOT | Yes, linear attention | Yes, transformer feed-forward | Attention + MLP |
| CBraMod | Yes, spatial/temporal `MultiheadAttention` | Yes, `linear1/linear2` | Attention + MLP |
| CSBrain | Yes, inter-region/window `MultiheadAttention` | Yes, `linear1/linear2` | Attention + MLP + temporal Conv2d |
| CodeBrain | Yes, `MultiheadAttention` in SSSM | No classic transformer MLP in its TUAB path | Attention only, plus trainable classifier |

# modified plot scripts
now include for freeze backbone in /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/Benchmark/Plotting/plot_nmt_ood_results.py.

# test run whole pipeline
test run the whole pipeline with labram with all 5 seeds using epoch=2, from inputing tuab .h5 subset to yaml, load config, load finetuning, run finetuning (wave 1: fullfientuning, wave 2: lora, wave 3: freeze backbone), use outputs checkpoints for testing NMT subset inference tests, then evaluate with plots, output png/pdf . show me the dependency link and provide terminal command.

# test run output
from labram:
/nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/Benchmark/Outputs

for labram, finetuning logs and checkpoints saved under:
/nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/Benchmark/Outputs/freeze backbone/labram;
/nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/Benchmark/Outputs/full finetuning/labram;
/nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/Benchmark/Outputs/lora/labram;

for labram, NMT inference results saved under /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/Benchmark/Outputs/nmt_ood/labram:
- seed_*/log.txt
- seed_*/metrics.json
- seed_*/nmt_ood_results.json

for labram, plots saved under /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/Benchmark/Outputs/nmt_ood_plots/labram;

Note:
/nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/Benchmark/Outputs/nmt_ood/labram/seed_0/lora/strategy includes LoRA metadata created while rebuilding the LoRA parametrization before loading the checkpoint:
- lora_targets.json;
- trainable_params.json;
- trainable_params.txt

test run succeeded for labram.

# dependency link
Benchmark/Config/labram.yaml
├── TUAB H5 path and split index
├── pretrained LaBraM checkpoint
├── training defaults
└── NMT inference paths
        ↓
Benchmark/LoaderTraining/run_training_labram.py
        ↓ launches
EEG-FM/Labram/run_class_finetuning.py
├── loads YAML through _config_defaults()
├── reads H5 using Benchmark/DataLoader/loader_common.py
├── constructs original LaBraM model through get_models()
├── loads EEG-FM/Labram/checkpoints/labram-base.pth
└── calls Benchmark/FinetuningStrategy/lora_labram.py
        ├── full_finetune: all original parameters trainable
        ├── lora: architecture-safe parametrizations
        └── freeze_backbone: classifier/head trainable
        ↓
EEG-FM/Labram/engine_for_finetuning.py
        ↓
Benchmark/Outputs/{strategy}/labram/seed_{seed}/
├── log.txt
├── checkpoint.pth
└── checkpoint-best.pth
        ↓
Benchmark/Inference/run_inference_labram.py
├── rebuilds each strategy structure
├── loads each checkpoint-best.pth
├── evaluates the NMT subset
└── writes nmt_ood_results.json
        ↓
Benchmark/Plotting/plot_nmt_ood_results.py
        ↓
PNG + PDF plots

# modified inference scripts
comparison in inference scripts now include:
| Comparison | Question answered |
|---|---|
| `eegnet_minus_ft` | How does full fine-tuning compare with the baseline? |
| `lora_minus_ft` | Is LoRA better than full fine-tuning? |
| `lora_minus_freeze_backbone` | Is LoRA better than freeze backbone? |
| `lora_minus_eegnet` | Is LoRA better than EEGNet? |
| `freeze_backbone_minus_ft` | Is freeze backbone better than full fine-tuning? |
| `freeze_backbone_minus_eegnet` | Is freeze backbone better than EEGNet? |

# modified plot scripts
Plots include code to show seed=0, 42, 123, 256, 512

# example terminal command for training and finetuning
Consistent resource settings in AI station: 30/60 CPUs; 3 GPUs

this launches one strategy per GPU. Each GPU processes the five seeds sequentially.

cd /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol
set -u

ROOT=/nicoletye/workspace/Noise-Robustness-Evaluation-Protocol
PY=/nicoletye/venvs/eegpt_labram/bin/python

run_eegpt_wave() {
  STRATEGY="$1"
  FOLDER="$2"
  GPU="$3"

  for SEED in 0 42 123 256 512; do
    OUT="$ROOT/Benchmark/Outputs/$FOLDER/eegpt/seed_${SEED}"
    rm -rf "$OUT"
    mkdir -p "$OUT"

    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES="$GPU" \
    "$PY" Benchmark/LoaderTraining/run_training_eegpt.py \
      --config Benchmark/Config/eegpt.yaml \
      --finetune_strategy "$STRATEGY" \
      --output_dir "$OUT" \
      --epochs 2 \
      --warmup_epochs 0 \
      --train_samples 64 \
      --validation_samples 32 \
      --test_samples 32 \
      --batch_size 8 \
      --num_workers 0 \
      --seed "$SEED" \
      --device cuda:0 \
      --no_auto_resume \
      2>&1 | tee "$OUT/console.log"
  done
}

run_eegpt_wave full_finetune "full finetuning" 0 &
PID_FT=$!

run_eegpt_wave lora "lora" 1 &
PID_LORA=$!

run_eegpt_wave freeze_backbone "freeze backbone" 2 &
PID_FREEZE=$!

wait "$PID_FT"
wait "$PID_LORA"
wait "$PID_FREEZE"


# example terminal command for running NMT OOD inference
cd /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol
set -euo pipefail

ROOT=/nicoletye/workspace/Noise-Robustness-Evaluation-Protocol
PY=/nicoletye/venvs/eegpt_labram/bin/python

for SEED in 0 42 123 256 512; do
  OUT="$ROOT/Benchmark/Outputs/nmt_ood/labram/seed_${SEED}"
  mkdir -p "$OUT"

  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  PYTHONUNBUFFERED=1 \
  CUDA_VISIBLE_DEVICES=0 \
  "$PY" Benchmark/Inference/run_inference_labram.py \
    --config Benchmark/Config/labram.yaml \
    --ckpt_ft "$ROOT/Benchmark/Outputs/full finetuning/labram/seed_${SEED}/checkpoint-best.pth" \
    --ckpt_freeze_backbone "$ROOT/Benchmark/Outputs/freeze backbone/labram/seed_${SEED}/checkpoint-best.pth" \
    --ckpt_lora "$ROOT/Benchmark/Outputs/lora/labram/seed_${SEED}/checkpoint-best.pth" \
    --nmt_dir /nicoletye/workspace/nmt_scalp_eeg_labram_ood/test \
    --eegnet_json Benchmark/Config/eegnet_nmt_baseline.json \
    --output_dir "$OUT" \
    --max_samples 32 \
    --device cuda:0 \
    2>&1 | tee "$OUT/console.log"
done

# example terminal command for generating plots
cd /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol

/nicoletye/venvs/eegpt_labram/bin/python \
  Benchmark/Plotting/plot_nmt_ood_results.py \
  --input_glob \
    'Benchmark/Outputs/nmt_ood/labram/seed_*/nmt_ood_results.json' \
  --output_dir \
    Benchmark/Outputs/nmt_ood_plots/labram

# test run whole pipeline for all 6 EEG Fms
all 6 EEG FMs test run succeeded.

EEGPT Dependency
Benchmark/Config/eegpt.yaml
  -> Benchmark/LoaderTraining/run_training_eegpt.py
  -> EEG-FM/EEGPT/downstream_tueg/run_class_finetuning_EEGPT_change.py
     -> Benchmark/DataLoader/loader_common.py
     -> original EEGPTClassifier architecture
     -> original pretrained EEGPT checkpoint
     -> Benchmark/FinetuningStrategy/lora_eegpt.py
     -> inline train_one_epoch() and evaluate()
  -> Benchmark/Outputs/{strategy}/eegpt/seed_{seed}/
  -> Benchmark/Inference/run_inference_eegpt.py
  -> Benchmark/Outputs/nmt_ood/eegpt/seed_{seed}/
  -> Benchmark/Plotting/plot_nmt_ood_results.py
  -> Benchmark/Outputs/nmt_ood_plots/eegpt/

# added new folder and files
Renamed previous README file as /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/README_26Aug2026.md

Added new README file as /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/README_02Sep2026.md

Added 6 files under /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/Example for reference

# Commit and pushed to Github
/nicoletye/workspace/Noise-Robustness-Evaluation-Protocol



