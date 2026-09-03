# EEGPT two-epoch pipeline example

This smoke test uses 64 TUAB training samples, 32 validation samples, 32 test
samples, 32 NMT OOD samples, and seeds `0`, `42`, `123`, `256`, and `512`.

## Dependency flow

```text
Benchmark/Config/eegpt.yaml
  -> Benchmark/LoaderTraining/run_training_eegpt.py
  -> EEG-FM/EEGPT/downstream_tueg/run_class_finetuning_EEGPT_change.py
     -> Benchmark/DataLoader/loader_common.py
     -> original EEGPTClassifier architecture and pretrained checkpoint
     -> Benchmark/FinetuningStrategy/lora_eegpt.py
     -> inline train_one_epoch() and evaluate()
  -> Benchmark/Outputs/{strategy}/eegpt/seed_{seed}/
  -> Benchmark/Inference/run_inference_eegpt.py
  -> Benchmark/Outputs/nmt_ood/eegpt/seed_{seed}/
  -> Benchmark/Plotting/plot_nmt_ood_results.py
  -> Benchmark/Outputs/nmt_ood_plots/eegpt/
```

## Train three strategies

One strategy runs on each GPU; its five seeds run sequentially.

```bash
cd /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol
set -uo pipefail

ROOT=/nicoletye/workspace/Noise-Robustness-Evaluation-Protocol
PY=/nicoletye/venvs/eegpt_labram/bin/python

rm -rf \
  "$ROOT/Benchmark/Outputs/full finetuning/eegpt" \
  "$ROOT/Benchmark/Outputs/freeze backbone/eegpt" \
  "$ROOT/Benchmark/Outputs/lora/eegpt"

run_eegpt_wave() {
  STRATEGY="$1"
  FOLDER="$2"
  GPU="$3"

  for SEED in 0 42 123 256 512; do
    OUT="$ROOT/Benchmark/Outputs/$FOLDER/eegpt/seed_${SEED}"
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
```

## Run NMT OOD inference

```bash
cd /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol

ROOT=/nicoletye/workspace/Noise-Robustness-Evaluation-Protocol
PY=/nicoletye/venvs/eegpt_labram/bin/python
SEEDS=(0 42 123 256 512)

rm -rf "$ROOT/Benchmark/Outputs/nmt_ood/eegpt"

for START in 0 3; do
  for OFFSET in 0 1 2; do
    INDEX=$((START + OFFSET))
    [ "$INDEX" -ge "${#SEEDS[@]}" ] && continue

    SEED="${SEEDS[$INDEX]}"
    OUT="$ROOT/Benchmark/Outputs/nmt_ood/eegpt/seed_${SEED}"
    mkdir -p "$OUT"

    (
      PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
      PYTHONUNBUFFERED=1 \
      CUDA_VISIBLE_DEVICES="$OFFSET" \
      "$PY" Benchmark/Inference/run_inference_eegpt.py \
        --config Benchmark/Config/eegpt.yaml \
        --checkpoint "$ROOT/Benchmark/Outputs/full finetuning/eegpt/seed_${SEED}/checkpoint-best.pth" \
        --ckpt_freeze_backbone "$ROOT/Benchmark/Outputs/freeze backbone/eegpt/seed_${SEED}/checkpoint-best.pth" \
        --ckpt_lora "$ROOT/Benchmark/Outputs/lora/eegpt/seed_${SEED}/checkpoint-best.pth" \
        --nmt_dir /nicoletye/workspace/nmt_scalp_eeg_labram_ood/test \
        --eegnet_json Benchmark/Config/eegnet_nmt_baseline.json \
        --output_dir "$OUT" \
        --max_samples 32 \
        --batch_size 8 \
        --device cuda:0 \
        2>&1 | tee "$OUT/console.log"
    ) &
  done
  wait
done
```

## Generate plots

```bash
cd /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol

rm -rf Benchmark/Outputs/nmt_ood_plots/eegpt

/nicoletye/venvs/eegpt_labram/bin/python \
  Benchmark/Plotting/plot_nmt_ood_results.py \
  --input_glob \
    'Benchmark/Outputs/nmt_ood/eegpt/seed_*/nmt_ood_results.json' \
  --output_dir \
    Benchmark/Outputs/nmt_ood_plots/eegpt
```

Remove `--max_samples 32` to evaluate the complete NMT OOD dataset.
