# LaBraM two-epoch pipeline example

This smoke test uses 64 TUAB training samples, 32 validation samples, 32 test
samples, 32 NMT OOD samples, and seeds `0`, `42`, `123`, `256`, and `512`.

## Dependency flow

```text
Benchmark/Config/labram.yaml
  -> Benchmark/LoaderTraining/run_training_labram.py
  -> EEG-FM/Labram/run_class_finetuning.py
     -> Benchmark/DataLoader/loader_common.py
     -> original LaBraM model and pretrained checkpoint
     -> Benchmark/FinetuningStrategy/lora_labram.py
     -> EEG-FM/Labram/engine_for_finetuning.py
  -> Benchmark/Outputs/{strategy}/labram/seed_{seed}/
  -> Benchmark/Inference/run_inference_labram.py
  -> Benchmark/Outputs/nmt_ood/labram/seed_{seed}/
  -> Benchmark/Plotting/plot_nmt_ood_results.py
  -> Benchmark/Outputs/nmt_ood_plots/labram/
```

## Train three strategies

One strategy runs on each GPU; its five seeds run sequentially.

```bash
cd /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol
set -uo pipefail

ROOT=/nicoletye/workspace/Noise-Robustness-Evaluation-Protocol
PY=/nicoletye/venvs/eegpt_labram/bin/python

rm -rf \
  "$ROOT/Benchmark/Outputs/full finetuning/labram" \
  "$ROOT/Benchmark/Outputs/freeze backbone/labram" \
  "$ROOT/Benchmark/Outputs/lora/labram"

run_labram_wave() {
  STRATEGY="$1"
  FOLDER="$2"
  GPU="$3"

  for SEED in 0 42 123 256 512; do
    OUT="$ROOT/Benchmark/Outputs/$FOLDER/labram/seed_${SEED}"
    mkdir -p "$OUT"

    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES="$GPU" \
    "$PY" Benchmark/LoaderTraining/run_training_labram.py \
      --config Benchmark/Config/labram.yaml \
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
      --abs_pos_emb \
      --no_auto_resume \
      -- --save_ckpt_freq 1 \
      2>&1 | tee "$OUT/console.log"
  done
}

run_labram_wave full_finetune "full finetuning" 0 &
PID_FT=$!
run_labram_wave lora "lora" 1 &
PID_LORA=$!
run_labram_wave freeze_backbone "freeze backbone" 2 &
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

rm -rf "$ROOT/Benchmark/Outputs/nmt_ood/labram"

for START in 0 3; do
  for OFFSET in 0 1 2; do
    INDEX=$((START + OFFSET))
    [ "$INDEX" -ge "${#SEEDS[@]}" ] && continue

    SEED="${SEEDS[$INDEX]}"
    OUT="$ROOT/Benchmark/Outputs/nmt_ood/labram/seed_${SEED}"
    mkdir -p "$OUT"

    (
      PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
      PYTHONUNBUFFERED=1 \
      CUDA_VISIBLE_DEVICES="$OFFSET" \
      "$PY" Benchmark/Inference/run_inference_labram.py \
        --config Benchmark/Config/labram.yaml \
        --ckpt_ft "$ROOT/Benchmark/Outputs/full finetuning/labram/seed_${SEED}/checkpoint-best.pth" \
        --ckpt_freeze_backbone "$ROOT/Benchmark/Outputs/freeze backbone/labram/seed_${SEED}/checkpoint-best.pth" \
        --ckpt_lora "$ROOT/Benchmark/Outputs/lora/labram/seed_${SEED}/checkpoint-best.pth" \
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

rm -rf Benchmark/Outputs/nmt_ood_plots/labram

/nicoletye/venvs/eegpt_labram/bin/python \
  Benchmark/Plotting/plot_nmt_ood_results.py \
  --input_glob \
    'Benchmark/Outputs/nmt_ood/labram/seed_*/nmt_ood_results.json' \
  --output_dir \
    Benchmark/Outputs/nmt_ood_plots/labram
```

Remove `--max_samples 32` to evaluate the complete NMT OOD dataset.
