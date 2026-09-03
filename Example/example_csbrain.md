# CSBrain two-epoch pipeline example

This smoke test uses 8 TUAB training, validation, and test samples, 32 NMT OOD
samples, and seeds `0`, `42`, `123`, `256`, and `512`.

## Dependency flow

```text
Benchmark/Config/csbrain.yaml
  -> Benchmark/LoaderTraining/run_training_csbrain.py
  -> EEG-FM/CSBrain/finetune_main.py
     -> Benchmark/DataLoader/loader_common.py
     -> original CSBrain TUAB model and pretrained checkpoint
     -> Benchmark/FinetuningStrategy/lora_csbrain.py
     -> EEG-FM/CSBrain/finetune_trainer.py and finetune_evaluator.py
  -> Benchmark/Outputs/{strategy}/csbrain/seed_{seed}/
  -> Benchmark/Inference/run_inference_csbrain.py
  -> Benchmark/Outputs/nmt_ood/csbrain/seed_{seed}/
  -> Benchmark/Plotting/plot_nmt_ood_results.py
  -> Benchmark/Outputs/nmt_ood_plots/csbrain/
```

## Train three strategies

One strategy runs on each GPU; its five seeds run sequentially.

```bash
cd /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol
set -uo pipefail

ROOT=/nicoletye/workspace/Noise-Robustness-Evaluation-Protocol
PY=/nicoletye/venvs/cbramod_csbrain/bin/python

rm -rf \
  "$ROOT/Benchmark/Outputs/full finetuning/csbrain" \
  "$ROOT/Benchmark/Outputs/freeze backbone/csbrain" \
  "$ROOT/Benchmark/Outputs/lora/csbrain"

run_csbrain_wave() {
  STRATEGY="$1"
  FOLDER="$2"
  GPU="$3"

  for SEED in 0 42 123 256 512; do
    OUT="$ROOT/Benchmark/Outputs/$FOLDER/csbrain/seed_${SEED}"
    mkdir -p "$OUT"

    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4 \
    CUDA_VISIBLE_DEVICES="$GPU" \
    "$PY" Benchmark/LoaderTraining/run_training_csbrain.py \
      --config Benchmark/Config/csbrain.yaml \
      --finetune_strategy "$STRATEGY" \
      --output_dir "$OUT" \
      --epochs 2 \
      --train_samples 8 \
      --validation_samples 8 \
      --test_samples 8 \
      --batch_size 2 \
      --num_workers 0 \
      --seed "$SEED" \
      --cuda 0 \
      2>&1 | tee "$OUT/console.log"
  done
}

run_csbrain_wave full_finetune "full finetuning" 0 &
PID_FT=$!
run_csbrain_wave lora "lora" 1 &
PID_LORA=$!
run_csbrain_wave freeze_backbone "freeze backbone" 2 &
PID_FREEZE=$!

wait "$PID_FT"
wait "$PID_LORA"
wait "$PID_FREEZE"
```

## Run NMT OOD inference

```bash
cd /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol

ROOT=/nicoletye/workspace/Noise-Robustness-Evaluation-Protocol
PY=/nicoletye/venvs/cbramod_csbrain/bin/python
SEEDS=(0 42 123 256 512)

rm -rf "$ROOT/Benchmark/Outputs/nmt_ood/csbrain"

for START in 0 3; do
  for OFFSET in 0 1 2; do
    INDEX=$((START + OFFSET))
    [ "$INDEX" -ge "${#SEEDS[@]}" ] && continue

    SEED="${SEEDS[$INDEX]}"
    OUT="$ROOT/Benchmark/Outputs/nmt_ood/csbrain/seed_${SEED}"
    mkdir -p "$OUT"

    (
      PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
      PYTHONUNBUFFERED=1 \
      CUDA_VISIBLE_DEVICES="$OFFSET" \
      "$PY" Benchmark/Inference/run_inference_csbrain.py \
        --config Benchmark/Config/csbrain.yaml \
        --checkpoint "$ROOT/Benchmark/Outputs/full finetuning/csbrain/seed_${SEED}/checkpoint-best.pth" \
        --ckpt_freeze_backbone "$ROOT/Benchmark/Outputs/freeze backbone/csbrain/seed_${SEED}/checkpoint-best.pth" \
        --ckpt_lora "$ROOT/Benchmark/Outputs/lora/csbrain/seed_${SEED}/checkpoint-best.pth" \
        --nmt_dir /nicoletye/workspace/nmt_scalp_eeg_labram_ood/test \
        --eegnet_json Benchmark/Config/eegnet_nmt_baseline.json \
        --output_dir "$OUT" \
        --max_samples 32 \
        --batch_size 2 \
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

rm -rf Benchmark/Outputs/nmt_ood_plots/csbrain

/nicoletye/venvs/cbramod_csbrain/bin/python \
  Benchmark/Plotting/plot_nmt_ood_results.py \
  --input_glob \
    'Benchmark/Outputs/nmt_ood/csbrain/seed_*/nmt_ood_results.json' \
  --output_dir \
    Benchmark/Outputs/nmt_ood_plots/csbrain
```

Remove `--max_samples 32` to evaluate the complete NMT OOD dataset.
