# BIOT two-epoch pipeline example

This smoke test uses 64 TUAB training samples, 32 validation samples, 32 test
samples, 32 NMT OOD samples, and seeds `0`, `42`, `123`, `256`, and `512`.

## Dependency flow

```text
Benchmark/Config/biot.yaml
  -> Benchmark/LoaderTraining/run_training_biot.py
  -> EEG-FM/Biot/run_binary_supervised.py
     -> Benchmark/DataLoader/loader_common.py
     -> original BIOTClassifier architecture and pretrained checkpoint
     -> Benchmark/FinetuningStrategy/lora_biot.py
     -> original PyTorch Lightning training flow
  -> Benchmark/Outputs/{strategy}/biot/seed_{seed}/
  -> Benchmark/Inference/run_inference_biot.py
  -> Benchmark/Outputs/nmt_ood/biot/seed_{seed}/
  -> Benchmark/Plotting/plot_nmt_ood_results.py
  -> Benchmark/Outputs/nmt_ood_plots/biot/
```

## Train three strategies

One strategy runs on each GPU; its five seeds run sequentially. The runner adds
`--benchmark_output` so Lightning writes the standardized Benchmark layout.

```bash
cd /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol
set -uo pipefail

ROOT=/nicoletye/workspace/Noise-Robustness-Evaluation-Protocol
PY=/nicoletye/venvs/biot_env/bin/python

rm -rf \
  "$ROOT/Benchmark/Outputs/full finetuning/biot" \
  "$ROOT/Benchmark/Outputs/freeze backbone/biot" \
  "$ROOT/Benchmark/Outputs/lora/biot"

run_biot_wave() {
  STRATEGY="$1"
  FOLDER="$2"
  GPU="$3"

  for SEED in 0 42 123 256 512; do
    OUT="$ROOT/Benchmark/Outputs/$FOLDER/biot/seed_${SEED}"
    mkdir -p "$OUT"

    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES="$GPU" \
    "$PY" Benchmark/LoaderTraining/run_training_biot.py \
      --config Benchmark/Config/biot.yaml \
      --finetune_strategy "$STRATEGY" \
      --output_dir "$OUT" \
      --epochs 2 \
      --train_samples 64 \
      --validation_samples 32 \
      --test_samples 32 \
      --batch_size 8 \
      --num_workers 0 \
      --seed "$SEED" \
      --device cuda \
      2>&1 | tee "$OUT/console.log"
  done
}

run_biot_wave full_finetune "full finetuning" 0 &
PID_FT=$!
run_biot_wave lora "lora" 1 &
PID_LORA=$!
run_biot_wave freeze_backbone "freeze backbone" 2 &
PID_FREEZE=$!

wait "$PID_FT"
wait "$PID_LORA"
wait "$PID_FREEZE"
```

## Run NMT OOD inference

```bash
cd /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol

ROOT=/nicoletye/workspace/Noise-Robustness-Evaluation-Protocol
PY=/nicoletye/venvs/biot_env/bin/python
SEEDS=(0 42 123 256 512)

rm -rf "$ROOT/Benchmark/Outputs/nmt_ood/biot"

for START in 0 3; do
  for OFFSET in 0 1 2; do
    INDEX=$((START + OFFSET))
    [ "$INDEX" -ge "${#SEEDS[@]}" ] && continue

    SEED="${SEEDS[$INDEX]}"
    OUT="$ROOT/Benchmark/Outputs/nmt_ood/biot/seed_${SEED}"
    mkdir -p "$OUT"

    (
      PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
      PYTHONUNBUFFERED=1 \
      CUDA_VISIBLE_DEVICES="$OFFSET" \
      "$PY" Benchmark/Inference/run_inference_biot.py \
        --config Benchmark/Config/biot.yaml \
        --checkpoint "$ROOT/Benchmark/Outputs/full finetuning/biot/seed_${SEED}/checkpoint-best.pth" \
        --ckpt_freeze_backbone "$ROOT/Benchmark/Outputs/freeze backbone/biot/seed_${SEED}/checkpoint-best.pth" \
        --ckpt_lora "$ROOT/Benchmark/Outputs/lora/biot/seed_${SEED}/checkpoint-best.pth" \
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

rm -rf Benchmark/Outputs/nmt_ood_plots/biot

/nicoletye/venvs/biot_env/bin/python \
  Benchmark/Plotting/plot_nmt_ood_results.py \
  --input_glob \
    'Benchmark/Outputs/nmt_ood/biot/seed_*/nmt_ood_results.json' \
  --output_dir \
    Benchmark/Outputs/nmt_ood_plots/biot
```

Remove `--max_samples 32` to evaluate the complete NMT OOD dataset.
