# CBraMod two-epoch pipeline example

This smoke test uses 8 TUAB training, validation, and test samples, 32 NMT OOD
samples, and seeds `0`, `42`, `123`, `256`, and `512`.

## Dependency flow

```text
Benchmark/Config/cbramod.yaml
  -> Benchmark/LoaderTraining/run_training_cbramod.py
  -> EEG-FM/CBraMod/finetune_main.py
     -> Benchmark/DataLoader/loader_common.py
     -> original CBraMod TUAB model and pretrained checkpoint
     -> Benchmark/FinetuningStrategy/lora_cbramod.py
     -> EEG-FM/CBraMod/finetune_trainer.py and finetune_evaluator.py
  -> Benchmark/Outputs/{strategy}/cbramod/seed_{seed}/
  -> Benchmark/Inference/run_inference_cbramod.py
  -> Benchmark/Outputs/nmt_ood/cbramod/seed_{seed}/
  -> Benchmark/Plotting/plot_nmt_ood_results.py
  -> Benchmark/Outputs/nmt_ood_plots/cbramod/
```

## Train three strategies

One strategy runs on each GPU; its five seeds run sequentially.

```bash
cd /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol
set -uo pipefail

ROOT=/nicoletye/workspace/Noise-Robustness-Evaluation-Protocol
PY=/nicoletye/venvs/cbramod_csbrain/bin/python

rm -rf \
  "$ROOT/Benchmark/Outputs/full finetuning/cbramod" \
  "$ROOT/Benchmark/Outputs/freeze backbone/cbramod" \
  "$ROOT/Benchmark/Outputs/lora/cbramod"

run_cbramod_wave() {
  STRATEGY="$1"
  FOLDER="$2"
  GPU="$3"

  for SEED in 0 42 123 256 512; do
    OUT="$ROOT/Benchmark/Outputs/$FOLDER/cbramod/seed_${SEED}"
    mkdir -p "$OUT"

    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4 \
    CUDA_VISIBLE_DEVICES="$GPU" \
    "$PY" Benchmark/LoaderTraining/run_training_cbramod.py \
      --config Benchmark/Config/cbramod.yaml \
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

run_cbramod_wave full_finetune "full finetuning" 0 &
PID_FT=$!
run_cbramod_wave lora "lora" 1 &
PID_LORA=$!
run_cbramod_wave freeze_backbone "freeze backbone" 2 &
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

rm -rf "$ROOT/Benchmark/Outputs/nmt_ood/cbramod"

for START in 0 3; do
  for OFFSET in 0 1 2; do
    INDEX=$((START + OFFSET))
    [ "$INDEX" -ge "${#SEEDS[@]}" ] && continue

    SEED="${SEEDS[$INDEX]}"
    OUT="$ROOT/Benchmark/Outputs/nmt_ood/cbramod/seed_${SEED}"
    mkdir -p "$OUT"

    (
      PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
      PYTHONUNBUFFERED=1 \
      CUDA_VISIBLE_DEVICES="$OFFSET" \
      "$PY" Benchmark/Inference/run_inference_cbramod.py \
        --config Benchmark/Config/cbramod.yaml \
        --checkpoint "$ROOT/Benchmark/Outputs/full finetuning/cbramod/seed_${SEED}/checkpoint-best.pth" \
        --ckpt_freeze_backbone "$ROOT/Benchmark/Outputs/freeze backbone/cbramod/seed_${SEED}/checkpoint-best.pth" \
        --ckpt_lora "$ROOT/Benchmark/Outputs/lora/cbramod/seed_${SEED}/checkpoint-best.pth" \
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

rm -rf Benchmark/Outputs/nmt_ood_plots/cbramod

/nicoletye/venvs/cbramod_csbrain/bin/python \
  Benchmark/Plotting/plot_nmt_ood_results.py \
  --input_glob \
    'Benchmark/Outputs/nmt_ood/cbramod/seed_*/nmt_ood_results.json' \
  --output_dir \
    Benchmark/Outputs/nmt_ood_plots/cbramod
```

Remove `--max_samples 32` to evaluate the complete NMT OOD dataset.
