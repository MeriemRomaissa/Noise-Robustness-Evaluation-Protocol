#!/usr/bin/env bash
set -euo pipefail

cd /nicoletye/workspace/unified_tuab

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

GPU_LIST="${GPU_LIST:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DRY_RUN="${DRY_RUN:-0}"

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "ERROR: GPU_LIST is empty" >&2
  exit 2
fi
if [[ "$MAX_PARALLEL" -gt "${#GPUS[@]}" ]]; then
  MAX_PARALLEL="${#GPUS[@]}"
fi

INDEX_NPZ="reports/all6_unified60_subset_epoch15_5seed_1gpu_dev_v1/all6_fixed_subset_seed42_index.npz"
H5_PATH="data/canonical_tuab_full.h5"
OUTPUT_ROOT="outputs/biot_unified60_ablation_seed42_1gpu"
REPORT_DIR="reports/biot_unified60_ablation_seed42_1gpu"
SMOKE_ROOT="outputs/biot_unified60_ablation_seed42_1gpu_smoke"
LAUNCH_LOG="$REPORT_DIR/launch.log"
SUMMARY_SCRIPT="scripts/eegfm_adapters/summarize_biot_unified60_ablation_seed42.py"
PYTHON="/nicoletye/venvs/biot_env/bin/python"
JOB_SCRIPT="scripts/eegfm_adapters/run_biot_unified60_meriem_strict_job.py"

mkdir -p "$OUTPUT_ROOT" "$REPORT_DIR"
: > "$LAUNCH_LOG"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LAUNCH_LOG"
}

command_text() {
  /nicoletye/venvs/eegnet/bin/python - "$@" <<'PY'
import shlex, sys
print(shlex.join(sys.argv[1:]))
PY
}

wait_for_slot() {
  local rc
  while [[ "$(jobs -pr | wc -l)" -ge "$MAX_PARALLEL" ]]; do
    set +e
    wait -n
    rc=$?
    set -e
    if [[ "$rc" -ne 0 ]]; then
      log "A background BIOT ablation job exited with rc=$rc"
    fi
  done
}

run_job() {
  local job_index="$1"
  local name="$2"
  local input_variant="$3"
  local recipe="$4"
  local use_pretrained="$5"
  local gpu="${GPUS[$((job_index % ${#GPUS[@]}))]}"
  local out_dir="$OUTPUT_ROOT/$name"
  local terminal_log="$out_dir/terminal.log"
  mkdir -p "$out_dir"
  local cmd=(
    "$PYTHON" "$JOB_SCRIPT"
    --strategy full_finetune
    --seed 42
    --output_dir "$out_dir"
    --epochs 15
    --batch_size 64
    --lr 0.0001
    --weight_decay 0.0005
    --device cuda:0
    --num_workers "$NUM_WORKERS"
    --pin_memory
    --index_npz "$INDEX_NPZ"
    --h5 "$H5_PATH"
    --train_n 8192
    --val_n 2048
    --test_n 2048
    --biot_input_variant "$input_variant"
    --biot_recipe "$recipe"
    --biot_use_pretrained "$use_pretrained"
    --token_size 200
    --hop_length 100
  )
  local text
  text="$(command_text "${cmd[@]}")"
  log "JOB $name gpu=$gpu input=$input_variant recipe=$recipe pretrained=$use_pretrained"
  log "COMMAND $text"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '%s\n' "$text" > "$out_dir/command.txt"
    return 0
  fi
  wait_for_slot
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "${cmd[@]}" 2>&1 | tee "$terminal_log"
  ) &
}

log "BIOT unified60 ablation seed42 full launcher"
log "GPU_LIST=$GPU_LIST parsed=${GPUS[*]} MAX_PARALLEL=$MAX_PARALLEL NUM_WORKERS=$NUM_WORKERS DRY_RUN=$DRY_RUN"
if [[ "$DRY_RUN" != "1" && ! -f "$SMOKE_ROOT/B1_bipolar_scratch_recipe/metrics.json" ]]; then
  log "WARNING: smoke B1 metrics not found at $SMOKE_ROOT/B1_bipolar_scratch_recipe/metrics.json"
  log "WARNING: run scripts/eegfm_adapters/run_biot_unified60_ablation_seed42_1gpu_smoke.sh first."
fi

run_job 0 B0_endpoint_scratch_recipe current github_vanilla 0
run_job 1 B1_bipolar_scratch_recipe bipolar_qnorm github_vanilla 0
run_job 2 B2_endpoint_pretrained_recipe current github_pretrained 1
run_job 3 B3_bipolar_pretrained_recipe bipolar_qnorm github_pretrained 1

if [[ "$DRY_RUN" != "1" ]]; then
  wait
  /nicoletye/venvs/eegnet/bin/python "$SUMMARY_SCRIPT" --output_root "$OUTPUT_ROOT" --report_dir "$REPORT_DIR" | tee -a "$LAUNCH_LOG"
fi

log "DONE"
