#!/usr/bin/env bash
set -euo pipefail

cd /nicoletye/workspace/unified_tuab

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

GPU_LIST="${GPU_LIST:-0,1}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
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
OUTPUT_ROOT="outputs/codebrain_unified60_ablation_seed42_2gpu"
REPORT_DIR="reports/codebrain_unified60_ablation_seed42_2gpu"
SMOKE_ROOT="outputs/codebrain_unified60_ablation_seed42_2gpu_smoke"
LAUNCH_LOG="$REPORT_DIR/launch.log"
SUMMARY_SCRIPT="scripts/eegfm_adapters/summarize_codebrain_unified60_ablation_seed42.py"
PYTHON="/nicoletye/venvs/cbramod_csbrain/bin/python"
JOB_SCRIPT="scripts/eegfm_adapters/run_codebrain_unified60_meriem_strict_job.py"

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
      log "A background ablation job exited with rc=$rc"
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
    --num_workers 2
    --pin_memory
    --index_npz "$INDEX_NPZ"
    --h5 "$H5_PATH"
    --train_n 8192
    --val_n 2048
    --test_n 2048
    --codebrain_input_variant "$input_variant"
    --codebrain_recipe "$recipe"
    --codebrain_use_pretrained "$use_pretrained"
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

log "CodeBrain unified60 ablation seed42 full launcher"
log "GPU_LIST=$GPU_LIST parsed=${GPUS[*]} MAX_PARALLEL=$MAX_PARALLEL DRY_RUN=$DRY_RUN"
if [[ "$DRY_RUN" != "1" && ! -f "$SMOKE_ROOT/C3_bipolar_author_recipe/metrics.json" ]]; then
  log "WARNING: smoke C3 metrics not found at $SMOKE_ROOT/C3_bipolar_author_recipe/metrics.json"
  log "WARNING: run scripts/eegfm_adapters/run_codebrain_unified60_ablation_seed42_2gpu_smoke.sh first."
fi

run_job 0 C0_endpoint_original_recipe endpoint_current original_wrapper 1
run_job 1 C1_bipolar_original_recipe bipolar_div100 original_wrapper 1
run_job 2 C2_endpoint_author_recipe endpoint_current author_default 1
run_job 3 C3_bipolar_author_recipe bipolar_div100 author_default 1
run_job 4 C4_bipolar_author_recipe_checkpoint_off bipolar_div100 author_default 0

if [[ "$DRY_RUN" != "1" ]]; then
  wait
  /nicoletye/venvs/eegnet/bin/python "$SUMMARY_SCRIPT" --output_root "$OUTPUT_ROOT" --report_dir "$REPORT_DIR" | tee -a "$LAUNCH_LOG"
fi

log "DONE"
