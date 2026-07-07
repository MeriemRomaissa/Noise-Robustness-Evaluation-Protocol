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
OUTPUT_ROOT="outputs/cbramod_unified60_ablation_seed42_2gpu"
SMOKE_ROOT="outputs/cbramod_unified60_ablation_seed42_2gpu_smoke"
REPORT_DIR="reports/all6_original_repo_vs_90job_unified60_matchedsplit_epoch15_seed42_v1"
LAUNCH_LOG="$REPORT_DIR/cbramod_unified60_ablation_seed42_2gpu_launch.log"
SUMMARY_SCRIPT="scripts/eegfm_adapters/summarize_cbramod_unified60_ablation_seed42.py"
PYTHON="/nicoletye/venvs/cbramod_csbrain/bin/python"
JOB_SCRIPT="scripts/eegfm_adapters/run_cbramod_unified60_meriem_strict_job.py"

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
  local mapping="$5"
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
    --cbramod_input_variant "$input_variant"
    --cbramod_recipe "$recipe"
    --cbramod_checkpoint_mapping "$mapping"
  )
  local text
  text="$(command_text "${cmd[@]}")"
  log "JOB $name gpu=$gpu input=$input_variant recipe=$recipe checkpoint_mapping=$mapping"
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

log "CBraMod unified60 ablation seed42 full launcher"
log "GPU_LIST=$GPU_LIST parsed=${GPUS[*]} MAX_PARALLEL=$MAX_PARALLEL DRY_RUN=$DRY_RUN"
if [[ "$DRY_RUN" != "1" && ! -f "$SMOKE_ROOT/C3_adapter_plus_author_recipe/metrics.json" ]]; then
  log "WARNING: smoke C3 metrics not found at $SMOKE_ROOT/C3_adapter_plus_author_recipe/metrics.json"
  log "WARNING: run scripts/eegfm_adapters/run_cbramod_unified60_ablation_seed42_2gpu_smoke.sh first if you want checkpoint smoke validation."
fi

run_job 0 C0_current_unified_reference current unified backbone_prefix
run_job 1 C1_adapter_only bipolar_div100 unified backbone_prefix
run_job 2 C2_author_recipe_only current author backbone_prefix
run_job 3 C3_adapter_plus_author_recipe bipolar_div100 author backbone_prefix
run_job 4 C4_checkpoint_off_control bipolar_div100 author none

if [[ "$DRY_RUN" != "1" ]]; then
  wait
  /nicoletye/venvs/eegnet/bin/python "$SUMMARY_SCRIPT" --output_root "$OUTPUT_ROOT" | tee -a "$LAUNCH_LOG"
fi

log "DONE"
