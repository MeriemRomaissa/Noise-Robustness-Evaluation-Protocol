#!/usr/bin/env bash
set -euo pipefail

cd /nicoletye/workspace/unified_tuab

MODELS=(labram biot eegpt cbramod csbrain codebrain)
STRATEGIES=(full_finetune linear_probe lora)
SEEDS=(0 42 123 256 512)

GPU_LIST="${GPU_LIST:-0,1,2}"
GPU_LIST_CLEAN="${GPU_LIST//[[:space:]]/}"
IFS=',' read -r -a GPUS <<< "$GPU_LIST_CLEAN"
if [[ "${#GPUS[@]}" -eq 0 || -z "${GPUS[0]:-}" ]]; then
  echo "ERROR: GPU_LIST is empty. Example: GPU_LIST=0,1,2" >&2
  exit 2
fi

DEFAULT_MAX_PARALLEL="${#GPUS[@]}"
MAX_PARALLEL="${MAX_PARALLEL:-$DEFAULT_MAX_PARALLEL}"
if [[ "$MAX_PARALLEL" -lt 1 ]]; then
  echo "ERROR: MAX_PARALLEL must be >= 1" >&2
  exit 2
fi
if [[ "$MAX_PARALLEL" -gt "${#GPUS[@]}" ]]; then
  echo "ERROR: MAX_PARALLEL=$MAX_PARALLEL exceeds GPU_LIST size=${#GPUS[@]} ($GPU_LIST_CLEAN)" >&2
  exit 2
fi

DRY_RUN="${DRY_RUN:-0}"
FRESH="${FRESH:-0}"
EPOCHS="${EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SUBSET_SEED="${SUBSET_SEED:-42}"
TRAIN_N="${TRAIN_N:-8192}"
VAL_N="${VAL_N:-2048}"
TEST_N="${TEST_N:-2048}"
LR="${LR:-0.0001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0005}"
LABRAM_LR="${LABRAM_LR:-0.0005}"
LABRAM_WEIGHT_DECAY="${LABRAM_WEIGHT_DECAY:-0.05}"
LABRAM_CHECKPOINT="/nicoletye/workspace/Noise Robustness Evaluation Protocol/EEG-FM/Labram/checkpoints/labram-base.pth"
MODEL_FILTER="${MODEL_FILTER:-}"
STRATEGY_FILTER="${STRATEGY_FILTER:-}"
SEED_FILTER="${SEED_FILTER:-}"
JOB_LIMIT="${JOB_LIMIT:-0}"

OUTPUT_ROOT="outputs/all6_unified60_subset_epoch15_5seed_1gpu_dev_v1"
REPORT_ROOT="reports/all6_unified60_subset_epoch15_5seed_1gpu_dev_v1"
STATUS_DIR="$OUTPUT_ROOT/task_status"
JOB_TABLE="$STATUS_DIR/job_status.tsv"
TOP_LOG="$OUTPUT_ROOT/task_terminal_3gpu.log"
INDEX_NPZ="$REPORT_ROOT/all6_fixed_subset_seed42_index.npz"
INDEX_SUMMARY="$REPORT_ROOT/all6_fixed_subset_seed42_index_summary.json"
SPLIT_CSV="reports/labram_exact_original_processed_split/canonical_h5_labram_exact_original_processed_split_index.csv"
H5_PATH="data/canonical_tuab_full.h5"

export PYTHONUNBUFFERED=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TMPDIR="${TMPDIR:-/nicoletye/workspace/tmp}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

if [[ "$FRESH" == "1" && "$DRY_RUN" != "1" && -d "$OUTPUT_ROOT" ]]; then
  echo "FRESH=1 deleting only new dev subset output root: $OUTPUT_ROOT"
  rm -rf "$OUTPUT_ROOT"
fi
mkdir -p "$TMPDIR" "$STATUS_DIR" "$REPORT_ROOT"

matches_filter() {
  local value="$1" filter="$2"
  if [[ -z "$filter" || "$filter" == "all" ]]; then
    return 0
  fi
  local filter_clean="${filter//[[:space:]]/}"
  local parts
  IFS=',' read -r -a parts <<< "$filter_clean"
  local part
  for part in "${parts[@]}"; do
    if [[ "$value" == "$part" ]]; then
      return 0
    fi
  done
  return 1
}

planned_job_count() {
  local count=0 model strategy seed
  for model in "${MODELS[@]}"; do
    matches_filter "$model" "$MODEL_FILTER" || continue
    for strategy in "${STRATEGIES[@]}"; do
      matches_filter "$strategy" "$STRATEGY_FILTER" || continue
      for seed in "${SEEDS[@]}"; do
        matches_filter "$seed" "$SEED_FILTER" || continue
        count=$((count + 1))
        if [[ "$JOB_LIMIT" -gt 0 && "$count" -ge "$JOB_LIMIT" ]]; then
          echo "$count"
          return 0
        fi
      done
    done
  done
  echo "$count"
}

: > "$TOP_LOG"

log() {
  echo "$*" | tee -a "$TOP_LOG"
}

command_text_for_cmd() {
  /nicoletye/venvs/eegnet/bin/python -c 'import shlex, sys; print(shlex.join(sys.argv[1:]))' "$@"
}

append_status() {
  local row="$1"
  {
    flock 9
    printf '%s\n' "$row" >> "$JOB_TABLE"
  } 9>"$STATUS_DIR/job_status.lock"
}

JOB_COUNT="$(planned_job_count)"
cat > "$STATUS_DIR/launcher_config.env" <<EOF
launcher=3gpu_parallel
output_root=$OUTPUT_ROOT
report_root=$REPORT_ROOT
gpu_list=$GPU_LIST_CLEAN
gpu_array=${GPUS[*]}
max_parallel=$MAX_PARALLEL
job_count=$JOB_COUNT
dry_run=$DRY_RUN
fresh=$FRESH
epochs=$EPOCHS
batch_size=$BATCH_SIZE
num_workers=$NUM_WORKERS
train_n=$TRAIN_N
val_n=$VAL_N
test_n=$TEST_N
subset_seed=$SUBSET_SEED
model_filter=$MODEL_FILTER
strategy_filter=$STRATEGY_FILTER
seed_filter=$SEED_FILTER
job_limit=$JOB_LIMIT
index_npz=$INDEX_NPZ
split_csv=$SPLIT_CSV
h5_path=$H5_PATH
labram_checkpoint=$LABRAM_CHECKPOINT
EOF

log "===== All-6 unified60 subset epoch15 5-seed 3GPU dev benchmark ====="
log "output_root=$OUTPUT_ROOT"
log "report_root=$REPORT_ROOT"
log "GPU_LIST=$GPU_LIST_CLEAN parsed_gpus=${GPUS[*]} MAX_PARALLEL=$MAX_PARALLEL"
log "DRY_RUN=$DRY_RUN FRESH=$FRESH EPOCHS=$EPOCHS BATCH_SIZE=$BATCH_SIZE NUM_WORKERS=$NUM_WORKERS"
log "subset train=$TRAIN_N val=$VAL_N test=$TEST_N subset_seed=$SUBSET_SEED"
log "filters MODEL_FILTER=${MODEL_FILTER:-all} STRATEGY_FILTER=${STRATEGY_FILTER:-all} SEED_FILTER=${SEED_FILTER:-all} JOB_LIMIT=$JOB_LIMIT"
log "job_count=$JOB_COUNT"
log "visible_nvidia_smi_gpus:"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader | tee "$STATUS_DIR/visible_gpus.txt" | tee -a "$TOP_LOG" || true
else
  log "nvidia-smi not found"
fi

if [[ ! -f "$INDEX_NPZ" ]]; then
  /nicoletye/venvs/eegnet/bin/python scripts/eegfm_adapters/build_all6_unified60_subset_epoch15_fixed_index.py \
    --split_csv "$SPLIT_CSV" \
    --output_npz "$INDEX_NPZ" \
    --summary_json "$INDEX_SUMMARY" \
    --train_n "$TRAIN_N" \
    --val_n "$VAL_N" \
    --test_n "$TEST_N" \
    --subset_seed "$SUBSET_SEED" | tee -a "$TOP_LOG"
fi

printf 'job_id\tmodel\tstrategy\tseed\tgpu\tstatus\toutput_dir\tterminal_log\tcommand\n' > "$JOB_TABLE"

cmd_for_job() {
  local model="$1" strategy="$2" seed="$3" out_dir="$4"
  case "$model" in
    labram)
      printf '%s\0' \
        /nicoletye/venvs/eegpt_labram/bin/python scripts/eegfm_adapters/eegfm_small_subset_train_worker.py \
        --model LaBraM \
        --repo_path "/nicoletye/workspace/Noise Robustness Evaluation Protocol/EEG-FM/Labram" \
        --venv_path /nicoletye/venvs/eegpt_labram \
        --index_npz "$INDEX_NPZ" \
        --h5 "$H5_PATH" \
        --output_dir "$out_dir" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LABRAM_LR" \
        --weight_decay "$LABRAM_WEIGHT_DECAY" \
        --seed "$seed" \
        --device cuda:0 \
        --num_workers "$NUM_WORKERS" \
        --pin_memory \
        --train_n 0 \
        --val_n 0 \
        --test_n 0 \
        --finetune_strategy "$strategy" \
        --lora_rank 2 \
        --lora_alpha 8 \
        --lora_target meriem_exact \
        --strict_checkpoint_load \
        --pretrained_checkpoint "$LABRAM_CHECKPOINT" \
        --strategy_report_path "$out_dir/strategy_param_audit"
      ;;
    biot)
      printf '%s\0' \
        /nicoletye/venvs/biot_env/bin/python scripts/eegfm_adapters/run_biot_unified60_meriem_strict_job.py \
        --strategy "$strategy" --seed "$seed" --output_dir "$out_dir" \
        --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --lr "$LR" --weight_decay "$WEIGHT_DECAY" \
        --device cuda:0 --num_workers "$NUM_WORKERS" --pin_memory \
        --index_npz "$INDEX_NPZ" --h5 "$H5_PATH" --train_n 0 --val_n 0 --test_n 0 \
        --lora_rank 2 --lora_alpha 8 --lora_target meriem_exact
      ;;
    eegpt)
      printf '%s\0' \
        /nicoletye/venvs/eegpt_labram/bin/python scripts/eegfm_adapters/run_eegpt_unified60_meriem_strict_job.py \
        --strategy "$strategy" --seed "$seed" --output_dir "$out_dir" \
        --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --lr "$LR" --weight_decay "$WEIGHT_DECAY" \
        --device cuda:0 --num_workers "$NUM_WORKERS" --pin_memory \
        --index_npz "$INDEX_NPZ" --h5 "$H5_PATH" --train_n 0 --val_n 0 --test_n 0 \
        --lora_rank 2 --lora_alpha 8 --lora_target meriem_exact
      ;;
    cbramod)
      printf '%s\0' \
        /nicoletye/venvs/cbramod_csbrain/bin/python scripts/eegfm_adapters/run_cbramod_unified60_meriem_strict_job.py \
        --strategy "$strategy" --seed "$seed" --output_dir "$out_dir" \
        --index_npz "$INDEX_NPZ" --h5 "$H5_PATH" --epochs "$EPOCHS" \
        --train_n 0 --val_n 0 --test_n 0 --batch_size "$BATCH_SIZE" \
        --lr "$LR" --weight_decay "$WEIGHT_DECAY" --device cuda:0 --num_workers "$NUM_WORKERS" --pin_memory
      ;;
    csbrain)
      printf '%s\0' \
        /nicoletye/venvs/cbramod_csbrain/bin/python scripts/eegfm_adapters/run_csbrain_unified60_meriem_strict_job.py \
        --strategy "$strategy" --seed "$seed" --output_dir "$out_dir" \
        --index_npz "$INDEX_NPZ" --h5 "$H5_PATH" --epochs "$EPOCHS" \
        --train_n 0 --val_n 0 --test_n 0 --batch_size "$BATCH_SIZE" \
        --lr "$LR" --weight_decay "$WEIGHT_DECAY" --device cuda:0 --num_workers "$NUM_WORKERS" --pin_memory
      ;;
    codebrain)
      printf '%s\0' \
        /nicoletye/venvs/cbramod_csbrain/bin/python scripts/eegfm_adapters/run_codebrain_unified60_meriem_strict_job.py \
        --strategy "$strategy" --seed "$seed" --output_dir "$out_dir" \
        --index_npz "$INDEX_NPZ" --h5 "$H5_PATH" --epochs "$EPOCHS" \
        --train_n 0 --val_n 0 --test_n 0 --batch_size "$BATCH_SIZE" \
        --lr "$LR" --weight_decay "$WEIGHT_DECAY" --device cuda:0 --num_workers "$NUM_WORKERS" --pin_memory
      ;;
    *)
      echo "Unknown model: $model" >&2
      return 2
      ;;
  esac
}

wait_for_slot() {
  local rc
  while [[ "$(jobs -pr | wc -l)" -ge "$MAX_PARALLEL" ]]; do
    set +e
    wait -n
    rc=$?
    set -e
    if [[ "$rc" -ne 0 ]]; then
      fail_count=$((fail_count + 1))
    fi
  done
}

run_job_background() {
  local job_id="$1" model="$2" strategy="$3" seed="$4" gpu="$5" out_dir="$6" terminal_log="$7"
  shift 7
  local cmd=("$@")
  local command_text
  command_text="$(command_text_for_cmd "${cmd[@]}")"

  if [[ "$FRESH" != "1" && -f "$out_dir/metrics.json" ]] && grep -q '"status"[[:space:]]*:[[:space:]]*"PASS"' "$out_dir/metrics.json"; then
    append_status "$job_id"$'\t'"$model"$'\t'"$strategy"$'\t'"$seed"$'\t'"$gpu"$'\tSKIP_PASS\t'"$out_dir"$'\t'"$terminal_log"$'\t'"$command_text"
    log "SKIP_PASS job=$job_id model=$model strategy=$strategy seed=$seed output=$out_dir"
    return 0
  fi

  if [[ "$FRESH" == "1" && -d "$out_dir" ]]; then
    rm -rf "$out_dir"
  fi
  mkdir -p "$out_dir"
  append_status "$job_id"$'\t'"$model"$'\t'"$strategy"$'\t'"$seed"$'\t'"$gpu"$'\tSTART\t'"$out_dir"$'\t'"$terminal_log"$'\t'"$command_text"
  log "START job=$job_id model=$model strategy=$strategy seed=$seed gpu=$gpu terminal_log=$terminal_log"

  (
    set +e
    {
      echo "===== START job=$job_id model=$model strategy=$strategy seed=$seed gpu=$gpu ====="
      echo "timestamp=$(date '+%Y-%m-%d %H:%M:%S %Z')"
      echo "CUDA_VISIBLE_DEVICES=$gpu"
      echo "command=$command_text"
      CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}"
      rc=$?
      echo "exit_code=$rc"
      echo "timestamp=$(date '+%Y-%m-%d %H:%M:%S %Z')"
      if [[ "$rc" -eq 0 ]]; then
        append_status "$job_id"$'\t'"$model"$'\t'"$strategy"$'\t'"$seed"$'\t'"$gpu"$'\tDONE\t'"$out_dir"$'\t'"$terminal_log"$'\t'"$command_text"
      else
        append_status "$job_id"$'\t'"$model"$'\t'"$strategy"$'\t'"$seed"$'\t'"$gpu"$'\tFAIL\t'"$out_dir"$'\t'"$terminal_log"$'\t'"$command_text"
      fi
      exit "$rc"
    } > "$terminal_log" 2>&1
  ) &
}

job_id=0
fail_count=0
stop_requested=0
for model in "${MODELS[@]}"; do
  [[ "$stop_requested" == "1" ]] && break
  matches_filter "$model" "$MODEL_FILTER" || continue
  for strategy in "${STRATEGIES[@]}"; do
    [[ "$stop_requested" == "1" ]] && break
    matches_filter "$strategy" "$STRATEGY_FILTER" || continue
    for seed in "${SEEDS[@]}"; do
      matches_filter "$seed" "$SEED_FILTER" || continue
      if [[ "$JOB_LIMIT" -gt 0 && "$job_id" -ge "$JOB_LIMIT" ]]; then
        stop_requested=1
        break
      fi

      gpu="${GPUS[$((job_id % ${#GPUS[@]}))]}"
      out_dir="$OUTPUT_ROOT/$model/$strategy/seed_$seed"
      terminal_log="$out_dir/terminal.log"
      mkdir -p "$out_dir"
      mapfile -d '' cmd < <(cmd_for_job "$model" "$strategy" "$seed" "$out_dir")
      command_text="$(command_text_for_cmd "${cmd[@]}")"

      if [[ "$DRY_RUN" == "1" ]]; then
        append_status "$job_id"$'\t'"$model"$'\t'"$strategy"$'\t'"$seed"$'\t'"$gpu"$'\tDRY_RUN\t'"$out_dir"$'\t'"$terminal_log"$'\t'"$command_text"
        log "DRY_RUN job=$job_id model=$model strategy=$strategy seed=$seed gpu=$gpu command=$command_text"
      else
        wait_for_slot
        run_job_background "$job_id" "$model" "$strategy" "$seed" "$gpu" "$out_dir" "$terminal_log" "${cmd[@]}"
      fi
      job_id=$((job_id + 1))
    done
  done
done

if [[ "$DRY_RUN" != "1" ]]; then
  while [[ "$(jobs -pr | wc -l)" -gt 0 ]]; do
    set +e
    wait -n
    rc=$?
    set -e
    if [[ "$rc" -ne 0 ]]; then
      fail_count=$((fail_count + 1))
    fi
  done
fi

/nicoletye/venvs/eegnet/bin/python scripts/eegfm_adapters/summarize_all6_unified60_subset_epoch15_5seed_1gpu_dev.py || true
log "planned_jobs=$job_id"
log "job_status=$JOB_TABLE"
log "fail_count=$fail_count"
exit "$fail_count"
