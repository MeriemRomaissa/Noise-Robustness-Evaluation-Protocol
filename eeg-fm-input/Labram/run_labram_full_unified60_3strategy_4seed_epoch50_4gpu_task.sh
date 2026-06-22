#!/usr/bin/env bash
set -euo pipefail

cd /nicoletye/workspace/unified_tuab

SEEDS=(42 123 256 512)
STRATEGIES=(full_finetune linear_probe lora_meriem_exact)
GPU_LIST="${GPU_LIST:-0,1,2,3}"
GPU_LIST_CLEAN="${GPU_LIST//[[:space:]]/}"
IFS=',' read -r -a GPUS <<< "$GPU_LIST_CLEAN"
if [[ "${#GPUS[@]}" -eq 0 || -z "${GPUS[0]:-}" ]]; then
  echo "ERROR: GPU_LIST is empty. Example: GPU_LIST=0,1,2" >&2
  exit 2
fi
if [[ -z "${MAX_PARALLEL+x}" ]]; then
  MAX_PARALLEL="${#GPUS[@]}"
fi
FRESH="${FRESH:-0}"
DRY_RUN="${DRY_RUN:-0}"
OUTPUT_ROOT="outputs/labram_full_unified60_3strategy_4seed_epoch50_v1"
REPORT_ROOT="reports/labram_full_unified60_3strategy_4seed_epoch50_v1"
STATUS_DIR="$OUTPUT_ROOT/task_status"
JOB_TABLE="$STATUS_DIR/job_status.tsv"
LAUNCHER_CONFIG="$STATUS_DIR/launcher_config.env"
WRAPPER="scripts/eegfm_adapters/run_labram_full_unified60_strategy_seeded_epoch50.py"

export PYTHONUNBUFFERED=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TMPDIR="${TMPDIR:-/nicoletye/workspace/tmp}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
mkdir -p "$TMPDIR" "$STATUS_DIR" "$REPORT_ROOT"
JOB_COUNT=$((${#SEEDS[@]} * ${#STRATEGIES[@]}))

cat > "$LAUNCHER_CONFIG" <<EOF
output_root=$OUTPUT_ROOT
report_root=$REPORT_ROOT
gpu_list=$GPU_LIST_CLEAN
parsed_gpus=${GPUS[*]}
max_parallel=$MAX_PARALLEL
job_count=$JOB_COUNT
fresh=$FRESH
dry_run=$DRY_RUN
EOF

echo "===== LaBraM unified60 3-strategy 4-seed epoch50 4GPU task ====="
echo "timestamp=$(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "output_root=$OUTPUT_ROOT"
echo "report_root=$REPORT_ROOT"
echo "fresh=$FRESH dry_run=$DRY_RUN"
echo "GPU_LIST=$GPU_LIST_CLEAN"
echo "parsed_gpus=${GPUS[*]}"
echo "MAX_PARALLEL=$MAX_PARALLEL"
echo "job_count=$JOB_COUNT"
echo "seeds=${SEEDS[*]}"
echo "strategies=${STRATEGIES[*]}"
echo "visible_nvidia_smi_gpus:"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
  nvidia-smi || true
else
  echo "nvidia-smi not found"
fi

printf 'job_id\tstrategy\tseed\tgpu\tstatus\toutput_dir\tterminal_log\tcommand\n' > "$JOB_TABLE"

job_id=0
running=0
fail_count=0

run_one_job() {
  local jid="$1" strategy="$2" seed="$3" gpu="$4" out_dir="$5" report_dir="$6" terminal_log="$7"
  mkdir -p "$out_dir" "$report_dir"
  local cmd=(
    /nicoletye/venvs/eegpt_labram/bin/python
    "$WRAPPER"
    --strategy "$strategy"
    --seed "$seed"
    --output_root "$out_dir"
    --report_dir "$report_dir"
    --epochs 50
    --batch_size 64
    --num_workers 4
    --device cuda
  )
  local command_text
  command_text="${cmd[*]}"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '%s\t%s\t%s\t%s\tDRY_RUN\t%s\t%s\t%s\n' "$jid" "$strategy" "$seed" "$gpu" "$out_dir" "$terminal_log" "$command_text" >> "$JOB_TABLE"
    echo "DRY_RUN job=$jid strategy=$strategy seed=$seed gpu=$gpu command=$command_text"
    return 0
  fi
  if [[ "$FRESH" != "1" && -f "$out_dir/metrics.json" ]] && grep -q '"status"[[:space:]]*:[[:space:]]*"PASS"' "$out_dir/metrics.json"; then
    printf '%s\t%s\t%s\t%s\tSKIP_PASS\t%s\t%s\t%s\n' "$jid" "$strategy" "$seed" "$gpu" "$out_dir" "$terminal_log" "$command_text" >> "$JOB_TABLE"
    echo "SKIP_PASS job=$jid strategy=$strategy seed=$seed output=$out_dir"
    return 0
  fi
  if [[ "$FRESH" == "1" && -d "$out_dir" ]]; then
    echo "FRESH=1 deleting job-local output only: $out_dir"
    rm -rf "$out_dir"
    mkdir -p "$out_dir" "$report_dir"
  fi
  printf '%s\t%s\t%s\t%s\tRUNNING\t%s\t%s\t%s\n' "$jid" "$strategy" "$seed" "$gpu" "$out_dir" "$terminal_log" "$command_text" >> "$JOB_TABLE"
  {
    echo "===== START job=$jid strategy=$strategy seed=$seed gpu=$gpu ====="
    echo "timestamp=$(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "output_dir=$out_dir"
    echo "report_dir=$report_dir"
    echo "command=$command_text"
    set +e
    CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}"
    rc=$?
    set -e
    echo "exit_code=$rc"
    exit "$rc"
  } > "$terminal_log" 2>&1
}

for strategy in "${STRATEGIES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    strategy_dir="$strategy"
    out_dir="$OUTPUT_ROOT/$strategy_dir/seed_$seed"
    report_dir="$REPORT_ROOT/$strategy_dir/seed_$seed"
    terminal_log="$out_dir/terminal.log"
    gpu="${GPUS[$((job_id % ${#GPUS[@]}))]}"
    run_one_job "$job_id" "$strategy" "$seed" "$gpu" "$out_dir" "$report_dir" "$terminal_log" &
    running=$((running + 1))
    job_id=$((job_id + 1))
    if [[ "$running" -ge "$MAX_PARALLEL" ]]; then
      set +e
      wait -n
      rc=$?
      set -e
      if [[ "$rc" -ne 0 ]]; then fail_count=$((fail_count + 1)); fi
      running=$((running - 1))
    fi
  done
done

while [[ "$running" -gt 0 ]]; do
  set +e
  wait -n
  rc=$?
  set -e
  if [[ "$rc" -ne 0 ]]; then fail_count=$((fail_count + 1)); fi
  running=$((running - 1))
done

/nicoletye/venvs/eegnet/bin/python scripts/eegfm_adapters/summarize_labram_full_unified60_3strategy_4seed_epoch50.py || true

echo "===== Final job metrics status ====="
find "$OUTPUT_ROOT" -mindepth 3 -maxdepth 3 -name metrics.json -print | sort | while read -r metrics; do
  /nicoletye/venvs/eegnet/bin/python - "$metrics" <<'PY'
import json, sys
p=sys.argv[1]
d=json.load(open(p))
print(f"{p}\tstatus={d.get('status')}\tbest_val_bal={d.get('best_val_balanced_accuracy')}\ttest_bal={d.get('test_balanced_accuracy')}")
PY
done
echo "job_table=$JOB_TABLE"
echo "fail_count=$fail_count"
exit "$fail_count"
