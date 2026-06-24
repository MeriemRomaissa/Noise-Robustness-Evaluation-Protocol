#!/usr/bin/env bash
set -uo pipefail

cd /nicoletye/workspace/unified_tuab

OUTPUT_ROOT="/nicoletye/workspace/unified_tuab/outputs/labram_notch_ablation_subset_v1_strict_epoch20_2gpu"
REPORT_ROOT="/nicoletye/workspace/unified_tuab/reports/labram_notch_ablation_subset_v1_strict_epoch20_2gpu"
UNIFIED_50HZ_BUILD_DIR="/nicoletye/workspace/unified_tuab/outputs/labram_notch_ablation_subset_v1/unified_50hz"
BRANCHES=(original_pkl_50hz unified_50hz unified_60hz)

echo "===== LaBraM strict epoch20 subset ablation monitor ====="
echo "timestamp=$(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "output_root=${OUTPUT_ROOT}"
echo "report_root=${REPORT_ROOT}"

echo
echo "----- active processes -----"
ps -eo pid,ppid,etime,pcpu,pmem,args | grep -E "run_labram_notch_ablation_subset_strict_epoch20_branch.py|run_labram_notch_ablation_subset_strict_epoch20_2gpu_task.sh" | grep -v grep || true

echo
echo "----- nvidia-smi -----"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi not found"
fi

echo
echo "----- task status -----"
if [[ -f "${OUTPUT_ROOT}/task_status.json" ]]; then
  python - <<PY
import json
p="${OUTPUT_ROOT}/task_status.json"
with open(p) as f: m=json.load(f)
for k in ["status", "partial_run", "unified_50hz_build_status", "branches_launched", "branches_skipped", "branches_completed", "skip_pass_branches", "reason", "updated_at"]:
    if k in m:
        print(f"{k}=", m.get(k))
PY
else
  echo "task_status.json missing"
fi

echo
echo "----- unified_50hz build status -----"
if [[ -f "${UNIFIED_50HZ_BUILD_DIR}/build_status.json" ]]; then
  cat "${UNIFIED_50HZ_BUILD_DIR}/build_status.json"
else
  echo "build_status.json missing"
fi
echo "--- unified_50hz build log tail ---"
tail -n 40 "${UNIFIED_50HZ_BUILD_DIR}/build_unified_50hz_subset.log" 2>/dev/null || true

for branch in "${BRANCHES[@]}"; do
  dir="${OUTPUT_ROOT}/${branch}"
  echo
  echo "===== ${branch} ====="
  if [[ -f "${dir}/branch_status.json" ]]; then
    echo "--- branch_status.json ---"
    cat "${dir}/branch_status.json"
  else
    echo "branch_status.json missing"
  fi
  if [[ -f "${dir}/metrics.json" ]]; then
    python - <<PY
import json
p="${dir}/metrics.json"
with open(p) as f: m=json.load(f)
print("status=", m.get("status"))
print("epochs_completed=", m.get("epochs_completed"))
print("best_epoch=", m.get("best_epoch"))
print("best_val_balanced_accuracy=", m.get("best_val_balanced_accuracy"))
print("test_balanced_accuracy=", m.get("test_balanced_accuracy"))
PY
  else
    echo "metrics.json missing"
  fi
  if [[ -f "${dir}/epoch_metrics.csv" ]]; then
    echo "latest_epoch:"
    tail -n 1 "${dir}/epoch_metrics.csv"
  fi
  echo "checkpoint_count=$(find "${dir}/checkpoints" -name '*.pt' -type f 2>/dev/null | wc -l)"
  echo "--- terminal.log tail ---"
  tail -n 20 "${dir}/terminal.log" 2>/dev/null || true
  echo "--- train.log tail ---"
  tail -n 20 "${dir}/train.log" 2>/dev/null || true
done

echo
echo "----- master task log tail -----"
tail -n 80 "${OUTPUT_ROOT}/task_terminal.log" 2>/dev/null || true

echo
echo "----- summary -----"
cat "${REPORT_ROOT}/summary_report.md" 2>/dev/null || true
