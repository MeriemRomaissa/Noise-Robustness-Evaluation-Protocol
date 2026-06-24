#!/usr/bin/env bash
set -Eeuo pipefail

cd /nicoletye/workspace/unified_tuab

DRY_CHECK_ONLY="${DRY_CHECK_ONLY:-0}"
if [[ "${1:-}" == "--dry_check_only" ]]; then
  DRY_CHECK_ONLY=1
fi

export PYTHONUNBUFFERED=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TMPDIR=/nicoletye/workspace/tmp
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
mkdir -p "$TMPDIR"

OUTPUT_ROOT="/nicoletye/workspace/unified_tuab/outputs/labram_notch_ablation_subset_v1_strict_epoch20_2gpu"
REPORT_ROOT="/nicoletye/workspace/unified_tuab/reports/labram_notch_ablation_subset_v1_strict_epoch20_2gpu"
TASK_LOG="${OUTPUT_ROOT}/task_terminal.log"
TASK_STATUS="${OUTPUT_ROOT}/task_status.json"
MANIFEST="/nicoletye/workspace/unified_tuab/reports/labram_notch_ablation_subset_v1/labram_notch_ablation_exact_subset_manifest.csv"
EXACT_SPLIT="/nicoletye/workspace/unified_tuab/reports/labram_exact_original_processed_split/canonical_h5_labram_exact_original_processed_split_index.csv"
H5="/nicoletye/workspace/unified_tuab/data/canonical_tuab_full.h5"
CHECKPOINT="/nicoletye/workspace/Noise Robustness Evaluation Protocol/EEG-FM/Labram/checkpoints/labram-base.pth"
REPO="/nicoletye/workspace/Noise Robustness Evaluation Protocol/EEG-FM/Labram"
PROCESSED_ROOT="/nicoletye/datasets/TUAB/processed"
UNIFIED_50HZ_DIR="/nicoletye/workspace/unified_tuab/outputs/labram_notch_ablation_subset_v1/unified_50hz"
UNIFIED_50HZ_NPZ="${UNIFIED_50HZ_NPZ:-${UNIFIED_50HZ_DIR}/unified_50hz_subset.npz}"
UNIFIED_50HZ_BUILD_STATUS="${UNIFIED_50HZ_DIR}/build_status.json"
UNIFIED_50HZ_BUILD_LOG="${UNIFIED_50HZ_DIR}/build_unified_50hz_subset.log"
GPU_IDS_CSV="${GPU_IDS:-0,1}"
MAX_PARALLEL=2
FRESH="${FRESH:-0}"
SKIP_PASS="${SKIP_PASS:-1}"
ALLOW_PARTIAL="${ALLOW_PARTIAL:-0}"
BUILD_MISSING_UNIFIED_50HZ="${BUILD_MISSING_UNIFIED_50HZ:-1}"
RUN_ONLY_BRANCH="${RUN_ONLY_BRANCH:-}"
ALL_BRANCHES=(original_pkl_50hz unified_50hz unified_60hz)
BRANCHES=()

mkdir -p "$OUTPUT_ROOT" "$REPORT_ROOT" "$UNIFIED_50HZ_DIR"
exec > >(tee -a "$TASK_LOG") 2>&1

json_status() {
  local status="$1"
  shift || true
  python - "$TASK_STATUS" "$status" "$@" <<'PY'
import json, sys, time
path, status, *items = sys.argv[1:]
payload = {}
try:
    with open(path) as f:
        payload = json.load(f)
except Exception:
    pass
payload.update({"status": status, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")})
for item in items:
    if "=" in item:
        k, v = item.split("=", 1)
        if "," in v:
            payload[k] = [x for x in v.split(",") if x]
        elif v.lower() in {"true", "false"}:
            payload[k] = v.lower() == "true"
        else:
            payload[k] = v
with open(path, "w") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")
PY
}

branch_status() {
  local branch="$1"
  local status="$2"
  local reason="${3:-}"
  local dir="${OUTPUT_ROOT}/${branch}"
  mkdir -p "$dir"
  python - "$dir/branch_status.json" "$branch" "$status" "$reason" <<'PY'
import json, sys, time
path, branch, status, reason = sys.argv[1:]
payload = {"branch": branch, "status": status, "reason": reason, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
with open(path, "w") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")
PY
}

echo "===== LaBraM strict epoch20 subset ablation 2-GPU task ====="
echo "timestamp=$(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "hostname=$(hostname)"
echo "cwd=$(pwd)"
echo "dry_check_only=${DRY_CHECK_ONLY}"
echo "allow_partial=${ALLOW_PARTIAL}"
echo "build_missing_unified_50hz=${BUILD_MISSING_UNIFIED_50HZ}"
echo "run_only_branch=${RUN_ONLY_BRANCH}"
echo "fresh=${FRESH}"
echo "skip_pass=${SKIP_PASS}"
echo "output_root=${OUTPUT_ROOT}"
echo "report_root=${REPORT_ROOT}"
echo "GPU_IDS=${GPU_IDS_CSV}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

json_status "STARTED" "dry_check_only=${DRY_CHECK_ONLY}" "allow_partial=${ALLOW_PARTIAL}" "build_missing_unified_50hz=${BUILD_MISSING_UNIFIED_50HZ}" "run_only_branch=${RUN_ONLY_BRANCH}" "output_root=${OUTPUT_ROOT}"

if [[ -n "$RUN_ONLY_BRANCH" ]]; then
  case "$RUN_ONLY_BRANCH" in
    original_pkl_50hz|unified_50hz|unified_60hz) BRANCHES=("$RUN_ONLY_BRANCH") ;;
    *) echo "ERROR unknown RUN_ONLY_BRANCH=${RUN_ONLY_BRANCH}"; json_status "FAIL" "reason=unknown RUN_ONLY_BRANCH"; exit 1 ;;
  esac
else
  BRANCHES=("${ALL_BRANCHES[@]}")
fi

PRECHECK_FAIL=0
for path in "$EXACT_SPLIT" "$CHECKPOINT" "$H5"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR missing required file: $path"
    PRECHECK_FAIL=1
  fi
done
for path in "$REPO" "$PROCESSED_ROOT"; do
  if [[ ! -d "$path" ]]; then
    echo "ERROR missing required directory: $path"
    PRECHECK_FAIL=1
  fi
done
if [[ ! -f scripts/eegfm_adapters/run_labram_unified_h5_strict_original_engine.py ]]; then
  echo "ERROR strict runner not found"
  PRECHECK_FAIL=1
fi

if [[ ! -f "$MANIFEST" ]]; then
  echo "Subset manifest missing; generating it now."
  /nicoletye/venvs/eegnet/bin/python scripts/eegfm_adapters/build_labram_exact_subset_manifest.py \
    --exact_split_csv "$EXACT_SPLIT" \
    --output_dir /nicoletye/workspace/unified_tuab/reports/labram_notch_ablation_subset_v1 \
    --train_n 8192 \
    --val_n 2048 \
    --test_n 2048 \
    --seed 42
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "ERROR missing subset manifest after generation attempt: $MANIFEST"
  PRECHECK_FAIL=1
fi

UNIFIED_50HZ_BUILD_STATUS_VALUE="NOT_NEEDED"
if [[ " ${BRANCHES[*]} " == *" unified_50hz "* && ( ! -f "$UNIFIED_50HZ_NPZ" || "$FRESH" == "1" ) ]]; then
  if [[ "$BUILD_MISSING_UNIFIED_50HZ" == "1" && "$DRY_CHECK_ONLY" != "1" ]]; then
    echo "Attempting unified_50hz subset build..."
    build_extra_args=()
    if [[ "$FRESH" == "1" ]]; then
      build_extra_args+=(--force)
    fi
    set +e
    /nicoletye/venvs/eegnet/bin/python scripts/eegfm_adapters/build_labram_unified_50hz_subset_from_raw_exact_manifest.py \
      --subset_manifest "$MANIFEST" \
      --raw_edf_root /nicoletye/workspace/tuh_data/TUAB/v3.0.1/edf \
      --output_npz "$UNIFIED_50HZ_NPZ" \
      --notch_freq 50 \
      "${build_extra_args[@]}" > "$UNIFIED_50HZ_BUILD_LOG" 2>&1
    build_rc=$?
    set -e
    if [[ "$build_rc" -eq 0 && -f "$UNIFIED_50HZ_NPZ" ]]; then
      UNIFIED_50HZ_BUILD_STATUS_VALUE="PASS"
      python - "$UNIFIED_50HZ_BUILD_STATUS" "$UNIFIED_50HZ_NPZ" <<'PY'
import json, sys, time
path, npz = sys.argv[1:]
with open(path, "w") as f:
    json.dump({"status": "PASS", "subset_npz": npz, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    f.write("\n")
PY
    else
      UNIFIED_50HZ_BUILD_STATUS_VALUE="BUILD_FAIL"
      python - "$UNIFIED_50HZ_BUILD_STATUS" "$UNIFIED_50HZ_BUILD_LOG" "$build_rc" <<'PY'
import json, sys, time
path, log, rc = sys.argv[1:]
with open(path, "w") as f:
    json.dump({"status": "BUILD_FAIL", "returncode": int(rc), "log_path": log, "reason": "unified_50hz subset build failed", "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    f.write("\n")
PY
    fi
  elif [[ "$BUILD_MISSING_UNIFIED_50HZ" == "1" && "$DRY_CHECK_ONLY" == "1" ]]; then
    UNIFIED_50HZ_BUILD_STATUS_VALUE="PENDING_BUILD_DRY_CHECK"
    python - "$UNIFIED_50HZ_BUILD_STATUS" "$UNIFIED_50HZ_NPZ" <<'PY'
import json, sys, time
path, npz = sys.argv[1:]
with open(path, "w") as f:
    json.dump({"status": "PENDING_BUILD_DRY_CHECK", "expected_subset_npz": npz, "reason": "dry check only; build not launched", "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    f.write("\n")
PY
  else
    UNIFIED_50HZ_BUILD_STATUS_VALUE="MISSING"
  fi
elif [[ -f "$UNIFIED_50HZ_NPZ" ]]; then
  UNIFIED_50HZ_BUILD_STATUS_VALUE="AVAILABLE"
fi

LAUNCH_BRANCHES=()
SKIPPED_BRANCHES=()
for branch in "${BRANCHES[@]}"; do
  if [[ "$branch" == "unified_50hz" && ! -f "$UNIFIED_50HZ_NPZ" ]]; then
    echo "SKIP branch=unified_50hz reason=missing_or_failed_build"
    branch_status "unified_50hz" "${UNIFIED_50HZ_BUILD_STATUS_VALUE}" "unified_50hz subset build failed or data missing"
    SKIPPED_BRANCHES+=("unified_50hz")
    continue
  fi
  LAUNCH_BRANCHES+=("$branch")
done

if [[ "${#LAUNCH_BRANCHES[@]}" -lt "${#BRANCHES[@]}" ]]; then
  if [[ "$ALLOW_PARTIAL" != "1" && "$DRY_CHECK_ONLY" != "1" ]]; then
    echo "Precheck failed and ALLOW_PARTIAL is not enabled. Stopping before training."
    json_status "FAIL" "unified_50hz_build_status=${UNIFIED_50HZ_BUILD_STATUS_VALUE}" "partial_run=false" "branches_launched=" "branches_skipped=${SKIPPED_BRANCHES[*]}" "reason=missing branch data"
    exit 1
  fi
fi

echo "Running branch dry checks..."
DRY_FAIL=0
for branch in "${LAUNCH_BRANCHES[@]}"; do
  set +e
  /nicoletye/venvs/eegpt_labram/bin/python scripts/eegfm_adapters/run_labram_notch_ablation_subset_strict_epoch20_branch.py \
    --branch "$branch" \
    --subset_manifest "$MANIFEST" \
    --split_csv "$EXACT_SPLIT" \
    --output_root "$OUTPUT_ROOT" \
    --h5 "$H5" \
    --processed_root "$PROCESSED_ROOT" \
    --unified_50hz_npz "$UNIFIED_50HZ_NPZ" \
    --repo_path "$REPO" \
    --finetune "$CHECKPOINT" \
    --device cuda \
    --dry_check_only
  rc=$?
  set -e
  echo "dry_check branch=${branch} rc=${rc}"
  if [[ "$rc" -ne 0 ]]; then
    DRY_FAIL=1
  fi
done

partial=false
if [[ "${#SKIPPED_BRANCHES[@]}" -gt 0 ]]; then
  partial=true
fi
json_status "PRECHECK_DONE" "unified_50hz_build_status=${UNIFIED_50HZ_BUILD_STATUS_VALUE}" "partial_run=${partial}" "branches_launched=${LAUNCH_BRANCHES[*]}" "branches_skipped=${SKIPPED_BRANCHES[*]}"

if [[ "$DRY_CHECK_ONLY" == "1" ]]; then
  echo "DRY_CHECK_ONLY complete. dry_fail=${DRY_FAIL} skipped=${SKIPPED_BRANCHES[*]:-none}"
  if [[ "$PRECHECK_FAIL" -ne 0 || "$DRY_FAIL" -ne 0 ]]; then
    json_status "DRY_CHECK_FAIL" "unified_50hz_build_status=${UNIFIED_50HZ_BUILD_STATUS_VALUE}" "partial_run=${partial}" "branches_launched=${LAUNCH_BRANCHES[*]}" "branches_skipped=${SKIPPED_BRANCHES[*]}" "reason=dry check failed"
    exit 1
  fi
  json_status "DRY_CHECK_PASS" "unified_50hz_build_status=${UNIFIED_50HZ_BUILD_STATUS_VALUE}" "partial_run=${partial}" "branches_launched=${LAUNCH_BRANCHES[*]}" "branches_skipped=${SKIPPED_BRANCHES[*]}"
  exit 0
fi

if [[ "$PRECHECK_FAIL" -ne 0 || "$DRY_FAIL" -ne 0 ]]; then
  echo "Required precheck failed. Stopping before launching training."
  json_status "FAIL" "unified_50hz_build_status=${UNIFIED_50HZ_BUILD_STATUS_VALUE}" "partial_run=${partial}" "branches_launched=${LAUNCH_BRANCHES[*]}" "branches_skipped=${SKIPPED_BRANCHES[*]}" "reason=required precheck failed"
  exit 1
fi

if [[ "${#LAUNCH_BRANCHES[@]}" -eq 0 ]]; then
  echo "No branches available to launch."
  json_status "FAIL" "unified_50hz_build_status=${UNIFIED_50HZ_BUILD_STATUS_VALUE}" "partial_run=${partial}" "branches_launched=" "branches_skipped=${SKIPPED_BRANCHES[*]}" "reason=no branches available"
  exit 1
fi

IFS=',' read -r -a GPU_QUEUE <<< "$GPU_IDS_CSV"
if [[ "${#GPU_QUEUE[@]}" -lt 1 ]]; then
  echo "ERROR expected at least 1 GPU ID, got: $GPU_IDS_CSV"
  exit 1
fi

declare -a RUNNING_PIDS=()
declare -A PID_BRANCH=()
declare -A PID_GPU=()
AVAILABLE_GPUS=("${GPU_QUEUE[@]:0:$MAX_PARALLEL}")
FAILED=0
COMPLETED_BRANCHES=()
SKIP_PASS_BRANCHES=()

launch_branch() {
  local branch="$1"
  local gpu="$2"
  local out_dir="${OUTPUT_ROOT}/${branch}"
  mkdir -p "$out_dir"
  if [[ "$SKIP_PASS" == "1" && "$FRESH" != "1" && -f "${out_dir}/metrics.json" ]] && grep -q '"status"[[:space:]]*:[[:space:]]*"PASS"' "${out_dir}/metrics.json"; then
    echo "SKIP_PASS branch=${branch}"
    branch_status "$branch" "SKIP_PASS" "existing PASS metrics and SKIP_PASS=1"
    SKIP_PASS_BRANCHES+=("$branch")
    AVAILABLE_GPUS+=("$gpu")
    return
  fi
  branch_status "$branch" "QUEUED" "assigned gpu ${gpu}"
  echo "LAUNCH branch=${branch} gpu=${gpu} log=${out_dir}/terminal.log"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    /nicoletye/venvs/eegpt_labram/bin/python scripts/eegfm_adapters/run_labram_notch_ablation_subset_strict_epoch20_branch.py \
      --branch "$branch" \
      --subset_manifest "$MANIFEST" \
      --split_csv "$EXACT_SPLIT" \
      --output_root "$OUTPUT_ROOT" \
      --h5 "$H5" \
      --processed_root "$PROCESSED_ROOT" \
      --unified_50hz_npz "$UNIFIED_50HZ_NPZ" \
      --repo_path "$REPO" \
      --finetune "$CHECKPOINT" \
      --epochs 20 \
      --batch_size 64 \
      --lr 5e-4 \
      --weight_decay 0.05 \
      --update_freq 1 \
      --warmup_epochs 5 \
      --layer_decay 0.65 \
      --drop_path 0.1 \
      --save_ckpt_freq 1 \
      --disable_rel_pos_bias \
      --abs_pos_emb \
      --disable_qkv_bias \
      --seed 0 \
      --device cuda \
      --num_workers 4 \
      --pin_memory
  ) > "${out_dir}/terminal.log" 2>&1 &
  local pid=$!
  RUNNING_PIDS+=("$pid")
  PID_BRANCH[$pid]="$branch"
  PID_GPU[$pid]="$gpu"
}

wait_for_one() {
  local new_running=()
  while true; do
    for pid in "${RUNNING_PIDS[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null; then
        if wait "$pid"; then
          echo "DONE branch=${PID_BRANCH[$pid]} gpu=${PID_GPU[$pid]} status=complete"
          COMPLETED_BRANCHES+=("${PID_BRANCH[$pid]}")
        else
          FAILED=1
          echo "DONE branch=${PID_BRANCH[$pid]} gpu=${PID_GPU[$pid]} status=failed"
        fi
        AVAILABLE_GPUS+=("${PID_GPU[$pid]}")
      else
        new_running+=("$pid")
      fi
    done
    if [[ "${#new_running[@]}" -lt "${#RUNNING_PIDS[@]}" ]]; then
      RUNNING_PIDS=("${new_running[@]}")
      return
    fi
    new_running=()
    sleep 10
  done
}

json_status "RUNNING" "unified_50hz_build_status=${UNIFIED_50HZ_BUILD_STATUS_VALUE}" "partial_run=${partial}" "branches_launched=${LAUNCH_BRANCHES[*]}" "branches_skipped=${SKIPPED_BRANCHES[*]}"
for branch in "${LAUNCH_BRANCHES[@]}"; do
  while [[ "${#AVAILABLE_GPUS[@]}" -eq 0 ]]; do
    wait_for_one
  done
  gpu="${AVAILABLE_GPUS[0]}"
  AVAILABLE_GPUS=("${AVAILABLE_GPUS[@]:1}")
  launch_branch "$branch" "$gpu"
done

while [[ "${#RUNNING_PIDS[@]}" -gt 0 ]]; do
  wait_for_one
done

/nicoletye/venvs/eegnet/bin/python scripts/eegfm_adapters/summarize_labram_notch_ablation_subset_strict_epoch20_2gpu.py || true

final_status="PASS"
if [[ "$FAILED" -ne 0 ]]; then
  final_status="FAIL"
elif [[ "$partial" == "true" ]]; then
  final_status="PARTIAL_PASS"
fi
reason=""
if [[ "$partial" == "true" ]]; then
  reason="unified_50hz subset build failed"
fi
json_status "$final_status" "unified_50hz_build_status=${UNIFIED_50HZ_BUILD_STATUS_VALUE}" "partial_run=${partial}" "branches_launched=${LAUNCH_BRANCHES[*]}" "branches_skipped=${SKIPPED_BRANCHES[*]}" "branches_completed=${COMPLETED_BRANCHES[*]}" "skip_pass_branches=${SKIP_PASS_BRANCHES[*]}" "reason=${reason}"
echo "Task complete status=${final_status} failed=${FAILED} partial=${partial}"
exit "$FAILED"
