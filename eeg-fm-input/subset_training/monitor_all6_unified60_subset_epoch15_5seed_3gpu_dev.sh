#!/usr/bin/env bash
set -u -o pipefail
cd /nicoletye/workspace/unified_tuab || exit 1

OUT_ROOT="outputs/all6_unified60_subset_epoch15_5seed_1gpu_dev_v1"
REPORT_ROOT="reports/all6_unified60_subset_epoch15_5seed_1gpu_dev_v1"
STATUS_DIR="$OUT_ROOT/task_status"

echo "== All-6 unified60 subset epoch15 3GPU dev monitor =="
if [[ -f "$STATUS_DIR/launcher_config.env" ]]; then
  cat "$STATUS_DIR/launcher_config.env"
else
  echo "no launcher_config.env yet"
fi
echo
echo "== nvidia-smi =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
else
  echo "nvidia-smi not found"
fi
echo
echo "== active processes =="
pgrep -af "eegfm_small_subset_train_worker|run_(biot|eegpt|cbramod|csbrain|codebrain)_unified60_meriem_strict_job" || true
echo
echo "== job status tail =="
if [[ -f "$STATUS_DIR/job_status.tsv" ]]; then
  tail -60 "$STATUS_DIR/job_status.tsv"
else
  echo "no job_status.tsv"
fi
echo
echo "== status counts from job_status.tsv =="
python - <<'PY'
from collections import Counter
from pathlib import Path
p = Path("outputs/all6_unified60_subset_epoch15_5seed_1gpu_dev_v1/task_status/job_status.tsv")
if not p.exists():
    print("no job_status.tsv")
else:
    counts = Counter()
    with p.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        status_i = header.index("status") if "status" in header else 5
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > status_i:
                counts[parts[status_i]] += 1
    print(dict(counts))
PY
echo
echo "== PASS/FAIL/INCOMPLETE metrics counts =="
python - <<'PY'
import json
from collections import Counter, defaultdict
from pathlib import Path
root = Path("outputs/all6_unified60_subset_epoch15_5seed_1gpu_dev_v1")
models = ["labram", "biot", "eegpt", "cbramod", "csbrain", "codebrain"]
counts = Counter()
by_model = defaultdict(Counter)
for model in models:
    for strategy in ["full_finetune", "linear_probe", "lora"]:
        for seed in [0, 42, 123, 256, 512]:
            p = root / model / strategy / f"seed_{seed}" / "metrics.json"
            if not p.exists():
                status = "INCOMPLETE"
            else:
                try:
                    status = json.loads(p.read_text()).get("status", "UNKNOWN")
                except Exception:
                    status = "READ_ERROR"
            counts[status] += 1
            by_model[model][status] += 1
print("total", dict(counts))
for model in models:
    print(model, dict(by_model[model]))
PY
echo
echo "== latest metrics =="
find "$OUT_ROOT" -name metrics.json -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -12 | while read -r _ p; do
  echo "--- $p"
  grep -E '"status"|"epochs_completed"|"best_epoch"|"best_val_balanced_accuracy"|"test_accuracy"|"test_balanced_accuracy"|"test_auroc"|"test_auprc"|"strategy_classification"|"lora_placement"' "$p" | head -24
done
echo
echo "== latest terminal logs =="
find "$OUT_ROOT" -name terminal.log -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -6 | while read -r _ p; do
  echo "--- $p"
  tail -25 "$p"
done
echo
echo "summary command: /nicoletye/venvs/eegnet/bin/python scripts/eegfm_adapters/summarize_all6_unified60_subset_epoch15_5seed_1gpu_dev.py"
echo "report_root=$REPORT_ROOT"
