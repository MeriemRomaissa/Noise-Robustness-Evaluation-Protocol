# Script Reference Audit: All-6 Unified60 Subset Epoch-15 Dev Benchmark

This audit documents which existing Meriem-strict or Meriem-mostly-strict implementation each model references. The new development launcher uses only the new output root and does not modify model repositories, canonical H5 data, raw EDF files, or existing full benchmark outputs.

| Model | Full Benchmark Output Root Referenced | Launcher/Job Script Referenced | Worker Script Used | Strategy Utility Used | Checkpoint Path Used | Strategy Classification | LoRA Target Interpretation | Strictness Label | Caveat |
|---|---|---|---|---|---|---|---|---|---|
| LaBraM | `outputs/labram_full_unified60_3strategy_4seed_epoch50_v1` | Full reference: `run_labram_full_unified60_3strategy_4seed_epoch50_4gpu_task.sh`; dev subset command uses `eegfm_small_subset_train_worker.py` with `--lora_target meriem_exact` because the full strict engine is full-dataset only. | `scripts/eegfm_adapters/eegfm_small_subset_train_worker.py` | `scripts/eegfm_adapters/finetune_strategy_utils.py` | `/nicoletye/workspace/Noise Robustness Evaluation Protocol/EEG-FM/Labram/checkpoints/labram-base.pth`, passed explicitly with `--pretrained_checkpoint` | full fine-tune: Meriem strict reference; linear probe: head-only; LoRA: Meriem exact/reference placement | `meriem_exact` | `MERIEM_EXACT` | Dev subset uses the current adapter worker with fixed index NPZ; checkpoint loading maps LaBraM `student.*` checkpoint keys to adapter `model.*` keys. |
| BIOT | `outputs/biot_full_unified60_meriem_strict_3strategy_5seed_epoch50_v1` | `scripts/eegfm_adapters/run_biot_unified60_meriem_strict_job.py` | `scripts/eegfm_adapters/eegfm_small_subset_train_worker.py` | `scripts/eegfm_adapters/finetune_strategy_utils.py` | `/nicoletye/workspace/Noise Robustness Evaluation Protocol/EEG-FM/Biot/pretrained-models/EEG-PREST-16-channels.ckpt` | full fine-tune: `STRICT_MATCH`; linear probe: `FUNCTIONAL_MATCH_EXACT_TRAINABLE_SET`; LoRA: `FUNCTIONAL_MATCH_WITH_ARCHITECTURE_DIFFERENCE` | architecture-native equivalent LoRA preserving Meriem PEFT semantics | `MERIEM_MOSTLY_STRICT_ARCHITECTURE_NATIVE` | BIOT architecture differs from LaBraM, so LoRA target names are native equivalents rather than physical LaBraM modules. |
| EEGPT | `outputs/eegpt_full_unified60_meriem_strict_3strategy_5seed_epoch50_v1` | `scripts/eegfm_adapters/run_eegpt_unified60_meriem_strict_job.py` | `scripts/eegfm_adapters/eegfm_small_subset_train_worker.py` | `scripts/eegfm_adapters/finetune_strategy_utils.py` | `/nicoletye/workspace/Noise Robustness Evaluation Protocol/EEG-FM/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt` | full fine-tune: `FUNCTIONAL_MATCH_WITH_HEAD_DIFFERENCE`; linear probe: `FUNCTIONAL_MATCH_EXACT_TRAINABLE_SET`; LoRA: `MERIEM_STRICT_ARCHITECTURE_NATIVE_EQUIVALENT` | architecture-native equivalent LoRA preserving Meriem PEFT semantics | `MERIEM_STRICT` | EEGPT uses an adapter-side classifier/head convention, matching the full benchmark script. |
| CBraMod | `outputs/cbramod_full_unified60_meriem_strict_3strategy_5seed_epoch50_v1` | `scripts/eegfm_adapters/run_cbramod_unified60_meriem_strict_job.py` | `scripts/eegfm_adapters/eegfm_small_subset_train_worker_remaining_models.py` | `scripts/eegfm_adapters/finetune_strategy_utils_remaining_models.py` | `/nicoletye/workspace/Noise Robustness Evaluation Protocol/EEG-FM/CBraMod/pretrained_weights/pretrained_weights.pth` | full fine-tune, linear probe, and LoRA use isolated remaining-model Meriem-strict logic | architecture-native equivalent LoRA preserving Meriem PEFT semantics | `MERIEM_STRICT` | Uses isolated remaining-model worker only. |
| CSBrain | `outputs/csbrain_full_unified60_meriem_strict_3strategy_5seed_epoch50_v1` | `scripts/eegfm_adapters/run_csbrain_unified60_meriem_strict_job.py` | `scripts/eegfm_adapters/eegfm_small_subset_train_worker_remaining_models.py` | `scripts/eegfm_adapters/finetune_strategy_utils_remaining_models.py` | `/nicoletye/workspace/Noise Robustness Evaluation Protocol/EEG-FM/CSBrain/downloaded_weights/pth/CSBrain.pth` | full fine-tune, linear probe, and LoRA use isolated remaining-model Meriem-strict logic | architecture-native equivalent LoRA preserving Meriem PEFT semantics | `MERIEM_STRICT` | Uses isolated remaining-model worker only. |
| CodeBrain | `outputs/codebrain_full_unified60_meriem_strict_3strategy_5seed_epoch50_v1` | `scripts/eegfm_adapters/run_codebrain_unified60_meriem_strict_job.py` | `scripts/eegfm_adapters/eegfm_small_subset_train_worker_remaining_models.py` | `scripts/eegfm_adapters/finetune_strategy_utils_remaining_models.py` | `/nicoletye/workspace/Noise Robustness Evaluation Protocol/EEG-FM/Codebrain/Checkpoints/CodeBrain.pth` | full fine-tune and linear probe use isolated Meriem-strict logic; LoRA is partial/native-equivalent | `MERIEM_STRICT_ARCHITECTURE_NATIVE_EQUIVALENT_PARTIAL` | `MERIEM_MOSTLY_STRICT_ARCHITECTURE_NATIVE` | CodeBrain SSSM-style architecture does not expose exact LaBraM q/k/v + MLP placement. Do not report CodeBrain LoRA as exact. |

## Fixed Subset Control

All jobs use:

- `reports/all6_unified60_subset_epoch15_5seed_1gpu_dev_v1/all6_fixed_subset_seed42_index.npz`
- `train_n=0`, `val_n=0`, `test_n=0`

The fixed index is built from:

- `reports/labram_exact_original_processed_split/canonical_h5_labram_exact_original_processed_split_index.csv`

with subset seed `42`, train `8192`, validation `2048`, and test `2048`. This prevents training seeds from changing subset membership.

## Audit Conclusion

No model is mapped to an older non-Meriem-strict debug-only strategy interpretation. The only nuance is LaBraM: the full strict engine remains the reference, while the subset development benchmark uses the current adapter worker with Meriem-exact LoRA, explicit `labram-base.pth` checkpoint loading, and the fixed H5 index so it can run the same subset size as the non-LaBraM adapters.

## Development Filters

The launcher supports `MODEL_FILTER`, `STRATEGY_FILTER`, `SEED_FILTER`, and `JOB_LIMIT` for safe development debugging. These filters only reduce the dev benchmark job plan; default behavior remains all 90 jobs.

## 3GPU Parallel Launcher

The 3GPU launcher uses the same model-to-script mapping shown above. It changes only scheduling:

- `GPU_LIST=0,1,2`
- `MAX_PARALLEL=3`
- one worker process per visible GPU slot
- same fixed subset index
- same output root for resume

The Meriem-strict mapping, LaBraM explicit `labram-base.pth` checkpoint, and CodeBrain partial/native LoRA caveat are unchanged.
