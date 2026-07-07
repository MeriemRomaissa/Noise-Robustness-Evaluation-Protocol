# EEG-FM Equivalence Issue Solved

This README belongs under:

```text
unified-eegfm-benchmark/equivalence_issue_solved/
```

It records the ablation tests used to diagnose and resolve the non-equivalence between:

```text
original reproduced EEG-FM pipelines
vs
our unified TUAB preprocessing pipeline / unified60 H5 pipeline
```

Main report file:

```text
eegfm_equivalence_report.pdf
```

This folder is intended to be a lightweight, focused record. It should include the report PDF, this README, and optionally small summary files or archival copies of ablation scripts. It should not include raw EDF files, H5 datasets, PKL datasets, checkpoints, full training outputs, TensorBoard logs, or unrelated code.

---

## 1. Included EEG-FMs

Included in this equivalence-resolution record:

- EEGPT
- BIOT
- CBraMod
- CodeBrain
- CSBrain

LaBraM is excluded because its equivalence issue was solved earlier.

---

## 2. Comparison Rule

The fair comparison is:

```text
original reproduction on the matched subset
vs
unified60 ablation row on the same matched subset
```

Paper TUAB results are only sanity/reference values. They are not direct comparison targets because they usually use a different full-dataset or author-side setting.

Common matched-subset setting:

| Item | Value |
|---|---|
| Task | TUAB binary classification |
| Seed | 42 |
| Epochs | 15 |
| Train / validation / test windows | 8192 / 2048 / 2048 |
| Main metrics | Balanced accuracy and AUROC |
| Data comparison | original reproduction branch vs canonical unified60 H5 branch |

---

## 3. Final Cross-Model Summary

| Model | Original reproduction reference | Debugged unified60 fair row | Main factor resolved | Final status |
|---|---:|---:|---|---|
| EEGPT | B-Acc 0.7140 / AUROC 0.7948 | A3: B-Acc 0.7042 / AUROC 0.8267 | Checkpoint mapping / effective initialization behavior | Reproduction-matched after ablation |
| BIOT | B-Acc 0.7627 / AUROC 0.8512 | B1: B-Acc 0.7646 / AUROC 0.8449 | True bipolar input + scratch author-style recipe | Equivalent after patch |
| CBraMod | B-Acc 0.7891 / AUROC 0.8544 | C3: B-Acc 0.7896 / AUROC 0.8693 | Bipolar `/100` adapter + author recipe + checkpoint mapping | Equivalent after patch |
| CodeBrain | B-Acc 0.7653 / AUROC 0.8510 | C1: B-Acc 0.7897 / AUROC 0.8490 | Wrapper-compatible bipolar adapter + original-wrapper recipe | Equivalent after patch |
| CSBrain | S3: B-Acc 0.7933 / AUROC 0.8599 | S4: B-Acc 0.7884 / AUROC 0.8685 | True bipolar montage + stable effective scale | Equivalent after scale/montage/recipe patch |

---

# 4. Model-by-Model Pipeline and Settings

Each section below lists the original reproduction flow, the previous unified60 issue, the final fair unified60 flow, and the exact configuration used by the fair row. This avoids leaving only row names such as `A3`, `B1`, or `C1` without explaining what those rows mean.

---

## 4.1 EEGPT

### Original reproduction pipeline

| Component | Setting |
|---|---|
| Data source | Original PKL reproduction |
| Dataset scope | Matched subset |
| Input channels | 23-channel TUAB-style input |
| Input shape | `[B, 23, 2000]` |
| Model | EEGPT |
| Learning rate | `5e-4` |
| Weight decay | `0.05` |
| DataLoader workers | `2` |
| Train `drop_last` | `True` |
| Checkpoint behavior | Original-wrapper observed behavior; logs suggested encoder keys may not have been fully used |
| Best epoch | 13 |
| Test B-Acc | 0.7140 |
| Test AUROC | 0.7948 |

Flow:

```text
Original matched subset
-> original PKL/reproduction input
-> 23-channel TUAB-style tensor
-> input shape [B, 23, 2000]
-> EEGPT model
-> original optimizer/DataLoader recipe
-> original reproduction reference
```

### Previous unified60 issue

| Component | Previous unified60 / A0 |
|---|---|
| Data source | Canonical unified60 H5 |
| Input shape | `[B, 23, 2000]` |
| Checkpoint mapping | ON, `encoder.*` remapped to `model.target_encoder.*` |
| Loaded keys | 102 |
| Learning rate | `1e-4` |
| Weight decay | `0.0005` |
| Workers | `8` |
| Train `drop_last` | `False` |
| Best epoch | 2 |
| Test B-Acc / AUROC | 0.6906 / 0.7729 |
| Interpretation | Reproduced the old unified60 mismatch |

### Final fair/debugged unified60 row

Final row:

```text
A3_original_recipe_plus_checkpoint_none
```

| Component | A3 setting |
|---|---|
| Data source | Canonical unified60 H5 |
| Input channels | 23-channel TUAB-style input |
| Input shape | `[B, 23, 2000]` |
| Checkpoint mapping/loading | OFF |
| Loaded keys | 0 |
| Learning rate | `5e-4` |
| Weight decay | `0.05` |
| DataLoader workers | `2` |
| Train `drop_last` | `True` |
| Best epoch | 14 |
| Test B-Acc | 0.7042 |
| Test AUROC | 0.8267 |
| Interpretation | Closest reproduction-matched unified60 row; AUROC exceeds original reproduction |

Flow:

```text
Canonical unified60 H5 matched subset
-> 23-channel tensor
-> input shape [B, 23, 2000]
-> EEGPT model
-> original optimizer/DataLoader recipe
   lr = 5e-4
   weight_decay = 0.05
   workers = 2
   train_drop_last = True
-> checkpoint mapping/loading disabled
-> A3 reproduction-matched unified60 row
```

### EEGPT ablation rows

| Row | Data source | Checkpoint mapping | Recipe | Purpose |
|---|---|---|---|---|
| A0 | Unified60 H5 | ON, 102 keys | Current unified | Reproduce previous unified60 baseline |
| A1 | Unified60 H5 | ON, 102 keys | Original optimizer/DataLoader | Test whether original recipe alone fixes the gap |
| A2 | Unified60 H5 | OFF, 0 keys | Current unified | Test checkpoint mapping/loading effect |
| A3 | Unified60 H5 | OFF, 0 keys | Original optimizer/DataLoader | Final reproduction-matched row |

Main finding:

```text
EEGPT non-equivalence was mainly related to checkpoint mapping / effective initialization behavior, not montage or input shape.
```

---

## 4.2 BIOT

### Original reproduction pipeline

| Component | Setting |
|---|---|
| Data source | Original BIOT-style PKL from matched EDF subset |
| Input channels | True 16-channel bipolar montage |
| Window | 10 s at 200 Hz |
| Input shape | `[B, 16, 2000]` |
| Normalization | q95 absolute normalization |
| Model | BIOTClassifier |
| Loss | BCEWithLogitsLoss |
| Optimizer | Adam |
| Learning rate | `1e-3` |
| Weight decay | `1e-5` |
| Scheduler | None |
| Checkpoint | OFF / scratch |
| FFT/token size | 200 |
| Hop length | 100 |
| Best metric | Validation AUROC |
| Test B-Acc / AUROC | 0.7627 / 0.8512 |

Flow:

```text
Original matched EDF/PKL subset
-> true 16-channel bipolar montage
-> 10 s window at 200 Hz
-> input shape [B, 16, 2000]
-> q95 absolute normalization
-> BIOTClassifier
-> BCEWithLogitsLoss
-> Adam lr=1e-3, weight_decay=1e-5
-> no PREST checkpoint / scratch setting
-> best validation AUROC
```

### Previous unified60 issue

| Component | Previous unified60 behavior |
|---|---|
| Input | Endpoint/current channel behavior rather than true bipolar reconstruction |
| Loss | CrossEntropyLoss |
| Checkpoint | PREST/pretrained behavior ON |
| Test B-Acc / AUROC | 0.7536 / 0.8202 |
| Interpretation | Not fair because it did not match original scratch reproduction |

### Final fair/debugged unified60 row

Final row:

```text
B1_bipolar_scratch_recipe
```

| Component | B1 setting |
|---|---|
| Data source | Canonical unified60 H5 |
| Adapter | `bipolar_qnorm` |
| Input channels | True 16-channel bipolar reconstruction |
| Input shape | `[B, 16, 2000]` |
| Normalization | q95 absolute normalization |
| Model | BIOTClassifier |
| Loss | BCEWithLogitsLoss |
| Optimizer | Adam |
| Learning rate | `1e-3` |
| Weight decay | `1e-5` |
| Scheduler | None |
| Checkpoint | OFF / scratch |
| FFT/token size | 200 |
| Hop length | 100 |
| Best metric | Validation AUROC |
| Best epoch | 6 |
| Test B-Acc | 0.7646 |
| Test AUROC | 0.8449 |

Flow:

```text
Canonical unified60 H5 matched subset
-> true 16-channel bipolar reconstruction
-> bipolar_qnorm adapter
-> q95 absolute normalization
-> input shape [B, 16, 2000]
-> BIOTClassifier
-> BCEWithLogitsLoss
-> Adam lr=1e-3, weight_decay=1e-5
-> no PREST checkpoint / scratch setting
-> B1_bipolar_scratch_recipe
```

### BIOT ablation rows

| Row | Adapter | Recipe | Checkpoint | Purpose |
|---|---|---|---|---|
| B0 | Endpoint/current | Scratch/github vanilla | OFF | Current adapter under scratch recipe |
| B1 | True bipolar q95-normalized | Scratch/github vanilla | OFF | Final fair row |
| B2 | Endpoint/current | Pretrained control | ON | Pretrained endpoint control |
| B3 | True bipolar q95-normalized | Pretrained control | ON | Best performance control, but not fair scratch row |

Main finding:

```text
BIOT unified60 became equivalent after using true bipolar reconstruction, q95 normalization, BCE loss, Adam optimizer, and scratch/no-PREST training.
```

---

## 4.3 CBraMod

### Original reproduction pipeline

| Component | Setting |
|---|---|
| Data source | Original matched PKL subset |
| Input channels | 16 author-style bipolar derivations |
| Scale | `/100` |
| Model | CBraMod |
| Checkpoint | ON |
| Recipe | Author-style |
| Weight decay | `0.05` |
| Scheduler | Cosine per batch |
| Gradient clipping | `1.0` |
| Best metric | Validation AUROC |
| Best epoch | 5 |
| Test B-Acc / AUROC | 0.7891 / 0.8544 |

Flow:

```text
Original matched PKL subset
-> author-style 16-channel bipolar input
-> /100 scaling
-> CBraMod model
-> checkpoint ON
-> author-style recipe
   weight_decay = 0.05
   scheduler = cosine per batch
   gradient clipping = 1.0
   best metric = validation AUROC
-> original reproduction reference
```

### Previous unified60 issue

| Component | Previous unified60 behavior |
|---|---|
| Adapter | Endpoint/current channel selection |
| Recipe | Unified recipe |
| Weight decay | `0.0005` |
| Scheduler | None |
| Test B-Acc / AUROC | 0.7470 / 0.8189 |
| Interpretation | Not author-equivalent input |

### Final fair/debugged unified60 row

Final row:

```text
C3_adapter_plus_author_recipe
```

| Component | C3 setting |
|---|---|
| Data source | Canonical unified60 H5 |
| Adapter | `bipolar_div100` |
| Montage | True 16 bipolar derivations |
| Scale | `/100` |
| Model | CBraMod |
| Checkpoint | ON |
| Loaded keys | 209 |
| Checkpoint mapping | Backbone-prefix mapping |
| Model size | Approximately 69.3M parameters |
| Recipe | Author-style |
| Weight decay | `0.05` |
| Scheduler | Cosine per batch |
| Gradient clipping | `1.0` |
| Best metric | Validation AUROC |
| Best epoch | 5 |
| Test B-Acc | 0.7896 |
| Test AUROC | 0.8693 |

The 16 bipolar derivations are:

```text
FP1-F7, F7-T3, T3-T5, T5-O1,
FP2-F8, F8-T4, T4-T6, T6-O2,
FP1-F3, F3-C3, C3-P3, P3-O1,
FP2-F4, F4-C4, C4-P4, P4-O2
```

Flow:

```text
Canonical unified60 H5 matched subset
-> bipolar_div100 adapter
-> reconstruct 16 author-style bipolar derivations
-> /100 scaling
-> CBraMod model
-> checkpoint mapping ON, 209 keys
-> author-style recipe
-> C3_adapter_plus_author_recipe
```

### CBraMod ablation rows

| Row | Adapter | Recipe | Checkpoint | Purpose |
|---|---|---|---|---|
| C0 | Current/endpoint | Unified | ON | Reproduce old/current unified-style reference |
| C1 | `bipolar_div100` | Unified | ON | Test adapter effect |
| C2 | Current/endpoint | Author | ON | Test author recipe alone |
| C3 | `bipolar_div100` | Author | ON | Final fair row |
| C4 | `bipolar_div100` | Author | OFF | Test checkpoint contribution |

Main finding:

```text
The dominant CBraMod issue was the input adapter. Author recipe alone did not fix the wrong input. bipolar_div100 plus author recipe and checkpoint mapping made unified60 equivalent.
```

---

## 4.4 CodeBrain

### Original reproduction pipeline

| Component | Setting |
|---|---|
| Data source | Original matched EDF subset |
| Input | Wrapper-compatible CodeBrain TUAB input |
| Adapter / representation | Bipolar-div100-style 16-channel representation |
| Input shape | `[B, 16, 10, 200]` |
| Forward path | Wrapper-only flattened TUAB forward path |
| Recipe | `original_wrapper` |
| Checkpoint | ON |
| Loaded keys | 269 |
| Best metric | Validation AUROC |
| Best epoch | 2 |
| Test B-Acc / AUROC | 0.7653 / 0.8510 |

Flow:

```text
Original matched EDF subset
-> wrapper-compatible CodeBrain TUAB input
-> bipolar_div100-style 16-channel representation
-> input shape [B, 16, 10, 200]
-> wrapper-only flattened TUAB forward path
-> original_wrapper recipe
-> checkpoint ON, 269 keys
-> best validation AUROC
```

### Previous unified60 issue

| Component | Previous unified60 behavior |
|---|---|
| Adapter | Earlier/current unified adapter behavior |
| Recipe | Not the fair original-wrapper-equivalent setting |
| Test B-Acc / AUROC | 0.7437 / 0.8053 |
| Interpretation | Not equivalent, especially lower AUROC |

### Final fair/debugged unified60 row

Final row:

```text
C1_bipolar_original_recipe
```

| Component | C1 setting |
|---|---|
| Data source | Canonical unified60 H5 |
| Adapter | `bipolar_div100` |
| Montage | True bipolar reconstruction |
| Input shape | `[B, 16, 10, 200]` |
| Forward path | Adapter-side flattened TUAB forward |
| Recipe | `original_wrapper` |
| Checkpoint | ON |
| Loaded keys | 269 |
| Best metric | Validation AUROC |
| Best epoch | 3 |
| Test B-Acc | 0.7897 |
| Test AUROC | 0.8490 |

Flow:

```text
Canonical unified60 H5 matched subset
-> bipolar_div100 true bipolar reconstruction
-> adapter-side flattened TUAB forward path
-> input shape [B, 16, 10, 200]
-> original_wrapper recipe
-> checkpoint ON, 269 keys
-> C1_bipolar_original_recipe
```

### CodeBrain ablation rows

| Row | Adapter | Recipe | Checkpoint | Purpose |
|---|---|---|---|---|
| C0 | Current/endpoint | Original wrapper | ON | Current adapter under reproduction recipe |
| C1 | `bipolar_div100` | Original wrapper | ON | Final fair row |
| C2 | Current/endpoint | Author default | ON | Author-default endpoint control |
| C3 | `bipolar_div100` | Author default | ON | Author-default bipolar control |
| C4 | `bipolar_div100` | Author default | OFF | Checkpoint-off control |

Main finding:

```text
The fair CodeBrain comparison is not the raw/default author path. It is the original-wrapper-equivalent row. Once unified60 used bipolar_div100 and the original_wrapper recipe, it became equivalent.
```

---

## 4.5 CSBrain

### Original reproduction pipeline

| Component | Setting |
|---|---|
| Data source | Original matched PKL subset |
| Montage | True 16-channel bipolar input |
| Input shape | `[B, 16, 10, 200]` |
| Model | CSBrain |
| Checkpoint | ON |
| Loaded keys | 293 |
| Loss | BCEWithLogitsLoss |
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| Weight decay | `0.05` |
| Scheduler | CosineAnnealingLR per train batch |
| Eta min | `1e-6` |
| Gradient clipping | `1.0` |
| Best metric | Validation balanced accuracy |
| Scale settings tested | `*10000`, `*1000`, `/100` |
| Main original fair row | `S3_original_scale_div100` |
| Test B-Acc | 0.7933 |
| Test AUROC | 0.8599 |

Flow:

```text
Original matched PKL subset
-> true 16-channel bipolar input
-> input shape [B, 16, 10, 200]
-> /100 effective scale for strongest fair pair
-> CSBrain model
-> checkpoint ON, 293 keys
-> BCEWithLogitsLoss
-> AdamW lr=1e-4, weight_decay=0.05
-> CosineAnnealingLR per train batch, eta_min=1e-6
-> gradient clipping = 1.0
-> best validation balanced accuracy
-> S3_original_scale_div100
```

### Scale and control rows tested

| Scale / setting | Original row | Unified row | Interpretation |
|---|---|---|---|
| Strict author-loader `*10000` | S2 | S6 | Weak/unstable scale control |
| Scale-sanity `*1000` | S1 | S5 | Stable close/equivalent control |
| Effective-scale `/100` | S3 | S4 | Strongest final fair pair |
| Endpoint-current | N/A | S7 | Historical control only; not fair because not true bipolar |

### Final fair/debugged unified60 row

Final row:

```text
S4_unified_bipolar_div100
```

| Component | S4 setting |
|---|---|
| Data source | Canonical unified60 H5 |
| Montage | True 16-channel bipolar reconstruction |
| Input shape | `[B, 16, 10, 200]` |
| Scale | `/100` effective scale |
| Model | CSBrain |
| Checkpoint | ON |
| Loaded keys | 293 |
| Loss | BCEWithLogitsLoss |
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| Weight decay | `0.05` |
| Scheduler | CosineAnnealingLR per train batch |
| Eta min | `1e-6` |
| Gradient clipping | `1.0` |
| Best metric | Validation balanced accuracy |
| Best epoch | 2 |
| Test B-Acc | 0.7884 |
| Test AUROC | 0.8685 |

Flow:

```text
Canonical unified60 H5 matched subset
-> true 16-channel bipolar reconstruction
-> input shape [B, 16, 10, 200]
-> /100 effective scale
-> CSBrain model
-> checkpoint ON, 293 keys
-> BCEWithLogitsLoss
-> AdamW lr=1e-4, weight_decay=0.05
-> CosineAnnealingLR per train batch, eta_min=1e-6
-> gradient clipping = 1.0
-> best validation balanced accuracy
-> S4_unified_bipolar_div100
```

### CSBrain ablation rows

| Row | Data source | Montage | Scale | Purpose |
|---|---|---|---|---|
| S2 | Original PKL | True bipolar | `*10000` | Strict author-loader scale control |
| S6 | Unified H5 | True bipolar | `*10000` | Unified strict author-loader scale control |
| S1 | Original PKL | True bipolar | `*1000` | Scale-sanity original control |
| S5 | Unified H5 | True bipolar | `*1000` | Scale-sanity unified control |
| S3 | Original PKL | True bipolar | `/100` | Main original fair row |
| S4 | Unified H5 | True bipolar | `/100` | Main unified fair row |
| S7 | Unified H5 | Endpoint-current | identity | Historical control only |

Main finding:

```text
CSBrain unified60 became equivalent after aligning true bipolar montage, checkpoint, training recipe, and stable effective scale. S7 endpoint-current can train, but it is not used as the fair row because it is not true bipolar.
```

---

# 5. Terminal Commands Used / Reference Commands

These commands are included for traceability. They assume the working directory:

```text
/nicoletye/workspace/unified_tuab
```

## 5.1 General Command Pattern

```bash
cd /nicoletye/workspace/unified_tuab

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
/nicoletye/venvs/<ENV_NAME>/bin/python \
scripts/eegfm_adapters/<ABLATION_SCRIPT>.py \
  --epochs 15 \
  --batch_size 64 \
  --seed 42 \
  --device auto \
  2>&1 | tee reports/<REPORT_FOLDER>/<TERMINAL_LOG>.log
```

## 5.2 EEGPT

```bash
cd /nicoletye/workspace/unified_tuab

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
/nicoletye/venvs/eegpt_labram/bin/python \
scripts/eegfm_adapters/run_eegpt_unified60_ablation_seed42.py \
  --epochs 15 \
  --batch_size 64 \
  --seed 42 \
  --device auto \
  2>&1 | tee reports/eegpt_unified60_ablation_seed42_terminal.log
```

## 5.3 BIOT

```bash
cd /nicoletye/workspace/unified_tuab

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
/nicoletye/venvs/biot_env/bin/python \
scripts/eegfm_adapters/run_biot_unified60_ablation_seed42.py \
  --epochs 15 \
  --batch_size 64 \
  --seed 42 \
  --device auto \
  2>&1 | tee reports/biot_unified60_ablation_seed42_terminal.log
```

## 5.4 CBraMod

```bash
cd /nicoletye/workspace/unified_tuab

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
/nicoletye/venvs/cbramod_csbrain/bin/python \
scripts/eegfm_adapters/run_cbramod_unified60_ablation_seed42.py \
  --epochs 15 \
  --batch_size 64 \
  --seed 42 \
  --device auto \
  2>&1 | tee reports/cbramod_unified60_ablation_seed42_terminal.log
```

## 5.5 CodeBrain

```bash
cd /nicoletye/workspace/unified_tuab

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
/nicoletye/venvs/cbramod_csbrain/bin/python \
scripts/eegfm_adapters/run_codebrain_unified60_ablation_seed42.py \
  --epochs 15 \
  --batch_size 64 \
  --seed 42 \
  --device auto \
  2>&1 | tee reports/codebrain_unified60_ablation_seed42_terminal.log
```

## 5.6 CSBrain Final Ablation

```bash
cd /nicoletye/workspace/unified_tuab

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
NUM_WORKERS=8 \
/nicoletye/venvs/cbramod_csbrain/bin/python \
scripts/eegfm_adapters/run_csbrain_final_ablation_seed42_1gpu.py \
  --epochs 15 \
  --batch_size 64 \
  --num_workers 8 \
  --seed 42 \
  --device cuda:0 \
  2>&1 | tee reports/csbrain_forensic_audit_seed42/csbrain_final_ablation_seed42_1gpu_terminal.log
```

---

# 6. Scripts Used and Script Relationships

The ablation scripts used during debugging were:

```text
scripts/eegfm_adapters/run_eegpt_unified60_ablation_seed42.py
scripts/eegfm_adapters/run_biot_unified60_ablation_seed42.py
scripts/eegfm_adapters/run_cbramod_unified60_ablation_seed42.py
scripts/eegfm_adapters/run_codebrain_unified60_ablation_seed42.py
scripts/eegfm_adapters/build_csbrain_forensic_audit_seed42.py
scripts/eegfm_adapters/run_csbrain_smoke_ablation_seed42.py
scripts/eegfm_adapters/run_csbrain_final_ablation_seed42_1gpu.py
```

If copies of these scripts are added under `equivalence_issue_solved/`, they should be treated as archival snapshots of the exact ablation logic used during debugging.

## 6.1 General Script Relationship

```text
Original reproduction wrappers
        |
        v
Matched subset / Option 1 split validation
        |
        v
Unified60 H5 adapter construction
        |
        v
Model-specific ablation launcher
        |
        v
Ablation rows isolate one factor at a time:
  - adapter / montage
  - normalization / scaling
  - checkpoint loading
  - optimizer and scheduler recipe
  - best-metric selection
        |
        v
metrics.json / summary.csv / summary.md / summary.json
        |
        v
eegfm_equivalence_report.pdf
```

## 6.2 CSBrain-Specific Script Relationship

```text
build_csbrain_forensic_audit_seed42.py
        |
        v
Creates forensic audit package:
  - author repo audit
  - original wrapper audit
  - unified60 adapter audit
  - input distribution pair audit
  - smoke matrix design
        |
        v
run_csbrain_smoke_ablation_seed42.py
        |
        v
Runs S1-S7 smoke rows:
  - *10000 strict author-scale control
  - *1000 scale-sanity control
  - /100 debugged effective-scale pair
  - endpoint-current historical control
        |
        v
run_csbrain_final_ablation_seed42_1gpu.py
        |
        v
Runs final 15-epoch rows sequentially on 1 GPU
```

---

# 7. Final Report-Ready Conclusion

The EEG-FM non-equivalence issue was resolved by model-specific ablation rather than by changing the canonical unified60 dataset itself.

Across EEGPT, BIOT, CBraMod, CodeBrain, and CSBrain, the earlier mismatches were traced to implementation-level factors such as:

- endpoint-channel adapters,
- missing true bipolar reconstruction,
- normalization or scale mismatch,
- checkpoint mapping behavior,
- forward-path wrapper differences,
- optimizer / scheduler / DataLoader recipe differences,
- best-metric selection differences.

After applying the correct model-specific adapter and recipe settings, the patched unified60 rows became performance-equivalent to the original reproduction references on the matched subset.

Therefore, the previous non-equivalence should not be attributed to unified preprocessing failure. It should be attributed to EEG-FM-specific adapter, checkpoint, scale, and wrapper differences that have now been documented and resolved.
