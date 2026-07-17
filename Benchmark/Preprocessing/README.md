## Preprocessing:
Raw EDF Files (TUAB)
        ↓
    STAGE 1: build_canonical_tuab.py  ← Converts raw to unified format
        ↓
   canonical_tuab.h5  (Shared dataset)
        ↓
    STAGE 2: build_canonical_h5_max_coverage_split.py  ← Creates train/val/test splits
        ↓
   split_index.csv  (Tells models which windows to use)
        ↓
    Model Training (LaBraM, EEGPT, BIOT, etc.)

## Stage 1: build_canonical_tuab.py - "Create the Data"
Takes raw EEG recordings and converts them to a single, standardized H5 file.

input Raw TUAB EDF files:
/edf_root/
├── train/
│   ├── abnormal/
│   │   ├── 00000021_s002_t001.edf
│   │   └── ...
│   └── normal/
│       ├── 00000045_s001_t001.edf
│       └── ...
└── eval/
    ├── abnormal/
    └── normal/

output canonical_tuab.h5  (One file)
├── /eeg                (N, 23, 2000)  - All EEG windows
├── /metadata/
│   ├── label           (0=normal, 1=abnormal)
│   ├── split           ("train" or "test")
│   ├── subject_id
│   ├── recording_id
│   ├── source_path
│   └── ...
└── .attrs              (Preprocessing parameters)

## Stage 1 flow:
main
    ↓
check dependencies (h5py, mne, tqdm)
    ↓
run preprocessing pipeline
    ↓
process each EDF  ← LOOP over every .edf file
    ↓
    read EDF (mne.io.read_raw_edf)
    ↓
    match canonical channels (FP1, F7, etc.)
    ↓
    retain available channels
    ↓
    filter, notch and resample
    ↓
    convert to μV and assemble canonical order
    ↓
    zero-pad missing positions
    ↓
    create 10-second windows (sliding window)
    ↓
write and verify canonical H5

## Stage 2: build_canonical_h5_max_coverage_split.py - "Split the Data"
Assigns every window in the H5 file to train/validation/test splits.

input canonical_tuab.h5  (from Stage 1)

output canonical_h5_labram_referenced_max_coverage_split_index.csv
├── h5_index          (Row number in H5 file)
├── canonical_split   (train, val, or test)
├── label             (0 or 1)
├── group_id          (subject or recording ID)
└── ...              (Provenance metadata)

canonical_h5_labram_referenced_max_coverage_split_index.json  (Machine-readable)
canonical_h5_labram_referenced_max_coverage_split_summary.txt (Human-readable)

## Stage 2 flow：
main
    ↓
validate request (paths exist, H5 valid)
    ↓
load H5 metadata (read labels, splits, subject_ids)
    ↓
choose subject or recording grouping
    ↓
calculate LaBraM-referenced validation targets
    (How many normal/abnormal windows for validation?)
    ↓
summarize indivisible groups
    (Group windows by subject/recording)
    ↓
select validation groups
    (Search for groups that best match targets)
    ↓
assign canonical splits
    (Move selected groups from train → val)
    ↓
audit leakage
    (Check no group appears in multiple splits)
    ↓
verify assignments
    (All windows assigned, no duplicates)
    ↓
save CSV, JSON and text reports