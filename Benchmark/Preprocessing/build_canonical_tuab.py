#!/usr/bin/env python3
"""Stage 1 of the unified TUAB preprocessing pipeline.

Reads raw TUAB v3.0.1 EDF files, applies unified preprocessing, and writes
a single canonical HDF5 file:

    canonical_tuab.h5
        /eeg                  (N, 23, 2000)  float32  µV
        /metadata/sample_id
        /metadata/subject_id
        /metadata/recording_id
        /metadata/source_path
        /metadata/label
        /metadata/split
        /metadata/start_time
        /metadata/end_time
        /metadata/raw_hash
        /.attrs               preprocessing parameters

Architecture decisions enforced here:
  - ONE output file only (no Family A / Family B split)
  - Stores ONLY raw referential EEG; no bipolar derivations
  - Adapters are not called here; they run later at load time

Usage:
    python build_canonical_tuab.py --edf_root /path/to/TUAB/edf \
                                   --output /path/to/canonical_tuab.h5 \
                                   --log_dir /path/to/logs \
                                   --report_dir /path/to/reports

    # Force rebuild of an existing file:
    python build_canonical_tuab.py ... --overwrite

    # Dry run (scan EDFs, report counts, do not write HDF5):
    python build_canonical_tuab.py ... --dry_run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    import h5py
except ModuleNotFoundError:
    h5py = None
try:
    import mne
except ModuleNotFoundError:
    mne = None
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None

# Reproducibility constants shared by every recording.
SFREQ          = 200          # Hz — target sampling rate
BANDPASS_HZ    = (0.1, 75.0) # (l_freq, h_freq)
NOTCH_HZ       = 60.0        # power-line frequency for US recordings
WINDOW_SEC     = 10.0        # seconds per segment
WINDOW_SAMPLES = int(WINDOW_SEC * SFREQ)   # 2000
OVERLAP        = 0           # non-overlapping windows
UNITS          = "uV"
DTYPE          = np.float32
META_FIELDS    = {
    "sample_id":    None,  # string dtype is created per HDF5 file
    "subject_id":   None,
    "recording_id": None,
    "source_path":  None,
    "label":        np.int8,
    "split":        None,
    "start_time":   np.float32,
    "end_time":     np.float32,
    "raw_hash":     None,
}
STR_FIELDS = {"sample_id", "subject_id", "recording_id", "source_path", "split", "raw_hash"}

# Suppress MNE's verbose output globally; we do our own logging
if mne is not None:
    mne.set_log_level("ERROR")

# Channel order stored in every canonical window.
CANONICAL_CHANNELS = [
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1",  "O2",  "F7", "F8", "T3", "T4", "T5", "T6",
    "A1",  "A2",  "FZ", "CZ", "PZ", "T1", "T2",
]
N_CHANNELS = len(CANONICAL_CHANNELS)   # 23

# Normalisation map: various EDF naming conventions → canonical name
# Handles REF suffix, Fp vs FP, lowercase, etc.
_ALIAS_MAP: dict[str, str] = {}

def _build_alias_map() -> dict[str, str]:
    """Return a mapping from every expected raw EDF variant → canonical name."""
    aliases: dict[str, str] = {}
    for canon in CANONICAL_CHANNELS:
        variants = {
            canon,
            canon.lower(),
            canon.upper(),
            f"EEG {canon}-REF",
            f"EEG {canon.upper()}-REF",
            f"{canon}-REF",
            f"{canon.upper()}-REF",
            f"EEG {canon}-LE",
            f"{canon}-LE",
        }
        # Fp1 ↔ FP1 special case
        if canon in ("FP1", "FP2"):
            fp_variant = canon.replace("FP", "Fp")
            variants.update({
                fp_variant,
                f"EEG {fp_variant}-REF",
                f"{fp_variant}-REF",
            })
        for v in variants:
            aliases[v.strip()] = canon
    return aliases


_ALIAS_MAP = _build_alias_map()


def _normalise_channel_name(raw_name: str) -> str | None:
    """Map a raw EDF channel name to its canonical name, or None if unknown."""
    name = raw_name.strip()
    if name in _ALIAS_MAP:
        return _ALIAS_MAP[name]
    # Case-insensitive fallback
    name_upper = name.upper()
    for key, canon in _ALIAS_MAP.items():
        if key.upper() == name_upper:
            return canon
    return None


# Dataset labels and upstream split names come from the TUAB directory layout.
def _label_from_path(path: Path) -> int:
    """Derive binary label from EDF file path (normal=0, abnormal=1)."""
    parts = {p.lower() for p in path.parts}
    if "abnormal" in parts:
        return 1
    if "normal" in parts:
        return 0
    raise ValueError(f"Cannot determine label from path: {path}")


def _split_from_path(path: Path) -> str:
    """
    Map TUAB v3.0.1 directory structure to split name.

    edf/train/... → "train"
    edf/eval/...  → "test"   (eval is the official held-out set)
    """
    parts = [p.lower() for p in path.parts]
    if "train" in parts:
        return "train"
    if "eval" in parts:
        return "test"
    raise ValueError(f"Cannot determine split from path: {path}")


def _subject_id_from_path(path: Path) -> str:
    """
    Extract subject ID from TUAB EDF filename.

    TUAB filenames follow: {subject_id}_{session}_{segment}.edf
    e.g.  00000021_s002_t001.edf  → subject_id = "00000021"
    """
    return path.stem.split("_")[0]


def _recording_id_from_path(path: Path) -> str:
    """EDF filename without extension."""
    return path.stem


# Raw hashes make duplicate and provenance checks possible after preprocessing.
def _sha256_of_array(arr: np.ndarray) -> str:
    """Return hex SHA-256 of the raw float32 bytes of arr."""
    return hashlib.sha256(arr.astype(np.float32).tobytes()).hexdigest()


# One recording is filtered, resampled, windowed, and returned for H5 writing.
class RecordingProcessingError(RuntimeError):
    """Carry one structured EDF failure reason into the manifest reports."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


def process_edf(edf_path: Path, logger: logging.Logger) -> "ProcessedFile":
    """Apply the canonical signal protocol to one EDF recording."""
    label = _label_from_path(edf_path)
    split = _split_from_path(edf_path)
    raw = read_edf_recording(edf_path)
    channel_map = match_canonical_channels(raw.ch_names)
    missing_channels = [
        name for name, edf_name in channel_map.items() if edf_name is None
    ]
    if missing_channels:
        logger.info(
            "MISSING_CHANNELS file=%s missing=%s (zero-padded)",
            edf_path.name,
            missing_channels,
        )

    keep_available_channels(raw, channel_map, edf_path)
    warn_for_long_recording(raw, edf_path, logger)
    apply_canonical_filters(raw, edf_path)
    canonical_matrix = assemble_canonical_matrix(raw, channel_map, edf_path)
    windows, metadata = segment_canonical_recording(
        canonical_matrix,
        edf_path,
        split,
        label,
        missing_channels,
        logger,
    )
    return ProcessedFile(split, label, windows, metadata, missing_channels)


def read_edf_recording(edf_path: Path):
    """Read one EDF into memory or raise a structured corruption error."""
    try:
        return mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    except Exception as exc:
        raise RecordingProcessingError(
            "unreadable_edf",
            f"could not read EDF: {exc}",
        ) from exc


def match_canonical_channels(edf_channel_names: list[str]) -> dict[str, str | None]:
    """Map available EDF labels onto the fixed 23-channel canonical order."""
    channel_map: dict[str, str | None] = {
        name: None for name in CANONICAL_CHANNELS
    }
    for edf_name in edf_channel_names:
        canonical_name = _normalise_channel_name(edf_name)
        if canonical_name and channel_map.get(canonical_name) is None:
            channel_map[canonical_name] = edf_name
    return channel_map


def keep_available_channels(raw, channel_map: dict[str, str | None], edf_path: Path) -> None:
    """Restrict MNE processing to canonical channels present in the recording."""
    available = [name for name in channel_map.values() if name is not None]
    if not available:
        raise RecordingProcessingError(
            "no_canonical_channels",
            f"{edf_path.name} contains none of the 23 canonical channels",
        )
    try:
        raw.pick_channels(available, ordered=False)
    except Exception as exc:
        raise RecordingProcessingError(
            "channel_pick_error",
            f"could not select canonical channels in {edf_path.name}: {exc}",
        ) from exc


def warn_for_long_recording(raw, edf_path: Path, logger: logging.Logger) -> None:
    """Flag recordings likely to require unusually long preprocessing time."""
    duration_seconds = float(raw.times[-1])
    if duration_seconds > 3600:
        logger.warning(
            "LARGE_EDF file=%s duration=%.1fs",
            edf_path.name,
            duration_seconds,
        )


def apply_canonical_filters(raw, edf_path: Path) -> None:
    """Band-pass, remove 60 Hz line noise, and resample to 200 Hz."""
    try:
        raw.filter(
            l_freq=BANDPASS_HZ[0],
            h_freq=BANDPASS_HZ[1],
            method="fir",
            fir_window="hamming",
            verbose=False,
        )
        raw.notch_filter(freqs=NOTCH_HZ, method="fir", verbose=False)
        if abs(raw.info["sfreq"] - SFREQ) > 0.1:
            raw.resample(SFREQ, verbose=False)
    except Exception as exc:
        raise RecordingProcessingError(
            "signal_preprocessing_error",
            f"filtering or resampling failed for {edf_path.name}: {exc}",
        ) from exc


def assemble_canonical_matrix(
    raw,
    channel_map: dict[str, str | None],
    edf_path: Path,
) -> np.ndarray:
    """Convert volts to µV and zero-pad missing positions in canonical order."""
    try:
        edf_data, _ = raw[:]
        data_by_name = {
            channel_name: edf_data[index] * 1e6
            for index, channel_name in enumerate(raw.ch_names)
        }
        canonical = np.zeros(
            (N_CHANNELS, edf_data.shape[1]),
            dtype=DTYPE,
        )
        for index, canonical_name in enumerate(CANONICAL_CHANNELS):
            edf_name = channel_map[canonical_name]
            if edf_name is not None:
                canonical[index] = data_by_name[edf_name].astype(DTYPE)
        return canonical
    except Exception as exc:
        raise RecordingProcessingError(
            "canonical_assembly_error",
            f"could not assemble {edf_path.name} in canonical order: {exc}",
        ) from exc


def segment_canonical_recording(
    canonical_matrix: np.ndarray,
    edf_path: Path,
    split: str,
    label: int,
    missing_channels: list[str],
    logger: logging.Logger,
) -> tuple[list[np.ndarray], list[dict]]:
    """Create valid non-overlapping windows and their provenance metadata."""
    number_of_windows = canonical_matrix.shape[1] // WINDOW_SAMPLES
    if number_of_windows == 0:
        raise RecordingProcessingError(
            "recording_too_short",
            f"recording has {canonical_matrix.shape[1]} samples; need {WINDOW_SAMPLES}",
        )

    subject_id = _subject_id_from_path(edf_path)
    recording_id = _recording_id_from_path(edf_path)
    real_indices = [
        index
        for index, name in enumerate(CANONICAL_CHANNELS)
        if name not in missing_channels
    ]
    windows = []
    metadata = []

    for window_index in range(number_of_windows):
        start_sample = window_index * WINDOW_SAMPLES
        end_sample = start_sample + WINDOW_SAMPLES
        window = canonical_matrix[:, start_sample:end_sample].copy()
        if real_indices and not np.any(window[real_indices] != 0):
            logger.info(
                "SKIP_FLATLINE file=%s window=%d (all real channels flat)",
                edf_path.name,
                window_index,
            )
            continue

        windows.append(window)
        metadata.append({
            "sample_id": f"{subject_id}_{recording_id}_{window_index:06d}",
            "subject_id": subject_id,
            "recording_id": recording_id,
            "source_path": str(edf_path),
            "label": label,
            "split": split,
            "start_time": float(start_sample) / SFREQ,
            "end_time": float(end_sample) / SFREQ,
            "raw_hash": _sha256_of_array(window),
        })
    return windows, metadata


# H5 creation and resume checks are kept separate from signal processing.
def _create_hdf5_contents(f: h5py.File, edf_root: Path) -> None:
    """Initialise an open HDF5 file with canonical datasets and attributes."""
    str_dt = h5py.special_dtype(vlen=str)

    # Main EEG dataset — chunked per segment for efficient random access
    f.create_dataset(
        "/eeg",
        shape=(0, N_CHANNELS, WINDOW_SAMPLES),
        maxshape=(None, N_CHANNELS, WINDOW_SAMPLES),
        dtype=np.float32,
        chunks=(1, N_CHANNELS, WINDOW_SAMPLES),
        compression="gzip",
        compression_opts=4,
    )

    # Metadata datasets
    for name, dtype in META_FIELDS.items():
        f.create_dataset(
            f"/metadata/{name}",
            shape=(0,),
            maxshape=(None,),
            dtype=str_dt if name in STR_FIELDS else dtype,
            chunks=(512,),
        )

    # Root attributes — preprocessing provenance
    f.attrs["channel_names"]  = CANONICAL_CHANNELS
    f.attrs["sfreq"]          = SFREQ
    f.attrs["bandpass_hz"]    = list(BANDPASS_HZ)
    f.attrs["notch_hz"]       = NOTCH_HZ
    f.attrs["window_sec"]     = WINDOW_SEC
    f.attrs["window_samples"] = WINDOW_SAMPLES
    f.attrs["overlap"]        = OVERLAP
    f.attrs["units"]          = UNITS
    f.attrs["dtype"]          = "float32"
    f.attrs["created_at"]     = datetime.now(timezone.utc).isoformat()
    f.attrs["source_root"]    = str(edf_root)


def _init_hdf5(h5_path: Path, edf_root: Path) -> h5py.File:
    """Create and initialise the HDF5 file with empty resizable datasets."""
    f = h5py.File(str(h5_path), "w")
    _create_hdf5_contents(f, edf_root)
    return f


def _validate_hdf5_for_resume(f: h5py.File, edf_root: Path) -> None:
    """Raise ValueError if an existing HDF5 file is not compatible for resume."""
    if "/eeg" not in f or "/metadata/source_path" not in f:
        raise ValueError("missing /eeg or /metadata/source_path")
    eeg = f["/eeg"]
    if eeg.ndim != 3 or eeg.shape[1:] != (N_CHANNELS, WINDOW_SAMPLES):
        raise ValueError(f"incompatible /eeg shape {eeg.shape}")
    if eeg.dtype != np.dtype(np.float32):
        raise ValueError(f"incompatible /eeg dtype {eeg.dtype}")
    if str(f.attrs.get("source_root", "")) != str(edf_root):
        raise ValueError(
            f"source_root mismatch: existing={f.attrs.get('source_root')} requested={edf_root}"
        )
    for field in META_FIELDS:
        name = f"/metadata/{field}"
        if name not in f:
            raise ValueError(f"missing metadata dataset {name}")
        if f[name].shape[0] != eeg.shape[0]:
            raise ValueError(f"metadata length mismatch for {name}")


def _decode_h5_string(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _load_resume_state(f: h5py.File, stats: "BuildStats") -> set[str]:
    """Load existing source paths and aggregate stats from a resumable HDF5 file."""
    processed_sources = {_decode_h5_string(v) for v in f["/metadata/source_path"][:]}
    labels = f["/metadata/label"][:]
    splits = [_decode_h5_string(v) for v in f["/metadata/split"][:]]
    stats.total_windows = int(f["/eeg"].shape[0])
    for label in labels:
        label_i = int(label)
        stats.windows_by_label[label_i] = stats.windows_by_label.get(label_i, 0) + 1
    for split in splits:
        stats.windows_by_split[split] = stats.windows_by_split.get(split, 0) + 1
    return processed_sources


def _append_batch(
    f: h5py.File,
    windows: list[np.ndarray],
    metadatas: list[dict],
) -> None:
    """Append a batch of windows and their metadata to the open HDF5 file."""
    if not windows:
        return

    batch = np.stack(windows, axis=0)   # (B, 23, 2000)
    B = batch.shape[0]
    n_existing = f["/eeg"].shape[0]
    new_size = n_existing + B

    f["/eeg"].resize(new_size, axis=0)
    f["/eeg"][n_existing:new_size] = batch

    for field in metadatas[0].keys():
        ds = f[f"/metadata/{field}"]
        ds.resize(new_size, axis=0)
        if field in STR_FIELDS:
            ds[n_existing:new_size] = [m[field] for m in metadatas]
        elif field == "label":
            ds[n_existing:new_size] = np.array([m[field] for m in metadatas], dtype=np.int8)
        else:
            ds[n_existing:new_size] = np.array([m[field] for m in metadatas], dtype=np.float32)


# Build statistics are written alongside the dataset for auditability.
class BuildStats:
    def __init__(self):
        self.total_edfs     = 0
        self.processed_edfs = 0
        self.skipped_edfs   = 0
        self.resumed_edfs   = 0
        self.total_windows  = 0
        self.windows_by_split: dict[str, int] = {}
        self.windows_by_label: dict[int, int]  = {}
        self.missing_channel_counts: dict[str, int] = {}
        self.errors: list[str] = []

    def record_missing(self, channels: list[str]) -> None:
        for ch in channels:
            self.missing_channel_counts[ch] = self.missing_channel_counts.get(ch, 0) + 1

    def record_windows(self, metadatas: list[dict]) -> None:
        for m in metadatas:
            self.total_windows += 1
            split = m["split"]
            label = m["label"]
            self.windows_by_split[split] = self.windows_by_split.get(split, 0) + 1
            self.windows_by_label[label] = self.windows_by_label.get(label, 0) + 1

    def log_summary(self, logger: logging.Logger) -> None:
        logger.info("=" * 70)
        logger.info("BUILD SUMMARY")
        logger.info("=" * 70)
        logger.info("Total EDF files scanned : %d", self.total_edfs)
        logger.info("Successfully processed  : %d", self.processed_edfs)
        logger.info("Already present/resumed : %d", self.resumed_edfs)
        logger.info("Skipped (errors/short)  : %d", self.skipped_edfs)
        logger.info("Total windows stored    : %d", self.total_windows)
        logger.info("")
        logger.info("Windows per split:")
        for split, count in sorted(self.windows_by_split.items()):
            logger.info("  %-10s : %d", split, count)
        logger.info("")
        logger.info("Windows per label:")
        label_names = {0: "normal", 1: "abnormal"}
        for label, count in sorted(self.windows_by_label.items()):
            logger.info("  %d (%s) : %d", label, label_names.get(label, "?"), count)
        logger.info("")
        if self.missing_channel_counts:
            logger.info("Zero-padded channel occurrences (per channel across all files):")
            for ch, cnt in sorted(self.missing_channel_counts.items()):
                logger.info("  %-6s : %d files", ch, cnt)
        else:
            logger.info("No missing channels detected.")
        if self.errors:
            logger.info("")
            logger.info("Errors recorded:")
            for err in self.errors[:20]:
                logger.info("  %s", err)
            if len(self.errors) > 20:
                logger.info("  ... %d more", len(self.errors) - 20)
        logger.info("=" * 70)


# Paths are required so no researcher accidentally writes to a machine-specific default.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build canonical TUAB HDF5 dataset for unified EEG foundation model benchmarking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--edf_root",
        type=Path,
        required=True,
        help="Root directory of raw TUAB v3.0.1 EDF files (must contain train/ and eval/).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for canonical_tuab.h5. Parent directories will be created.",
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        required=True,
        help="Directory for build log files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Required flag to overwrite an existing canonical_tuab.h5. Without this flag the script exits if the output file already exists.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted build by appending EDFs not already present in /metadata/source_path.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Scan EDF files and report counts without writing the HDF5 file.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of windows to buffer in memory before flushing to HDF5.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="(Debug) Process at most this many EDF files then stop.",
    )
    parser.add_argument(
        "--report_dir",
        type=Path,
        required=True,
        help="Directory for manifest, skipped/error CSVs, and build summary reports.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional stable run name for log/report filenames, useful for tail -f in VS Code.",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=25,
        help="Log progress every N EDF files during non-dry-run builds.",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Write optional TensorBoard scalar logs if torch.utils.tensorboard is available.",
    )
    parser.add_argument(
        "--tb_dir",
        type=Path,
        default=None,
        help="TensorBoard log directory. Defaults to <log_dir>/tensorboard/<run_name-or-timestamp>.",
    )
    return parser.parse_args()


# Each run receives its own log and report identifiers.
def _safe_run_name(run_name: str | None, timestamp: str) -> str:
    if not run_name:
        return timestamp
    keep = []
    for ch in run_name:
        keep.append(ch if ch.isalnum() or ch in ("-", "_") else "_")
    return "".join(keep).strip("_") or timestamp


def setup_logging(log_dir: Path, dry_run: bool, run_name: str | None) -> tuple[logging.Logger, Path, str]:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = _safe_run_name(run_name, timestamp)
    prefix = "dryrun_" if dry_run else ""
    log_file = log_dir / f"{prefix}build_canonical_tuab_{run_id}.log"

    logger = logging.getLogger("build_canonical_tuab")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("Log file: %s", log_file)
    return logger, log_file, run_id


def _init_csv(path: Path, fieldnames: list[str]) -> tuple[object, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    fh.flush()
    return fh, writer


def _write_csv_row(fh, writer: csv.DictWriter, row: dict) -> None:
    writer.writerow(row)
    fh.flush()


def _make_tb_writer(args: argparse.Namespace, run_id: str, logger: logging.Logger):
    if not args.tensorboard:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:
        logger.warning("TensorBoard disabled; could not import SummaryWriter: %s", exc)
        return None
    tb_dir = args.tb_dir or (args.log_dir / "tensorboard" / run_id)
    tb_dir.mkdir(parents=True, exist_ok=True)
    logger.info("TensorBoard dir: %s", tb_dir)
    return SummaryWriter(log_dir=str(tb_dir))


def _write_summary_report(
    path: Path,
    args: argparse.Namespace,
    stats: BuildStats,
    log_file: Path,
    manifest_path: Path,
    skipped_path: Path,
    elapsed_s: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "TUAB Canonical Build Summary",
        f"created_at: {datetime.now(timezone.utc).isoformat()}",
        f"edf_root: {args.edf_root}",
        f"output: {args.output}",
        f"log_file: {log_file}",
        f"manifest_csv: {manifest_path}",
        f"skipped_csv: {skipped_path}",
        f"elapsed_seconds: {elapsed_s:.1f}",
        "",
        f"total_edfs_scanned: {stats.total_edfs}",
        f"processed_edfs: {stats.processed_edfs}",
        f"resumed_edfs: {stats.resumed_edfs}",
        f"skipped_edfs: {stats.skipped_edfs}",
        f"total_windows: {stats.total_windows}",
        "",
        "windows_by_split:",
    ]
    for split, count in sorted(stats.windows_by_split.items()):
        lines.append(f"  {split}: {count}")
    lines.append("")
    lines.append("windows_by_label:")
    for label, count in sorted(stats.windows_by_label.items()):
        lines.append(f"  {label}: {count}")
    lines.append("")
    lines.append("missing_channel_counts:")
    if stats.missing_channel_counts:
        for ch, count in sorted(stats.missing_channel_counts.items()):
            lines.append(f"  {ch}: {count}")
    else:
        lines.append("  none")
    if stats.errors:
        lines.append("")
        lines.append("errors:")
        for err in stats.errors:
            lines.append(f"  {err}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class FileTask:
    """Identity and progress information for one EDF file."""

    path: Path
    index: int
    total: int
    started_at: float


@dataclass(frozen=True)
class Failure:
    """One preprocessing outcome that must be written to the skip reports."""

    reason: str
    message: str
    split: str = ""
    label: object = ""
    missing_channels: tuple[str, ...] = ()
    traceback_text: str = ""
    include_in_error_summary: bool = False


@dataclass(frozen=True)
class ProcessedFile:
    """Windows and metadata accepted from one EDF recording."""

    split: str
    label: int
    windows: list
    metadata: list
    missing_channels: list[str]


@dataclass
class BuildContext:
    """Mutable output state shared while EDF files are processed."""

    h5_file: "h5py.File"
    stats: "BuildStats"
    window_buffer: list
    metadata_buffer: list
    processed_sources: set
    manifest_writer: tuple
    skipped_writer: tuple
    batch_size: int
    progress_every: int
    tb_writer: object
    logger: logging.Logger


def _flush_buffer(context: BuildContext) -> None:
    """Append buffered windows to H5, record the new size, then clear the buffer."""
    _append_batch(context.h5_file, context.window_buffer, context.metadata_buffer)
    context.h5_file.attrs["n_segments"] = context.h5_file["/eeg"].shape[0]
    context.h5_file.attrs["last_updated_at"] = datetime.now(timezone.utc).isoformat()
    context.h5_file.flush()
    context.window_buffer.clear()
    context.metadata_buffer.clear()


def _write_manifest(context: BuildContext, task: FileTask, status: str, **values) -> None:
    """Write one consistently shaped manifest record."""
    manifest_fh, manifest_writer = context.manifest_writer
    missing = values.get("missing_channels", ())
    row = {
        "edf_path": str(task.path),
        "status": status,
        "split": values.get("split", ""),
        "label": values.get("label", ""),
        "n_windows": values.get("n_windows", 0),
        "missing_channels": ",".join(missing),
        "message": values.get("message", ""),
        "elapsed_seconds": f"{time.time() - task.started_at:.3f}",
    }
    _write_csv_row(manifest_fh, manifest_writer, row)


def _record_failure(context: BuildContext, task: FileTask, failure: Failure) -> None:
    """Record one skipped EDF in statistics, manifest, and error CSV."""
    skipped_fh, skipped_writer = context.skipped_writer
    context.stats.skipped_edfs += 1
    if failure.include_in_error_summary:
        context.stats.errors.append(f"{task.path.name}: {failure.message}")
    _write_csv_row(skipped_fh, skipped_writer, {
        "edf_path": str(task.path),
        "reason": failure.reason,
        "message": failure.message,
        "traceback": failure.traceback_text,
    })
    _write_manifest(
        context,
        task,
        "skipped",
        split=failure.split,
        label=failure.label,
        missing_channels=failure.missing_channels,
        message=failure.message,
    )


def _record_processed(context: BuildContext, task: FileTask, result: ProcessedFile) -> None:
    """Update statistics and buffers for one successfully processed EDF."""
    stats = context.stats
    stats.processed_edfs += 1
    if result.missing_channels:
        stats.record_missing(result.missing_channels)
    stats.record_windows(result.metadata)
    context.processed_sources.add(str(task.path))
    context.window_buffer.extend(result.windows)
    context.metadata_buffer.extend(result.metadata)
    _write_manifest(
        context,
        task,
        "processed",
        split=result.split,
        label=result.label,
        n_windows=len(result.windows),
        missing_channels=result.missing_channels,
    )

    if len(context.window_buffer) >= context.batch_size:
        _flush_buffer(context)

    if context.tb_writer is not None:
        context.tb_writer.add_scalar("build/processed_edfs", stats.processed_edfs, task.index)
        context.tb_writer.add_scalar("build/skipped_edfs", stats.skipped_edfs, task.index)
        context.tb_writer.add_scalar("build/total_windows", stats.total_windows, task.index)

    if task.index % context.progress_every == 0 or task.index == task.total:
        context.logger.info(
            "PROGRESS files=%d/%d processed=%d resumed=%d skipped=%d windows=%d current=%s",
            task.index, task.total, stats.processed_edfs, stats.resumed_edfs,
            stats.skipped_edfs, stats.total_windows, task.path.name,
        )


def _process_one_file(task: FileTask, context: BuildContext) -> None:
    """Route one EDF to the appropriate resume, failure, or success handler."""
    if str(task.path) in context.processed_sources:
        _write_manifest(
            context,
            task,
            "resumed_existing",
            message="already present in HDF5 source_path metadata",
        )
        return

    split = ""
    label: object = ""
    try:
        split = _split_from_path(task.path)
        label = _label_from_path(task.path)
        result = process_edf(task.path, context.logger)
    except RecordingProcessingError as exc:
        context.logger.warning(
            "SKIP %s file=%s error=%s",
            exc.reason,
            task.path.name,
            exc,
        )
        failure = Failure(
            reason=exc.reason,
            message=str(exc),
            split=split,
            label=label,
        )
        _record_failure(context, task, failure)
        return
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        context.logger.warning("SKIP unhandled error: %s  error=%s", task.path, message)
        failure = Failure(
            reason="unhandled_exception",
            message=message,
            traceback_text=traceback.format_exc(),
            include_in_error_summary=True,
        )
        _record_failure(context, task, failure)
        return

    if not result.windows:
        failure = Failure(
            reason="no_windows",
            message="No valid windows returned after preprocessing/windowing.",
            split=result.split,
            label=result.label,
            missing_channels=tuple(result.missing_channels),
        )
        _record_failure(context, task, failure)
        return

    _record_processed(context, task, result)


def main() -> None:
    args = parse_args()
    require_preprocessing_dependencies()
    run_preprocessing_pipeline(args)


def require_preprocessing_dependencies() -> None:
    """Fail before scanning data when required preprocessing packages are absent."""
    missing = [name for name, module in (("h5py", h5py), ("mne", mne), ("tqdm", tqdm)) if module is None]
    if missing:
        raise RuntimeError(f"Install preprocessing dependencies before running: {', '.join(missing)}")


def run_preprocessing_pipeline(args: argparse.Namespace) -> None:
    """Validate one build request, process recordings, and verify the H5 output."""
    logger, log_file, run_id = setup_logging(args.log_dir, args.dry_run, args.run_name)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.report_dir / f"canonical_tuab_manifest_{run_id}.csv"
    skipped_path = args.report_dir / f"canonical_tuab_skipped_{run_id}.csv"
    summary_path = args.report_dir / f"canonical_tuab_summary_{run_id}.txt"
    tb_writer = _make_tb_writer(args, run_id, logger)

    if args.overwrite and args.resume:
        logger.error("Use only one of --overwrite or --resume, not both.")
        sys.exit(1)
    if args.progress_every < 1:
        logger.warning("--progress_every must be >= 1; using 1")
        args.progress_every = 1

    logger.info("=" * 70)
    logger.info("TUAB Canonical Dataset Builder")
    logger.info("=" * 70)
    logger.info("EDF root   : %s", args.edf_root)
    logger.info("Output     : %s", args.output)
    logger.info("Dry run    : %s", args.dry_run)
    logger.info("Overwrite  : %s", args.overwrite)
    logger.info("Resume     : %s", args.resume)
    logger.info("Report dir : %s", args.report_dir)
    logger.info("Preprocessing:")
    logger.info("  Bandpass : %.1f–%.1f Hz", *BANDPASS_HZ)
    logger.info("  Notch    : %.0f Hz", NOTCH_HZ)
    logger.info("  Resample : %d Hz", SFREQ)
    logger.info("  Window   : %.0f s (%d samples)", WINDOW_SEC, WINDOW_SAMPLES)
    logger.info("  Channels : %d  %s", N_CHANNELS, CANONICAL_CHANNELS)
    logger.info("=" * 70)

    # Fail fast on bad paths or an ambiguous overwrite/resume request.
    if not args.edf_root.is_dir():
        logger.error("EDF root does not exist: %s", args.edf_root)
        sys.exit(1)

    if args.output.exists() and not args.dry_run:
        if args.resume:
            logger.info("Resume requested; existing output will be opened for append: %s", args.output)
        elif not args.overwrite:
            logger.error(
                "Output file already exists: %s\n"
                "Pass --overwrite to replace it or --resume to append missing EDFs.",
                args.output,
            )
            sys.exit(1)
        else:
            logger.warning("Overwriting existing file: %s", args.output)
            args.output.unlink()
    elif args.resume and not args.dry_run:
        logger.info("Resume requested but output does not exist; starting a new build.")

    if not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    # Find every EDF under edf_root; a dry run stops here.
    logger.info("Scanning EDF files...")
    edf_paths = sorted(args.edf_root.rglob("*.edf"))

    if not edf_paths:
        logger.error("No EDF files found under %s", args.edf_root)
        sys.exit(1)

    if args.max_files is not None:
        edf_paths = edf_paths[: args.max_files]
        logger.info("DEBUG: limited to first %d files", args.max_files)

    logger.info("Found %d EDF files", len(edf_paths))

    # Quick sanity: confirm train/ and eval/ directories exist
    for expected_split in ("train", "eval"):
        if not any(expected_split in str(p) for p in edf_paths):
            logger.warning("No files found for expected directory: %s/", expected_split)

    stats = BuildStats()
    stats.total_edfs = len(edf_paths)

    if args.dry_run:
        logger.info("DRY RUN — scanning only, no HDF5 will be written.")
        for edf_path in tqdm(edf_paths, desc="Scanning", unit="file"):
            try:
                label = _label_from_path(edf_path)
                split = _split_from_path(edf_path)
                logger.debug("OK  split=%-5s label=%d  %s", split, label, edf_path.name)
            except ValueError as exc:
                logger.warning("SKIP %s: %s", edf_path.name, exc)
                stats.skipped_edfs += 1
                continue
            stats.processed_edfs += 1
        logger.info("Dry run complete. Scanned %d files.", stats.processed_edfs)
        sys.exit(0)

    # Create or resume the output file, then process each EDF in turn.
    t_start = time.time()
    window_buffer:   list[np.ndarray] = []
    metadata_buffer: list[dict]       = []
    h5_mode = "a" if args.resume and args.output.exists() else "w"
    processed_sources: set[str] = set()

    manifest_fh, manifest_writer = _init_csv(
        manifest_path,
        [
            "edf_path", "status", "split", "label", "n_windows",
            "missing_channels", "message", "elapsed_seconds",
        ],
    )
    skipped_fh, skipped_writer = _init_csv(
        skipped_path,
        ["edf_path", "reason", "message", "traceback"],
    )

    logger.info("Manifest CSV: %s", manifest_path)
    logger.info("Skipped/error CSV: %s", skipped_path)
    logger.info("Opening HDF5 with mode=%s: %s", h5_mode, args.output)

    try:
        with h5py.File(str(args.output), h5_mode) as h5f:
            if h5_mode == "w":
                _create_hdf5_contents(h5f, args.edf_root)
            else:
                _validate_hdf5_for_resume(h5f, args.edf_root)
                processed_sources = _load_resume_state(h5f, stats)
                stats.resumed_edfs = len(processed_sources)
                logger.info(
                    "Resume state: %d EDF source paths already present, %d windows stored.",
                    len(processed_sources), stats.total_windows,
                )

            context = BuildContext(
                h5_file=h5f, stats=stats, window_buffer=window_buffer, metadata_buffer=metadata_buffer,
                processed_sources=processed_sources, manifest_writer=(manifest_fh, manifest_writer),
                skipped_writer=(skipped_fh, skipped_writer), batch_size=args.batch_size,
                progress_every=args.progress_every, tb_writer=tb_writer, logger=logger,
            )
            pbar = tqdm(edf_paths, desc="Processing EDF files", unit="file", dynamic_ncols=True)
            for file_idx, edf_path in enumerate(pbar, start=1):
                pbar.set_postfix({"file": edf_path.name[:30], "windows": stats.total_windows})
                task = FileTask(edf_path, file_idx, len(edf_paths), time.time())
                _process_one_file(task, context)

            # Flush remaining
            if window_buffer:
                _append_batch(h5f, window_buffer, metadata_buffer)
                h5f.flush()

            final_n = h5f["/eeg"].shape[0]
            h5f.attrs["n_segments"] = final_n
            h5f.attrs["last_updated_at"] = datetime.now(timezone.utc).isoformat()
            h5f.flush()
    finally:
        manifest_fh.close()
        skipped_fh.close()
        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()

    t_elapsed = time.time() - t_start
    logger.info("HDF5 written: %s  (%.1f s)", args.output, t_elapsed)
    logger.info("Total segments stored: %d", stats.total_windows)

    # Log and save the run summary, then verify what was written.
    stats.log_summary(logger)
    _write_summary_report(
        summary_path, args, stats, log_file, manifest_path, skipped_path, t_elapsed
    )
    logger.info("Summary report: %s", summary_path)

    # Quick post-write sanity read
    logger.info("Post-write sanity check...")
    try:
        with h5py.File(str(args.output), "r") as h5f:
            eeg_shape = h5f["/eeg"].shape
            assert eeg_shape[1] == N_CHANNELS,    f"Channel dim wrong: {eeg_shape[1]}"
            assert eeg_shape[2] == WINDOW_SAMPLES, f"Time dim wrong: {eeg_shape[2]}"
            assert eeg_shape[0] == stats.total_windows, \
                f"Count mismatch: stored {eeg_shape[0]}, expected {stats.total_windows}"
            assert h5f["/eeg"].dtype == np.dtype(np.float32), f"dtype wrong: {h5f['/eeg'].dtype}"
            forbidden = []
            def _visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    lname = name.lower()
                    if "bipolar" in lname or "family" in lname:
                        forbidden.append(name)
            h5f.visititems(_visit)
            assert not forbidden, f"Forbidden datasets found: {forbidden}"
            logger.info("Sanity check PASSED — shape %s, dtype %s", eeg_shape, h5f["/eeg"].dtype)
    except Exception as exc:
        logger.error("Sanity check FAILED: %s", exc)
        sys.exit(1)

    logger.info("Done. Canonical dataset ready at: %s", args.output)


if __name__ == "__main__":
    main()
