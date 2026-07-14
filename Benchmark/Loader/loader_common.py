"""Shared data-loading tools for the six TUAB EEG foundation models.

Model modules define how one EEG window is prepared. This module handles the
unchanging work: locating samples, reading H5 or PKL data, and making batches.
"""

from __future__ import annotations

import csv
import pickle
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

CANONICAL_CHANNELS = 23
WINDOW_SAMPLES = 2000
BIPOLAR_PAIRS = [
    ("FP1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
    ("FP2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
    ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
]

WindowTransform = Callable[[Any, list[str] | None, dict], Any]


def build_unified60_loader(config: dict, split: str, transform: WindowTransform) -> DataLoader:
    """Build one train, validation, or test loader from the unified60 H5 data."""
    dataset = Unified60Dataset(config, split, transform)
    return make_dataloader(dataset, config, split)


def build_original_loader(config: dict, split: str, transform: WindowTransform) -> DataLoader:
    """Build one train, validation, or test loader from original PKL samples."""
    dataset = OriginalPklDataset(config, split, transform)
    return make_dataloader(dataset, config, split)


def make_dataloader(dataset: Dataset, config: dict, split: str) -> DataLoader:
    """Group samples into batches using the shared six-model loading policy.

    Training is shuffled by default. Validation and test data are not shuffled.
    Incomplete batches are retained unless ``drop_last=True`` is requested.
    """
    return DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 64)),
        shuffle=bool(config.get("shuffle", split == "train")),
        num_workers=int(config.get("num_workers", 0)),
        drop_last=bool(config.get("drop_last", False)),
        pin_memory=bool(config.get("pin_memory", False)),
    )


class Unified60Dataset(Dataset):
    """Read indexed referential EEG windows from the shared unified60 H5 file."""

    def __init__(self, config: dict, split: str, transform: WindowTransform):
        self.h5_path = str(require_config(config, "h5_path"))
        split_path = str(require_config(config, "split_index_path"))
        self.h5_indices, self.labels = read_split_rows(split_path, split)
        self.config = dict(config)
        self.transform = transform
        self._h5 = None

        with self._open_file() as h5_file:
            self.channel_names = read_h5_channels(h5_file, self.config)
        self._h5 = None  # Each DataLoader worker must open its own H5 handle.

    def _open_file(self):
        import h5py

        return h5py.File(self.h5_path, "r")

    def _get_file(self):
        if self._h5 is None:
            self._h5 = self._open_file()
        return self._h5

    def __len__(self) -> int:
        return len(self.h5_indices)

    def __getitem__(self, sample_number: int):
        h5_index = int(self.h5_indices[sample_number])
        window = self._get_file()["eeg"][h5_index].astype(np.float32)
        window = self.transform(window, self.channel_names, self.config)
        label = float(self.labels[sample_number])
        return torch.as_tensor(window, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

    def __del__(self):
        h5_file = getattr(self, "_h5", None)
        if h5_file is not None:
            try:
                h5_file.close()
            except Exception:
                pass


class OriginalPklDataset(Dataset):
    """Read EEG windows from sorted per-sample PKL files."""

    def __init__(self, config: dict, split: str, transform: WindowTransform):
        root = Path(require_config(config, "original_data_path"))
        self.files = sorted((root / split).glob("*.pkl"))
        if not self.files:
            raise FileNotFoundError(f"no PKL files found in {root / split}")
        self.config = dict(config)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, sample_number: int):
        sample = load_pkl_sample(self.files[sample_number])
        window = self.transform(sample["X"], None, self.config)
        label = float(sample["y"])
        return torch.as_tensor(window, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


def load_pkl_sample(path: Path) -> dict:
    """Load one trusted PKL sample and convert its EEG array to float32."""
    with path.open("rb") as file:
        sample = pickle.load(file)
    if "X" not in sample or "y" not in sample:
        raise KeyError(f"expected keys 'X' and 'y' in {path}")
    return {"X": np.asarray(sample["X"], dtype=np.float32), "y": sample["y"]}


def read_split_rows(path: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Return H5 row indices and labels for one canonical data split."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be train, val, or test; got {split!r}")

    indices: list[int] = []
    labels: list[int] = []
    with open(path, newline="", encoding="utf-8") as file:
        rows = csv.DictReader(file)
        required = {"h5_index", "canonical_split", "label"}
        missing = required - set(rows.fieldnames or [])
        if missing:
            raise ValueError(f"split index missing columns: {sorted(missing)}")

        for row in rows:
            if normalize_split_name(row["canonical_split"]) == split:
                indices.append(int(row["h5_index"]))
                labels.append(int(row["label"]))

    if not indices:
        raise ValueError(f"no rows found for split={split!r} in {path}")
    return np.asarray(indices, dtype=np.int64), np.asarray(labels, dtype=np.int64)


def normalize_split_name(raw_split: str) -> str:
    """"Rename the CSV split ``eval`` to ``test`` for consistency with the rest of the codebase."""
    normalized = raw_split.strip().lower()
    return "test" if normalized == "eval" else normalized


def read_h5_channels(h5_file, config: dict) -> list[str]:
    """Read the 23 canonical channel names from config or H5 metadata."""
    raw_names = config.get("channel_names")
    if not raw_names and "channel_names" in h5_file.attrs:
        raw_names = h5_file.attrs["channel_names"]
    if raw_names is None:
        raise ValueError("channel_names is required in config or H5 attributes")

    names = [decode_channel_name(name) for name in raw_names]
    if len(names) != CANONICAL_CHANNELS:
        raise ValueError(f"expected {CANONICAL_CHANNELS} channel names, got {len(names)}")
    return names


def decode_channel_name(name: Any) -> str:
    """Convert byte-valued H5 channel metadata into ordinary text."""
    if isinstance(name, bytes):
        return name.decode("utf-8", errors="replace").strip()
    return str(name).strip()


def reconstruct_bipolar_16(window: Any, channel_names: list[str] | None):
    """Convert 23 referential channels into the shared 16 bipolar derivations."""
    if channel_names is None:
        raise ValueError("channel_names is required for bipolar reconstruction")
    validate_window_shape(window, (CANONICAL_CHANNELS, WINDOW_SAMPLES), allow_batch=True)
    single_window = len(shape_of(window)) == 2
    batched = window.reshape(1, CANONICAL_CHANNELS, WINDOW_SAMPLES) if single_window else window
    lookup = build_channel_lookup(channel_names)

    bipolar_channels = []
    for left, right in BIPOLAR_PAIRS:
        if left not in lookup or right not in lookup:
            raise ValueError(f"missing bipolar channel {left}-{right}; available={sorted(lookup)}")
        bipolar_channels.append(batched[:, lookup[left], :] - batched[:, lookup[right], :])

    bipolar = stack_channels(bipolar_channels, batched)
    return bipolar[0] if single_window else bipolar


def reshape_to_patches(window: Any, channels: int):
    """Regroup a 10-second window into ten non-overlapping one-second patches."""
    validate_window_shape(window, (channels, WINDOW_SAMPLES), allow_batch=True)
    single_window = len(shape_of(window)) == 2
    batched = window.reshape(1, channels, WINDOW_SAMPLES) if single_window else window
    patched = batched.reshape(batched.shape[0], channels, 10, 200)
    return patched[0] if single_window else patched


def validate_window_shape(window: Any, expected: tuple[int, int], allow_batch: bool = False) -> None:
    """Raise a clear error when an EEG window has unexpected dimensions."""
    actual = shape_of(window)
    valid = actual == expected or (allow_batch and len(actual) == 3 and actual[1:] == expected)
    if not valid:
        batch_form = f" or [B,{expected[0]},{expected[1]}]" if allow_batch else ""
        raise ValueError(f"expected [{expected[0]},{expected[1]}]{batch_form}, got {list(actual)}")


def shape_of(value: Any) -> tuple[int, ...]:
    """Return an array or tensor shape as an ordinary tuple of integers."""
    shape = getattr(value, "shape", None)
    if shape is None:
        raise ValueError("expected an array or tensor with shape")
    return tuple(int(dimension) for dimension in shape)


def build_channel_lookup(channel_names: list[str]) -> dict[str, int]:
    """Map canonical and legacy temporal-channel names to array positions."""
    if len(channel_names) != CANONICAL_CHANNELS:
        raise ValueError(f"expected {CANONICAL_CHANNELS} channel names, got {len(channel_names)}")

    aliases = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
    lookup: dict[str, int] = {}
    for index, name in enumerate(channel_names):
        cleaned = clean_channel_name(name)
        lookup[cleaned] = index
        for legacy, modern in aliases.items():
            if cleaned == legacy:
                lookup.setdefault(modern, index)
            elif cleaned == modern:
                lookup.setdefault(legacy, index)
    return lookup


def clean_channel_name(name: Any) -> str:
    """Convert differently formatted channel labels into one standard name."""
    return decode_channel_name(name).upper().replace("EEG ", "").replace("-REF", "").strip()


def stack_channels(values: list[Any], like: Any):
    """Stack reconstructed channels without changing NumPy or PyTorch type."""
    if isinstance(like, torch.Tensor):
        return torch.stack(values, dim=1)
    return np.stack(values, axis=1)


def require_config(config: dict, key: str):
    """Return a required config setting or raise an error naming the missing setting."""
    if not config or not config.get(key):
        raise ValueError(f"config['{key}'] is required for loading")
    return config[key]
