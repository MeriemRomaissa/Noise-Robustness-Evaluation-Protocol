"""Shared loading infrastructure for the six TUAB EEG foundation models.

Model modules describe how one EEG window must be transformed. This module
owns the repeated mechanics: configuration validation, split parsing, H5 and
PKL access, channel lookup, shape checks, tensor conversion, and batching.

Scientific assumptions
----------------------
The unified60 H5 source contains 10-second, 200 Hz windows with 23 referential
channels in ``[23, 2000]`` order. Model transforms may preserve that montage or
reconstruct the documented 16-channel bipolar montage. Original PKL files for
the four bipolar models are already bipolar and must not be reconstructed again.
"""

from __future__ import annotations

import csv
import pickle
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

CANONICAL_CHANNELS = 23
WINDOW_SAMPLES = 2000
VALID_SPLITS = {"train", "val", "test"}
REQUIRED_SPLIT_COLUMNS = {"h5_index", "canonical_split", "label"}

BIPOLAR_PAIRS = [
    ("FP1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
    ("FP2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
    ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
]

WindowTransform = Callable[[Any, list[str] | None, dict], Any]


def build_h5_loader(
    config: dict, split: str, transform: WindowTransform
) -> DataLoader:
    """Connect a model transform to the shared unified60 dataset."""
    validate_loader_config(config, source="unified60")
    split = validate_split(split)
    dataset = Unified60Dataset(config, split, transform)
    return make_dataloader(dataset, config, split)


def build_pkl_loader(
    config: dict,
    split: str,
    transform: WindowTransform,
    expected_shape: tuple[int, ...],
) -> DataLoader:
    """Connect a model transform to the shared original-PKL dataset."""
    validate_loader_config(config, source="original")
    split = validate_split(split)
    dataset = OriginalPklDataset(config, split, transform, expected_shape)
    return make_dataloader(dataset, config, split)


def validate_loader_config(config: dict, source: str) -> None:
    """Validate shared configuration once before opening files or workers."""
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dictionary, got {type(config).__name__}")

    required_by_source = {
        "unified60": ("h5_path", "split_index_path"),
        "original": ("original_data_path",),
    }
    if source not in required_by_source:
        raise ValueError(
            f"source must be 'unified60' or 'original'; got {source!r}"
        )
    for key in required_by_source[source]:
        require_config(config, key)

    validate_positive_integer(config, "batch_size", default=64)
    validate_nonnegative_integer(config, "num_workers", default=0)
    if config.get("max_samples") is not None:
        validate_positive_integer(config, "max_samples", default=1)
    for key in ("shuffle", "drop_last", "pin_memory"):
        if key in config and not isinstance(config[key], bool):
            raise TypeError(
                f"config['{key}'] must be a boolean, "
                f"got {type(config[key]).__name__}"
            )


def validate_positive_integer(config: dict, key: str, default: int) -> int:
    """Return a positive integer configuration option or raise a clear error."""
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"config['{key}'] must be a positive integer; got {value!r}")
    return int(value)


def validate_nonnegative_integer(config: dict, key: str, default: int) -> int:
    """Return a non-negative integer option or raise a clear error."""
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(
            f"config['{key}'] must be a non-negative integer; got {value!r}"
        )
    return int(value)


def make_dataloader(dataset: Dataset, config: dict, split: str) -> DataLoader:
    """Group samples using the shared, configurable six-model loading policy.

    Training is shuffled by default; validation and test are not. The caller may
    explicitly override ``shuffle``. Incomplete batches are retained unless
    ``drop_last=True`` is requested, and pinned memory is opt-in.
    """
    split = validate_split(split)
    return DataLoader(
        dataset,
        batch_size=validate_positive_integer(config, "batch_size", 64),
        shuffle=config.get("shuffle", split == "train"),
        num_workers=validate_nonnegative_integer(config, "num_workers", 0),
        drop_last=config.get("drop_last", False),
        pin_memory=config.get("pin_memory", False),
    )


class Unified60Dataset(Dataset):
    """Read indexed ``[23,2000]`` referential windows from unified60 H5."""

    def __init__(self, config: dict, split: str, transform: WindowTransform):
        self.h5_path = str(require_config(config, "h5_path"))
        split_path = str(require_config(config, "split_index_path"))
        self.h5_indices, self.labels = read_split_rows(split_path, split)
        sample_limit = config.get("max_samples")
        if sample_limit is not None:
            self.h5_indices = self.h5_indices[: int(sample_limit)]
            self.labels = self.labels[: int(sample_limit)]
        self.config = dict(config)
        self.transform = transform
        self._h5 = None

        with self._open_file() as h5_file:
            if "eeg" not in h5_file:
                raise KeyError(f"H5 file {self.h5_path} does not contain dataset 'eeg'")
            if int(self.h5_indices.max()) >= len(h5_file["eeg"]):
                raise IndexError(
                    f"split index {int(self.h5_indices.max())} exceeds the last EEG "
                    f"row {len(h5_file['eeg']) - 1} in {self.h5_path}"
                )
            self.channel_names = read_h5_channel_names(h5_file, self.config)

    def _open_file(self):
        """Open the H5 source in read-only mode for the current process."""
        import h5py

        return h5py.File(self.h5_path, "r")

    def _get_file(self):
        """Reuse one H5 handle per process and therefore per loader worker."""
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
        return (
            torch.as_tensor(window, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )

    def __del__(self):
        h5_file = getattr(self, "_h5", None)
        if h5_file is not None:
            try:
                h5_file.close()
            except Exception:
                pass


class OriginalPklDataset(Dataset):
    """Read model-ready EEG windows from sorted, trusted per-sample PKLs."""

    def __init__(
        self,
        config: dict,
        split: str,
        transform: WindowTransform,
        expected_shape: tuple[int, ...],
    ):
        root = Path(require_config(config, "original_data_path"))
        self.files = sorted((root / split).glob("*.pkl"))
        sample_limit = config.get("max_samples")
        if sample_limit is not None:
            self.files = self.files[: int(sample_limit)]
        if not self.files:
            raise FileNotFoundError(f"no PKL files found in {root / split}")
        self.config = dict(config)
        self.transform = transform
        self.expected_shape = tuple(expected_shape)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, sample_number: int):
        path = self.files[sample_number]
        sample = load_pkl_sample(path)
        window = sample["X"]
        if window.shape != self.expected_shape:
            raise ValueError(
                f"expected X shape {self.expected_shape}, got {window.shape} in {path}"
            )
        window = self.transform(window, None, self.config)
        return (
            torch.as_tensor(window, dtype=torch.float32),
            torch.tensor(float(sample["y"]), dtype=torch.float32),
        )


def load_pkl_sample(path: Path) -> dict:
    """Load one trusted PKL sample and validate its required ``X`` and ``y`` keys."""
    try:
        with path.open("rb") as file:
            sample = pickle.load(file)
    except (OSError, pickle.UnpicklingError, EOFError) as error:
        raise ValueError(f"could not read PKL sample {path}: {error}") from error

    if not isinstance(sample, dict):
        raise TypeError(f"expected a dictionary in {path}, got {type(sample).__name__}")
    missing = {"X", "y"} - sample.keys()
    if missing:
        raise KeyError(f"missing keys {sorted(missing)} in {path}; expected 'X' and 'y'")
    return {"X": np.asarray(sample["X"], dtype=np.float32), "y": sample["y"]}


def validate_split(split: str) -> str:
    """Normalize a requested split and accept upstream ``eval`` as ``test``."""
    if not isinstance(split, str):
        raise TypeError(f"split must be a string, got {type(split).__name__}")
    normalized = normalize_split_name(split)
    if normalized not in VALID_SPLITS:
        raise ValueError(
            f"split must be 'train', 'val', or 'test'; got {split!r}"
        )
    return normalized


def normalize_split_name(raw_split: str) -> str:
    """Treat the upstream name ``eval`` as the canonical ``test`` split."""
    normalized = raw_split.strip().lower()
    return "test" if normalized == "eval" else normalized


def read_split_rows(path: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Return H5 indices and labels for one validated canonical data split."""
    split = validate_split(split)
    indices: list[int] = []
    labels: list[int] = []

    try:
        with open(path, newline="", encoding="utf-8") as file:
            rows = csv.DictReader(file)
            missing = REQUIRED_SPLIT_COLUMNS - set(rows.fieldnames or [])
            if missing:
                raise ValueError(
                    f"split index {path} is missing columns {sorted(missing)}; "
                    f"required={sorted(REQUIRED_SPLIT_COLUMNS)}"
                )
            for line_number, row in enumerate(rows, start=2):
                if normalize_split_name(row["canonical_split"]) != split:
                    continue
                try:
                    indices.append(int(row["h5_index"]))
                    labels.append(int(row["label"]))
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"invalid h5_index or label at {path}:{line_number}"
                    ) from error
    except OSError as error:
        raise FileNotFoundError(f"could not read split index {path}: {error}") from error

    if not indices:
        raise ValueError(f"no rows found for split={split!r} in {path}")
    if min(indices) < 0:
        raise ValueError(f"negative H5 index found for split={split!r} in {path}")
    return np.asarray(indices, dtype=np.int64), np.asarray(labels, dtype=np.int64)


def read_h5_channel_names(h5_file, config: dict | None = None) -> list[str]:
    """Read and validate 23 canonical channel names from config or H5 metadata."""
    config = config or {}
    raw_names = config.get("channel_names")
    if raw_names is None and "channel_names" in h5_file.attrs:
        raw_names = h5_file.attrs["channel_names"]
    if raw_names is None:
        raise ValueError(
            "channel names are required in config['channel_names'] or "
            "H5 attrs['channel_names']"
        )
    names = [decode_channel_name(name) for name in raw_names]
    if len(names) != CANONICAL_CHANNELS:
        raise ValueError(
            f"expected {CANONICAL_CHANNELS} channel names, got {len(names)}"
        )
    return names


def decode_channel_name(name: Any) -> str:
    """Convert byte-valued H5 channel metadata into ordinary stripped text."""
    if isinstance(name, bytes):
        return name.decode("utf-8", errors="replace").strip()
    return str(name).strip()


def clean_channel_name(name: Any) -> str:
    """Normalize EEG prefixes, reference suffixes, whitespace, and case."""
    return (
        decode_channel_name(name)
        .upper()
        .replace("EEG ", "")
        .replace("-REF", "")
        .strip()
    )


def build_channel_lookup(channel_names: Sequence[Any]) -> dict[str, int]:
    """Map canonical and legacy temporal-channel names to array positions."""
    if len(channel_names) != CANONICAL_CHANNELS:
        raise ValueError(
            f"expected {CANONICAL_CHANNELS} channel names, got {len(channel_names)}"
        )
    aliases = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
    lookup: dict[str, int] = {}
    for index, name in enumerate(channel_names):
        cleaned = clean_channel_name(name)
        if cleaned in lookup:
            raise ValueError(f"duplicate channel name {cleaned!r} at index {index}")
        lookup[cleaned] = index
        for legacy, modern in aliases.items():
            if cleaned == legacy:
                lookup.setdefault(modern, index)
            elif cleaned == modern:
                lookup.setdefault(legacy, index)
    return lookup


def reconstruct_bipolar(
    window: Any,
    channel_names: Sequence[Any] | None,
    pairs: Sequence[tuple[str, str]] = BIPOLAR_PAIRS,
):
    """Convert 23 referential channels into the requested bipolar derivations."""
    if channel_names is None:
        raise ValueError("channel_names is required for bipolar reconstruction")
    single_window = validate_window_shape(
        window, CANONICAL_CHANNELS, WINDOW_SAMPLES
    )
    batched = (
        window.reshape(1, CANONICAL_CHANNELS, WINDOW_SAMPLES)
        if single_window
        else window
    )
    lookup = build_channel_lookup(channel_names)

    bipolar_channels = []
    for left, right in pairs:
        if left not in lookup or right not in lookup:
            raise ValueError(
                f"missing bipolar channel {left}-{right}; "
                f"available={sorted(lookup)}"
            )
        bipolar_channels.append(
            batched[:, lookup[left], :] - batched[:, lookup[right], :]
        )

    bipolar = stack_channels(bipolar_channels, batched)
    return bipolar[0] if single_window else bipolar


def validate_window_shape(window: Any, n_channels: int, n_samples: int) -> bool:
    """Validate one window or batch and return whether it is unbatched."""
    actual = shape_of(window)
    expected = (n_channels, n_samples)
    if actual == expected:
        return True
    if len(actual) == 3 and actual[1:] == expected:
        return False
    raise ValueError(
        f"EEG window must be [{n_channels},{n_samples}] or "
        f"[B,{n_channels},{n_samples}]; got {list(actual)}"
    )


def reshape_to_patches(
    window: Any,
    n_channels: int,
    n_patches: int = 10,
    patch_samples: int = 200,
):
    """Regroup samples into patches without changing values or their order."""
    n_samples = n_patches * patch_samples
    single_window = validate_window_shape(window, n_channels, n_samples)
    batched = (
        window.reshape(1, n_channels, n_samples) if single_window else window
    )
    patched = batched.reshape(
        batched.shape[0], n_channels, n_patches, patch_samples
    )
    return patched[0] if single_window else patched


def check_batch_shape(
    batch: Any,
    allowed_trailing_shapes: Sequence[tuple[int, ...]],
    model_name: str,
) -> bool:
    """Confirm a completed batch matches one of a model's accepted shapes."""
    eeg = batch[0] if isinstance(batch, (tuple, list)) else batch
    actual = shape_of(eeg)
    valid = len(actual) >= 2 and any(
        actual[1:] == tuple(expected) for expected in allowed_trailing_shapes
    )
    if valid:
        return True
    expected_text = " or ".join(
        f"[B,{','.join(str(dimension) for dimension in expected)}]"
        for expected in allowed_trailing_shapes
    )
    raise ValueError(
        f"{model_name} batch must be {expected_text}; got {list(actual)}"
    )


def shape_of(value: Any) -> tuple[int, ...]:
    """Return an array or tensor shape as an ordinary tuple of integers."""
    shape = getattr(value, "shape", None)
    if shape is None:
        raise TypeError(
            f"expected a NumPy array or PyTorch tensor with a shape; "
            f"got {type(value).__name__}"
        )
    return tuple(int(dimension) for dimension in shape)


def is_torch_tensor(value: Any) -> bool:
    """Return whether a value is a PyTorch tensor."""
    return isinstance(value, torch.Tensor)


def stack_channels(values: list[Any], like: Any):
    """Stack channels without changing the NumPy or PyTorch container type."""
    if is_torch_tensor(like):
        return torch.stack(values, dim=1)
    return np.stack(values, axis=1)


def require_config(config: dict, key: str):
    """Return one required configuration value or identify what is missing."""
    if key not in config or config[key] is None or config[key] == "":
        raise ValueError(f"config['{key}'] is required for loading")
    return config[key]
