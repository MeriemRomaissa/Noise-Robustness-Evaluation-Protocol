"""Shared NMT OOD pickle loader for Benchmark inference.

The NMT preprocessor writes one pickle per 10-second segment:
``{"X": [23, 2000], "y": 0/1}``.  Model-specific inference scripts can use
this loader to read the common OOD format, then apply their own input transform.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

NMT_EXPECTED_SHAPE = (23, 2000)
NMT_DATASET_NAME = "NMT_OOD"

NmtTransform = Callable[[Any], Any]


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Load numpy>=2 pickles in older numpy environments."""

    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def load_pickle(path: Path) -> dict:
    """Read one NMT segment pickle with numpy version compatibility."""
    with Path(path).open("rb") as f:
        try:
            sample = pickle.load(f)
        except ModuleNotFoundError:
            f.seek(0)
            sample = _NumpyCompatUnpickler(f).load()
    if not isinstance(sample, dict):
        raise TypeError(f"NMT pickle must contain a dict, got {type(sample).__name__}: {path}")
    return sample


class NmtOodDataset(Dataset):
    """Read preprocessed NMT OOD pickles as base ``[23, 2000]`` windows."""

    def __init__(
        self,
        root: str | Path,
        transform: NmtTransform | None = None,
        max_samples: int | None = None,
    ):
        self.root = Path(root)
        self.files = sorted(self.root.glob("*.pkl"))
        if max_samples is not None:
            if max_samples < 1:
                raise ValueError(f"max_samples must be positive, got {max_samples!r}")
            self.files = self.files[: int(max_samples)]
        if not self.files:
            raise FileNotFoundError(f"no NMT .pkl files found in {self.root}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        sample = load_pickle(self.files[idx])
        if "X" not in sample:
            raise KeyError(f"NMT pickle is missing key 'X': {self.files[idx]}")
        y = sample.get("y", sample.get("label"))
        if y is None:
            raise KeyError(f"NMT pickle is missing key 'y' or 'label': {self.files[idx]}")

        x = np.asarray(sample["X"], dtype=np.float32)
        if x.shape != NMT_EXPECTED_SHAPE:
            raise ValueError(
                f"NMT sample shape must be {NMT_EXPECTED_SHAPE}, got {x.shape}: {self.files[idx]}"
            )
        if self.transform is not None:
            x = self.transform(x)
        return torch.as_tensor(x, dtype=torch.float32), int(y)


def build_nmt_ood_loader(
    nmt_dir: str | Path,
    batch_size: int = 64,
    transform: NmtTransform | None = None,
    max_samples: int | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Build a test-only DataLoader for preprocessed NMT OOD pickles."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size!r}")
    if num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {num_workers!r}")
    dataset = NmtOodDataset(nmt_dir, transform=transform, max_samples=max_samples)
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        drop_last=False,
        pin_memory=bool(pin_memory),
    )
