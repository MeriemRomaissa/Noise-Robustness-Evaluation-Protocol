"""Prepare original or unified60 TUAB windows for BIOT."""

import numpy as np
import torch

from loader_common import (
    BIPOLAR_PAIRS,
    build_original_loader as build_common_original_loader,
    build_unified60_loader as build_common_unified60_loader,
    reconstruct_bipolar_16,
    shape_of,
    validate_window_shape,
)


def get_loader_settings() -> dict:
    """Describe the preprocessing for BIOT."""
    return {
        "model": "BIOT",
        "channels": 16,
        "montage": "bipolar",
        "input_shape": [None, 16, 2000],
        "sampling_rate_hz": 200,
        "window_samples": 2000,
        "window_seconds": 10,
        "scale_or_normalization": "Per-window, per-channel q95 absolute normalization.",
    }


def build_unified60_loader(config: dict, split: str):
    """Load referential unified60 data, reconstruct bipolar channels, and normalize."""
    return build_common_unified60_loader(config, split, transform_unified60_input)


def build_original_loader(config: dict, split: str):
    """Load original bipolar PKL data and apply BIOT normalization."""
    return build_common_original_loader(config, split, transform_original_input)


def transform_unified60_input(window, channel_names=None, config=None):
    """Convert [23,2000] referential data into normalized [16,2000] bipolar data."""
    del config
    return q95_abs_normalize(reconstruct_bipolar_16(window, channel_names))


def transform_original_input(window, channel_names=None, config=None):
    """Normalize original [16,2000] bipolar data without reconstructing it again."""
    del channel_names, config
    validate_window_shape(window, (16, 2000), allow_batch=True)
    return q95_abs_normalize(window)


def q95_abs_normalize(window, eps: float = 1e-8):
    """Divide each channel by its 95th-percentile absolute amplitude."""
    if isinstance(window, torch.Tensor):
        scale = torch.quantile(window.detach().abs().float(), 0.95, dim=-1, keepdim=True)
        scale = scale.to(device=window.device, dtype=window.dtype)
        return window / torch.clamp(scale, min=eps)
    scale = np.quantile(np.abs(window), 0.95, axis=-1, keepdims=True)
    scale = scale.astype(window.dtype, copy=False)
    return window / np.maximum(scale, eps)


def validate_loader_batch(batch) -> bool:
    """Confirm that a BIOT batch is [B,16,2000]."""
    eeg = batch[0] if isinstance(batch, (tuple, list)) else batch
    shape = shape_of(eeg)
    if len(shape) == 3 and shape[1:] == (16, 2000):
        return True
    raise ValueError(f"BIOT batch must be [B,16,2000], got {list(shape)}")
