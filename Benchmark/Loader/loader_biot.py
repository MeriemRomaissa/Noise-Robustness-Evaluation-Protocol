"""Prepare bipolar TUAB windows for BIOT."""

import numpy as np
import torch

from loader_common import (
    build_h5_loader,
    build_pkl_loader,
    check_batch_shape,
    is_torch_tensor,
    reconstruct_bipolar,
)

N_CHANNELS = 16
WINDOW_SAMPLES = 2000
DEFAULT_NORMALIZATION_EPSILON = 1e-8


def build_unified60_loader(config, split):
    """Read referential unified60 windows and prepare BIOT input."""
    return build_h5_loader(config, split, transform_h5_window)


def build_original_loader(config, split):
    """Read original BIOT windows that are already bipolar."""
    expected_shape = (N_CHANNELS, WINDOW_SAMPLES)
    return build_pkl_loader(config, split, transform_pkl_window, expected_shape)


def transform_h5_window(window, channel_names, config=None):
    """Reconstruct bipolar signals and apply BIOT's robust normalization."""
    bipolar_window = reconstruct_bipolar(window, channel_names)
    epsilon = normalization_epsilon(config)
    return q95_abs_normalize(bipolar_window, epsilon)


def transform_pkl_window(window, channel_names=None, config=None):
    """Normalize original BIOT data without reconstructing its bipolar montage."""
    epsilon = normalization_epsilon(config)
    return q95_abs_normalize(window, epsilon)


def normalization_epsilon(config) -> float:
    """Read the numerical floor used when a channel has near-zero amplitude."""
    settings = config or {}
    return float(
        settings.get("normalization_epsilon", DEFAULT_NORMALIZATION_EPSILON)
    )


def q95_abs_normalize(window, epsilon=DEFAULT_NORMALIZATION_EPSILON):
    """Scale each channel by its 95th-percentile absolute amplitude."""
    if is_torch_tensor(window):
        scale = torch.quantile(
            window.detach().abs().float(),
            0.95,
            dim=-1,
            keepdim=True,
        )
        safe_scale = torch.clamp(scale.to(window), min=epsilon)
        return window / safe_scale

    scale = np.quantile(np.abs(window), 0.95, axis=-1, keepdims=True)
    safe_scale = np.maximum(
        scale.astype(window.dtype, copy=False),
        epsilon,
    )
    return window / safe_scale


def validate_loader_batch(batch):
    """Confirm BIOT receives ``[B,16,2000]``."""
    expected_shape = (N_CHANNELS, WINDOW_SAMPLES)
    return check_batch_shape(batch, [expected_shape], "BIOT")
