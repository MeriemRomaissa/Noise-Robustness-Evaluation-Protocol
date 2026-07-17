"""Prepare canonical TUAB windows for LaBraM."""

from loader_common import (
    build_h5_loader,
    build_pkl_loader,
    check_batch_shape,
    reshape_to_patches,
)

N_CHANNELS = 23
WINDOW_SAMPLES = 2000
N_PATCHES = 10
PATCH_SAMPLES = 200


def build_unified60_loader(config, split):
    """Read one unified60 split and prepare every window for LaBraM."""
    return build_h5_loader(config, split, transform_window)


def build_original_loader(config, split):
    """Read one original LaBraM split and require canonical input windows."""
    expected_shape = (N_CHANNELS, WINDOW_SAMPLES)
    return build_pkl_loader(config, split, transform_window, expected_shape)


def transform_window(window, channel_names=None, config=None):
    """Regroup samples into ten one-second patches without changing values."""
    return reshape_to_patches(
        window,
        n_channels=N_CHANNELS,
        n_patches=N_PATCHES,
        patch_samples=PATCH_SAMPLES,
    )


def validate_loader_batch(batch):
    """Confirm LaBraM receives ``[B,23,10,200]``."""
    expected_shape = (N_CHANNELS, N_PATCHES, PATCH_SAMPLES)
    return check_batch_shape(batch, [expected_shape], "LaBraM")
