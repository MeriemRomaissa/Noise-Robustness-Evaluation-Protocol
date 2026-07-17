"""Prepare bipolar TUAB windows for CBraMod."""

from loader_common import (
    build_h5_loader,
    build_pkl_loader,
    check_batch_shape,
    reconstruct_bipolar,
    reshape_to_patches,
)

N_CHANNELS = 16
WINDOW_SAMPLES = 2000
N_PATCHES = 10
PATCH_SAMPLES = 200
AMPLITUDE_DIVISOR = 100.0


def build_unified60_loader(config, split):
    """Read referential unified60 windows and prepare CBraMod input."""
    return build_h5_loader(config, split, transform_h5_window)


def build_original_loader(config, split):
    """Read original CBraMod windows that are already bipolar."""
    expected_shape = (N_CHANNELS, WINDOW_SAMPLES)
    return build_pkl_loader(config, split, transform_pkl_window, expected_shape)


def transform_h5_window(window, channel_names, config=None):
    """Reconstruct bipolar signals, scale amplitudes, and form model patches."""
    bipolar_window = reconstruct_bipolar(window, channel_names)
    scaled_window = bipolar_window / AMPLITUDE_DIVISOR
    return patch_window(scaled_window)


def transform_pkl_window(window, channel_names=None, config=None):
    """Scale and patch original CBraMod data without reconstructing its montage."""
    scaled_window = window / AMPLITUDE_DIVISOR
    return patch_window(scaled_window)


def patch_window(window):
    """Regroup 2,000 samples into ten non-overlapping 200-sample patches."""
    return reshape_to_patches(
        window,
        n_channels=N_CHANNELS,
        n_patches=N_PATCHES,
        patch_samples=PATCH_SAMPLES,
    )


def validate_loader_batch(batch):
    """Confirm CBraMod receives ``[B,16,10,200]``."""
    expected_shape = (N_CHANNELS, N_PATCHES, PATCH_SAMPLES)
    return check_batch_shape(batch, [expected_shape], "CBraMod")
