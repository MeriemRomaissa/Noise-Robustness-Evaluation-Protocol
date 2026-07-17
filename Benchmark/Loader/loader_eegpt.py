"""Prepare canonical TUAB windows for EEGPT."""

from loader_common import (
    build_h5_loader,
    build_pkl_loader,
    check_batch_shape,
    validate_window_shape,
)

N_CHANNELS = 23
WINDOW_SAMPLES = 2000


def build_unified60_loader(config, split):
    """Read one unified60 split and preserve EEGPT's canonical input."""
    return build_h5_loader(config, split, transform_window)


def build_original_loader(config, split):
    """Read one original EEGPT split and require canonical input windows."""
    expected_shape = (N_CHANNELS, WINDOW_SAMPLES)
    return build_pkl_loader(config, split, transform_window, expected_shape)


def transform_window(window, channel_names=None, config=None):
    """Validate referential input without scaling, re-referencing, or reshaping."""
    validate_window_shape(window, N_CHANNELS, WINDOW_SAMPLES)
    return window


def validate_loader_batch(batch):
    """Confirm EEGPT receives ``[B,23,2000]``."""
    expected_shape = (N_CHANNELS, WINDOW_SAMPLES)
    return check_batch_shape(batch, [expected_shape], "EEGPT")
