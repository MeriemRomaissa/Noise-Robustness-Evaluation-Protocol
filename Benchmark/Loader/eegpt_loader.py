"""Prepare original or unified60 TUAB windows for EEGPT."""

from loader_common import (
    build_original_loader as build_common_original_loader,
    build_unified60_loader as build_common_unified60_loader,
    shape_of,
    validate_window_shape,
)

EEGPT_USE_CHANNELS = [
    "FP1", "FPZ", "FP2", "F7", "F3", "FZ", "F4", "F8", "T7", "C3",
    "CZ", "C4", "T8", "P7", "P3", "PZ", "P4", "P8", "O1", "O2",
]


def get_loader_settings() -> dict:
    """Describe the preprocessing for EEGPT."""
    return {
        "model": "EEGPT",
        "channels": 23,
        "montage": "referential",
        "input_shape": [None, 23, 2000],
        "sampling_rate_hz": 200,
        "window_samples": 2000,
        "window_seconds": 10,
        "scale_or_normalization": "No bipolar reconstruction or scaling in this loader.",
    }


def build_unified60_loader(config: dict, split: str):
    """Load one unified60 split without changing its referential EEG values."""
    return build_common_unified60_loader(config, split, transform_unified60_input)


def build_original_loader(config: dict, split: str):
    """Load one original PKL split without changing its referential EEG values."""
    return build_common_original_loader(config, split, transform_unified60_input)


def transform_unified60_input(window, channel_names=None, config=None):
    """Validate and preserve one [23,2000] window or a batch of such windows."""
    del channel_names, config
    validate_window_shape(window, (23, 2000), allow_batch=True)
    return window


def validate_loader_batch(batch) -> bool:
    """Confirm that an EEGPT batch is [B,23,2000]."""
    eeg = batch[0] if isinstance(batch, (tuple, list)) else batch
    shape = shape_of(eeg)
    if len(shape) == 3 and shape[1:] == (23, 2000):
        return True
    raise ValueError(f"EEGPT batch must be [B,23,2000], got {list(shape)}")
