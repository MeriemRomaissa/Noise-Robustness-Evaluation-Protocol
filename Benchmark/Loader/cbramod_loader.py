"""Prepare original or unified60 TUAB windows for CBraMod."""

from loader_common import (
    BIPOLAR_PAIRS,
    build_original_loader as build_common_original_loader,
    build_unified60_loader as build_common_unified60_loader,
    reconstruct_bipolar_16,
    reshape_to_patches,
    shape_of,
    validate_window_shape,
)

def get_loader_settings() -> dict:
    """Describe the preprocessing for CBraMod."""
    return {
        "model": "CBraMod",
        "channels": 16,
        "montage": "bipolar",
        "input_shape": [None, 16, 10, 200],
        "sampling_rate_hz": 200,
        "window_samples": 2000,
        "window_seconds": 10,
        "scale_or_normalization": "Reconstruct bipolar, divide by 100, reshape to patches",
    }


def reshape_to_10x200(window):
    """Keep the established model-specific name for the shared patch operation."""
    return reshape_to_patches(window, channels=16)


def build_unified60_loader(config: dict, split: str):
    """Load unified60 data and apply the complete CBraMod input recipe."""
    return build_common_unified60_loader(config, split, transform_unified60_input)


def build_original_loader(config: dict, split: str):
    """Load original bipolar PKL data and apply scaling and patching."""
    return build_common_original_loader(config, split, transform_original_input)


def transform_unified60_input(window, channel_names=None, config=None):
    """Reconstruct, divide, and patch [23,2000] referential unified60 data."""
    del config
    bipolar = reconstruct_bipolar_16(window, channel_names)
    return reshape_to_patches(bipolar / 100.0, channels=16)


def transform_original_input(window, channel_names=None, config=None):
    """Divide and patch original [16,2000] bipolar data."""
    del channel_names, config
    validate_window_shape(window, (16, 2000), allow_batch=True)
    return reshape_to_patches(window / 100.0, channels=16)


def validate_loader_batch(batch) -> bool:
    """Confirm that a CBraMod batch is [B,16,10,200]."""
    eeg = batch[0] if isinstance(batch, (tuple, list)) else batch
    shape = shape_of(eeg)
    if len(shape) == 4 and shape[1:] == (16, 10, 200):
        return True
    raise ValueError(f"CBraMod batch must be [B,16,10,200], got {list(shape)}")
