"""Prepare original or unified60 TUAB windows for LaBraM."""

from loader_common import (
    build_original_loader as build_common_original_loader,
    build_unified60_loader as build_common_unified60_loader,
    reshape_to_patches,
    shape_of,
    validate_window_shape,
)

N_CHANNELS = 23
WINDOW_SAMPLES = 2000


def get_loader_settings() -> dict:
    """Describe the preprocessing for LaBraM."""
    return {
        "model": "LaBraM",
        "channels": 23,
        "montage": "referential",
        "input_shape": {"canonical": [None, 23, 2000], "model_ready": [None, 23, 10, 200]},
        "sampling_rate_hz": 200,
        "window_samples": 2000,
        "window_seconds": 10,
        "scale_or_normalization": "No scaling or re-referencing in this loader.",
    }


def build_unified60_loader(config: dict, split: str):
    """Load one unified60 split and prepare each referential window for LaBraM."""
    return build_common_unified60_loader(config, split, transform_input)


def build_original_loader(config: dict, split: str):
    """Load one original PKL split and prepare each referential window for LaBraM."""
    return build_common_original_loader(config, split, transform_input)


def transform_input(window, channel_names=None, config=None):
    """Validate [23,2000] input and optionally return [23,10,200] patches."""
    del channel_names
    config = config or {}
    validate_window_shape(window, (N_CHANNELS, WINDOW_SAMPLES), allow_batch=True)
    return reshape_to_patches(window, N_CHANNELS) if config.get("patched", True) else window


def to_canonical_input(window):
    """Validate one canonical [23,2000] window and preserve its values."""
    validate_window_shape(window, (N_CHANNELS, WINDOW_SAMPLES), allow_batch=False)
    return window


def to_patched_input(window):
    """Convert one canonical [23,2000] window into [23,10,200] patches."""
    validate_window_shape(window, (N_CHANNELS, WINDOW_SAMPLES), allow_batch=False)
    return reshape_to_patches(window, N_CHANNELS)


def transform_unified60_input(window, channel_names=None, config=None):
    """Keep the established public transform name used by existing callers."""
    return transform_input(window, channel_names, config)


def validate_loader_batch(batch) -> bool:
    """Confirm that a LaBraM batch has canonical or patched dimensions."""
    eeg = batch[0] if isinstance(batch, (tuple, list)) else batch
    shape = shape_of(eeg)
    if (len(shape) == 3 and shape[1:] == (23, 2000)) or (
        len(shape) == 4 and shape[1:] == (23, 10, 200)
    ):
        return True
    raise ValueError(f"LaBraM batch must be [B,23,2000] or [B,23,10,200], got {list(shape)}")
