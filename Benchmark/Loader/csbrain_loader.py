"""Prepare original or unified60 TUAB windows for CSBrain."""

from loader_common import (
    BIPOLAR_PAIRS,
    build_original_loader as build_common_original_loader,
    build_unified60_loader as build_common_unified60_loader,
    reconstruct_bipolar_16,
    reshape_to_patches,
    shape_of,
    validate_window_shape,
)

VALID_SCALE_MODES = {"mul10000", "mul1000", "div100"}


def get_loader_settings() -> dict:
    """Describe the preprocessing for CSBrain."""
    return {
        "model": "CSBrain",
        "channels": 16,
        "montage": "bipolar",
        "input_shape": [None, 16, 10, 200],
        "sampling_rate_hz": 200,
        "window_samples": 2000,
        "window_seconds": 10,
        "scale_or_normalization": "Choose mul10000, mul1000, or div100; default div100. Reshape to patches",
    }


def reshape_to_10x200(window):
    """Keep the established model-specific name for the shared patch operation."""
    return reshape_to_patches(window, channels=16)


def build_unified60_loader(config: dict, split: str):
    """Load unified60 data using the selected CSBrain scaling experiment."""
    return build_common_unified60_loader(config, split, transform_unified60_input)


def build_original_loader(config: dict, split: str):
    """Load original bipolar PKL data using the same selected scale mode."""
    return build_common_original_loader(config, split, transform_original_input)


def transform_unified60_input(window, channel_names=None, config=None):
    """Reconstruct, scale, and patch [23,2000] referential unified60 data."""
    config = config or {}
    bipolar = reconstruct_bipolar_16(window, channel_names)
    scaled = apply_csbrain_scale(bipolar, config.get("scale_mode", "div100"))
    return reshape_to_patches(scaled, channels=16)


def transform_original_input(window, channel_names=None, config=None):
    """Scale and patch original [16,2000] bipolar data without reconstructing it."""
    del channel_names
    config = config or {}
    validate_window_shape(window, (16, 2000), allow_batch=True)
    scaled = apply_csbrain_scale(window, config.get("scale_mode", "div100"))
    return reshape_to_patches(scaled, channels=16)


def apply_csbrain_scale(window, scale_mode: str = "div100"):
    """Paper used mul10000, ablation showed div100 is the most stable."""
    if scale_mode == "mul10000":
        return window * 10000.0
    if scale_mode == "mul1000":
        return window * 1000.0
    if scale_mode == "div100":
        return window / 100.0
    raise ValueError(f"unknown scale_mode={scale_mode!r}; expected {sorted(VALID_SCALE_MODES)}")


def validate_loader_batch(batch) -> bool:
    """Confirm that a CSBrain batch is [B,16,10,200]."""
    eeg = batch[0] if isinstance(batch, (tuple, list)) else batch
    shape = shape_of(eeg)
    if len(shape) == 4 and shape[1:] == (16, 10, 200):
        return True
    raise ValueError(f"CSBrain batch must be [B,16,10,200], got {list(shape)}")
