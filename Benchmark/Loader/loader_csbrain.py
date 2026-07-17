"""Prepare bipolar TUAB windows for CSBrain."""

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
DEFAULT_SCALE_MODE = "div100"
SCALE_FACTORS = {
    "mul10000": 10000.0,
    "mul1000": 1000.0,
    "div100": 0.01,
}


def build_unified60_loader(config, split):
    """Read referential unified60 windows and prepare CSBrain input."""
    return build_h5_loader(config, split, transform_h5_window)


def build_original_loader(config, split):
    """Read original CSBrain windows that are already bipolar."""
    expected_shape = (N_CHANNELS, WINDOW_SAMPLES)
    return build_pkl_loader(config, split, transform_pkl_window, expected_shape)


def transform_h5_window(window, channel_names, config=None):
    """Reconstruct bipolar signals, apply selected scaling, and form patches."""
    bipolar_window = reconstruct_bipolar(window, channel_names)
    scaled_window = apply_scale(bipolar_window, selected_scale_mode(config))
    return patch_window(scaled_window)


def transform_pkl_window(window, channel_names=None, config=None):
    """Scale and patch original CSBrain data without reconstructing its montage."""
    scaled_window = apply_scale(window, selected_scale_mode(config))
    return patch_window(scaled_window)


def selected_scale_mode(config) -> str:
    """Read the CSBrain amplitude condition selected in the YAML data section."""
    return (config or {}).get("scale_mode", DEFAULT_SCALE_MODE)


def apply_scale(window, mode: str):
    """Apply one named CSBrain amplitude condition."""
    if mode not in SCALE_FACTORS:
        raise ValueError(
            f"scale_mode must be one of {sorted(SCALE_FACTORS)}, got {mode!r}"
        )
    return window * SCALE_FACTORS[mode]


def patch_window(window):
    """Regroup 2,000 samples into ten non-overlapping 200-sample patches."""
    return reshape_to_patches(
        window,
        n_channels=N_CHANNELS,
        n_patches=N_PATCHES,
        patch_samples=PATCH_SAMPLES,
    )


def validate_loader_batch(batch):
    """Confirm CSBrain receives ``[B,16,10,200]``."""
    expected_shape = (N_CHANNELS, N_PATCHES, PATCH_SAMPLES)
    return check_batch_shape(batch, [expected_shape], "CSBrain")
