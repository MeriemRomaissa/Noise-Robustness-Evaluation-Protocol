"""Apply the reproducible 23-channel versus 16-signal LaBraM comparison."""

from channel_common import keep_16_and_zero_7, shape_of, validate_canonical_window

VALID_MODES = {"23channels", "16channels_zeropadded"}


def use_all_23_channels(window):
    """Keep every LaBraM channel unchanged for the control condition."""
    validate_canonical_window(window)
    return window


def use_16_channels_with_zero_padding(window, channel_names):
    """Keep 16 signals and zero seven positions in a canonical LaBraM window."""
    validate_canonical_window(window)
    return keep_16_and_zero_7(window, channel_names)


def apply_channel_case(window, channel_names, mode: str):
    """Apply the selected channel condition and restore LaBraM's input shape."""
    original_shape = shape_of(window)
    canonical_window = to_canonical_shape(window)

    if mode == "23channels":
        result = use_all_23_channels(canonical_window)
    elif mode == "16channels_zeropadded":
        result = use_16_channels_with_zero_padding(canonical_window, channel_names)
    else:
        raise ValueError(f"channel mode must be one of {sorted(VALID_MODES)}, got {mode!r}")

    return result.reshape(original_shape)


def to_canonical_shape(window):
    """View patched or unpatched LaBraM input as canonical 2,000-sample windows."""
    shape = shape_of(window)
    if shape == (23, 10, 200):
        return window.reshape(23, 2000)
    if len(shape) == 4 and shape[1:] == (23, 10, 200):
        return window.reshape(shape[0], 23, 2000)
    validate_canonical_window(window)
    return window


def describe_case_difference() -> str:
    """Explain the single experimental variable changed by this study case."""
    return "Keep all 23 signals, or zero seven positions before restoring the same LaBraM input shape."
