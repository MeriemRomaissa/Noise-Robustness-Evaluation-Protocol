"""Apply the reproducible 23-channel versus 16-signal EEGPT comparison."""

from channel_common import keep_16_and_zero_7, validate_canonical_window

VALID_MODES = {"23channels", "16channels_zeropadded"}


def use_all_23_channels(window):
    """Keep every EEGPT channel unchanged for the control condition."""
    validate_canonical_window(window)
    return window


def use_16_channels_with_zero_padding(window, channel_names):
    """Keep 16 signals and zero seven positions without changing EEGPT's shape."""
    validate_canonical_window(window)
    return keep_16_and_zero_7(window, channel_names)


def apply_channel_case(window, channel_names, mode: str):
    """Apply the channel condition selected in the experiment configuration."""
    if mode == "23channels":
        return use_all_23_channels(window)

    if mode == "16channels_zeropadded":
        return use_16_channels_with_zero_padding(window, channel_names)

    raise ValueError(f"channel mode must be one of {sorted(VALID_MODES)}, got {mode!r}")


def describe_case_difference() -> str:
    """Explain the single experimental variable changed by this study case."""
    return "Keep all 23 signals, or zero seven positions while retaining the same 23-channel EEGPT input shape."
