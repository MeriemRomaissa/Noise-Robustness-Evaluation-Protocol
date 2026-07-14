# EEGPT channel case study: compare using all 23 real channels vs 16 real channels
# with the other 7 zero-padded. EEGPT never reshapes into patches — output stays
# [B,23,2000] for both cases.

from channel_common import (
    build_16_signal_mask,
    get_keep_16_channels,
    shape_of,
    validate_23_channel_window,
    zero_masked_channels,
)

EEGPT_USE_CHANNELS = [
    "FP1", "FPZ", "FP2", "F7", "F3", "FZ", "F4", "F8", "T7", "C3",
    "CZ", "C4", "T8", "P7", "P3", "PZ", "P4", "P8", "O1", "O2",
]


# Summarizes this case study for anyone inspecting it — not used internally, just documentation.
def get_case_settings() -> dict:
    return {
        "model": "EEGPT",
        "purpose": "Compare all 23 real channels with 16 real channels plus 7 zero-padded channel positions.",
        "mode_23channels": "Use all 23 real referential channels; output remains [B,23,2000].",
        "mode_16channels_zeropadded": (
            "Keep 23-channel shape, keep 16 selected channel positions as real signal, "
            "and zero-pad the remaining 7 channel positions."
        ),
        "keep_16_channels": get_keep_16_channels(),
        "input_shape": [None, 23, 2000],
        "output_shape": [None, 23, 2000],
        "notes": [
            "This is not true 16-channel model input.",
            "The model still receives 23 channel positions.",
            "Do not reorder channels.",
            "Do not remove channels.",
            "Do not create [16,2000].",
            "Do not reconstruct bipolar channels.",
        ],
        "eegpt_use_channels": EEGPT_USE_CHANNELS,
    }


# Case 1: all 23 real channels, unchanged.
# channel_names isn't used here — kept so both apply_* functions share one call signature.
def apply_23channels_input(window, channel_names=None):
    validate_23_channel_window(window)
    return window


# Case 2: keep 16 real channels, zero out the other 7. Shape stays [23,2000] either way.
def apply_16channels_zeropadded_input(window, channel_names):
    unbatched = validate_23_channel_window(window)
    mask = build_16_signal_mask(channel_names)
    return zero_masked_channels(window, mask, unbatched)


# Checks a finished batch matches the shape EEGPT expects.
def validate_case_batch(batch, mode) -> bool:
    if mode not in {"23channels", "16channels_zeropadded"}:
        raise ValueError(f"mode must be '23channels' or '16channels_zeropadded', got {mode!r}")
    x = batch[0] if isinstance(batch, (tuple, list)) else batch
    shape = shape_of(x)
    if len(shape) == 3 and shape[1:] == (23, 2000):
        return True
    raise ValueError(f"EEGPT case batch must be [B,23,2000], got {list(shape)}")


# One-sentence explanation of what differs between the two cases, for logs/reports.
def describe_case_difference() -> str:
    return (
        "23channels keeps all 23 channel positions with real signal. "
        "16channels_zeropadded keeps the same 23-channel model input shape, "
        "keeps 16 selected channel positions as real signal, and sets the other "
        "7 channel positions to zero. This tests the effect of removing 7 real "
        "channels without changing the model input shape."
    )
