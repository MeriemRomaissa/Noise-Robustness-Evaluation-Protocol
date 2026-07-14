# LaBraM channel case study: compare using all 23 real channels vs 16 real channels
# with the other 7 zero-padded. Both cases keep the same 23-channel input shape —
# only the "16 zero-padded" case is a true channel ablation.

from channel_common import (
    build_16_signal_mask,
    get_keep_16_channels,
    shape_of,
    validate_23_channel_window,
    zero_masked_channels,
)


# Summarizes this case study for anyone inspecting it — not used internally, just documentation.
def get_case_settings() -> dict:
    return {
        "model": "LaBraM",
        "purpose": "Compare all 23 real channels with 16 real channels plus 7 zero-padded channel positions.",
        "mode_23channels": "Use all 23 real referential channels; default output is [B,23,10,200] (patched).",
        "mode_16channels_zeropadded": (
            "Keep 23-channel shape, keep 16 selected channel positions as real signal, "
            "zero-pad the remaining 7 channel positions, then apply the same LaBraM patching."
        ),
        "keep_16_channels": get_keep_16_channels(),
        "input_shape": [None, 23, 2000],
        "output_shape": {"patched": [None, 23, 10, 200], "unpatched": [None, 23, 2000]},
        "notes": [
            "This is not true 16-channel model input.",
            "The model still receives 23 channel positions.",
            "Do not reorder channels.",
            "Do not remove channels.",
            "Do not create [16,2000].",
            "Do not reconstruct bipolar channels.",
        ],
    }


# Reshapes a validated 23-channel window into LaBraM's patch format: 10 one-second
# patches of 200 samples each. Values are untouched, only regrouped.
def to_patched_shape(window):
    shape = shape_of(window)
    if len(shape) == 2:
        return window.reshape(23, 10, 200)
    return window.reshape(shape[0], 23, 10, 200)


# Leaves a validated 23-channel window's shape as-is — use this to get unpatched output.
def keep_canonical_shape(window):
    return window


# Case 1: all 23 real channels. Patched by default; pass shape_transform=keep_canonical_shape for [23,2000].
# channel_names isn't used here — kept so both apply_* functions share one call signature.
def apply_23channels_input(window, channel_names=None, shape_transform=to_patched_shape):
    validate_23_channel_window(window)
    return shape_transform(window)


# Case 2: keep 16 real channels, zero out the other 7, then shape the same way as case 1.
def apply_16channels_zeropadded_input(window, channel_names, shape_transform=to_patched_shape):
    unbatched = validate_23_channel_window(window)
    mask = build_16_signal_mask(channel_names)
    masked_window = zero_masked_channels(window, mask, unbatched)
    return shape_transform(masked_window)


# Checks a finished batch matches the shape LaBraM expects for the given case.
def validate_case_batch(batch, mode) -> bool:
    if mode not in {"23channels", "16channels_zeropadded"}:
        raise ValueError(f"mode must be '23channels' or '16channels_zeropadded', got {mode!r}")
    x = batch[0] if isinstance(batch, (tuple, list)) else batch
    shape = shape_of(x)
    if len(shape) == 3 and shape[1:] == (23, 2000):
        return True
    if len(shape) == 4 and shape[1:] == (23, 10, 200):
        return True
    raise ValueError(f"LaBraM case batch must be [B,23,2000] or [B,23,10,200], got {list(shape)}")


# One-sentence explanation of what differs between the two cases, for logs/reports.
def describe_case_difference() -> str:
    return (
        "23channels keeps all 23 channel positions with real signal. "
        "16channels_zeropadded keeps the same 23-channel model input shape, "
        "keeps 16 selected channel positions as real signal, and sets the other "
        "7 channel positions to zero. This tests the effect of removing 7 real "
        "channels without changing the model input shape."
    )
