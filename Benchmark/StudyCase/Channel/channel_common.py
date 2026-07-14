# Shared channel-selection helpers used by every model's 23ch-vs-16pad case study.
# Model scripts (labram_*, eegpt_*) import these instead of redefining them.

from typing import Any

# The 16 real channel positions kept in the "16 channels + 7 zero-padded" case.
# Don't add, remove, or reorder these — the other 7 positions are whatever's left over.
KEEP_16_REFERENTIAL = [
    "FP1", "FP2",
    "F7", "F3", "F4", "F8",
    "T3", "C3", "C4", "T4",
    "T5", "P3", "P4", "T6",
    "O1", "O2",
]


# Returns a fresh copy of the 16 kept channel names, so callers can't accidentally mutate the shared list.
def get_keep_16_channels() -> list[str]:
    return KEEP_16_REFERENTIAL[:]


# Normalizes one channel name so different naming styles (case, "EEG " prefix, "-REF" suffix) line up.
def clean_channel_name(name) -> str:
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    return str(name).strip().upper().replace("EEG ", "").replace("-REF", "")


# Maps each of the 23 channel names to its index, including old/modern 10-20 aliases
# (T3/T7, T4/T8, T5/P7, T6/P8) so either naming convention resolves to the same channel.
def build_channel_lookup(channel_names) -> dict[str, int]:
    aliases = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
    if len(channel_names) != 23:
        raise ValueError(f"Expected 23 channel names, got {len(channel_names)}")
    lookup: dict[str, int] = {}
    for idx, name in enumerate(channel_names):
        key = clean_channel_name(name)
        lookup[key] = idx
        for old, modern in aliases.items():
            if key == old:
                lookup.setdefault(modern, idx)
            elif key == modern:
                lookup.setdefault(old, idx)
    return lookup


# Builds a length-23 True/False mask: True where a channel stays real signal in the 16-channel case.
def build_16_signal_mask(channel_names) -> list[bool]:
    lookup = build_channel_lookup(channel_names)
    missing = [name for name in KEEP_16_REFERENTIAL if name not in lookup]
    if missing:
        raise ValueError(f"Missing selected 16-channel names: {missing}; available={sorted(lookup)}")
    mask = [False] * 23
    for name in KEEP_16_REFERENTIAL:
        mask[lookup[name]] = True
    return mask


# Reads an array/tensor's shape as a plain tuple of ints.
def shape_of(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise ValueError("Expected an array or tensor with shape.")
    return tuple(int(dim) for dim in shape)


# True if value is a torch tensor rather than a NumPy array.
def is_torch_tensor(value: Any) -> bool:
    return value.__class__.__module__.startswith("torch")


# Copies an array/tensor so masking never modifies the caller's original window.
def copy_window(value):
    return value.clone() if is_torch_tensor(value) else value.copy()


# Confirms window is [23,2000] or [B,23,2000]. Returns True if it's a single unbatched window.
def validate_23_channel_window(window) -> bool:
    shape = shape_of(window)
    if len(shape) == 2:
        if shape != (23, 2000):
            raise ValueError(f"Expected [23,2000] or [B,23,2000], got {list(shape)}")
        return True
    if len(shape) == 3 and shape[1:] == (23, 2000):
        return False
    raise ValueError(f"Expected [23,2000] or [B,23,2000], got {list(shape)}")


# Returns a copy of window with every channel position where mask is False set to zero.
def zero_masked_channels(window, mask, unbatched: bool):
    out = copy_window(window)
    for idx, keep in enumerate(mask):
        if keep:
            continue
        if unbatched:
            out[idx, :] = 0
        else:
            out[:, idx, :] = 0
    return out
