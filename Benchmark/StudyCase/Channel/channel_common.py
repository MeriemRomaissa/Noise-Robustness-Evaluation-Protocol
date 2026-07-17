"""Shared channel-ablation rules for LaBraM and EEGPT."""

from typing import Any, Sequence

KEEP_16_REFERENTIAL = (
    "FP1", "FP2", "F7", "F3", "F4", "F8", "T3", "C3",
    "C4", "T4", "T5", "P3", "P4", "T6", "O1", "O2",
)

# Original PKLs follow this fixed order when channel metadata is unavailable.
CANONICAL_CHANNEL_ORDER = (
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "A1", "A2", "FZ", "CZ", "PZ",
    "T1", "T2",
)


def normalize_channel_name(name: Any) -> str:
    """Normalize common TUAB channel labels without changing channel order."""
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    return str(name).strip().upper().replace("EEG ", "").replace("-REF", "")


def channel_lookup(channel_names: Sequence[Any]) -> dict[str, int]:
    """Map canonical and legacy temporal names to their array positions."""
    if len(channel_names) != 23:
        raise ValueError(f"Expected 23 channel names, got {len(channel_names)}")
    aliases = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
    lookup = {}
    for index, name in enumerate(channel_names):
        cleaned = normalize_channel_name(name)
        lookup[cleaned] = index
        for legacy, modern in aliases.items():
            if cleaned == legacy:
                lookup.setdefault(modern, index)
            elif cleaned == modern:
                lookup.setdefault(legacy, index)
    return lookup


def shape_of(value: Any) -> tuple[int, ...]:
    """Return an array or tensor shape as plain integers."""
    if not hasattr(value, "shape"):
        raise TypeError(f"Expected an array or tensor, got {type(value).__name__}")
    return tuple(int(dimension) for dimension in value.shape)


def validate_canonical_window(window: Any) -> bool:
    """Validate ``[23,2000]`` or ``[B,23,2000]`` and report if unbatched."""
    shape = shape_of(window)
    if shape == (23, 2000):
        return True
    if len(shape) == 3 and shape[1:] == (23, 2000):
        return False
    raise ValueError(f"Expected [23,2000] or [B,23,2000], got {list(shape)}")


def keep_16_and_zero_7(window: Any, channel_names: Sequence[Any] | None):
    """Return a copy with seven excluded positions zeroed and shape unchanged."""
    unbatched = validate_canonical_window(window)
    names = channel_names or CANONICAL_CHANNEL_ORDER
    lookup = channel_lookup(names)
    missing = [name for name in KEEP_16_REFERENTIAL if name not in lookup]
    if missing:
        raise ValueError(f"Cannot apply channel ablation; missing channels: {missing}")
    keep_indices = {lookup[name] for name in KEEP_16_REFERENTIAL}
    output = window.clone() if value_is_tensor(window) else window.copy()
    channel_axis = 0 if unbatched else 1
    for index in range(23):
        if index not in keep_indices:
            slices = [slice(None)] * len(output.shape)
            slices[channel_axis] = index
            output[tuple(slices)] = 0
    return output


def value_is_tensor(value: Any) -> bool:
    """Recognize tensors without importing PyTorch in this small helper."""
    return value.__class__.__module__.startswith("torch")
