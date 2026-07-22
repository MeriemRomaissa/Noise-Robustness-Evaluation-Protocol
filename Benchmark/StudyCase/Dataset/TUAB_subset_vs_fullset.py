"""Resolve TUAB subset-vs-full split limits without changing split membership."""

VALID_TUAB_MODES = {"subset_tuab", "full_tuab"}
SUBSET_LIMITS = {
    "train_samples": 8192,
    "validation_samples": 2048,
    "test_samples": 2048,
}


def apply_tuab_mode(config: dict) -> dict:
    """Set runtime sample caps from study_case.tuab_mode."""
    study_case = config.setdefault("study_case", {})
    mode = study_case.get("tuab_mode")
    if mode not in VALID_TUAB_MODES:
        raise ValueError(
            "study_case.tuab_mode must be one of "
            f"{sorted(VALID_TUAB_MODES)}, got {mode!r}"
        )

    data = config.setdefault("data", {})
    if mode == "subset_tuab":
        data.update(SUBSET_LIMITS)
    else:
        data["train_samples"] = None
        data["validation_samples"] = None
        data["test_samples"] = None
    return config
