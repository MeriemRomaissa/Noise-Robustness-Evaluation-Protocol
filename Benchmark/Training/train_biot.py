"""Connect BIOT to the shared pretrained EEG benchmark workflow."""

from pathlib import Path

from torch import nn

from training_common import (
    ModelSpec,
    add_repo_to_import_path,
    load_matching_weights,
    run_experiments,
)


BIOT_CHECKPOINT_PREFIXES = (
    "module.biot.",
    "model.biot.",
    "biot.",
    "module.",
    "model.",
)


def build_biot_classifier(config: dict) -> nn.Module:
    """Build the one-logit BIOT TUAB classifier from YAML model choices."""
    add_repo_to_import_path(config)
    from model.biot import BIOTClassifier

    settings = config.get("model", {})
    return BIOTClassifier(
        n_classes=1,
        n_channels=16,
        n_fft=int(settings.get("n_fft", 200)),
        hop_length=int(settings.get("hop_length", 100)),
        depth=int(settings.get("depth", 4)),
        heads=int(settings.get("heads", 8)),
    )


def map_biot_checkpoint_key(source_key: str) -> str:
    """Remove a known training-wrapper prefix from a PREST encoder key."""
    for prefix in BIOT_CHECKPOINT_PREFIXES:
        if source_key.startswith(prefix):
            return source_key.removeprefix(prefix)
    return source_key


def load_biot_checkpoint(model: nn.Module, checkpoint_path: Path) -> dict:
    """Load mandatory PREST weights into BIOT's encoder and report the match."""
    return load_matching_weights(
        target=model.biot,
        checkpoint_path=checkpoint_path,
        key_mapper=map_biot_checkpoint_key,
    )


def create_biot_specification() -> ModelSpec:
    """Declare how BIOT connects to the shared benchmark workflow."""
    return ModelSpec(
        name="BIOT",
        loader_module="loader_biot",
        build_model=build_biot_classifier,
        load_checkpoint=load_biot_checkpoint,
    )


def main() -> None:
    """Run BIOT with its reader-controlled YAML configuration."""
    config_path = Path(__file__).resolve().parents[1] / "Config" / "biot.yaml"
    run_experiments(create_biot_specification(), str(config_path))


if __name__ == "__main__":
    main()
