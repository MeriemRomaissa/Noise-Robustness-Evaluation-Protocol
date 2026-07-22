"""Connect BIOT to the shared pretrained EEG benchmark workflow."""

from pathlib import Path

from torch import nn

from training_common import (
    BIOT_ENCODER_CHECKPOINT_PREFIXES,
    ModelSpec,
    add_repo_to_import_path,
    load_prefixed_checkpoint,
    run_experiments,
)


def build_biot_classifier(config: dict) -> nn.Module:
    """Build the one-logit BIOT TUAB classifier from YAML model choices."""
    add_repo_to_import_path(config)
    from model.biot import BIOTClassifier

    settings = config["model"]
    return BIOTClassifier(
        n_classes=int(settings["n_classes"]),
        n_channels=int(settings["n_channels"]),
        emb_size=int(settings["emb_size"]),
        n_fft=int(settings["n_fft"]),
        hop_length=int(settings["hop_length"]),
        depth=int(settings["depth"]),
        heads=int(settings["heads"]),
    )


def load_biot_checkpoint(model: nn.Module, checkpoint_path: Path) -> dict:
    """Load mandatory PREST weights into BIOT's encoder and report the match."""
    return load_prefixed_checkpoint(
        target=model.biot,
        checkpoint_path=checkpoint_path,
        prefixes=BIOT_ENCODER_CHECKPOINT_PREFIXES,
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
