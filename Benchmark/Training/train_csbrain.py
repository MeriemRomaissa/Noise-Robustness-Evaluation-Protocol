"""Connect CSBrain to the shared pretrained EEG benchmark workflow."""

from pathlib import Path
from types import SimpleNamespace

from torch import nn

from training_common import (
    ModelSpec,
    add_repo_to_import_path,
    load_matching_weights,
    run_experiments,
)


BACKBONE_CHECKPOINT_PREFIXES = (
    "module.backbone.",
    "model.backbone.",
    "backbone.",
    "module.",
)


def build_csbrain_classifier(config: dict) -> nn.Module:
    """Build the author CSBrain classifier used by the TUAB experiment."""
    add_repo_to_import_path(config)
    from models.model_for_tuab import Model

    # Construct the architecture without implicit weight loading. The shared
    # workflow loads the configured checkpoint and records every matched key.
    author_settings = SimpleNamespace(
        model="CSBrain",
        use_pretrained_weights=False,
        dropout=0.1,
        cuda=0,
        foundation_dir="",
    )
    return Model(author_settings)


def map_csbrain_checkpoint_key(source_key: str) -> str:
    """Translate a checkpoint key into CSBrain's downstream backbone namespace."""
    for prefix in BACKBONE_CHECKPOINT_PREFIXES:
        if source_key.startswith(prefix):
            return source_key.removeprefix(prefix)
    return source_key


def load_csbrain_checkpoint(model: nn.Module, checkpoint_path: Path) -> dict:
    """Load mandatory pretrained parameters into the CSBrain backbone."""
    return load_matching_weights(
        target=model.backbone,
        checkpoint_path=checkpoint_path,
        key_mapper=map_csbrain_checkpoint_key,
    )


def create_csbrain_specification() -> ModelSpec:
    """Declare how CSBrain connects to the shared benchmark workflow."""
    return ModelSpec(
        name="CSBrain",
        loader_module="loader_csbrain",
        build_model=build_csbrain_classifier,
        load_checkpoint=load_csbrain_checkpoint,
    )


def main() -> None:
    """Run CSBrain with its reader-controlled YAML configuration."""
    config_path = Path(__file__).resolve().parents[1] / "Config" / "csbrain.yaml"
    run_experiments(create_csbrain_specification(), str(config_path))


if __name__ == "__main__":
    main()
