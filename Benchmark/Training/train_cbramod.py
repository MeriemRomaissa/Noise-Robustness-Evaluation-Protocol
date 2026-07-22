"""Connect CBraMod to the shared pretrained EEG benchmark workflow."""

from pathlib import Path

from torch import nn

from training_common import (
    COMMON_BACKBONE_CHECKPOINT_PREFIXES,
    ModelSpec,
    add_repo_to_import_path,
    load_prefixed_checkpoint,
    namespace_from_config,
    run_experiments,
)


CBRAMOD_AUTHOR_FIELDS = {
    "use_pretrained_weights": ("use_pretrained_weights", bool),
    "classifier": "variant",
    "dropout": ("dropout", float),
    "cuda": ("cuda", int),
    "foundation_dir": "foundation_dir",
}


def build_cbramod_classifier(config: dict) -> nn.Module:
    """Build the author CBraMod TUAB classifier selected in the YAML."""
    add_repo_to_import_path(config)
    from models.model_for_tuab import Model

    settings = config["model"]
    # Construct the architecture without implicit weight loading. The shared
    # workflow loads the configured checkpoint and records every matched key.
    author_settings = namespace_from_config(settings, CBRAMOD_AUTHOR_FIELDS)
    return Model(author_settings)


def load_cbramod_checkpoint(model: nn.Module, checkpoint_path: Path) -> dict:
    """Load mandatory pretrained parameters into the CBraMod backbone."""
    return load_prefixed_checkpoint(
        target=model.backbone,
        checkpoint_path=checkpoint_path,
        prefixes=COMMON_BACKBONE_CHECKPOINT_PREFIXES,
    )


def create_cbramod_specification() -> ModelSpec:
    """Declare how CBraMod connects to the shared benchmark workflow."""
    return ModelSpec(
        name="CBraMod",
        loader_module="loader_cbramod",
        build_model=build_cbramod_classifier,
        load_checkpoint=load_cbramod_checkpoint,
    )


def main() -> None:
    """Run CBraMod with its reader-controlled YAML configuration."""
    config_path = Path(__file__).resolve().parents[1] / "Config" / "cbramod.yaml"
    run_experiments(create_cbramod_specification(), str(config_path))


if __name__ == "__main__":
    main()
