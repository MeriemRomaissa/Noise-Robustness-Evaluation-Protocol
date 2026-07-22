"""Connect CodeBrain to the shared pretrained EEG benchmark workflow."""

import os
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


CODEBRAIN_AUTHOR_FIELDS = {
    "downstream_dataset": "downstream_dataset",
    "num_of_classes": ("num_of_classes", int),
    "use_pretrained_weights": ("use_pretrained_weights", bool),
    "dropout": ("dropout", float),
    "cuda": ("cuda", int),
    "foundation_dir": "foundation_dir",
    "n_layer": ("n_layer", int),
    "codebook_size_t": ("codebook_size_t", int),
    "codebook_size_f": ("codebook_size_f", int),
    "codebook_dim": ("codebook_dim", int),
}


class CodeBrainClassifier(nn.Module):
    """Adapt CodeBrain's backbone and TUAB head to the benchmark interface."""

    def __init__(self, author_model: nn.Module):
        super().__init__()
        self.model = author_model

    def forward(self, eeg):
        batch_size, channels, patches, patch_width = eeg.shape
        features = self.model.backbone(eeg)
        flattened = features.contiguous().view(
            batch_size,
            channels * patches * patch_width,
        )
        return self.model.classifier(flattened).reshape(batch_size)


def build_codebrain_classifier(config: dict) -> CodeBrainClassifier:
    """Build the corrected author CodeBrain TUAB model and wrapper."""
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    add_repo_to_import_path(config)

    try:
        from Models.model_for_tuab import Model
    except ImportError:
        from models.model_for_tuab import Model

    settings = config["model"]
    # Construct the architecture without implicit weight loading. The shared
    # workflow loads the configured checkpoint and records every matched key.
    author_settings = namespace_from_config(settings, CODEBRAIN_AUTHOR_FIELDS)
    return CodeBrainClassifier(Model(author_settings))


def load_codebrain_checkpoint(model: CodeBrainClassifier, checkpoint_path: Path) -> dict:
    """Load mandatory pretrained parameters into the CodeBrain backbone."""
    return load_prefixed_checkpoint(
        target=model.model.backbone,
        checkpoint_path=checkpoint_path,
        prefixes=COMMON_BACKBONE_CHECKPOINT_PREFIXES,
    )


def create_codebrain_specification() -> ModelSpec:
    """Declare how CodeBrain connects to the shared benchmark workflow."""
    return ModelSpec(
        name="CodeBrain",
        loader_module="loader_codebrain",
        build_model=build_codebrain_classifier,
        load_checkpoint=load_codebrain_checkpoint,
    )


def main() -> None:
    """Run CodeBrain with its reader-controlled YAML configuration."""
    config_path = Path(__file__).resolve().parents[1] / "Config" / "codebrain.yaml"
    run_experiments(create_codebrain_specification(), str(config_path))


if __name__ == "__main__":
    main()
