"""Connect CodeBrain to the shared pretrained EEG benchmark workflow."""

import os
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

    # Construct the architecture without implicit weight loading. The shared
    # workflow loads the configured checkpoint and records every matched key.
    author_settings = SimpleNamespace(
        downstream_dataset="TUAB",
        num_of_classes=2,
        use_pretrained_weights=False,
        dropout=0.1,
        cuda=0,
        foundation_dir="",
        n_layer=8,
        codebook_size_t=4096,
        codebook_size_f=4096,
        codebook_dim=32,
    )
    return CodeBrainClassifier(Model(author_settings))


def map_codebrain_checkpoint_key(source_key: str) -> str:
    """Translate a checkpoint key into CodeBrain's downstream backbone namespace."""
    for prefix in BACKBONE_CHECKPOINT_PREFIXES:
        if source_key.startswith(prefix):
            return source_key.removeprefix(prefix)
    return source_key


def load_codebrain_checkpoint(model: CodeBrainClassifier, checkpoint_path: Path) -> dict:
    """Load mandatory pretrained parameters into the CodeBrain backbone."""
    return load_matching_weights(
        target=model.model.backbone,
        checkpoint_path=checkpoint_path,
        key_mapper=map_codebrain_checkpoint_key,
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
