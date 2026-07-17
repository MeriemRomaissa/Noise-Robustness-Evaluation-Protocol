"""Connect LaBraM to the shared pretrained EEG benchmark workflow."""

from pathlib import Path

import torch
from torch import nn

from training_common import (
    ModelSpec,
    add_repo_to_import_path,
    load_matching_weights,
    run_experiments,
)


class LaBraMClassifier(nn.Module):
    """Adapt the author LaBraM classifier to the benchmark's model(eeg) interface."""

    def __init__(self, author_model: nn.Module):
        super().__init__()
        self.model = author_model

    def forward(self, eeg):
        input_channels = build_labram_channel_indices(eeg)
        return self.model(eeg, input_chans=input_channels)


def build_labram_classifier(config: dict) -> LaBraMClassifier:
    """Build the LaBraM architecture used by the matched TUAB experiment."""
    add_repo_to_import_path(config)
    import modeling_finetune

    # Weight loading remains false here because the shared workflow loads and
    # audits the mandatory checkpoint immediately after model construction.
    author_model = modeling_finetune.labram_base_d6_patch200_200(
        pretrained=False,
        num_classes=2,
        EEG_size=2000,
        drop_path_rate=0.0,
        init_values=0.1,
        qkv_bias=True,
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
    )
    return LaBraMClassifier(author_model)


def build_labram_channel_indices(eeg) -> torch.Tensor:
    """Follow the author fine-tuning interface for LaBraM channel indices."""
    number_of_eeg_channels = eeg.shape[1]
    return torch.arange(number_of_eeg_channels + 1, device=eeg.device)


def map_labram_checkpoint_key(source_key: str) -> str:
    """Translate an author checkpoint key into this wrapper's model namespace."""
    if source_key.startswith("student."):
        return "model." + source_key.removeprefix("student.")
    if source_key.startswith("model."):
        return source_key
    return "model." + source_key


def load_labram_checkpoint(model: LaBraMClassifier, checkpoint_path: Path) -> dict:
    """Load and report shape-compatible pretrained LaBraM parameters."""
    return load_matching_weights(
        target=model,
        checkpoint_path=checkpoint_path,
        key_mapper=map_labram_checkpoint_key,
    )


def create_labram_specification() -> ModelSpec:
    """Declare how LaBraM connects to the shared benchmark workflow."""
    return ModelSpec(
        name="LaBraM",
        loader_module="loader_labram",
        build_model=build_labram_classifier,
        load_checkpoint=load_labram_checkpoint,
        study_case_module="Labram_23ch_vs_16ch",
    )


def main() -> None:
    """Run LaBraM with its reader-controlled YAML configuration."""
    config_path = Path(__file__).resolve().parents[1] / "Config" / "labram.yaml"
    run_experiments(create_labram_specification(), str(config_path))


if __name__ == "__main__":
    main()
