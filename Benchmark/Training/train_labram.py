"""Connect LaBraM to the shared pretrained EEG benchmark workflow."""

import importlib
from pathlib import Path
import sys

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

    def __init__(self, author_model: nn.Module, input_chans: torch.Tensor):
        super().__init__()
        self.model = author_model
        self.register_buffer("input_chans", input_chans.long(), persistent=False)

    def forward(self, eeg):
        input_channels = build_labram_channel_indices(eeg, self.input_chans)
        return self.model(eeg, input_chans=input_channels)


def build_labram_classifier(config: dict) -> LaBraMClassifier:
    """Build the LaBraM architecture used by the matched TUAB experiment."""
    add_repo_to_import_path(config)
    import modeling_finetune
    import utils

    settings = config["model"]
    model_builder = getattr(modeling_finetune, settings["name"])
    # Weight loading remains false here because the shared workflow loads and
    # audits the mandatory checkpoint immediately after model construction.
    author_model = model_builder(
        pretrained=False,
        num_classes=int(settings["num_classes"]),
        EEG_size=int(settings["EEG_size"]),
        drop_rate=float(settings["drop_rate"]),
        attn_drop_rate=float(settings["attn_drop_rate"]),
        drop_path_rate=float(settings["drop_path_rate"]),
        init_values=float(settings["init_values"]),
        qkv_bias=bool(settings["qkv_bias"]),
        use_abs_pos_emb=bool(settings["use_abs_pos_emb"]),
        use_rel_pos_bias=bool(settings["use_rel_pos_bias"]),
    )
    input_chans = torch.tensor(
        utils.get_input_chans(load_canonical_channel_names()),
        dtype=torch.long,
    )
    return LaBraMClassifier(author_model, input_chans)


def build_labram_channel_indices(eeg, input_chans: torch.Tensor) -> torch.Tensor:
    """Move LaBraM's author-derived channel indices to the batch device."""
    return input_chans.to(device=eeg.device)


def load_canonical_channel_names() -> list[str]:
    """Read the 23-channel order used by canonical TUAB preprocessing."""
    preprocessing_dir = Path(__file__).resolve().parents[1] / "Preprocessing"
    if str(preprocessing_dir) not in sys.path:
        sys.path.insert(0, str(preprocessing_dir))
    canonical_builder = importlib.import_module("build_canonical_tuab")
    return list(canonical_builder.CANONICAL_CHANNELS)


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
