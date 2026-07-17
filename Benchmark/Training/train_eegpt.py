"""Connect EEGPT to the shared pretrained EEG benchmark workflow."""

from pathlib import Path

from torch import nn

from training_common import (
    ModelSpec,
    add_repo_to_import_path,
    load_matching_weights,
    run_experiments,
)


EEGPT_CHANNEL_NAMES = [
    "FP1", "FPZ", "FP2", "F7", "F3", "FZ", "F4", "F8", "T7", "C3",
    "CZ", "C4", "T8", "P7", "P3", "PZ", "P4", "P8", "O1", "O2",
]


class EEGPTClassifierWrapper(nn.Module):
    """Adapt the author EEGPT classifier to the shared benchmark interface."""

    def __init__(self, author_model: nn.Module):
        super().__init__()
        self.model = author_model

    def forward(self, eeg):
        return self.model(eeg)


def build_eegpt_classifier(config: dict) -> EEGPTClassifierWrapper:
    """Build the checkpoint-compatible EEGPT classifier used for TUAB."""
    add_repo_to_import_path(config)
    from downstream_tueg.Modules.models.EEGPT_mcae_finetune_change import EEGPTClassifier

    author_model = EEGPTClassifier(
        num_classes=1,
        in_channels=23,
        img_size=[20, 2000],
        use_channels_names=EEGPT_CHANNEL_NAMES,
        use_chan_conv=True,
        use_mean_pooling=True,
    )
    return EEGPTClassifierWrapper(author_model)


def map_eegpt_checkpoint_key(source_key: str) -> str | None:
    """Map pretrained encoder parameters into EEGPT's target encoder."""
    if not source_key.startswith("encoder."):
        return None
    encoder_key = source_key.removeprefix("encoder.")
    return f"model.target_encoder.{encoder_key}"


def load_eegpt_checkpoint(model: EEGPTClassifierWrapper, checkpoint_path: Path) -> dict:
    """Load and report shape-compatible pretrained EEGPT parameters."""
    return load_matching_weights(
        target=model,
        checkpoint_path=checkpoint_path,
        key_mapper=map_eegpt_checkpoint_key,
    )


def create_eegpt_specification() -> ModelSpec:
    """Declare how EEGPT connects to the shared benchmark workflow."""
    return ModelSpec(
        name="EEGPT",
        loader_module="loader_eegpt",
        build_model=build_eegpt_classifier,
        load_checkpoint=load_eegpt_checkpoint,
        study_case_module="EEGPT_23ch_vs_16ch",
    )


def main() -> None:
    """Run EEGPT with its reader-controlled YAML configuration."""
    config_path = Path(__file__).resolve().parents[1] / "Config" / "eegpt.yaml"
    run_experiments(create_eegpt_specification(), str(config_path))


if __name__ == "__main__":
    main()
