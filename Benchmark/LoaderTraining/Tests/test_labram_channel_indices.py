"""Regression tests for LaBraM's anatomical channel-position indices."""

from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINING_DIR = REPO_ROOT / "Benchmark" / "Training"
PREPROCESSING_DIR = REPO_ROOT / "Benchmark" / "Preprocessing"
CHANNEL_DIR = REPO_ROOT / "Benchmark" / "StudyCase" / "Channel"
LABRAM_DIR = REPO_ROOT / "EEG-FM" / "Labram"

for directory in (TRAINING_DIR, PREPROCESSING_DIR, CHANNEL_DIR, LABRAM_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import utils
from build_canonical_tuab import CANONICAL_CHANNELS
from Labram_23ch_vs_16ch import apply_channel_case
from train_labram import build_labram_channel_indices, build_labram_classifier


EXPECTED_CANONICAL_CHANNELS = [
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
    "A1", "A2", "FZ", "CZ", "PZ", "T1", "T2",
]
EXPECTED_LABRAM_INPUT_CHANS = [
    0, 1, 3, 18, 22, 40, 44, 62, 66, 81, 83, 16, 24,
    89, 91, 90, 92, 95, 96, 20, 42, 64, 113, 114,
]


def labram_test_config() -> dict:
    """Return only the LaBraM settings needed to build the classifier."""
    return {
        "paths": {"model_repo": str(LABRAM_DIR)},
        "model": {
            "name": "labram_base_patch200_200",
            "num_classes": 1,
            "EEG_size": 2000,
            "drop_rate": 0.0,
            "attn_drop_rate": 0.0,
            "drop_path_rate": 0.0,
            "init_values": 0.1,
            "qkv_bias": False,
            "use_abs_pos_emb": True,
            "use_rel_pos_bias": False,
        },
    }


def test_author_indices_for_canonical_channels_are_fixed():
    """Guard the exact canonical channel order and author 10-20 index mapping."""
    assert len(CANONICAL_CHANNELS) == 23
    assert list(CANONICAL_CHANNELS) == EXPECTED_CANONICAL_CHANNELS
    assert utils.get_input_chans(CANONICAL_CHANNELS) == EXPECTED_LABRAM_INPUT_CHANS


def test_benchmark_classifier_stores_author_channel_indices():
    """Confirm the benchmark wrapper caches the author-derived LaBraM indices."""
    classifier = build_labram_classifier(labram_test_config())
    assert classifier.input_chans.tolist() == EXPECTED_LABRAM_INPUT_CHANS

    eeg = torch.randn(2, 23, 10, 200)
    indices = build_labram_channel_indices(eeg, classifier.input_chans)
    assert indices.device == eeg.device
    assert indices.tolist() == EXPECTED_LABRAM_INPUT_CHANS


def test_labram_forward_uses_same_indices_for_both_channel_modes():
    """Check that channel ablation changes amplitudes, not LaBraM positions."""
    classifier = build_labram_classifier(labram_test_config()).eval()
    eeg = torch.randn(1, 23, 10, 200)
    all_channels = apply_channel_case(eeg, CANONICAL_CHANNELS, "23channels")
    zero_padded = apply_channel_case(eeg, CANONICAL_CHANNELS, "16channels_zeropadded")

    with torch.no_grad():
        output_23 = classifier(all_channels)
        output_16_padded = classifier(zero_padded)

    assert classifier.input_chans.tolist() == EXPECTED_LABRAM_INPUT_CHANS
    assert output_23.shape == output_16_padded.shape
