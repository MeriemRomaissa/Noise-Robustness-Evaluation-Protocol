"""
Mock tests for the unified logging changes.

Run from any model directory, e.g.:
    cd CSBrain && python ../test_mock.py
    cd CBraMod && python ../test_mock.py
    cd EEGMamba && python ../test_mock.py
    cd Codebrain/Downstream && python ../../test_mock.py

Requirements: torch, sklearn, numpy (already in each model's env)
"""

import sys
import os
import json
import tempfile
import importlib

# Add CWD so "cd CSBrain && python ../test_mock.py" can import finetune_evaluator etc.
sys.path.insert(0, os.getcwd())
# Codebrain uses package-style imports (from Downstream.xxx), so also add the parent dir.
if os.path.basename(os.getcwd()) == "Downstream":
    sys.path.insert(0, os.path.dirname(os.getcwd()))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"


# ---------------------------------------------------------------------------
# Minimal mock objects
# ---------------------------------------------------------------------------

class MockBinaryModel(nn.Module):
    """1-output linear model — mimics a binary classifier."""
    def __init__(self, in_features=32):
        super().__init__()
        self.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.fc(x.view(x.shape[0], -1)[:, :32]).squeeze(-1)


def make_binary_loader(n=120, in_features=32, batch_size=32, pos_ratio=0.6):
    X = torch.randn(n, in_features)
    # Skewed labels so balanced_acc != accuracy
    y = (torch.rand(n) < pos_ratio).long()
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, drop_last=False)


class MockParams:
    downstream_dataset = "TUAB"
    epochs = 2
    lr = 1e-4
    weight_decay = 0.05
    clip_value = 1.0
    optimizer = "AdamW"
    multi_lr = False
    frozen = False
    label_smoothing = 0.1
    model = "mock"
    seed = 42
    dropout = 0.1
    model_dir = ""
    log_dir = ""
    output_dir = ""
    cuda = 0          # Codebrain uses params.cuda for device selection
    file_name = os.devnull  # Codebrain writes plain-text logs to params.file_name
    save_ckpt_freq = 0  # periodic checkpoint-N.pth (0=disabled)


# ---------------------------------------------------------------------------
# Test 1: evaluator returns a dict with correct keys
# ---------------------------------------------------------------------------

def test_evaluator_returns_dict():
    from finetune_evaluator import Evaluator

    loader = make_binary_loader()
    model = MockBinaryModel().cuda()
    params = MockParams()

    evaluator = Evaluator(params, loader)
    result = evaluator.get_metrics_for_binaryclass(model)

    expected_keys = {"balanced_accuracy", "accuracy", "pr_auc", "roc_auc", "loss", "cm"}
    missing = expected_keys - set(result.keys())
    extra   = set(result.keys()) - expected_keys

    assert not missing, f"Missing keys: {missing}"
    assert not extra,   f"Unexpected extra keys: {extra}"
    print(f"{PASS} evaluator returns dict with correct keys: {sorted(result.keys())}")


# ---------------------------------------------------------------------------
# Test 2: loss is None without criterion, float with criterion
# ---------------------------------------------------------------------------

def test_evaluator_loss_behaviour():
    from finetune_evaluator import Evaluator

    loader = make_binary_loader()
    model = MockBinaryModel().cuda()
    params = MockParams()
    criterion = nn.BCEWithLogitsLoss().cuda()

    evaluator = Evaluator(params, loader)

    r_no_crit = evaluator.get_metrics_for_binaryclass(model)
    assert r_no_crit["loss"] is None, f"Expected None, got {r_no_crit['loss']}"
    print(f"{PASS} loss is None when criterion not passed")

    r_with_crit = evaluator.get_metrics_for_binaryclass(model, criterion)
    assert isinstance(r_with_crit["loss"], float) and r_with_crit["loss"] > 0, \
        f"Expected positive float, got {r_with_crit['loss']}"
    print(f"{PASS} loss is a positive float when criterion passed: {r_with_crit['loss']:.4f}")


# ---------------------------------------------------------------------------
# Test 3: balanced_accuracy != accuracy (class imbalance detected)
# ---------------------------------------------------------------------------

def test_evaluator_balanced_vs_unweighted():
    from finetune_evaluator import Evaluator
    from sklearn.metrics import balanced_accuracy_score, accuracy_score

    # Force a very skewed loader so the two metrics differ noticeably
    loader = make_binary_loader(n=200, pos_ratio=0.9)
    # Biased model: always predicts positive (large positive bias)
    model = MockBinaryModel().cuda()
    with torch.no_grad():
        model.fc.bias.fill_(10.0)
        model.fc.weight.fill_(0.0)

    params = MockParams()
    evaluator = Evaluator(params, loader)
    result = evaluator.get_metrics_for_binaryclass(model)

    bal  = result["balanced_accuracy"]
    unw  = result["accuracy"]
    print(f"{PASS} balanced_accuracy={bal:.4f}  accuracy={unw:.4f}  (should differ on skewed data)")
    # On 90% positive data, always-positive model: accuracy≈0.9, balanced_acc≈0.5
    assert abs(unw - bal) > 0.1, \
        f"Expected bal_acc and acc to differ by >0.1 on skewed data, got {bal:.4f} vs {unw:.4f}"
    print(f"{PASS} balanced_accuracy and accuracy correctly differ on class-imbalanced data")


# ---------------------------------------------------------------------------
# Test 4: log.txt is written with all expected keys after 2 epochs
# ---------------------------------------------------------------------------

def test_trainer_log_txt():
    from finetune_evaluator import Evaluator
    from finetune_trainer import Trainer

    with tempfile.TemporaryDirectory() as tmpdir:
        params = MockParams()
        params.output_dir = tmpdir
        params.log_dir = ""   # skip TensorBoard
        params.model_dir = tmpdir
        params.epochs = 2

        train_loader = make_binary_loader(n=64,  batch_size=16)
        val_loader   = make_binary_loader(n=32,  batch_size=16)
        test_loader  = make_binary_loader(n=32,  batch_size=16)
        data_loader  = {"train": train_loader, "val": val_loader, "test": test_loader}

        model = MockBinaryModel()
        trainer = Trainer(params, data_loader, model)
        trainer.train_for_binaryclass()

        log_path = os.path.join(tmpdir, "log.txt")
        assert os.path.exists(log_path), "log.txt was not created"

        lines = open(log_path).read().strip().splitlines()
        assert len(lines) == 2, f"Expected 2 lines (one per epoch), got {len(lines)}"
        print(f"{PASS} log.txt created with {len(lines)} line(s)")

        required_keys = {
            "train_loss", "train_lr", "train_weight_decay",
            "train_class_acc", "train_grad_norm",
            "val_pr_auc", "val_roc_auc", "val_accuracy", "val_balanced_accuracy", "val_loss",
            "test_pr_auc", "test_roc_auc", "test_accuracy", "test_balanced_accuracy", "test_loss",
            "epoch", "n_parameters",
        }

        for i, line in enumerate(lines):
            entry = json.loads(line)
            missing = required_keys - set(entry.keys())
            assert not missing, f"Epoch {i}: missing keys in log.txt: {missing}"
            assert entry["epoch"] == i, f"epoch field wrong: {entry['epoch']}"
            assert isinstance(entry["n_parameters"], int) and entry["n_parameters"] > 0

        print(f"{PASS} log.txt entries contain all required keys")
        print(f"     Sample entry (epoch 0):")
        sample = json.loads(lines[0])
        for k, v in sorted(sample.items()):
            if k != "cm":
                print(f"       {k}: {v}")


# ---------------------------------------------------------------------------
# Test 5: n_parameters is computed and matches torch
# ---------------------------------------------------------------------------

def test_n_parameters():
    from finetune_trainer import Trainer

    params = MockParams()
    train_loader = make_binary_loader(n=32, batch_size=16)
    data_loader  = {"train": train_loader, "val": train_loader, "test": train_loader}

    model = MockBinaryModel()
    expected = sum(p.numel() for p in model.parameters() if p.requires_grad)

    trainer = Trainer(params, data_loader, model)
    assert trainer.n_parameters == expected, \
        f"n_parameters mismatch: {trainer.n_parameters} vs {expected}"
    print(f"{PASS} n_parameters = {trainer.n_parameters} (matches torch count)")


# ---------------------------------------------------------------------------
# Test 6: checkpoint files have LaBraM-compatible rich format
# ---------------------------------------------------------------------------

def test_checkpoint_format():
    from finetune_trainer import Trainer

    with tempfile.TemporaryDirectory() as tmpdir:
        params = MockParams()
        params.output_dir = tmpdir
        params.epochs = 3
        params.save_ckpt_freq = 2   # should produce checkpoint-1.pth at epoch 2

        train_loader = make_binary_loader(n=64, batch_size=16)
        val_loader   = make_binary_loader(n=32, batch_size=16)
        test_loader  = make_binary_loader(n=32, batch_size=16)
        data_loader  = {"train": train_loader, "val": val_loader, "test": test_loader}

        model = MockBinaryModel()
        trainer = Trainer(params, data_loader, model)
        trainer.train_for_binaryclass()

        # Rolling checkpoint must exist
        ckpt_path = os.path.join(tmpdir, 'checkpoint.pth')
        assert os.path.exists(ckpt_path), "checkpoint.pth not found"
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        required_keys = {'model', 'optimizer', 'epoch', 'args'}
        assert required_keys == set(ckpt.keys()), f"checkpoint keys: {set(ckpt.keys())}"
        assert isinstance(ckpt['model'], dict), "model state_dict should be a dict"
        assert isinstance(ckpt['optimizer'], dict), "optimizer state should be a dict"
        assert isinstance(ckpt['epoch'], int), "epoch should be int"
        print(f"{PASS} checkpoint.pth has LaBraM-compatible format: {sorted(ckpt.keys())}")

        # Best checkpoint must exist
        best_path = os.path.join(tmpdir, 'checkpoint-best.pth')
        assert os.path.exists(best_path), "checkpoint-best.pth not found"
        best_ckpt = torch.load(best_path, map_location='cpu', weights_only=False)
        assert set(best_ckpt.keys()) == required_keys
        print(f"{PASS} checkpoint-best.pth saved with same format (best epoch={best_ckpt['epoch']})")

        # Periodic checkpoint at epoch index 1 (epoch 2, since save_ckpt_freq=2)
        periodic_path = os.path.join(tmpdir, 'checkpoint-1.pth')
        assert os.path.exists(periodic_path), \
            f"checkpoint-1.pth not found (save_ckpt_freq=2 should create it at epoch 2)"
        print(f"{PASS} checkpoint-1.pth created by save_ckpt_freq=2")

        # Verify the model weights can be loaded back
        loaded_model = MockBinaryModel()
        loaded_model.load_state_dict(best_ckpt['model'])
        print(f"{PASS} model weights loadable from checkpoint-best.pth")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all():
    tests = [
        test_evaluator_returns_dict,
        test_evaluator_loss_behaviour,
        test_evaluator_balanced_vs_unweighted,
        test_n_parameters,
        test_trainer_log_txt,
        test_checkpoint_format,
    ]

    print(f"\nRunning mock tests from: {os.getcwd()}\n{'='*55}")
    passed = 0
    failed = 0
    for t in tests:
        print(f"\n--- {t.__name__} ---")
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"{FAIL} FAILED: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{'='*55}")
    print(f"Results: {passed} passed, {failed} failed\n")
    sys.exit(failed)


if __name__ == "__main__":
    run_all()
