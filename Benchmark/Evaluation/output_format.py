"""Output files used by training and later result analysis.

The Benchmark keeps its structured JSON artifacts, and also writes the simple
LaBraM-compatible files that downstream analysis scripts expect:

- one JSON object per epoch in `log.txt`
- `checkpoint.pth` for the latest epoch
- `checkpoint-<epoch>.pth` for each epoch
- `checkpoint-best.pth` for the best validation epoch
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def epoch_log_record(
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    n_parameters: int,
) -> dict[str, float | int]:
    """Build one `log.txt` JSON line for LaBraM-style analysis."""
    record: dict[str, float | int] = {
        "epoch": int(epoch),
        "n_parameters": int(n_parameters),
    }
    record.update({f"train_{key}": value for key, value in train_metrics.items()})
    record.update({f"val_{key}": value for key, value in val_metrics.items()})
    record.update({f"test_{key}": value for key, value in test_metrics.items()})

    if "train_accuracy" in record:
        record["train_class_acc"] = record["train_accuracy"]
    return record


def record_log_txt(run_dir: Path, record: dict[str, Any]) -> None:
    """Append one JSON line to `log.txt`."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "log.txt").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def trainable_parameter_count(model) -> int:
    """Match LaBraM's `n_parameters`: trainable parameters only."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def checkpoint_payload(
    model,
    optimizer,
    scheduler,
    epoch: int | str,
    record: dict[str, Any],
    config: dict,
) -> dict[str, Any]:
    """Create one portable checkpoint payload."""
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "log_record": record,
        "config": config,
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    return payload


def save_epoch_checkpoints(
    run_dir: Path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    record: dict[str, Any],
    config: dict,
) -> None:
    """Save latest and per-epoch checkpoints in LaBraM-compatible names."""
    payload = checkpoint_payload(model, optimizer, scheduler, epoch, record, config)
    torch.save(payload, run_dir / "checkpoint.pth")
    torch.save(payload, run_dir / f"checkpoint-{int(epoch)}.pth")


def save_best_checkpoint(
    run_dir: Path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    record: dict[str, Any],
    config: dict,
) -> Path:
    """Save the validation-selected checkpoint and return its path."""
    payload = checkpoint_payload(model, optimizer, scheduler, "best", record, config)
    payload["best_epoch"] = int(epoch)
    path = run_dir / "checkpoint-best.pth"
    torch.save(payload, path)
    torch.save(payload, run_dir / "best_model.pt")
    return path
