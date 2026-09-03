#!/usr/bin/env python3
"""CBraMod NMT OOD evaluation."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import torch
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
CBRAMOD_ROOT = REPO_ROOT / "EEG-FM" / "CBraMod"
BENCHMARK_LOADER = REPO_ROOT / "Benchmark" / "DataLoader"
DEFAULT_CONFIG = REPO_ROOT / "Benchmark" / "Config" / "cbramod.yaml"
FINETUNING_STRATEGY_ROOT = REPO_ROOT / "Benchmark" / "FinetuningStrategy"

sys.path.insert(0, str(FINETUNING_STRATEGY_ROOT))
sys.path.insert(0, str(BENCHMARK_LOADER))
sys.path.insert(0, str(CBRAMOD_ROOT))

from _lora_common import DEFAULT_LORA_SETTINGS, strategy_freeze_backbone
from lora_cbramod import apply_lora_strategy as apply_model_lora_strategy
from loader_cbramod import transform_h5_window
from models import model_for_tuab


MODEL_NAME = "CBraMod"
MODEL_LABEL = "CBraMod"
METRIC_KEYS = ("test_accuracy", "test_balanced_accuracy", "test_roc_auc", "test_pr_auc")
TUAB_CH = [
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "A1", "A2", "FZ", "CZ", "PZ", "T1", "T2",
]

class _NumpyCompatUnpickler(pickle.Unpickler):
    """Load numpy>=2 pickles in older numpy environments."""
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def load_pickle(path: Path):
    with path.open("rb") as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError:
            f.seek(0)
            return _NumpyCompatUnpickler(f).load()


class NmtPickleDataset(Dataset):
    """Read NMT samples and adapt them to CBraMod's TUAB input shape."""
    def __init__(self, root: Path, max_samples: int | None = None):
        self.root = Path(root)
        self.files = sorted(self.root.glob("*.pkl"))
        if max_samples is not None:
            self.files = self.files[:max_samples]
        if not self.files:
            raise FileNotFoundError(f"no NMT .pkl files found in {self.root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = load_pickle(self.files[idx])
        x = transform_h5_window(sample["X"], TUAB_CH, {})
        y = sample.get("y", sample.get("label"))
        return torch.as_tensor(x, dtype=torch.float32), int(y)


def load_config(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_repo_path(value: str | Path | None) -> Path | None:
    """Resolve YAML paths consistently, independent of the launch directory."""
    if value in (None, ""):
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def validate_inference_inputs(nmt_dir: Path | None, eegnet_json: Path | None,
                              checkpoints: dict[str, Path | None]) -> None:
    problems = []
    if nmt_dir is None or not nmt_dir.is_dir():
        problems.append(f"inference.nmt_dir is not a directory: {nmt_dir}")
    if eegnet_json is None or not eegnet_json.is_file():
        problems.append(f"inference.eegnet_json is not a file: {eegnet_json}")
    if checkpoints["full_finetune"] is None:
        problems.append("inference.checkpoints.full_finetune is required")
    for strategy, path in checkpoints.items():
        if path is not None and not path.is_file():
            problems.append(f"inference.checkpoints.{strategy} is not a file: {path}")
    if problems:
        raise FileNotFoundError("Invalid inference configuration:\n- " + "\n- ".join(problems))


def build_model(config):
    fixed_model = config.get("fixed_recipe", {}).get("model", {})
    params = SimpleNamespace(
        downstream_dataset=fixed_model.get("downstream_dataset", "TUAB"),
        num_of_classes=int(fixed_model.get("num_of_classes", 2)),
        classifier=fixed_model.get("classifier", "all_patch_reps"),
        dropout=float(fixed_model.get("dropout", 0.1)),
        use_pretrained_weights=False,
        foundation_dir="",
        cuda=0,
    )
    return model_for_tuab.Model(params)


def load_checkpoint(model, checkpoint_path: str, checkpoint=None):
    checkpoint = checkpoint if checkpoint is not None else torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(clean_state_dict(state_dict, model.state_dict()), strict=False)
    model._benchmark_checkpoint_epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    return model


def apply_checkpoint_strategy(model, strategy: str, output_dir: Path, lora_settings: dict | None = None):
    if strategy in {"full_ft", "full_finetune", "original"}:
        return None
    if strategy == "freeze_backbone":
        args = SimpleNamespace(model_name=MODEL_NAME, model=MODEL_NAME, allow_head_guess=True)
        return strategy_freeze_backbone(model, args, model_name=MODEL_NAME)
    if strategy == "lora":
        return apply_model_lora_strategy(
            model,
            lora_settings=lora_settings or DEFAULT_LORA_SETTINGS,
            output_dir=output_dir,
            allow_head_guess=True,
        )
    raise ValueError(f"Unknown inference checkpoint strategy: {strategy}")


def load_strategy_checkpoint(model, checkpoint_path: str, strategy: str, output_dir: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    apply_checkpoint_strategy(
        model,
        strategy,
        output_dir=output_dir / strategy,
        lora_settings=checkpoint_lora_settings(checkpoint) if strategy == "lora" else None,
    )
    return load_checkpoint(model, checkpoint_path, checkpoint=checkpoint)


def checkpoint_lora_settings(checkpoint) -> dict:
    """Rebuild the exact LoRA shape recorded by the training checkpoint."""
    saved_args = checkpoint.get("args") if isinstance(checkpoint, dict) else None
    get_value = saved_args.get if isinstance(saved_args, dict) else lambda key, default=None: getattr(saved_args, key, default)
    return {
        "rank": get_value("lora_rank", DEFAULT_LORA_SETTINGS["rank"]),
        "alpha": get_value("lora_alpha", DEFAULT_LORA_SETTINGS["alpha"]),
        "layers": get_value("lora_layers", DEFAULT_LORA_SETTINGS["layers"]),
        "target": get_value("lora_target", DEFAULT_LORA_SETTINGS["target"]),
        "init_scale": get_value("lora_init_scale", DEFAULT_LORA_SETTINGS["init_scale"]),
    }


def load_eegnet_results(json_path: Path) -> tuple[dict, dict]:
    with Path(json_path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    data = data.get("eegnet", data)
    clean_metrics = data.get("tuab_clean")
    if not isinstance(clean_metrics, dict):
        raise ValueError(f"EEGNet JSON must contain a tuab_clean object: {json_path}")
    return normalize_eegnet_nmt_metrics(data.get("nmt_ood", data)), clean_metrics


def load_clean_tuab_metrics(checkpoint_path: Path, checkpoint_epoch) -> dict:
    """Read clean TUAB metrics corresponding to the selected best checkpoint."""
    log_path = checkpoint_path.parent / "log.txt"
    if not log_path.is_file():
        raise FileNotFoundError(f"standardized training log not found beside checkpoint: {log_path}")
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    matching = [record for record in records if record.get("epoch") == checkpoint_epoch]
    if not matching:
        matching = [record for record in records if record.get("val_roc_auc") is not None]
    if not matching:
        raise ValueError(f"no validation ROC-AUC records found in {log_path}")
    record = max(matching, key=lambda item: float(item["val_roc_auc"]))
    return {
        "accuracy": record.get("test_accuracy"),
        "balanced_accuracy": record.get("test_balanced_accuracy"),
        "roc_auc": record.get("test_roc_auc"),
        "pr_auc": record.get("test_pr_auc"),
        "loss": record.get("test_loss"),
        "epoch": record.get("epoch"),
    }


def normalize_eegnet_nmt_metrics(metrics: dict) -> dict:
    return {
        "test_loss": metrics.get("test_loss") or metrics.get("loss"),
        "test_accuracy": metrics.get("test_accuracy") or metrics.get("accuracy"),
        "test_balanced_accuracy": metrics.get("test_balanced_accuracy") or metrics.get("balanced_accuracy"),
        "test_roc_auc": metrics.get("test_roc_auc") or metrics.get("roc_auc") or metrics.get("auc"),
        "test_pr_auc": metrics.get("test_pr_auc") or metrics.get("pr_auc") or metrics.get("auprc"),
        "n_samples": metrics.get("n_samples"),
    }


def clean_state_dict(state_dict: dict, model_state: dict) -> dict:
    cleaned = {}
    for key, value in state_dict.items():
        name = key[7:] if key.startswith("module.") else key
        if name in model_state and tuple(model_state[name].shape) == tuple(value.shape):
            cleaned[name] = value
    return cleaned


def logits_to_probability(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim > 1 and logits.shape[-1] == 2:
        return torch.softmax(logits, dim=-1)[:, 1]
    return torch.sigmoid(logits.view(-1))


def loss_for_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if logits.ndim > 1 and logits.shape[-1] == 2:
        return torch.nn.CrossEntropyLoss()(logits, target.long())
    return torch.nn.BCEWithLogitsLoss()(logits.view(-1), target.float().view(-1))


def run_nmt_inference(model, dataset, device, batch_size, threshold):
    from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, roc_auc_score

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=(device.type == "cuda"))
    labels, probs, losses = [], [], []

    model.to(device).eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.float().to(device)
            y = y.to(device)
            logits = model(x)
            losses.append(float(loss_for_logits(logits, y).item()))
            labels.append(y.view(-1).cpu().numpy())
            probs.append(logits_to_probability(logits).cpu().numpy())

    y_true = np.concatenate(labels).astype(int)
    y_prob = np.concatenate(probs)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "test_loss": float(np.mean(losses)),
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "test_roc_auc": safe_metric(roc_auc_score, y_true, y_prob),
        "test_pr_auc": safe_metric(average_precision_score, y_true, y_prob),
        "n_samples": int(len(y_true)),
    }


def safe_metric(fn, y_true, y_prob):
    try:
        return float(fn(y_true, y_prob))
    except ValueError:
        return None


def benchmark_log_record(model_name: str, metrics: dict, checkpoint_path: str, dataset_name: str):
    return {
        "train_loss": None,
        "train_lr": None,
        "train_weight_decay": None,
        "train_class_acc": None,
        "train_balanced_accuracy": None,
        "train_grad_norm": None,
        "train_min_lr": None,
        "train_loss_scale": None,
        "val_roc_auc": None,
        "val_pr_auc": None,
        "val_accuracy": None,
        "val_balanced_accuracy": None,
        "val_loss": None,
        **metrics,
        "epoch": None,
        "mode": "inference",
        "dataset": dataset_name,
        "model": model_name,
        "checkpoint": checkpoint_path,
    }


def write_nmt_outputs(output_dir: Path, results: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [
        benchmark_log_record("CBraMod Full FT", results["full_ft"]["nmt_ood"],
                             results["full_ft"]["checkpoint"], "NMT_OOD"),
    ]
    if "freeze_backbone" in results:
        records.append(
            benchmark_log_record("CBraMod Freeze Backbone", results["freeze_backbone"]["nmt_ood"],
                                 results["freeze_backbone"]["checkpoint"], "NMT_OOD")
        )
    if "lora" in results:
        records.append(
            benchmark_log_record("CBraMod LoRA", results["lora"]["nmt_ood"],
                                 results["lora"]["checkpoint"], "NMT_OOD")
        )
    records.append(
        benchmark_log_record("EEGNet", results["eegnet"]["nmt_ood"],
                             results["eegnet"]["source_json"], "NMT_OOD")
    )
    with (output_dir / "log.txt").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with (output_dir / "nmt_ood_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def metric_delta(left: dict, right: dict, key: str):
    if left.get(key) is None or right.get(key) is None:
        return None
    return float(left[key] - right[key])


def add_metric_deltas(comparison: dict, left_name: str, left_metrics: dict, right_name: str, right_metrics: dict) -> None:
    for key in METRIC_KEYS:
        comparison[f"{left_name}_minus_{right_name}_{key}"] = metric_delta(left_metrics, right_metrics, key)


def parse_args():
    parser = argparse.ArgumentParser(description="Run CBraMod NMT OOD inference.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--ckpt_freeze_backbone", default=None)
    parser.add_argument("--ckpt_lora", default=None)
    parser.add_argument("--eegnet_json", type=Path, default=None)
    parser.add_argument("--nmt_dir", type=Path, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--classification_threshold", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    inference = config.get("inference", {})
    checkpoint_config = inference.get("checkpoints", {})
    evaluation = config.get("evaluation", {})
    checkpoints = {
        "full_finetune": resolve_repo_path(args.checkpoint or checkpoint_config.get("full_finetune")),
        "freeze_backbone": resolve_repo_path(args.ckpt_freeze_backbone or checkpoint_config.get("freeze_backbone")),
        "lora": resolve_repo_path(args.ckpt_lora or checkpoint_config.get("lora")),
    }
    nmt_dir = resolve_repo_path(args.nmt_dir or inference.get("nmt_dir"))
    eegnet_json = resolve_repo_path(args.eegnet_json or inference.get("eegnet_json"))
    output_dir = resolve_repo_path(args.output_dir or inference.get("output"))
    if output_dir is None:
        raise ValueError("inference.output must be set in the YAML config")
    threshold = args.classification_threshold if args.classification_threshold is not None else evaluation.get("classification_threshold", 0.5)
    max_samples = args.max_samples if args.max_samples is not None else inference.get("max_samples")
    device_name = args.device if args.device is not None else inference.get("device")

    if args.dry_run:
        print(f"config={args.config}")
        for strategy, path in checkpoints.items():
            print(f"checkpoint.{strategy}={path} exists={bool(path and path.is_file())}")
        print(f"eegnet_json={eegnet_json} exists={bool(eegnet_json and eegnet_json.is_file())}")
        print(f"nmt_dir={nmt_dir} exists={bool(nmt_dir and nmt_dir.is_dir())}")
        print(f"output_dir={output_dir}")
        print(f"max_samples={max_samples}")
        print(f"device={device_name or 'auto'}")
        return

    validate_inference_inputs(nmt_dir, eegnet_json, checkpoints)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = NmtPickleDataset(nmt_dir, max_samples=max_samples)
    batch_size = args.batch_size if args.batch_size is not None else inference.get("batch_size", config.get("loader", {}).get("batch_size", 64))
    model = load_checkpoint(build_model(config), checkpoints["full_finetune"])
    metrics = run_nmt_inference(model, dataset, device, batch_size, float(threshold))
    eegnet_metrics, eegnet_clean = load_eegnet_results(eegnet_json)
    results = {
        "description": "CBraMod full fine-tuning vs EEGNet NMT OOD inference from Benchmark/Inference.",
        "nmt_ood": metrics,
        "checkpoint": str(checkpoints["full_finetune"]),
        "full_ft": {
            "tuab_clean": load_clean_tuab_metrics(checkpoints["full_finetune"], model._benchmark_checkpoint_epoch),
            "nmt_ood": metrics,
            "checkpoint": str(checkpoints["full_finetune"]),
        },
        "eegnet": {
            "tuab_clean": eegnet_clean,
            "nmt_ood": eegnet_metrics,
            "source_json": str(eegnet_json),
        },
        "comparison": {
            "eegnet_minus_full_ft_test_accuracy": metric_delta(eegnet_metrics, metrics, "test_accuracy"),
            "eegnet_minus_full_ft_test_balanced_accuracy": metric_delta(
                eegnet_metrics, metrics, "test_balanced_accuracy"
            ),
            "eegnet_minus_full_ft_test_roc_auc": metric_delta(eegnet_metrics, metrics, "test_roc_auc"),
            "eegnet_minus_full_ft_test_pr_auc": metric_delta(eegnet_metrics, metrics, "test_pr_auc"),
            "full_ft_minus_eegnet_test_accuracy": metric_delta(metrics, eegnet_metrics, "test_accuracy"),
            "full_ft_minus_eegnet_test_balanced_accuracy": metric_delta(
                metrics, eegnet_metrics, "test_balanced_accuracy"
            ),
            "full_ft_minus_eegnet_test_roc_auc": metric_delta(metrics, eegnet_metrics, "test_roc_auc"),
            "full_ft_minus_eegnet_test_pr_auc": metric_delta(metrics, eegnet_metrics, "test_pr_auc"),
        },
    }
    if checkpoints["freeze_backbone"]:
        model_freeze = load_strategy_checkpoint(build_model(config), checkpoints["freeze_backbone"], "freeze_backbone", output_dir)
        freeze_metrics = run_nmt_inference(model_freeze, dataset, device, batch_size, float(threshold))
        results["freeze_backbone"] = {
            "tuab_clean": load_clean_tuab_metrics(checkpoints["freeze_backbone"], model_freeze._benchmark_checkpoint_epoch),
            "nmt_ood": freeze_metrics,
            "checkpoint": str(checkpoints["freeze_backbone"]),
        }
        add_metric_deltas(results["comparison"], "freeze_backbone", freeze_metrics, "full_ft", metrics)
        add_metric_deltas(results["comparison"], "freeze_backbone", freeze_metrics, "eegnet", eegnet_metrics)
    if checkpoints["lora"]:
        model_lora = load_strategy_checkpoint(build_model(config), checkpoints["lora"], "lora", output_dir)
        lora_metrics = run_nmt_inference(model_lora, dataset, device, batch_size, float(threshold))
        results["lora"] = {
            "tuab_clean": load_clean_tuab_metrics(checkpoints["lora"], model_lora._benchmark_checkpoint_epoch),
            "nmt_ood": lora_metrics,
            "checkpoint": str(checkpoints["lora"]),
        }
        add_metric_deltas(results["comparison"], "lora", lora_metrics, "full_ft", metrics)
        add_metric_deltas(results["comparison"], "lora", lora_metrics, "eegnet", eegnet_metrics)
    write_nmt_outputs(output_dir, results)
    print(f"Wrote {output_dir / 'nmt_ood_results.json'}")
    print(f"Wrote {output_dir / 'log.txt'}")


if __name__ == "__main__":
    main()
