#!/usr/bin/env python3
"""LaBraM NMT OOD evaluation.

Runs the NMT OOD comparison used by the LaBraM protocol scripts. Optional
freeze-backbone and LoRA checkpoints are compared with full fine-tuning and
precomputed EEGNet baseline metrics when supplied.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
LABRAM_ROOT = REPO_ROOT / "EEG-FM" / "Labram"
DEFAULT_CONFIG = REPO_ROOT / "Benchmark" / "Config" / "labram.yaml"
FINETUNING_STRATEGY_ROOT = REPO_ROOT / "Benchmark" / "FinetuningStrategy"

sys.path.insert(0, str(FINETUNING_STRATEGY_ROOT))
sys.path.insert(0, str(LABRAM_ROOT))

from _lora_common import DEFAULT_LORA_SETTINGS, strategy_freeze_backbone
from lora_labram import apply_lora_strategy as apply_model_lora_strategy
from run_class_finetuning import get_models
from utils import get_input_chans


MODEL_NAME = "LaBraM"
MODEL_LABEL = "LaBraM"
METRIC_KEYS = ("test_accuracy", "test_balanced_accuracy", "test_roc_auc", "test_pr_auc")
TUAB_CH = [
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "A1", "A2", "FZ", "CZ", "PZ", "T1", "T2",
]

NMT_BASE = Path("/home/meriem-ubuntu/Projects/Foundation-models/nmt_scalp_eeg_labram_ood")
NMT_CKPT_BASE = NMT_BASE / "Checkpoints"


# TUAB clean metrics are optional context for the NMT OOD report. They are not
# recomputed here.
CLEAN_TUAB = {
    "ft": {"accuracy": 0.8228, "balanced_accuracy": 0.8186, "roc_auc": 0.9040, "epoch": 1},
    "lora": {"accuracy": 0.8186, "balanced_accuracy": 0.8144, "roc_auc": 0.8981, "epoch": 22},
    "eegnet": {"accuracy": 0.7869, "balanced_accuracy": 0.7821, "roc_auc": 0.8536, "epoch": 11},
}


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
    """Read NMT samples saved as {'X': [23,2000], 'y': 0/1} pickles."""
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
        return torch.FloatTensor(sample["X"]), int(sample["y"])


def load_config(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model():
    model_args = SimpleNamespace(
        model="labram_base_patch200_200",
        nb_classes=1,
        input_size=200,
        qkv_bias=False,
        rel_pos_bias=True,
        abs_pos_emb=True,
        layer_scale_init_value=0.1,
        drop=0.0,
        attn_drop_rate=0.0,
        drop_path=0.1,
        use_mean_pooling=True,
        init_scale=0.001,
        dataset="TUAB",
    )
    return get_models(model_args)


def load_checkpoint(model, checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=False)
    return model


def apply_checkpoint_strategy(model, strategy: str, output_dir: Path):
    if strategy in {"full_ft", "full_finetune", "original"}:
        return None
    if strategy == "freeze_backbone":
        args = SimpleNamespace(model_name=MODEL_NAME, model=MODEL_NAME, allow_head_guess=True)
        return strategy_freeze_backbone(model, args, model_name=MODEL_NAME)
    if strategy == "lora":
        return apply_model_lora_strategy(
            model,
            lora_settings=DEFAULT_LORA_SETTINGS,
            output_dir=output_dir,
            allow_head_guess=True,
        )
    raise ValueError(f"Unknown inference checkpoint strategy: {strategy}")


def load_strategy_checkpoint(checkpoint_path: str, strategy: str, output_dir: Path):
    model = build_model()
    apply_checkpoint_strategy(
        model,
        strategy,
        output_dir=output_dir / strategy,
    )
    return load_checkpoint(model, checkpoint_path)


def load_eegnet_nmt_results(json_path: Path) -> dict:
    with Path(json_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_eegnet_nmt_metrics(metrics: dict) -> dict:
    return {
        "test_loss": metrics.get("test_loss") or metrics.get("loss"),
        "test_accuracy": metrics.get("test_accuracy") or metrics.get("accuracy"),
        "test_balanced_accuracy": metrics.get("test_balanced_accuracy") or metrics.get("balanced_accuracy"),
        "test_roc_auc": metrics.get("test_roc_auc") or metrics.get("roc_auc") or metrics.get("auc"),
        "test_pr_auc": metrics.get("test_pr_auc") or metrics.get("pr_auc") or metrics.get("auprc"),
        "n_samples": metrics.get("n_samples"),
    }


def run_nmt_inference(model, dataset, device, batch_size, threshold):
    from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, roc_auc_score

    input_chans = get_input_chans(TUAB_CH)
    criterion = torch.nn.BCEWithLogitsLoss()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=(device.type == "cuda"))
    labels, probs, losses = [], [], []

    model.to(device).eval()
    with torch.no_grad():
        for x, y in loader:
            # NMT pickles are [B,23,2000]. Match the original OOD script:
            # scale by 100 and regroup to LaBraM patches.
            x = x.float().to(device) / 100.0
            x = rearrange(x, "B N (A T) -> B N A T", T=200)
            y = y.float().to(device)
            logits = model(x, input_chans=input_chans).view(-1)
            target = y.view(-1)
            losses.append(float(criterion(logits, target).item()))
            labels.append(target.cpu().numpy())
            probs.append(torch.sigmoid(logits).cpu().numpy())

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
        benchmark_log_record("LaBraM Full FT", results["ft"]["nmt_ood"], results["ft"]["checkpoint"], "NMT_OOD"),
    ]
    if "lora" in results:
        records.append(
            benchmark_log_record("LaBraM LoRA", results["lora"]["nmt_ood"],
                                 results["lora"]["checkpoint"], "NMT_OOD")
        )
    if "freeze_backbone" in results:
        records.append(
            benchmark_log_record("LaBraM Freeze Backbone", results["freeze_backbone"]["nmt_ood"],
                                 results["freeze_backbone"]["checkpoint"], "NMT_OOD")
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


def parse_args():
    parser = argparse.ArgumentParser(description="Run LaBraM NMT OOD inference.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ckpt_ft", default=str(NMT_CKPT_BASE / "checkpoint-best-original-42-TUAB.pth"))
    parser.add_argument("--ckpt_freeze_backbone", default=None,
                        help="Optional freeze-backbone checkpoint.")
    parser.add_argument("--ckpt_lora", default=None,
                        help="Optional LoRA checkpoint. If omitted, compare Full FT vs EEGNet only.")
    parser.add_argument("--eegnet_json", type=Path, default=NMT_BASE / "nmt_eegnet_results.json",
                        help="Precomputed EEGNet NMT metrics JSON.")
    parser.add_argument("--nmt_dir", type=Path, default=NMT_BASE / "test")
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
    paths = config.get("paths", {})
    evaluation = config.get("evaluation", {})
    output_dir = Path(args.output_dir or paths.get("output", "outputs/labram"))
    threshold = args.classification_threshold or evaluation.get("classification_threshold", 0.5)

    if args.dry_run:
        print(f"config={args.config}")
        print(f"ckpt_ft={args.ckpt_ft}")
        print(f"ckpt_freeze_backbone={args.ckpt_freeze_backbone}")
        print(f"ckpt_lora={args.ckpt_lora}")
        print(f"eegnet_json={args.eegnet_json}")
        print(f"nmt_dir={args.nmt_dir}")
        print(f"output_dir={output_dir}")
        return

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = NmtPickleDataset(args.nmt_dir, max_samples=args.max_samples)
    batch_size = args.batch_size or config.get("loader", {}).get("batch_size", 64)
    eegnet_metrics = normalize_eegnet_nmt_metrics(load_eegnet_nmt_results(args.eegnet_json))
    model_ft = load_checkpoint(build_model(), args.ckpt_ft)
    ft_metrics = run_nmt_inference(model_ft, dataset, device, batch_size, float(threshold))
    results = {
        "description": (
            "LaBraM NMT OOD inference from Benchmark/Inference. "
            "NMT pickles are scaled and patched like the original LaBraM OOD script."
        ),
        "ft": {
            "tuab_clean": CLEAN_TUAB["ft"],
            "nmt_ood": ft_metrics,
            "checkpoint": args.ckpt_ft,
        },
        "eegnet": {
            "tuab_clean": CLEAN_TUAB["eegnet"],
            "nmt_ood": eegnet_metrics,
            "source_json": str(args.eegnet_json),
        },
        "comparison": {
            "eegnet_minus_ft_test_accuracy": metric_delta(eegnet_metrics, ft_metrics, "test_accuracy"),
            "eegnet_minus_ft_test_balanced_accuracy": metric_delta(eegnet_metrics, ft_metrics, "test_balanced_accuracy"),
            "eegnet_minus_ft_test_roc_auc": metric_delta(eegnet_metrics, ft_metrics, "test_roc_auc"),
            "eegnet_minus_ft_test_pr_auc": metric_delta(eegnet_metrics, ft_metrics, "test_pr_auc"),
        },
    }
    if args.ckpt_lora:
        model_lora = load_strategy_checkpoint(args.ckpt_lora, "lora", output_dir)
        lora_metrics = run_nmt_inference(model_lora, dataset, device, batch_size, float(threshold))
        results["lora"] = {
            "tuab_clean": CLEAN_TUAB["lora"],
            "nmt_ood": lora_metrics,
            "checkpoint": args.ckpt_lora,
        }
        results["comparison"].update({
            "lora_minus_ft_test_accuracy": metric_delta(lora_metrics, ft_metrics, "test_accuracy"),
            "lora_minus_ft_test_balanced_accuracy": metric_delta(lora_metrics, ft_metrics, "test_balanced_accuracy"),
            "lora_minus_ft_test_roc_auc": metric_delta(lora_metrics, ft_metrics, "test_roc_auc"),
            "lora_minus_ft_test_pr_auc": metric_delta(lora_metrics, ft_metrics, "test_pr_auc"),
            "lora_minus_eegnet_test_accuracy": metric_delta(lora_metrics, eegnet_metrics, "test_accuracy"),
            "lora_minus_eegnet_test_balanced_accuracy": metric_delta(
                lora_metrics, eegnet_metrics, "test_balanced_accuracy"
            ),
            "lora_minus_eegnet_test_roc_auc": metric_delta(lora_metrics, eegnet_metrics, "test_roc_auc"),
            "lora_minus_eegnet_test_pr_auc": metric_delta(lora_metrics, eegnet_metrics, "test_pr_auc"),
        })
    if args.ckpt_freeze_backbone:
        model_freeze = load_strategy_checkpoint(args.ckpt_freeze_backbone, "freeze_backbone", output_dir)
        freeze_metrics = run_nmt_inference(model_freeze, dataset, device, batch_size, float(threshold))
        results["freeze_backbone"] = {
            "nmt_ood": freeze_metrics,
            "checkpoint": args.ckpt_freeze_backbone,
        }
        add_metric_deltas(results["comparison"], "freeze_backbone", freeze_metrics, "ft", ft_metrics)
        add_metric_deltas(results["comparison"], "freeze_backbone", freeze_metrics, "eegnet", eegnet_metrics)
    write_nmt_outputs(output_dir, results)
    print(f"Wrote {output_dir / 'nmt_ood_results.json'}")
    print(f"Wrote {output_dir / 'log.txt'}")


def metric_delta(left: dict, right: dict, key: str):
    if left.get(key) is None or right.get(key) is None:
        return None
    return float(left[key] - right[key])


def add_metric_deltas(comparison: dict, left_name: str, left_metrics: dict, right_name: str, right_metrics: dict) -> None:
    for key in METRIC_KEYS:
        comparison[f"{left_name}_minus_{right_name}_{key}"] = metric_delta(left_metrics, right_metrics, key)


if __name__ == "__main__":
    main()
