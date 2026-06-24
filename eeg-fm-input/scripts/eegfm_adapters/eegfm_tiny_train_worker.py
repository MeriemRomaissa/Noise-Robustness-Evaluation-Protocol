#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-model tiny training smoke worker for canonical TUAB EEG-FM adapters."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from canonical_h5_subset import select_tuab_16, to_patches


DEFAULT_PRETRAINED_CHECKPOINTS = {
    "LaBraM": "checkpoints/labram-base.pth",
    "EEGPT": "checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt",
    "BIOT": "pretrained-models/EEG-PREST-16-channels.ckpt",
    "CBraMod": "pretrained_weights/pretrained_weights.pth",
    "CSBrain": "pth/CSBrain.pth",
    "CodeBrain": "Checkpoints/CodeBrain.pth",
}


def shape_of(value) -> list[int] | None:
    return list(value.shape) if hasattr(value, "shape") else None


class NoOpTensorCuda:
    """Keep repo-internal .cuda() calls on CPU for adapter-only smoke tests."""

    def __enter__(self):
        self._orig_cuda = torch.Tensor.cuda
        torch.Tensor.cuda = lambda tensor, *args, **kwargs: tensor
        return self

    def __exit__(self, exc_type, exc, tb):
        torch.Tensor.cuda = self._orig_cuda
        return False


class LaBraMWrapper(nn.Module):
    def __init__(self, repo_path: str):
        super().__init__()
        sys.path.insert(0, repo_path)
        import modeling_finetune

        self.model = modeling_finetune.labram_base_d6_patch200_200(
            pretrained=False,
            num_classes=2,
            EEG_size=2000,
            drop_path_rate=0.0,
            init_values=0.1,
            qkv_bias=True,
            use_abs_pos_emb=True,
            use_rel_pos_bias=False,
        )

    def forward(self, x):
        input_chans = torch.arange(0, x.shape[1] + 1, device=x.device)
        return self.model(x, input_chans=input_chans)


class EEGPTWrapper(nn.Module):
    """Native TUAB EEGPT classifier wrapper using checkpoint-compatible config."""

    native_config = {
        "class": "EEGPTClassifier",
        "num_classes": 1,
        "in_channels": 23,
        "img_size": [20, 2000],
        "patch_size": 64,
        "patch_stride": 64,
        "embed_dim": 512,
        "embed_num": 4,
        "depth": 8,
        "num_heads": 8,
        "mlp_ratio": 4.0,
        "use_chan_conv": True,
        "use_mean_pooling": True,
    }

    def __init__(self, repo_path: str, channel_names: list[str], input_shape: tuple[int, int]):
        super().__init__()
        sys.path.insert(0, repo_path)
        from downstream_tueg.Modules.models.EEGPT_mcae_finetune_change import EEGPTClassifier

        use_channels_names = [
            "FP1",
            "FPZ",
            "FP2",
            "F7",
            "F3",
            "FZ",
            "F4",
            "F8",
            "T7",
            "C3",
            "CZ",
            "C4",
            "T8",
            "P7",
            "P3",
            "PZ",
            "P4",
            "P8",
            "O1",
            "O2",
        ]
        self.model = EEGPTClassifier(
            num_classes=1,
            in_channels=input_shape[0],
            img_size=[len(use_channels_names), input_shape[1]],
            use_channels_names=use_channels_names,
            use_chan_conv=True,
            use_mean_pooling=True,
        )
        self.input_channel_names = list(channel_names)

    def forward(self, x):
        return self.model(x)


def resolve_codebrain_repo_path(repo_path: str) -> Path:
    repo = Path(repo_path)
    if repo.exists():
        return repo
    parent = repo.parent
    for candidate_name in ["Codebrain", "CodeBrain", "codebrain"]:
        candidate = parent / candidate_name
        if candidate.exists():
            return candidate
    fallback = Path("/nicoletye/workspace/Noise Robustness Evaluation Protocol/EEG-FM/Codebrain")
    return fallback if fallback.exists() else repo


def import_codebrain_tuab_model(repo_path: str):
    repo = resolve_codebrain_repo_path(repo_path)
    for path in [repo, repo / "Models", repo / "models"]:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))

    try:
        from Models.model_for_tuab import Model

        return Model
    except Exception:
        pass

    try:
        from models.model_for_tuab import Model

        return Model
    except Exception:
        pass

    candidates = list(repo.glob("**/model_for_tuab.py"))
    if not candidates:
        raise ModuleNotFoundError(f"Could not find model_for_tuab.py under CodeBrain repo path: {repo}")
    model_file = candidates[0]
    spec = importlib.util.spec_from_file_location("codebrain_model_for_tuab_adapter", model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {model_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Model


class CodeBrainWrapper(nn.Module):
    def __init__(self, repo_path: str):
        super().__init__()
        os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
        Model = import_codebrain_tuab_model(repo_path)

        param = SimpleNamespace(
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
        self.model = Model(param)

    def forward(self, x):
        bz, ch_num, seq_len, patch_size = x.shape
        context = NoOpTensorCuda() if x.device.type == "cpu" else nullcontext()
        with context:
            feats = self.model.backbone(x)
        feats = feats.contiguous().view(bz, ch_num * seq_len * patch_size)
        out = self.model.classifier(feats)
        return out.contiguous().view(bz)


class FeatureHeadWrapper(nn.Module):
    def __init__(self, base: nn.Module, feature_dim: int, n_classes: int = 2):
        super().__init__()
        self.base = base
        self.head = nn.Linear(feature_dim, n_classes)

    def forward(self, x):
        features = self.base(x)
        return self.head(features.reshape(features.shape[0], -1))


class nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def load_tiny_npz(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    return {
        "train_x": torch.from_numpy(data["train_x"].astype(np.float32)),
        "train_y": torch.from_numpy(data["train_y"].astype(np.int64)),
        "val_x": torch.from_numpy(data["val_x"].astype(np.float32)),
        "val_y": torch.from_numpy(data["val_y"].astype(np.int64)),
        "test_x": torch.from_numpy(data["test_x"].astype(np.float32)),
        "test_y": torch.from_numpy(data["test_y"].astype(np.int64)),
        "channel_names": [str(x) for x in data["channel_names"].tolist()],
        "split_counts": json.loads(str(data["split_counts_json"][0])),
        "tiny_counts": json.loads(str(data["tiny_counts_json"][0])),
    }


def eegpt_adapt(raw: torch.Tensor, channel_names: list[str]) -> tuple[torch.Tensor, dict]:
    sys_modules = sys.modules
    # CHANNEL_DICT is imported inside the EEGPT venv after repo_path is on sys.path.
    module = sys_modules.get("downstream_tueg.Modules.models.EEGPT_mcae_finetune_change")
    channel_dict = getattr(module, "CHANNEL_DICT", None)
    if channel_dict is None:
        from downstream_tueg.Modules.models.EEGPT_mcae_finetune_change import CHANNEL_DICT

        channel_dict = CHANNEL_DICT

    name_map = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
    keep_indices = []
    final_names = []
    renamed = []
    dropped = []
    for idx, name in enumerate(channel_names):
        src = name.upper()
        dst = name_map.get(src, src)
        if dst != src:
            renamed.append({"from": src, "to": dst})
        if dst in channel_dict:
            keep_indices.append(idx)
            final_names.append(dst)
        else:
            dropped.append(src)
    return raw[:, keep_indices, :], {
        "eegpt_channel_names": final_names,
        "eegpt_selected_channel_indices": keep_indices,
        "eegpt_renamed_channels": renamed,
        "eegpt_dropped_channels": dropped,
    }


def adapt_inputs(model_name: str, raw: torch.Tensor, channel_names: list[str], repo_path: str) -> tuple[torch.Tensor, dict]:
    if model_name == "LaBraM":
        return to_patches(raw, 200), {"adapter": "LaBraM [B,23,10,200]"}
    if model_name == "EEGPT":
        # Native TUAB EEGPT uses all 23 canonical channels as input and an
        # internal channel convolution maps them to the 20 EEGPT channel set.
        use_channels_names = [
            "FP1", "FPZ", "FP2", "F7", "F3", "FZ", "F4", "F8", "T7", "C3",
            "CZ", "C4", "T8", "P7", "P3", "PZ", "P4", "P8", "O1", "O2",
        ]
        return raw, {
            "adapter": "EEGPT native TUAB [B,23,2000] with internal 23->20 channel convolution",
            "eegpt_input_channel_names": list(channel_names),
            "eegpt_use_channels_names": use_channels_names,
        }
    if model_name == "BIOT":
        x16, indices, names = select_tuab_16(raw, channel_names)
        return x16, {"adapter": "BIOT [B,16,2000]", "selected_channel_indices": indices, "selected_channel_names": names}
    if model_name in {"CBraMod", "CSBrain", "CodeBrain"}:
        x16, indices, names = select_tuab_16(raw, channel_names)
        return to_patches(x16, 200), {
            "adapter": f"{model_name} [B,16,10,200]",
            "selected_channel_indices": indices,
            "selected_channel_names": names,
        }
    raise ValueError(f"Unsupported model: {model_name}")


def build_model(model_name: str, repo_path: str, adapted_train: torch.Tensor, adapter_meta: dict) -> nn.Module:
    if model_name == "LaBraM":
        return LaBraMWrapper(repo_path)
    if model_name == "EEGPT":
        return EEGPTWrapper(repo_path, adapter_meta["eegpt_input_channel_names"], tuple(adapted_train.shape[1:]))
    if model_name == "BIOT":
        sys.path.insert(0, repo_path)
        from model.biot import BIOTClassifier

        return BIOTClassifier(n_classes=2, n_channels=16, n_fft=200, hop_length=200, depth=4, heads=8)
    if model_name == "CBraMod":
        sys.path.insert(0, repo_path)
        from models.model_for_tuab import Model

        param = SimpleNamespace(use_pretrained_weights=False, classifier="avgpooling_patch_reps", dropout=0.1, cuda=0, foundation_dir="")
        return Model(param)
    if model_name == "CSBrain":
        sys.path.insert(0, repo_path)
        from models.model_for_tuab import Model

        param = SimpleNamespace(model="CSBrain", use_pretrained_weights=False, dropout=0.1, cuda=0, foundation_dir="")
        return Model(param)
    if model_name == "CodeBrain":
        return CodeBrainWrapper(repo_path)
    raise ValueError(f"Unsupported model: {model_name}")


def default_pretrained_checkpoint(model_name: str, repo_path: str) -> Path | None:
    rel = DEFAULT_PRETRAINED_CHECKPOINTS.get(model_name)
    if not rel:
        return None
    return Path(repo_path) / rel


def _safe_torch_load(path: Path, map_location: str = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _state_dict_from_checkpoint(obj):
    if isinstance(obj, dict):
        for key in ["state_dict", "model", "model_state_dict", "module"]:
            value = obj.get(key)
            if isinstance(value, dict):
                return value
    return obj if isinstance(obj, dict) else {}


def load_pretrained_checkpoint_if_available(model: nn.Module, model_name: str, repo_path: str, checkpoint_override: str = "") -> dict:
    """Adapter-side checkpoint loading audit.

    This loads only when the current lightweight adapter architecture is known
    to match the checkpoint safely. It reports explicit BLOCKED/PARTIAL states
    instead of silently pretending strict pretrained initialization happened.
    """

    checkpoint_path = Path(checkpoint_override) if checkpoint_override else default_pretrained_checkpoint(model_name, repo_path)
    report = {
        "model": model_name,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else "",
        "checkpoint_exists": bool(checkpoint_path and checkpoint_path.exists()),
        "checkpoint_loaded": False,
        "checkpoint_status": "NO_DEFAULT_CHECKPOINT",
        "loaded_key_count": 0,
        "model_key_count": len(model.state_dict()),
        "missing_key_count": None,
        "unexpected_key_count": None,
        "notes": [],
    }
    if checkpoint_path is None:
        return report
    if not checkpoint_path.exists():
        report["checkpoint_status"] = "MISSING"
        report["notes"].append("Expected pretrained checkpoint path is absent on AI Station.")
        return report

    if model_name == "BIOT" and hasattr(model, "biot"):
        state = _safe_torch_load(checkpoint_path, map_location="cpu")
        load_result = model.biot.load_state_dict(state, strict=True)
        report.update(
            {
                "checkpoint_loaded": True,
                "checkpoint_status": "LOADED_STRICT",
                "loaded_key_count": len(state) if isinstance(state, dict) else 0,
                "missing_key_count": len(getattr(load_result, "missing_keys", [])),
                "unexpected_key_count": len(getattr(load_result, "unexpected_keys", [])),
            }
        )
        return report

    if model_name == "LaBraM" and hasattr(model, "model"):
        checkpoint = _safe_torch_load(checkpoint_path, map_location="cpu")
        state = _state_dict_from_checkpoint(checkpoint)
        mapped = {}
        skipped_shape = []
        skipped_not_in_model = []
        model_state = model.state_dict()
        for key, value in state.items():
            if key.startswith("student."):
                mapped_key = "model." + key[len("student.") :]
            elif key.startswith("model."):
                mapped_key = key
            else:
                continue
            if mapped_key in model_state and hasattr(value, "shape") and tuple(model_state[mapped_key].shape) == tuple(value.shape):
                mapped[mapped_key] = value
            elif mapped_key in model_state and hasattr(value, "shape"):
                skipped_shape.append(
                    {
                        "checkpoint_key": key,
                        "model_key": mapped_key,
                        "checkpoint_shape": list(value.shape),
                        "model_shape": list(model_state[mapped_key].shape),
                    }
                )
            else:
                skipped_not_in_model.append(key)
        load_result = model.load_state_dict(mapped, strict=False)
        missing = list(getattr(load_result, "missing_keys", []))
        unexpected = list(getattr(load_result, "unexpected_keys", []))
        backbone_missing = [name for name in missing if name.startswith("model.") and not name.startswith("model.head.")]
        report.update(
            {
                "checkpoint_loaded": len(mapped) > 0,
                "checkpoint_status": "LOADED_BACKBONE_FILTERED_HEAD_INITIALIZED" if len(mapped) > 0 else "PRESENT_BACKBONE_LOAD_FAILED",
                "loaded_key_count": len(mapped),
                "model_key_count": len(model_state),
                "missing_key_count": len(missing),
                "unexpected_key_count": len(unexpected),
                "missing_keys": missing,
                "unexpected_keys": unexpected,
                "backbone_missing_key_count": len(backbone_missing),
                "backbone_missing_keys": backbone_missing[:200],
                "skipped_shape_mismatch": skipped_shape[:80],
                "skipped_not_in_model_count": len(skipped_not_in_model),
                "skipped_not_in_model_sample": skipped_not_in_model[:80],
                "checkpoint_top_level_keys": list(checkpoint.keys())[:30] if isinstance(checkpoint, dict) else [],
                "inferred_checkpoint_prefix": "student.",
            }
        )
        report["notes"].append("LaBraM checkpoint loaded by mapping student.* keys to adapter model.* keys; task head is initialized separately.")
        return report

    if model_name == "EEGPT" and hasattr(model, "model") and hasattr(model.model, "target_encoder"):
        checkpoint = _safe_torch_load(checkpoint_path, map_location="cpu")
        state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else {}
        mapped = {}
        skipped_shape = []
        model_state = model.state_dict()
        for key, value in state.items():
            if not key.startswith("encoder."):
                continue
            mapped_key = "model.target_encoder." + key[len("encoder.") :]
            if mapped_key in model_state and hasattr(value, "shape") and tuple(model_state[mapped_key].shape) == tuple(value.shape):
                mapped[mapped_key] = value
            elif mapped_key in model_state:
                skipped_shape.append({"checkpoint_key": key, "model_key": mapped_key, "checkpoint_shape": list(value.shape), "model_shape": list(model_state[mapped_key].shape)})
        load_result = model.load_state_dict(mapped, strict=False)
        missing = list(getattr(load_result, "missing_keys", []))
        unexpected = list(getattr(load_result, "unexpected_keys", []))
        backbone_missing = [name for name in missing if name.startswith("model.target_encoder.")]
        report.update(
            {
                "checkpoint_loaded": len(mapped) > 0 and len(backbone_missing) == 0,
                "checkpoint_status": "LOADED_BACKBONE_STRICT_HEAD_INITIALIZED" if len(mapped) > 0 and len(backbone_missing) == 0 else "PRESENT_BACKBONE_LOAD_INCOMPLETE",
                "loaded_key_count": len(mapped),
                "model_key_count": len(model_state),
                "missing_key_count": len(missing),
                "unexpected_key_count": len(unexpected),
                "missing_keys": missing,
                "unexpected_keys": unexpected,
                "backbone_missing_key_count": len(backbone_missing),
                "backbone_missing_keys": backbone_missing,
                "skipped_shape_mismatch": skipped_shape,
                "checkpoint_top_level_keys": list(checkpoint.keys())[:30] if isinstance(checkpoint, dict) else [],
                "inferred_checkpoint_prefix": "encoder.",
            }
        )
        if report["checkpoint_loaded"]:
            report["notes"].append("Native EEGPT target_encoder loaded from checkpoint encoder.* keys; TUAB chan_conv/head are task-specific initialized modules.")
        else:
            report["notes"].append("EEGPT checkpoint was present but native target_encoder did not load completely.")
        return report

    if model_name == "EEGPT":
        report["checkpoint_status"] = "PRESENT_NOT_LOADED_ARCHITECTURE_MISMATCH"
        report["notes"].append("EEGPT checkpoint exists, but model is not the native checkpoint-compatible EEGPTClassifier wrapper.")
        return report

    report["checkpoint_status"] = "PRESENT_LOAD_NOT_IMPLEMENTED"
    report["notes"].append("Checkpoint exists, but adapter-side safe loading is not implemented for this model.")
    return report


def infer_loss_and_maybe_head(model: nn.Module, sample_x: torch.Tensor) -> tuple[nn.Module, str, list[int], bool]:
    model.eval()
    with torch.no_grad():
        out = model(sample_x)
    original_shape = shape_of(out)
    if out.ndim == 2 and out.shape[1] == 2:
        return model, "CrossEntropyLoss", original_shape, False
    if out.ndim == 1 or (out.ndim == 2 and out.shape[1] == 1):
        return model, "BCEWithLogitsLoss", original_shape, False
    feature_dim = int(np.prod(out.shape[1:]))
    return FeatureHeadWrapper(model, feature_dim, n_classes=2), "CrossEntropyLoss", original_shape, True


def compute_loss(logits: torch.Tensor, labels: torch.Tensor, loss_type: str) -> torch.Tensor:
    if loss_type == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()(logits, labels.long())
    if loss_type == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss()(logits.reshape(-1), labels.float())
    raise ValueError(f"Unsupported loss type: {loss_type}")


def run_training_smoke(args: argparse.Namespace) -> dict:
    batch = load_tiny_npz(args.tiny_npz)
    channel_names = batch["channel_names"]
    train_x, train_meta = adapt_inputs(args.model, batch["train_x"], channel_names, args.repo_path)
    val_x, _ = adapt_inputs(args.model, batch["val_x"], channel_names, args.repo_path)

    device = torch.device(args.device)
    model = build_model(args.model, args.repo_path, train_x, train_meta).to(device)
    sample_x = train_x[: min(args.batch_size, len(train_x))].to(device)
    model, loss_type, raw_output_shape, used_temp_head = infer_loss_and_maybe_head(model, sample_x)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    loader = DataLoader(TensorDataset(train_x, batch["train_y"]), batch_size=args.batch_size, shuffle=False)
    initial_loss = None
    backward_status = "NOT_RUN"
    optimizer_step_status = "NOT_RUN"
    output_shape = None
    finite = False

    model.train()
    for step, (x, y) in enumerate(loader):
        if step >= args.max_train_batches:
            break
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        output_shape = shape_of(logits)
        loss = compute_loss(logits, y, loss_type)
        if initial_loss is None:
            initial_loss = float(loss.detach().cpu())
            finite = bool(math.isfinite(initial_loss))
        loss.backward()
        backward_status = "PASS"
        optimizer.step()
        optimizer_step_status = "PASS"

    model.eval()
    with torch.no_grad():
        vx = val_x[: min(args.batch_size, len(val_x))].to(device)
        vy = batch["val_y"][: min(args.batch_size, len(val_x))].to(device)
        val_logits = model(vx)
        val_loss = compute_loss(val_logits, vy, loss_type)

    status = "PASS" if finite and backward_status == "PASS" and optimizer_step_status == "PASS" else "FAIL"
    notes = []
    if used_temp_head:
        notes.append("Adapter-side temporary linear classification head used because model forward returned feature-shaped output.")
    if args.model == "CodeBrain":
        notes.append("Used adapter-side flattened TUAB forward; original CodeBrain TUAB forward path is not used.")

    return {
        "model": args.model,
        "venv_path": args.venv_path,
        "repo_path": args.repo_path,
        "python_path": sys.executable,
        "python_version": platform.python_version(),
        "train_subset_count": int(len(train_x)),
        "val_subset_count": int(len(val_x)),
        "test_subset_count": int(len(batch["test_x"])),
        "input_raw_shape": shape_of(batch["train_x"]),
        "adapted_input_shape": shape_of(train_x),
        "raw_model_output_shape_before_temp_head": raw_output_shape,
        "output_shape": output_shape,
        "loss_type": loss_type,
        "initial_train_loss": initial_loss,
        "loss_is_finite": finite,
        "backward_status": backward_status,
        "optimizer_step_status": optimizer_step_status,
        "validation_forward_status": "PASS",
        "validation_output_shape": shape_of(val_logits),
        "validation_loss": float(val_loss.detach().cpu()),
        "temporary_head_used": used_temp_head,
        "adapter_meta": train_meta,
        "status": status,
        "notes": " ".join(notes),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--repo_path", required=True)
    parser.add_argument("--venv_path", default="")
    parser.add_argument("--tiny_npz", required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_train_batches", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    result = {
        "model": args.model,
        "venv_path": args.venv_path,
        "repo_path": args.repo_path,
        "python_path": sys.executable,
        "status": "FAIL",
    }
    try:
        result.update(run_training_smoke(args))
    except Exception as exc:
        result.update(
            {
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
