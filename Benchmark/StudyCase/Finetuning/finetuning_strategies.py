#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LoRA helpers for Benchmark EEG-FM study cases.

This is helper code, not a connector or runner. Training and inference scripts
build an EEG-FM model first, then call this file only when LoRA is requested.
Full fine-tuning stays in the original EEG-FM training scripts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.utils import parametrize


LORA_TARGET = "lora_module"
LORA_EXCLUDE_TOKENS = ("classifier", "head", "lm_head", "prediction")
DEFAULT_LORA_SETTINGS = {
    "rank": 2,
    "alpha": 8.0,
    "layers": "all",
    "target": LORA_TARGET,
    "init_scale": 0.01,
}


def _init_lora_a_(param: torch.Tensor, init_scale: float | None = None) -> None:
    if init_scale is not None:
        with torch.no_grad():
            param.copy_(torch.randn_like(param) * float(init_scale))
    else:
        nn.init.kaiming_uniform_(param, a=5 ** 0.5)


class LoRALinear(nn.Module):
    """Adapter-side LoRA wrapper for nn.Linear without editing model repos."""

    def __init__(self, base: nn.Linear, rank: int = 2, alpha: float = 8.0, init_scale: float | None = None):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        device = base.weight.device
        dtype = base.weight.dtype
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        # LoRA layers may be inserted after model.to(device), so match the base layer.
        self.lora_a = nn.Linear(base.in_features, self.rank, bias=False, device=device, dtype=dtype)
        self.lora_b = nn.Linear(self.rank, base.out_features, bias=False, device=device, dtype=dtype)
        _init_lora_a_(self.lora_a.weight, init_scale=init_scale)
        nn.init.zeros_(self.lora_b.weight)
        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(x)) * self.scaling


class LoRAWeightParametrization(nn.Module):
    """LoRA parametrization for weights that must remain readable as `.weight`."""

    def __init__(self, weight: torch.Tensor, rank: int = 2, alpha: float = 8.0, init_scale: float | None = None):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        out_features = int(weight.shape[0])
        in_features = int(weight.numel() // out_features)
        self.lora_a = nn.Parameter(torch.empty(self.rank, in_features, device=weight.device, dtype=weight.dtype))
        self.lora_b = nn.Parameter(torch.zeros(out_features, self.rank, device=weight.device, dtype=weight.dtype))
        _init_lora_a_(self.lora_a, init_scale=init_scale)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        update = (self.lora_b @ self.lora_a).view_as(weight) * self.scaling
        return weight + update


class LoRAConv2dParametrization(LoRAWeightParametrization):
    """Named subclass for Conv2d LoRA reporting."""


def _can_require_grad(param: torch.Tensor) -> bool:
    return bool(param.is_floating_point() or param.is_complex())


def _set_all_trainable(model: nn.Module, trainable: bool) -> list[str]:
    skipped: list[str] = []
    for name, param in model.named_parameters():
        if trainable and _can_require_grad(param):
            param.requires_grad = True
        else:
            param.requires_grad = False
            if trainable:
                skipped.append(name)
    return skipped


def _set_named_trainable(model: nn.Module, trainable_names: set[str]) -> list[str]:
    skipped: list[str] = []
    for name, param in model.named_parameters():
        if name not in trainable_names:
            continue
        if _can_require_grad(param):
            param.requires_grad = True
        else:
            param.requires_grad = False
            skipped.append(name)
    return skipped


def _param_counts(model: nn.Module) -> dict[str, Any]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "frozen_params": int(total - trainable),
        "trainable_percent": float((trainable / total) * 100.0) if total else 0.0,
    }


def _trainable_param_examples(model: nn.Module, limit: int = 30) -> list[str]:
    return [name for name, param in model.named_parameters() if param.requires_grad][:limit]


def _param_names_with_prefix(model: nn.Module, prefixes: tuple[str, ...]) -> list[str]:
    names: list[str] = []
    for name, _ in model.named_parameters():
        if any(name == prefix.rstrip(".") or name.startswith(prefix) for prefix in prefixes):
            names.append(name)
    return names


def _safe_head_name(name: str) -> bool:
    lowered = name.lower()
    parts = lowered.split(".")
    if lowered.startswith("head."):
        return True
    return bool({"classifier", "clshead", "classification_head", "out_proj_head"} & set(parts))


def find_head_parameter_names(model: nn.Module, allow_head_guess: bool = False, model_name: str = "") -> list[str]:
    """Return downstream classifier/head parameters."""

    if model_name == "LaBraM":
        names = _param_names_with_prefix(model, ("model.head.", "head."))
    elif model_name == "EEGPT":
        names = _param_names_with_prefix(model, ("model.head.", "head."))
    elif model_name == "BIOT":
        names = _param_names_with_prefix(model, ("classifier.clshead.", "classifier."))
    elif model_name in {"CBraMod", "CSBrain"}:
        names = _param_names_with_prefix(model, ("classifier.",))
    elif model_name == "CodeBrain":
        names = _param_names_with_prefix(model, ("model.classifier.", "classifier."))
    else:
        names = [name for name, _ in model.named_parameters() if _safe_head_name(name)]
    if names:
        return names

    if not allow_head_guess:
        return []

    linear_modules = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    if not linear_modules:
        return []
    last_name = linear_modules[-1][0]
    prefix = f"{last_name}."
    return [name for name, _ in model.named_parameters() if name.startswith(prefix)]


def _replace_module(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    parent = root
    parts = dotted_name.split(".")
    for part in parts[:-1]:
        if part.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    leaf = parts[-1]
    if leaf.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(leaf)] = new_module
    else:
        setattr(parent, leaf, new_module)


def _is_lora_excluded(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in LORA_EXCLUDE_TOKENS)


def _is_attention_out_proj_name(name: str) -> bool:
    lowered = name.lower()
    return lowered == "out_proj" or lowered.endswith(".out_proj")


def _freeze_parametrization_original(module: nn.Module, parameter_name: str) -> None:
    plist = getattr(module, "parametrizations", None)
    if plist is None or not hasattr(plist, parameter_name):
        return
    original = getattr(getattr(plist, parameter_name), "original", None)
    if original is not None:
        original.requires_grad = False


def _register_lora_parametrization(
    module: nn.Module,
    parameter_name: str,
    rank: int,
    alpha: float,
    conv: bool = False,
    init_scale: float | None = None,
) -> None:
    weight = getattr(module, parameter_name)
    cls = LoRAConv2dParametrization if conv else LoRAWeightParametrization
    parametrize.register_parametrization(module, parameter_name, cls(weight.detach(), rank=rank, alpha=alpha, init_scale=init_scale))
    _freeze_parametrization_original(module, parameter_name)


def _leaf_name(name: str) -> str:
    return name.rsplit(".", 1)[-1].lower()


def _is_qkv_linear_name(name: str) -> bool:
    leaf = _leaf_name(name)
    return leaf == "qkv" or leaf in {"to_q", "to_k", "to_v", "query", "key", "value"}


def _is_mlp_linear_name(name: str) -> bool:
    lowered = name.lower()
    leaf = _leaf_name(lowered)
    return (
        leaf in {"fc1", "fc2", "linear1", "linear2", "w1", "w2"}
        or lowered.endswith(".fn.fn.w1")
        or lowered.endswith(".fn.fn.w2")
    )


def _is_strict_temporal_conv_name(model_name: str, name: str) -> bool:
    lowered = name.lower()
    if model_name == "LaBraM":
        return any(lowered.endswith(f"patch_embed.{conv}") for conv in ("conv1", "conv2", "conv3"))
    return "temembed" in lowered or "temporal" in lowered


def _apply_lora_module_policy(
    model: nn.Module,
    model_name: str,
    rank: int,
    alpha: float,
    init_scale: float,
) -> tuple[list[str], list[str], dict[str, Any]]:
    notes: list[str] = []
    target_names: list[str] = []
    strict_summary: dict[str, Any] = {
        "strict_lora_supported": "partial",
        "qkv_or_qkv_equivalent_adapted": False,
        "mlp_adapted": False,
        "temporal_conv2d_adapted": False,
        "temporal_conv2d_status": "no equivalent",
        "attention_output_projection_adapted": False,
        "classifier_head_trainable": False,
        "non_equivalent_targets_used": [],
        "remaining_deviations": [],
        "qkv_mlp_rank": int(rank),
        "conv_rank": 4,
        "alpha": float(alpha),
        "lora_a_init": "torch.randn(...)*0.01",
        "lora_b_init": "zeros",
        "bias_lora": False,
        "model_specific_function": f"lora_{model_name.lower()}",
    }

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.MultiheadAttention) and getattr(module, "in_proj_weight", None) is not None:
            _register_lora_parametrization(module, "in_proj_weight", rank=rank, alpha=alpha, init_scale=init_scale)
            target_names.append(f"{name}.in_proj_weight")
            strict_summary["qkv_or_qkv_equivalent_adapted"] = True

    for name, module in list(model.named_modules()):
        if not name:
            continue
        lowered = name.lower()
        if model_name == "EEGPT" and ".reconstructor" in lowered:
            continue
        if isinstance(module, nn.Linear):
            if _is_lora_excluded(name) or "spectral_proj" in lowered or "patch_embedding.projection" in lowered:
                continue
            if _is_attention_out_proj_name(name) or lowered.endswith(".attn.proj") or lowered.endswith(".attention.out_proj"):
                continue
            if _is_qkv_linear_name(name):
                # Parametrization preserves qkv.weight for repos that use F.linear(..., qkv.weight).
                _register_lora_parametrization(module, "weight", rank=rank, alpha=alpha, init_scale=init_scale)
                target_names.append(f"{name}.weight")
                strict_summary["qkv_or_qkv_equivalent_adapted"] = True
            elif _is_mlp_linear_name(name):
                _replace_module(model, name, LoRALinear(module, rank=rank, alpha=alpha, init_scale=init_scale))
                target_names.append(name)
                strict_summary["mlp_adapted"] = True
        elif isinstance(module, nn.Conv2d) and _is_strict_temporal_conv_name(model_name, name):
            _register_lora_parametrization(module, "weight", rank=4, alpha=alpha, conv=True, init_scale=init_scale)
            target_names.append(f"{name}.weight")
            strict_summary["temporal_conv2d_adapted"] = True
            strict_summary["temporal_conv2d_status"] = "adapted"

    if not strict_summary["qkv_or_qkv_equivalent_adapted"]:
        strict_summary["remaining_deviations"].append("No safe qkv/q/k/v equivalent target was identified.")
    if not strict_summary["mlp_adapted"]:
        strict_summary["remaining_deviations"].append("No MLP/feed-forward fc1/fc2 or linear1/linear2 equivalent target was identified.")
    if not strict_summary["temporal_conv2d_adapted"]:
        notes.append("No true temporal Conv2d stem equivalent was identified for LoRA.")

    if strict_summary["qkv_or_qkv_equivalent_adapted"] and strict_summary["mlp_adapted"]:
        strict_summary["strict_lora_supported"] = "yes"
    if not target_names:
        strict_summary["strict_lora_supported"] = "no"
        notes.append("No LoRA targets were found.")
    return target_names, notes, strict_summary


def lora_labram(model: nn.Module, rank: int = 2, alpha: float = 8.0, init_scale: float = 0.01):
    """LaBraM LoRA: qkv, fc1/fc2, temporal conv1/2/3; no attention output."""
    return _apply_lora_module_policy(model, "LaBraM", rank, alpha, init_scale)


def lora_eegpt(model: nn.Module, rank: int = 2, alpha: float = 8.0, init_scale: float = 0.01):
    """EEGPT LoRA: qkv and MLP equivalents; reconstruction modules excluded."""
    return _apply_lora_module_policy(model, "EEGPT", rank, alpha, init_scale)


def lora_biot(model: nn.Module, rank: int = 2, alpha: float = 8.0, init_scale: float = 0.01):
    """BIOT LoRA: q/k/v plus feed-forward w1/w2 equivalents."""
    return _apply_lora_module_policy(model, "BIOT", rank, alpha, init_scale)


def lora_cbramod(model: nn.Module, rank: int = 2, alpha: float = 8.0, init_scale: float = 0.01):
    """CBraMod LoRA: MHA in-projection plus linear1/linear2; out_proj skipped."""
    return _apply_lora_module_policy(model, "CBraMod", rank, alpha, init_scale)


def lora_csbrain(model: nn.Module, rank: int = 2, alpha: float = 8.0, init_scale: float = 0.01):
    """CSBrain LoRA: MHA in-projection, linear1/linear2, temporal conv equivalents."""
    return _apply_lora_module_policy(model, "CSBrain", rank, alpha, init_scale)


def lora_codebrain(model: nn.Module, rank: int = 2, alpha: float = 8.0, init_scale: float = 0.01):
    """CodeBrain LoRA: qkv equivalent where available; MLP equivalent may be partial."""
    return _apply_lora_module_policy(model, "CodeBrain", rank, alpha, init_scale)


LORA_BY_MODEL = {
    "LaBraM": lora_labram,
    "EEGPT": lora_eegpt,
    "BIOT": lora_biot,
    "CBraMod": lora_cbramod,
    "CSBrain": lora_csbrain,
    "CodeBrain": lora_codebrain,
}


def _resolve_lora_settings(settings: dict[str, Any] | None) -> dict[str, Any]:
    raw = settings or {}
    resolved = {**DEFAULT_LORA_SETTINGS, **raw}
    target = raw.get("target", raw.get("target_modules", raw.get("lora_target", DEFAULT_LORA_SETTINGS["target"])))
    resolved["target"] = str(target or DEFAULT_LORA_SETTINGS["target"])
    resolved["rank"] = int(resolved["rank"])
    resolved["alpha"] = float(resolved["alpha"])
    resolved["layers"] = str(resolved.get("layers", "all"))
    resolved["init_scale"] = float(resolved["init_scale"]) if resolved.get("init_scale") is not None else None
    if resolved["rank"] <= 0:
        raise ValueError("LoRA rank must be positive")
    if resolved["layers"] != "all":
        raise ValueError("LoRA layers currently supports only 'all'.")
    if resolved["target"] != LORA_TARGET:
        raise ValueError(f"Unsupported LoRA target {resolved['target']!r}; use {LORA_TARGET!r}.")
    return resolved


def _report_dir(output_dir: str | Path = "", strategy_report_path: str | Path = "") -> Path | None:
    if strategy_report_path:
        path = Path(strategy_report_path)
    elif output_dir:
        path = Path(output_dir) / "strategy"
    else:
        return None
    if path.suffix:
        path = path.parent
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_lora_reports(summary: dict[str, Any], output_dir: str | Path = "", strategy_report_path: str | Path = "") -> None:
    path = _report_dir(output_dir=output_dir, strategy_report_path=strategy_report_path)
    if path is None:
        return
    (path / "trainable_params.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"model: {summary['model_name']}",
        "strategy: lora",
        f"total_params: {summary['total_params']}",
        f"trainable_params: {summary['trainable_params']}",
        f"trainable_percent: {summary['trainable_percent']:.6f}",
        "trainable_examples:",
    ]
    lines.extend(f"- {name}" for name in summary.get("trainable_param_examples", []))
    if summary.get("head_param_names"):
        lines.append("head_param_names:")
        lines.extend(f"- {name}" for name in summary["head_param_names"])
    if summary.get("lora_target_names"):
        lines.append("lora_target_names:")
        lines.extend(f"- {name}" for name in summary["lora_target_names"])
    if summary.get("notes"):
        lines.append("notes:")
        lines.extend(f"- {note}" for note in summary["notes"])
    (path / "trainable_params.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {
        "model_name": summary["model_name"],
        "lora_rank": summary.get("lora_rank"),
        "lora_alpha": summary.get("lora_alpha"),
        "lora_target": summary.get("lora_target"),
        "lora_target_names": summary.get("lora_target_names", []),
        "notes": summary.get("notes", []),
    }
    (path / "lora_targets.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def apply_lora_strategy(
    model: nn.Module,
    model_name: str,
    lora_settings: dict[str, Any] | None = None,
    output_dir: str | Path = "",
    strategy_report_path: str | Path = "",
    allow_head_guess: bool = False,
) -> dict[str, Any]:
    """Apply model-specific LoRA in-place and return a JSON-safe summary."""

    settings = _resolve_lora_settings(lora_settings)
    if model_name not in LORA_BY_MODEL:
        raise ValueError(f"No model-specific LoRA policy is defined for {model_name!r}")

    _set_all_trainable(model, False)
    target_names, notes, strict_summary = LORA_BY_MODEL[model_name](
        model,
        rank=settings["rank"],
        alpha=settings["alpha"],
        init_scale=settings["init_scale"],
    )
    if not target_names:
        raise RuntimeError(f"No LoRA targets found for {model_name}.")

    head_names = find_head_parameter_names(model, allow_head_guess=allow_head_guess, model_name=model_name)
    skipped = _set_named_trainable(model, set(head_names))
    if not head_names:
        notes.append("No classifier/head parameters were found; LoRA adapters are trainable but head is unchanged.")
    if skipped:
        notes.append("non-floating parameters cannot require gradients and were skipped")

    strict_summary["classifier_head_trainable"] = bool(head_names)
    reference_match = (
        strict_summary.get("qkv_or_qkv_equivalent_adapted") is True
        and strict_summary.get("mlp_adapted") is True
        and strict_summary.get("temporal_conv2d_adapted") is True
        and strict_summary.get("attention_output_projection_adapted") is False
        and bool(head_names)
        and strict_summary.get("qkv_mlp_rank") == 2
        and strict_summary.get("conv_rank") == 4
        and float(strict_summary.get("alpha", 0.0)) == 8.0
        and strict_summary.get("lora_a_init") == "torch.randn(...)*0.01"
        and strict_summary.get("lora_b_init") == "zeros"
        and strict_summary.get("bias_lora") is False
    )
    strict_summary["meriem_reference_status"] = "REFERENCE_STRATEGY_MATCH" if reference_match else "REFERENCE_STRATEGY_MISMATCH"

    summary: dict[str, Any] = {
        "model_name": model_name,
        "strategy": "lora",
        "head_param_names": head_names,
        "lora_target_names": target_names,
        "skipped_non_float_param_names": skipped,
        "skipped_non_float_param_count": len(skipped),
        "lora_rank": settings["rank"],
        "lora_alpha": settings["alpha"],
        "lora_target": settings["target"],
        "strict_lora_summary": strict_summary,
        "notes": notes,
    }
    summary.update(_param_counts(model))
    summary["trainable_param_examples"] = _trainable_param_examples(model)
    if summary["trainable_params"] <= 0:
        raise RuntimeError(f"LoRA produced zero trainable parameters for {model_name}.")

    _write_lora_reports(summary, output_dir=output_dir, strategy_report_path=strategy_report_path)
    print(
        f"[lora_strategy] {model_name}: "
        f"{summary['trainable_params']}/{summary['total_params']} trainable "
        f"({summary['trainable_percent']:.4f}%)"
    )
    return summary
