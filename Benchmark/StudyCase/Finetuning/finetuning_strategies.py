#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fine-tuning strategy helpers for Benchmark EEG-FM study cases."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
from torch.nn.utils import parametrize


STRATEGIES = {"full_finetune", "freeze_backbone", "linear_probe", "lora"}
STRATEGY_ALIASES = {"freeze_backbone": "linear_probe"}
HEAD_TOKENS = ("classifier", "head", "fc", "prediction", "out_proj_head")
MERIEM_EXACT_LORA_TARGETS = {"meriem_exact", "labram_meriem_exact", "meriem_reference_exact"}
STRICT_LORA_TARGETS = {"lora_module", "labram_reference", "strict_labram_reference"} | MERIEM_EXACT_LORA_TARGETS
LORA_AUTO_TOKENS = (
    "qkv",
    "query",
    "key",
    "value",
    "attn",
    "attention",
    "proj",
    "mlp",
    "ffn",
    "linear",
)
LORA_EXCLUDE_TOKENS = ("classifier", "head", "lm_head", "prediction")
DEFAULT_LORA_SETTINGS = {
    "rank": 2,
    "alpha": 8.0,
    "target": "lora_module",
}


def _init_lora_a_(param: torch.Tensor, init_scale: float | None = None) -> None:
    if init_scale is not None:
        with torch.no_grad():
            param.copy_(torch.randn_like(param) * float(init_scale))
    else:
        nn.init.kaiming_uniform_(param, a=5 ** 0.5)


class LoRALinear(nn.Module):
    """Small adapter-side LoRA wrapper for nn.Linear without editing model repos."""

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
        # LoRA modules are inserted after model.to(device), so match the wrapped layer.
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
        self.weight_shape = tuple(weight.shape)
        out_features = int(weight.shape[0])
        in_features = int(weight.numel() // out_features)
        self.lora_a = nn.Parameter(torch.empty(self.rank, in_features, device=weight.device, dtype=weight.dtype))
        self.lora_b = nn.Parameter(torch.zeros(out_features, self.rank, device=weight.device, dtype=weight.dtype))
        _init_lora_a_(self.lora_a, init_scale=init_scale)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        update = (self.lora_b @ self.lora_a).view_as(weight) * self.scaling
        return weight + update


class LoRAConv2dParametrization(LoRAWeightParametrization):
    """Named subclass for strict-reference Conv2d LoRA reporting."""


def _get_arg(args: Any, name: str, default: Any = None) -> Any:
    return getattr(args, name, default) if args is not None else default


def _set_arg(args: Any, name: str, value: Any) -> None:
    if args is not None and hasattr(args, name):
        setattr(args, name, value)


def _can_require_grad(param: torch.Tensor) -> bool:
    return bool(param.is_floating_point() or param.is_complex())


def _set_all_trainable(model: nn.Module, trainable: bool) -> list[str]:
    skipped: list[str] = []
    for name, param in model.named_parameters():
        if trainable:
            if _can_require_grad(param):
                param.requires_grad = True
            else:
                param.requires_grad = False
                skipped.append(name)
        else:
            param.requires_grad = False
    return skipped


def _set_named_trainable(model: nn.Module, trainable_names: set[str]) -> list[str]:
    skipped: list[str] = []
    for name, param in model.named_parameters():
        if name in trainable_names:
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


def _is_legacy_head_name(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in HEAD_TOKENS)


def _is_lora_excluded(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in LORA_EXCLUDE_TOKENS)


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
    """Return true downstream classifier/head parameters.

    This intentionally avoids broad substrings like ``fc`` and unscoped
    ``head`` because they catch backbone MLP/global_fc/lm_head parameters.
    """

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


def _get_parent_module(root: nn.Module, dotted_name: str) -> nn.Module | None:
    if "." not in dotted_name:
        return root
    parent = root
    for part in dotted_name.split(".")[:-1]:
        try:
            if part.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)
        except Exception:
            return None
    return parent


def _is_attention_out_proj_name(name: str) -> bool:
    lowered = name.lower()
    return lowered == "out_proj" or lowered.endswith(".out_proj")


def _filter_attention_out_proj_lora_names(model: nn.Module, names: list[str]) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    skipped = 0
    for name in names:
        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        parent = _get_parent_module(model, name)
        parent_suggests_attention = "attention" in parent_name.lower() or "attn" in parent_name.lower()
        is_mha_out_proj = isinstance(parent, nn.MultiheadAttention) and _is_attention_out_proj_name(name)
        if _is_attention_out_proj_name(name) and (is_mha_out_proj or parent_suggests_attention):
            skipped += 1
            continue
        kept.append(name)
    if skipped:
        return kept, [
            "Skipped MultiheadAttention out_proj LoRA because PyTorch MultiheadAttention accesses out_proj.weight directly."
        ]
    return kept, []


def _is_labram_qkv_name(name: str) -> bool:
    lowered = name.lower()
    return lowered.endswith(".qkv") or ".qkv" in lowered or lowered == "qkv"


def _filter_model_specific_lora_names(names: list[str], model_name: str) -> tuple[list[str], list[str]]:
    notes: list[str] = []
    if model_name == "LaBraM":
        kept = [name for name in names if not _is_labram_qkv_name(name)]
        if len(kept) != len(names):
            notes.append("Skipped LaBraM qkv LoRA because LaBraM accesses qkv.weight directly.")
        return kept, notes
    return names, notes


def _candidate_lora_module_names(model: nn.Module, model_name: str, target: str = "auto") -> tuple[list[str], list[str]]:
    notes: list[str] = []
    linear_names = [
        name
        for name, module in model.named_modules()
        if name and isinstance(module, nn.Linear) and not _is_lora_excluded(name)
    ]
    linear_names, out_proj_notes = _filter_attention_out_proj_lora_names(model, linear_names)
    notes.extend(out_proj_notes)
    linear_names, model_notes = _filter_model_specific_lora_names(linear_names, model_name)
    notes.extend(model_notes)
    if not linear_names:
        return [], ["No non-head nn.Linear modules were found for LoRA."]

    target = (target or "auto").strip()
    if target in {"auto", "legacy_auto", ""}:
        names = [name for name in linear_names if any(token in name.lower() for token in LORA_AUTO_TOKENS)]
        if not names:
            names = linear_names
            notes.append("Auto target matching found no token-specific modules; using all non-head Linear modules.")
        return names, notes

    if target in {"all", "all_linear", "*"}:
        return linear_names, notes

    requested = [item.strip() for item in target.split(",") if item.strip()]
    names = [name for name in linear_names if any(item in name for item in requested)]
    missing = [item for item in requested if not any(item in name for name in linear_names)]
    if missing:
        notes.append(f"Requested LoRA target substrings not found: {missing}")
    return names, notes


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


def _apply_strict_labram_reference_lora(
    model: nn.Module,
    model_name: str,
    rank: int,
    alpha: float,
    meriem_exact: bool = False,
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
        "meriem_exact": bool(meriem_exact),
        "qkv_mlp_rank": int(rank),
        "conv_rank": 4 if meriem_exact else int(max(rank, 2)),
        "alpha": float(alpha),
        "lora_a_init": "torch.randn(...)*0.01" if meriem_exact else "kaiming_uniform",
        "lora_b_init": "zeros",
        "bias_lora": False,
    }
    init_scale = 0.01 if meriem_exact else None
    conv_rank = 4 if meriem_exact else max(rank, 2)

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
            _register_lora_parametrization(module, "weight", rank=conv_rank, alpha=alpha, conv=True, init_scale=init_scale)
            target_names.append(f"{name}.weight")
            strict_summary["temporal_conv2d_adapted"] = True
            strict_summary["temporal_conv2d_status"] = "adapted"

    if not strict_summary["qkv_or_qkv_equivalent_adapted"]:
        strict_summary["remaining_deviations"].append("No safe qkv/q/k/v equivalent target was identified.")
    if not strict_summary["mlp_adapted"]:
        strict_summary["remaining_deviations"].append("No MLP/feed-forward fc1/fc2 or linear1/linear2 equivalent target was identified.")
    if not strict_summary["temporal_conv2d_adapted"]:
        notes.append("No true temporal Conv2d stem equivalent was identified for strict LoRA.")

    if strict_summary["qkv_or_qkv_equivalent_adapted"] and strict_summary["mlp_adapted"]:
        strict_summary["strict_lora_supported"] = "yes"
    if not target_names:
        strict_summary["strict_lora_supported"] = "no"
        notes.append("No strict LaBraM-reference LoRA targets were found.")
    return target_names, notes, strict_summary


def _report_dir(args: Any) -> Path | None:
    raw = _get_arg(args, "strategy_report_path", "")
    if raw:
        path = Path(raw)
    else:
        output_dir = _get_arg(args, "output_dir", "")
        if not output_dir:
            return None
        path = Path(output_dir) / "strategy"
        _set_arg(args, "strategy_report_path", str(path))
    if path.suffix:
        path = path.parent
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_strategy_reports(summary: dict[str, Any], args: Any) -> None:
    path = _report_dir(args)
    if path is None:
        return
    (path / "trainable_params.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"model: {summary['model_name']}",
        f"strategy: {summary['strategy']}",
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
    if summary["strategy"] == "lora":
        payload = {
            "model_name": summary["model_name"],
            "lora_rank": summary.get("lora_rank"),
            "lora_alpha": summary.get("lora_alpha"),
            "lora_target": summary.get("lora_target"),
            "lora_target_names": summary.get("lora_target_names", []),
            "notes": summary.get("notes", []),
        }
        (path / "lora_targets.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def apply_finetuning_strategy(model: nn.Module, config: dict[str, Any], model_name: str) -> dict[str, Any]:
    """Benchmark config entry point; apply policy before optimizer creation."""
    fine_tuning = config.get("fine_tuning", {})
    strategy = str(fine_tuning.get("strategy", "full_finetune"))
    lora_settings = {**DEFAULT_LORA_SETTINGS, **(fine_tuning.get("lora") or {})}
    lora_target = lora_settings.get(
        "target",
        lora_settings.get("target_modules", lora_settings.get("lora_target", DEFAULT_LORA_SETTINGS["target"])),
    )
    paths = config.get("paths", {})
    args = SimpleNamespace(
        output_dir=str(paths.get("output", "")),
        strategy_report_path=str(fine_tuning.get("strategy_report_path", "")),
        lora_rank=int(lora_settings.get("rank", DEFAULT_LORA_SETTINGS["rank"])),
        lora_alpha=float(lora_settings.get("alpha", DEFAULT_LORA_SETTINGS["alpha"])),
        lora_target=str(lora_target),
        allow_head_guess=bool(fine_tuning.get("allow_head_guess", False)),
    )
    return apply_finetune_strategy(model, strategy, model_name, args)


def apply_finetune_strategy(model: nn.Module, strategy: str, model_name: str, args: Any = None) -> dict[str, Any]:
    """Apply a fine-tuning strategy in-place and return a JSON-safe summary."""

    if strategy not in STRATEGIES:
        raise ValueError(f"Unsupported fine-tuning strategy {strategy!r}; expected one of {sorted(STRATEGIES)}")
    strategy = STRATEGY_ALIASES.get(strategy, strategy)

    allow_head_guess = bool(_get_arg(args, "allow_head_guess", False))
    summary: dict[str, Any] = {
        "model_name": model_name,
        "strategy": strategy,
        "head_param_names": [],
        "lora_target_names": [],
        "skipped_non_float_param_names": [],
        "skipped_non_float_param_count": 0,
        "trainable_param_examples": [],
        "notes": [],
    }

    if strategy == "full_finetune":
        skipped = _set_all_trainable(model, True)
        summary["skipped_non_float_param_names"] = skipped
        summary["skipped_non_float_param_count"] = len(skipped)
        if skipped:
            summary["notes"].append("non-floating parameters cannot require gradients and were skipped")

    elif strategy == "linear_probe":
        _set_all_trainable(model, False)
        head_names = find_head_parameter_names(model, allow_head_guess=allow_head_guess, model_name=model_name)
        if not head_names:
            raise RuntimeError(
                f"Could not identify classifier/head parameters for {model_name}. "
                "Pass --allow_head_guess for last-Linear fallback or add a model-specific head rule."
            )
        head_set = set(head_names)
        skipped = _set_named_trainable(model, head_set)
        summary["head_param_names"] = head_names
        summary["skipped_non_float_param_names"] = skipped
        summary["skipped_non_float_param_count"] = len(skipped)
        if skipped:
            summary["notes"].append("non-floating parameters cannot require gradients and were skipped")

    elif strategy == "lora":
        _set_all_trainable(model, False)
        rank = int(_get_arg(args, "lora_rank", 2))
        alpha = float(_get_arg(args, "lora_alpha", 8.0))
        target = str(_get_arg(args, "lora_target", "auto"))
        strict_summary: dict[str, Any] = {}
        if target in STRICT_LORA_TARGETS:
            target_names, notes, strict_summary = _apply_strict_labram_reference_lora(
                model,
                model_name,
                rank,
                alpha,
                meriem_exact=target in MERIEM_EXACT_LORA_TARGETS,
            )
        else:
            target_names, notes = _candidate_lora_module_names(model, model_name, target)
        summary["notes"].extend(notes)
        if not target_names:
            raise RuntimeError(f"No LoRA targets found for {model_name} with target={target!r}.")
        if target not in STRICT_LORA_TARGETS:
            for name in target_names:
                module = dict(model.named_modules())[name]
                _replace_module(model, name, LoRALinear(module, rank=rank, alpha=alpha))
        head_names = find_head_parameter_names(model, allow_head_guess=allow_head_guess, model_name=model_name)
        head_set = set(head_names)
        skipped = _set_named_trainable(model, head_set)
        summary.update(
            {
                "head_param_names": head_names,
                "skipped_non_float_param_names": skipped,
                "skipped_non_float_param_count": len(skipped),
                "lora_rank": rank,
                "lora_alpha": alpha,
                "lora_target": target,
                "lora_target_names": target_names,
                "strict_lora_summary": strict_summary,
            }
        )
        if strict_summary:
            strict_summary["classifier_head_trainable"] = bool(head_names)
            if strict_summary.get("meriem_exact"):
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
        if not head_names:
            summary["notes"].append("No classifier/head parameters were found; LoRA adapters are trainable but head is unchanged.")
        if skipped:
            summary["notes"].append("non-floating parameters cannot require gradients and were skipped")

    summary.update(_param_counts(model))
    summary["trainable_param_examples"] = _trainable_param_examples(model)
    if summary["trainable_params"] <= 0:
        raise RuntimeError(f"Strategy {strategy} produced zero trainable parameters for {model_name}.")
    _write_strategy_reports(summary, args)
    print(
        f"[finetune_strategy] {model_name} {strategy}: "
        f"{summary['trainable_params']}/{summary['total_params']} trainable "
        f"({summary['trainable_percent']:.4f}%)"
    )
    return summary
