"""Shared fine-tuning strategy policies for the TUAB benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.utils import parametrize


VALID_STRATEGIES = {"full_finetune", "freeze_backbone", "lora"}
HEAD_PREFIXES = {
    "LaBraM": ("model.head.", "head."),
    "EEGPT": ("model.head.", "head."),
    "BIOT": ("classifier.clshead.", "classifier."),
    "CBraMod": ("classifier.",),
    "CSBrain": ("classifier.",),
    "CodeBrain": ("model.classifier.", "classifier."),
}
LORA_EXCLUDE_TOKENS = ("classifier", "head", "lm_head", "prediction")
LORA_MODULE_TARGET_ALIASES = {"lora_module"}
LORA_ROLE_TARGETS = {"attention_only", "mlp_only", "attention_and_mlp"}
DEFAULT_LORA_SETTINGS = {
    "rank": 8,
    "alpha": 8,
    "layers": "all",
    "target": "attention_and_mlp",
    "init_scale": 0.01,
}
MODEL_FINETUNING_SOURCES = {
    "LaBraM": [
        "run_class_finetuning.py",
        "engine_for_finetuning.py",
        "Protocol_Evaluation/finetune_strategies.py",
    ],
    "EEGPT": [
        "downstream_tueg/run_class_finetuning_EEGPT_change.py",
        "downstream_tueg/finetune_TUAB_EEGPT.sh",
    ],
    "BIOT": [
        "README.md",
        "model/biot.py",
    ],
    "CBraMod": [
        "finetune_main.py",
        "finetune_trainer.py",
        "models/model_for_tuab.py",
    ],
    "CSBrain": [
        "finetune_main.py",
        "finetune_trainer.py",
        "models/model_for_tuab.py",
        "sh/finetune_CSBrain_TUAB.sh",
    ],
    "CodeBrain": [
        "Downstream/finetune_main.py",
        "Downstream/finetune_trainer.py",
        "Models/model_for_tuab.py",
    ],
}


# Strategy entry points.

def apply_finetuning_strategy(
    model: nn.Module,
    config: dict,
    model_name: str,
) -> dict[str, Any]:
    """Apply the config-selected parameter policy before optimizer creation."""
    settings = config["fine_tuning"]
    strategy = settings["strategy"]
    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"fine_tuning.strategy must be one of {sorted(VALID_STRATEGIES)}, "
            f"got {strategy!r}"
        )

    if strategy == "full_finetune":
        summary = full_finetune(model, model_name, config)
    elif strategy == "freeze_backbone":
        summary = freeze_backbone(model, model_name, config)
    else:
        summary = lora(model, model_name, settings.get("lora", {}), config)

    summary.update(parameter_summary(model))
    if summary["trainable_parameters"] <= 0:
        raise RuntimeError(f"{strategy} selected zero trainable parameters for {model_name}")
    return {"strategy": strategy, "model_name": model_name, **summary}


def full_finetune(model: nn.Module, model_name: str, config: dict) -> dict[str, Any]:
    """Train every floating-point or complex parameter in the author model architecture."""
    skipped = set_all_trainable(model, True)
    notes = []
    if skipped:
        notes.append("non-floating parameters cannot require gradients and were skipped")
    return {
        "head_parameter_names": [],
        "lora_target_names": [],
        "skipped_non_float_param_names": skipped,
        "notes": notes,
        "dependency_flow": strategy_dependency_flow(model_name, config, "full_finetune"),
        "model_specific_function": "full_finetune",
    }


def freeze_backbone(model: nn.Module, model_name: str, config: dict) -> dict[str, Any]:
    """Freeze the author backbone and train only the model-specific final task head."""
    set_all_trainable(model, False)
    head_names = head_parameter_names(model, model_name)
    if not head_names:
        raise RuntimeError(
            f"Could not identify classifier/head parameters for {model_name}; "
            "freeze_backbone would otherwise train the wrong tensors."
        )
    skipped = set_named_trainable(model, set(head_names))
    return {
        "head_parameter_names": head_names,
        "lora_target_names": [],
        "skipped_non_float_param_names": skipped,
        "notes": [
            "freeze_backbone starts from the full fine-tuning architecture and re-enables only the model-specific final classifier/task head",
            "freeze_backbone is the benchmark linear-probing policy",
        ],
        "dependency_flow": strategy_dependency_flow(model_name, config, "freeze_backbone"),
        "model_specific_function": "freeze_backbone",
    }


def lora(model: nn.Module, model_name: str, settings: dict, config: dict) -> dict[str, Any]:
    """Freeze base weights, dispatch model-specific LoRA, then train the task head."""
    settings = validate_lora_settings(settings)
    if model_name not in LORA_BY_MODEL:
        raise ValueError(f"No model-specific LoRA policy is defined for {model_name!r}")
    set_all_trainable(model, False)
    target_names, notes, lora_summary = LORA_BY_MODEL[model_name](model, settings)
    if not target_names:
        raise RuntimeError(
            f"No LoRA targets found for {model_name} with "
            f"layers={settings['layers']!r}, target={settings['target']!r}"
        )

    head_names = head_parameter_names(model, model_name)
    skipped = set_named_trainable(model, set(head_names))
    if not head_names:
        notes.append("No classifier/head parameters were found; only LoRA parameters are trainable.")
    if skipped:
        notes.append("non-floating parameters cannot require gradients and were skipped")
    lora_summary["classifier_head_trainable"] = bool(head_names)

    return {
        "head_parameter_names": head_names,
        "lora_target_names": target_names,
        "skipped_non_float_param_names": skipped,
        "notes": notes,
        "lora_rank": settings["rank"],
        "lora_alpha": settings["alpha"],
        "lora_layers": settings["layers"],
        "lora_target": settings["target"],
        "lora_init_scale": settings["init_scale"],
        "lora_policy_summary": lora_summary,
        "dependency_flow": strategy_dependency_flow(model_name, config, "lora"),
    }


# Model-specific LoRA policies.

def lora_labram(model: nn.Module, settings: dict[str, Any]) -> tuple[list[str], list[str], dict[str, Any]]:
    """LaBraM LoRA: qkv, fc1/fc2, TemporalConv conv1/2/3; no attention out projection."""
    return apply_lora_role_targets(model, "LaBraM", settings, skip_reconstructor=False, allow_temporal_conv=True)


def lora_eegpt(model: nn.Module, settings: dict[str, Any]) -> tuple[list[str], list[str], dict[str, Any]]:
    """EEGPT LoRA: safe qkv/MLP equivalents, excluding reconstruction modules."""
    return apply_lora_role_targets(model, "EEGPT", settings, skip_reconstructor=True, allow_temporal_conv=True)


def lora_biot(model: nn.Module, settings: dict[str, Any]) -> tuple[list[str], list[str], dict[str, Any]]:
    """BIOT LoRA: attention equivalents plus feed-forward w1/w2 modules."""
    return apply_lora_role_targets(model, "BIOT", settings, skip_reconstructor=False, allow_temporal_conv=True)


def lora_cbramod(model: nn.Module, settings: dict[str, Any]) -> tuple[list[str], list[str], dict[str, Any]]:
    """CBraMod LoRA: MHA in-projection plus linear1/linear2; out_proj is skipped."""
    return apply_lora_role_targets(model, "CBraMod", settings, skip_reconstructor=False, allow_temporal_conv=True)


def lora_csbrain(model: nn.Module, settings: dict[str, Any]) -> tuple[list[str], list[str], dict[str, Any]]:
    """CSBrain LoRA: MHA in-projection plus linear1/linear2; out_proj is skipped."""
    return apply_lora_role_targets(model, "CSBrain", settings, skip_reconstructor=False, allow_temporal_conv=True)


def lora_codebrain(model: nn.Module, settings: dict[str, Any]) -> tuple[list[str], list[str], dict[str, Any]]:
    """CodeBrain LoRA: partial if no MLP/feed-forward equivalent is identifiable."""
    return apply_lora_role_targets(model, "CodeBrain", settings, skip_reconstructor=False, allow_temporal_conv=True)


LORA_BY_MODEL = {
    "LaBraM": lora_labram,
    "EEGPT": lora_eegpt,
    "BIOT": lora_biot,
    "CBraMod": lora_cbramod,
    "CSBrain": lora_csbrain,
    "CodeBrain": lora_codebrain,
}


# LoRA target selection.

def apply_lora_role_targets(
    model: nn.Module,
    model_name: str,
    settings: dict[str, Any],
    skip_reconstructor: bool,
    allow_temporal_conv: bool,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Apply model-specific qkv/MLP-equivalent roles or explicit module substrings."""
    target = settings["target"]
    rank = settings["rank"]
    alpha = settings["alpha"]
    init_scale = settings["init_scale"]
    if target in LORA_MODULE_TARGET_ALIASES:
        target = "attention_and_mlp"
    if target in LORA_ROLE_TARGETS:
        return apply_author_target_roles(
            model,
            model_name,
            target,
            rank,
            alpha,
            init_scale,
            skip_reconstructor=skip_reconstructor,
            allow_temporal_conv=allow_temporal_conv,
        )

    requested = [item.strip() for item in target.split(",") if item.strip()]
    if not requested:
        raise ValueError(
            "fine_tuning.lora.target must be one of "
            f"{sorted(LORA_ROLE_TARGETS)} or name module substrings."
        )
    target_names = []
    for name, module in list(model.named_modules()):
        if not name or not isinstance(module, nn.Linear):
            continue
        lowered = name.lower()
        if skip_reconstructor and ".reconstructor" in lowered:
            continue
        if is_head_name(name) or ".base" in lowered or is_attention_output(name):
            continue
        if any(item in name for item in requested):
            replace_module(model, name, LoRALinear(module, rank, alpha, init_scale))
            target_names.append(name)

    missing = [item for item in requested if not any(item in name for name in target_names)]
    notes = [f"Requested LoRA target substrings not found: {missing}"] if missing else []
    return target_names, notes, {"model_specific_lora_supported": "not_requested"}


def apply_author_target_roles(
    model: nn.Module,
    model_name: str,
    target: str,
    rank: int,
    alpha: float,
    init_scale: float | None,
    skip_reconstructor: bool,
    allow_temporal_conv: bool,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Use each model's attention and MLP-equivalent target roles."""
    adapt_attention = target in {"attention_only", "attention_and_mlp"}
    adapt_mlp = target in {"mlp_only", "attention_and_mlp"}
    target_names: list[str] = []
    notes: list[str] = []
    summary: dict[str, Any] = {
        "model_specific_lora_supported": "partial",
        "qkv_or_qkv_equivalent_adapted": False,
        "mlp_adapted": False,
        "temporal_conv2d_adapted": False,
        "attention_output_projection_adapted": False,
        "remaining_deviations": [],
        "target_resolution": "model_specific_attention_mlp_equivalent_roles",
        "qkv_mlp_rank": int(rank),
        "conv_rank": 4,
        "alpha": float(alpha),
        "lora_a_init": "torch.randn(...)*0.01" if init_scale == 0.01 else "kaiming_uniform",
        "lora_b_init": "zeros",
        "bias_lora": False,
        "model_specific_function": f"lora_{model_name.lower()}",
    }

    for name, module in list(model.named_modules()):
        if (
            adapt_attention
            and isinstance(module, nn.MultiheadAttention)
            and getattr(module, "in_proj_weight", None) is not None
        ):
            register_lora_parametrization(module, "in_proj_weight", rank, alpha, init_scale)
            target_names.append(f"{name}.in_proj_weight")
            summary["qkv_or_qkv_equivalent_adapted"] = True

    for name, module in list(model.named_modules()):
        if not name:
            continue
        lowered = name.lower()
        if skip_reconstructor and ".reconstructor" in lowered:
            continue
        if isinstance(module, nn.Linear):
            if is_lora_excluded(name) or "spectral_proj" in lowered or "patch_embedding.projection" in lowered:
                continue
            if is_attention_output(name):
                continue
            if adapt_attention and is_qkv_linear_name(name):
                register_lora_parametrization(module, "weight", rank, alpha, init_scale)
                target_names.append(f"{name}.weight")
                summary["qkv_or_qkv_equivalent_adapted"] = True
            elif adapt_mlp and is_mlp_linear_name(name):
                replace_module(model, name, LoRALinear(module, rank, alpha, init_scale))
                target_names.append(name)
                summary["mlp_adapted"] = True
        elif (
            allow_temporal_conv
            and target == "attention_and_mlp"
            and isinstance(module, nn.Conv2d)
            and is_temporal_conv_name(model_name, name)
        ):
            register_lora_parametrization(module, "weight", 4, alpha, init_scale)
            target_names.append(f"{name}.weight")
            summary["temporal_conv2d_adapted"] = True

    if adapt_attention and not summary["qkv_or_qkv_equivalent_adapted"]:
        summary["remaining_deviations"].append("No qkv/q/k/v equivalent target was identified.")
    if adapt_mlp and not summary["mlp_adapted"]:
        summary["remaining_deviations"].append("No MLP/feed-forward equivalent target was identified.")
    if target == "attention_and_mlp" and not summary["temporal_conv2d_adapted"]:
        notes.append("No true temporal Conv2d equivalent was identified for this model.")
    if (
        (not adapt_attention or summary["qkv_or_qkv_equivalent_adapted"])
        and (not adapt_mlp or summary["mlp_adapted"])
    ):
        summary["model_specific_lora_supported"] = "yes"
    if model_name == "CodeBrain" and not summary["mlp_adapted"]:
        notes.append("CodeBrain LoRA is partial: qkv adapted but no MLP equivalent was identified.")
    return target_names, notes, summary


# LoRA modules.

class LoRALinear(nn.Module):
    """Low-rank update for normal Linear calls without editing author repos."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, init_scale: float | None = 0.01):
        super().__init__()
        device = base.weight.device
        dtype = base.weight.dtype
        self.base = base
        self.scaling = float(alpha) / float(rank)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False, device=device, dtype=dtype)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False, device=device, dtype=dtype)
        init_lora_a(self.lora_a.weight, init_scale)
        nn.init.zeros_(self.lora_b.weight)
        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(x)) * self.scaling


class LoRAWeightParametrization(nn.Module):
    """Weight LoRA for modules whose forward reads `.weight` directly."""

    def __init__(self, weight: torch.Tensor, rank: int, alpha: float, init_scale: float | None = 0.01):
        super().__init__()
        self.scaling = float(alpha) / float(rank)
        out_features = int(weight.shape[0])
        in_features = int(weight.numel() // out_features)
        self.lora_a = nn.Parameter(torch.empty(rank, in_features, device=weight.device, dtype=weight.dtype))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank, device=weight.device, dtype=weight.dtype))
        init_lora_a(self.lora_a, init_scale)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        update = (self.lora_b @ self.lora_a).view_as(weight) * self.scaling
        return weight + update


def register_lora_parametrization(
    module: nn.Module,
    parameter_name: str,
    rank: int,
    alpha: float,
    init_scale: float | None = 0.01,
) -> None:
    weight = getattr(module, parameter_name)
    parametrize.register_parametrization(
        module,
        parameter_name,
        LoRAWeightParametrization(weight.detach(), rank, alpha, init_scale),
    )
    original = getattr(module.parametrizations, parameter_name).original
    original.requires_grad = False


def init_lora_a(param: torch.Tensor, init_scale: float | None) -> None:
    if init_scale is None:
        nn.init.kaiming_uniform_(param, a=5 ** 0.5)
    else:
        with torch.no_grad():
            param.copy_(torch.randn_like(param) * float(init_scale))


# Trainability helpers.

def set_all_trainable(model: nn.Module, trainable: bool) -> list[str]:
    """Set gradients safely; integer/bool tensors can never require grad."""
    skipped = []
    for name, param in model.named_parameters():
        if trainable and can_require_grad(param):
            param.requires_grad = True
        elif trainable:
            param.requires_grad = False
            skipped.append(name)
        else:
            param.requires_grad = False
    return skipped


def set_named_trainable(model: nn.Module, trainable_names: set[str]) -> list[str]:
    """Enable only named floating-point or complex parameters."""
    skipped = []
    for name, param in model.named_parameters():
        if name not in trainable_names:
            continue
        if can_require_grad(param):
            param.requires_grad = True
        else:
            param.requires_grad = False
            skipped.append(name)
    return skipped


def can_require_grad(param: torch.Tensor) -> bool:
    return bool(param.is_floating_point() or param.is_complex())


def head_parameter_names(model: nn.Module, model_name: str) -> list[str]:
    """Return audited task-head parameter names for each benchmark wrapper."""
    prefixes = HEAD_PREFIXES.get(model_name, ())
    names = [
        name
        for name, _ in model.named_parameters()
        if any(name == prefix.rstrip(".") or name.startswith(prefix) for prefix in prefixes)
    ]
    if names:
        return names
    return [name for name, _ in model.named_parameters() if is_head_name(name)]


def is_head_name(name: str) -> bool:
    parts = name.lower().split(".")
    return bool({"classifier", "clshead", "classification_head", "head"} & set(parts))


def is_lora_excluded(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in LORA_EXCLUDE_TOKENS)


def is_attention_output(name: str) -> bool:
    lowered = name.lower()
    return lowered.endswith(".out_proj") or lowered.endswith(".attn.proj") or lowered.endswith(".attention.out_proj")


def is_qkv_linear_name(name: str) -> bool:
    leaf = name.rsplit(".", 1)[-1].lower()
    return leaf in {"qkv", "to_q", "to_k", "to_v", "query", "key", "value"}


def is_mlp_linear_name(name: str) -> bool:
    lowered = name.lower()
    leaf = lowered.rsplit(".", 1)[-1]
    return (
        leaf in {"fc1", "fc2", "linear1", "linear2", "w1", "w2"}
        or lowered.endswith(".fn.fn.w1")
        or lowered.endswith(".fn.fn.w2")
    )


def is_temporal_conv_name(model_name: str, name: str) -> bool:
    lowered = name.lower()
    if model_name == "LaBraM":
        return any(lowered.endswith(f"patch_embed.{item}") for item in ("conv1", "conv2", "conv3"))
    return "temembed" in lowered or "temporal" in lowered


def normalize_lora_layers(layers: Any) -> str:
    return str(layers or "all").strip()


def normalize_lora_target(target: Any) -> str:
    if isinstance(target, (list, tuple)):
        return ",".join(str(item) for item in target)
    return str(target or "attention_and_mlp").strip()


# Audit summaries and provenance.

def strategy_dependency_flow(model_name: str, config: dict, strategy: str) -> dict[str, Any]:
    return {
        "config_path": config.get("_config_path", ""),
        "config_field": "fine_tuning.strategy",
        "selected_strategy": strategy,
        "training_entry": "Benchmark/Training/training_common.py::configure_fine_tuning",
        "studycase_entry": "Benchmark/StudyCase/Finetuning/finetuning_strategies.py::apply_finetuning_strategy",
        "model_repo": str(config.get("paths", {}).get("model_repo", "")),
        "eegfm_source_files": model_source_files(model_name, config),
        "source_read_mode": "path/provenance inspection; author training scripts are not imported or executed",
    }


def model_source_files(model_name: str, config: dict) -> list[dict[str, Any]]:
    repo = Path(str(config.get("paths", {}).get("model_repo", ""))).expanduser()
    files = []
    for relative_path in MODEL_FINETUNING_SOURCES.get(model_name, []):
        path = repo / relative_path
        files.append(
            {
                "relative_path": relative_path,
                "path": str(path),
                "exists": path.is_file(),
            }
        )
    return files


def validate_lora_settings(settings: dict | None) -> dict[str, Any]:
    resolved = {**DEFAULT_LORA_SETTINGS, **(settings or {})}
    rank = int(resolved["rank"])
    alpha = float(resolved["alpha"])
    layers = normalize_lora_layers(resolved.get("layers", resolved.get("lora_layers", "all")))
    target = normalize_lora_target(resolved.get("target", resolved.get("lora_target")))
    init_scale = float(resolved["init_scale"]) if resolved.get("init_scale") is not None else None
    if target in LORA_MODULE_TARGET_ALIASES:
        target = "attention_and_mlp"
    if rank <= 0:
        raise ValueError("fine_tuning.lora.rank must be positive")
    if layers != "all":
        raise ValueError("fine_tuning.lora.layers currently matches LaBraM's supported default: 'all'")
    return {
        **resolved,
        "rank": rank,
        "alpha": alpha,
        "layers": layers,
        "target": target,
        "init_scale": init_scale,
    }


def replace_module(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    parent = root
    parts = dotted_name.split(".")
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)) else getattr(parent, part)
    leaf = parts[-1]
    if leaf.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(leaf)] = new_module
    else:
        setattr(parent, leaf, new_module)


def parameter_summary(model: nn.Module) -> dict[str, Any]:
    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    frozen_names = [name for name, param in model.named_parameters() if not param.requires_grad]
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "trainable_parameters": int(trainable),
        "frozen_parameters": int(total - trainable),
        "total_parameters": int(total),
        "trainable_percent": float((trainable / total) * 100.0) if total else 0.0,
        "trainable_parameter_tensors": trainable_names,
        "frozen_parameter_tensors": frozen_names,
    }
