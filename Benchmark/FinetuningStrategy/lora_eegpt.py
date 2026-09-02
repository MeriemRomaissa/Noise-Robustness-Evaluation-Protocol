#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EEGPT LoRA study-case helper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import nn

try:
    from ._lora_common import apply_model_lora, apply_lora_module_policy, apply_strategy_for_model, lora_settings_from_args
except ImportError:
    from _lora_common import apply_model_lora, apply_lora_module_policy, apply_strategy_for_model, lora_settings_from_args


MODEL_NAME = "EEGPT"


def lora_eegpt(model: nn.Module, rank: int = 2, alpha: float = 8.0, init_scale: float = 0.01):
    """EEGPT LoRA: qkv and MLP equivalents; reconstruction modules excluded."""
    return apply_lora_module_policy(model, MODEL_NAME, rank=rank, alpha=alpha, init_scale=init_scale)


def apply_lora_strategy(
    model: nn.Module,
    lora_settings: dict[str, Any] | None = None,
    output_dir: str | Path = "",
    strategy_report_path: str | Path = "",
    allow_head_guess: bool = False,
) -> dict[str, Any]:
    return apply_model_lora(
        model,
        MODEL_NAME,
        lora_settings=lora_settings,
        output_dir=output_dir,
        strategy_report_path=strategy_report_path,
        allow_head_guess=allow_head_guess,
    )


def strategy_lora(model: nn.Module, args: Any = None) -> dict[str, Any]:
    """Reference-style entry point used after model construction."""
    return apply_lora_strategy(
        model,
        lora_settings=lora_settings_from_args(args),
        output_dir=getattr(args, "output_dir", ""),
        strategy_report_path=getattr(args, "strategy_report_path", ""),
        allow_head_guess=bool(getattr(args, "allow_head_guess", False)),
    )


def apply_strategy(model: nn.Module, args: Any = None) -> dict[str, Any]:
    return apply_strategy_for_model(model, args, MODEL_NAME, apply_lora_strategy)
