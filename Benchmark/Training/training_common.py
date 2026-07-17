"""Run one auditable training protocol across all six EEG foundation models.

Model-specific files provide model construction and checkpoint-name rules. This
module owns the decisions that must remain identical across the benchmark:
configuration validation, deterministic seeds, loader construction, mandatory
pretrained initialization, fine-tuning, validation-based model selection,
single final test evaluation, and cross-seed reporting.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class ModelSpec:
    """Declare the model-specific components required by the shared protocol."""

    name: str
    loader_module: str
    build_model: Callable[[dict], nn.Module]
    load_checkpoint: Callable[[nn.Module, Path], dict]
    study_case_module: str | None = None


@dataclass(frozen=True)
class EpochContext:
    """Hold the model state shared by every batch in one data split."""

    model: nn.Module
    device: torch.device
    config: dict
    transform: Callable[[Any], Any]


@dataclass(frozen=True)
class Optimization:
    """Hold state used only while updating model parameters."""

    optimizer: torch.optim.Optimizer
    scheduler: Any = None


@dataclass(frozen=True)
class SeedExperiment:
    """Collect the prepared components for one reproducible seed run."""

    spec: ModelSpec
    config: dict
    seed: int
    run_dir: Path
    model: nn.Module
    loaders: dict[str, Any]
    contexts: dict[str, EpochContext]
    optimization: Optimization
    checkpoint_report: dict
    fine_tuning_report: dict


@dataclass(frozen=True)
class ModelSelection:
    """Record the checkpoint selected exclusively on validation performance."""

    checkpoint_path: Path
    best_epoch: int
    best_value: float
    history: list[dict]


def run_experiments(spec: ModelSpec, default_config: str) -> None:
    """Run every configured seed and summarize final test performance."""
    arguments = parse_arguments(default_config)
    config = load_config(arguments.config)
    seeds = [int(seed) for seed in config["training"]["seeds"]]

    seed_results = [run_seed(spec, config, seed) for seed in seeds]
    output_dir = Path(config["paths"]["output"]).expanduser()

    save_json(output_dir / "all_seed_results.json", seed_results)
    summary = summarize_seed_results(
        seed_results,
        metrics=config["evaluation"]["report_metrics"],
    )
    save_json(output_dir / "summary.json", summary)


def run_seed(spec: ModelSpec, config: dict, seed: int) -> dict:
    """Prepare, train, select, test, and record one deterministic experiment."""
    experiment = prepare_seed_experiment(spec, config, seed)
    selection = train_and_select_best_model(experiment)
    test_metrics = evaluate_selected_model(experiment, selection)
    result = build_seed_result(experiment, selection, test_metrics)
    save_seed_artifacts(experiment, selection, result)
    return result


def prepare_seed_experiment(spec: ModelSpec, config: dict, seed: int) -> SeedExperiment:
    """Build every component needed before the first training epoch begins."""
    set_random_seed(seed)
    device = select_device()
    run_dir = create_seed_output_directory(config, seed)
    save_visible_config(config, run_dir)

    loaders = build_loaders(config, spec.loader_module)
    transforms = build_study_case_transforms(spec, config, loaders)
    model, checkpoint_report = build_pretrained_model(spec, config)
    model.to(device)

    fine_tuning_report = configure_fine_tuning(model, config)
    optimization = build_optimization(model, config, len(loaders["train"]))
    contexts = build_epoch_contexts(model, device, config, transforms)

    return SeedExperiment(
        spec=spec,
        config=config,
        seed=seed,
        run_dir=run_dir,
        model=model,
        loaders=loaders,
        contexts=contexts,
        optimization=optimization,
        checkpoint_report=checkpoint_report,
        fine_tuning_report=fine_tuning_report,
    )


def train_and_select_best_model(experiment: SeedExperiment) -> ModelSelection:
    """Train all epochs and select a checkpoint using validation data only."""
    config = experiment.config
    selection_metric = config["evaluation"]["selection_metric"]
    best_value = -math.inf
    best_epoch = 0
    history = []
    checkpoint_path = experiment.run_dir / "best_model.pt"

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_metrics = train_one_epoch(
            experiment.contexts["train"],
            experiment.loaders["train"],
            experiment.optimization,
        )
        validation_metrics = evaluate_one_split(
            experiment.contexts["val"],
            experiment.loaders["val"],
        )
        step_epoch_scheduler(experiment.optimization, config)

        history.append(format_epoch_record(epoch, train_metrics, validation_metrics))
        current_value = validation_metrics[selection_metric]
        if current_value > best_value:
            best_value = current_value
            best_epoch = epoch
            save_selected_checkpoint(
                checkpoint_path,
                experiment.model,
                epoch,
                validation_metrics,
            )

    if best_epoch == 0:
        raise RuntimeError("Training finished without selecting a validation checkpoint")

    return ModelSelection(
        checkpoint_path=checkpoint_path,
        best_epoch=best_epoch,
        best_value=best_value,
        history=history,
    )


def evaluate_selected_model(
    experiment: SeedExperiment,
    selection: ModelSelection,
) -> dict[str, float]:
    """Reload the validation-selected checkpoint and evaluate the test split once."""
    selected_checkpoint = safe_torch_load(selection.checkpoint_path)
    experiment.model.load_state_dict(selected_checkpoint["model"], strict=True)
    return evaluate_one_split(
        experiment.contexts["test"],
        experiment.loaders["test"],
    )


def build_seed_result(
    experiment: SeedExperiment,
    selection: ModelSelection,
    test_metrics: dict,
) -> dict:
    """Create the concise final record returned for one seed."""
    return {
        "model": experiment.spec.name,
        "seed": experiment.seed,
        "best_epoch": selection.best_epoch,
        "selection_metric": experiment.config["evaluation"]["selection_metric"],
        "best_validation_value": selection.best_value,
        "test": test_metrics,
    }


def save_seed_artifacts(
    experiment: SeedExperiment,
    selection: ModelSelection,
    result: dict,
) -> None:
    """Save the checkpoint audit, adaptation report, history, and final result."""
    save_json(experiment.run_dir / "checkpoint_load.json", experiment.checkpoint_report)
    save_json(experiment.run_dir / "fine_tuning.json", experiment.fine_tuning_report)
    save_json(experiment.run_dir / "history.json", selection.history)
    save_json(experiment.run_dir / "result.json", result)


def summarize_seed_results(seed_results: list[dict], metrics: list[str]) -> dict:
    """Report the mean and sample standard deviation across configured seeds."""
    summary = {}
    for metric in metrics:
        values = [result["test"][metric] for result in seed_results]
        summary[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
    return summary


def parse_arguments(default_config: str) -> argparse.Namespace:
    """Accept an alternate YAML path while keeping all choices inside YAML."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config)
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    """Read YAML and reject invalid settings before expensive model setup."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install PyYAML to read the training configuration.") from exc

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")

    config["_config_path"] = str(config_path)
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    """Validate required values, controlled choices, and numeric ranges."""
    required = {
        "paths": [
            "model_repo", "checkpoint", "h5_file", "split_index",
            "original_data", "output",
        ],
        "data": [
            "source", "train_samples", "validation_samples", "test_samples",
        ],
        "loader": [
            "batch_size", "num_workers", "shuffle_train",
            "drop_last_train", "pin_memory",
        ],
        "training": [
            "seeds", "epochs", "loss", "optimizer", "learning_rate",
            "weight_decay", "scheduler",
        ],
        "fine_tuning": ["strategy"],
        "evaluation": [
            "selection_metric", "classification_threshold", "report_metrics",
        ],
    }

    missing = find_missing_config_values(config, required)
    if missing:
        raise ValueError("Missing configuration values: " + ", ".join(missing))

    validate_config_choices(config)
    validate_numeric_config(config)
    validate_lora_config(config)


def find_missing_config_values(config: dict, required: dict[str, list[str]]) -> list[str]:
    """Return every absent required YAML value in reader-facing dotted form."""
    missing = []
    for section, keys in required.items():
        values = config.get(section)
        if not isinstance(values, dict):
            missing.extend(f"{section}.{key}" for key in keys)
            continue
        for key in keys:
            if values.get(key) is None:
                missing.append(f"{section}.{key}")
    return sorted(set(missing))


def validate_config_choices(config: dict) -> None:
    """Reject misspelled experiment choices and show the accepted values."""
    choices = {
        "data.source": (config["data"]["source"], {"unified60", "original"}),
        "training.loss": (
            config["training"]["loss"],
            {"cross_entropy", "bce_with_logits"},
        ),
        "training.optimizer": (
            config["training"]["optimizer"],
            {"adam", "adamw", "sgd"},
        ),
        "training.scheduler": (
            str(config["training"]["scheduler"]).lower(),
            {"none", "cosine"},
        ),
        "fine_tuning.strategy": (
            config["fine_tuning"]["strategy"],
            {"full_finetune", "linear_probe", "lora"},
        ),
        "evaluation.selection_metric": (
            config["evaluation"]["selection_metric"],
            {"balanced_accuracy", "auroc", "auprc"},
        ),
    }

    for name, (value, allowed) in choices.items():
        if value not in allowed:
            raise ValueError(f"{name} must be one of {sorted(allowed)}, got {value!r}")

    supported_metrics = {"balanced_accuracy", "auroc", "auprc"}
    unknown_metrics = set(config["evaluation"]["report_metrics"]) - supported_metrics
    if unknown_metrics:
        raise ValueError(f"Unknown evaluation.report_metrics: {sorted(unknown_metrics)}")


def validate_numeric_config(config: dict) -> None:
    """Check numeric values that would otherwise fail late during training."""
    training = config["training"]
    loader = config["loader"]
    evaluation = config["evaluation"]

    if not training["seeds"]:
        raise ValueError("training.seeds must contain at least one integer")
    if int(training["epochs"]) < 1:
        raise ValueError("training.epochs must be positive")
    if float(training["learning_rate"]) <= 0:
        raise ValueError("training.learning_rate must be positive")
    if float(training["weight_decay"]) < 0:
        raise ValueError("training.weight_decay cannot be negative")
    if int(loader["batch_size"]) < 1:
        raise ValueError("loader.batch_size must be positive")
    if int(loader["num_workers"]) < 0:
        raise ValueError("loader.num_workers cannot be negative")

    threshold = float(evaluation["classification_threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("evaluation.classification_threshold must be between 0 and 1")


def validate_lora_config(config: dict) -> None:
    """Require LoRA parameters only when the researcher selects LoRA."""
    if config["fine_tuning"]["strategy"] != "lora":
        return

    settings = config["fine_tuning"].get("lora")
    required = ("rank", "alpha", "dropout")
    if not isinstance(settings, dict):
        raise ValueError("fine_tuning.lora settings are required when LoRA is selected")

    missing = [key for key in required if settings.get(key) is None]
    if missing:
        raise ValueError("Missing LoRA settings: " + ", ".join(missing))
    if int(settings["rank"]) <= 0:
        raise ValueError("fine_tuning.lora.rank must be positive")
    if not 0.0 <= float(settings["dropout"]) <= 1.0:
        raise ValueError("fine_tuning.lora.dropout must be between 0 and 1")


def select_device() -> torch.device:
    """Use a visible CUDA device when available and otherwise use CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_seed_output_directory(config: dict, seed: int) -> Path:
    """Create the directory that owns every artifact from one seed."""
    run_dir = Path(config["paths"]["output"]).expanduser() / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_visible_config(config: dict, run_dir: Path) -> None:
    """Record reader-controlled settings without private runtime keys."""
    visible_config = {
        key: value for key, value in config.items() if not key.startswith("_")
    }
    save_json(run_dir / "config.json", visible_config)


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, CPU, and every visible CUDA device."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_pretrained_model(spec: ModelSpec, config: dict) -> tuple[nn.Module, dict]:
    """Construct one model and prove that pretrained parameters reached it."""
    model = spec.build_model(config)
    checkpoint_path = resolve_checkpoint(config)
    checkpoint_report = spec.load_checkpoint(model, checkpoint_path)
    loaded_keys = int(checkpoint_report.get("loaded_keys", 0))
    if loaded_keys <= 0:
        raise RuntimeError(f"{spec.name} did not load pretrained backbone weights")
    return model, checkpoint_report


def add_repo_to_import_path(config: dict) -> Path:
    """Make the configured author repository importable without installing it."""
    repository = Path(config["paths"]["model_repo"]).expanduser().resolve()
    if not repository.is_dir():
        raise FileNotFoundError(f"Model repository not found: {repository}")

    import_candidates = (repository, repository / "Models", repository / "models")
    for path in import_candidates:
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return repository


def resolve_checkpoint(config: dict) -> Path:
    """Find the mandatory checkpoint using documented repository-relative paths."""
    configured_path = Path(str(config["paths"]["checkpoint"])).expanduser()
    repository = Path(config["paths"]["model_repo"]).expanduser()
    candidates = (
        [configured_path]
        if configured_path.is_absolute()
        else [configured_path, repository / configured_path, repository.parent / configured_path]
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Mandatory pretrained checkpoint not found. Checked: {checked}")


def safe_torch_load(path: Path) -> Any:
    """Load checkpoints on CPU across supported PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Find tensor parameters inside common author checkpoint containers."""
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a dictionary of tensors")

    for key in ("state_dict", "model", "model_state_dict", "module", "student"):
        value = checkpoint.get(key)
        if isinstance(value, dict) and any(torch.is_tensor(item) for item in value.values()):
            return value

    if any(torch.is_tensor(item) for item in checkpoint.values()):
        return checkpoint
    raise ValueError("No tensor state dictionary was found in the checkpoint")


def load_matching_weights(
    target: nn.Module,
    checkpoint_path: Path,
    key_mapper: Callable[[str], str | None] | None = None,
) -> dict:
    """Load shape-compatible pretrained weights and report every unmatched key."""
    state = extract_state_dict(safe_torch_load(checkpoint_path))
    target_state = target.state_dict()
    matched = {}
    shape_mismatches = []

    for source_key, value in state.items():
        target_key = key_mapper(source_key) if key_mapper else source_key
        if target_key not in target_state or not torch.is_tensor(value):
            continue
        if tuple(value.shape) == tuple(target_state[target_key].shape):
            matched[target_key] = value
        else:
            shape_mismatches.append(source_key)

    if not matched:
        raise RuntimeError(f"Checkpoint {checkpoint_path} loaded zero compatible keys")

    load_result = target.load_state_dict(matched, strict=False)
    return {
        "path": str(checkpoint_path),
        "loaded_keys": len(matched),
        "target_keys": len(target_state),
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "shape_mismatches": shape_mismatches,
    }


def loader_config(config: dict, split: str) -> dict:
    """Translate reader-facing YAML into the stable shared-loader interface."""
    data = config["data"]
    loader = config["loader"]
    paths = config["paths"]
    sample_key = {
        "train": "train_samples",
        "val": "validation_samples",
        "test": "test_samples",
    }[split]

    translated = {
        "h5_path": paths["h5_file"],
        "split_index_path": paths["split_index"],
        "original_data_path": paths["original_data"],
        "batch_size": loader["batch_size"],
        "num_workers": loader["num_workers"],
        "pin_memory": loader["pin_memory"],
        "shuffle": loader["shuffle_train"] if split == "train" else False,
        "drop_last": loader["drop_last_train"] if split == "train" else False,
        "max_samples": data[sample_key],
    }

    reserved_data_keys = {
        "source", "train_samples", "validation_samples", "test_samples",
    }
    model_specific_data = {
        key: value for key, value in data.items() if key not in reserved_data_keys
    }
    translated.update(model_specific_data)
    return translated


def build_loaders(config: dict, module_name: str) -> dict[str, Any]:
    """Build train, validation, and test loaders through one model adapter."""
    loader_directory = Path(__file__).resolve().parents[1] / "Loader"
    if str(loader_directory) not in sys.path:
        sys.path.insert(0, str(loader_directory))

    loader_module = importlib.import_module(module_name)
    if config["data"]["source"] == "unified60":
        build_loader = loader_module.build_unified60_loader
    else:
        build_loader = loader_module.build_original_loader

    return {
        split: build_loader(loader_config(config, split), split)
        for split in ("train", "val", "test")
    }


def build_study_case_transforms(
    spec: ModelSpec,
    config: dict,
    loaders: dict[str, Any],
) -> dict[str, Callable[[Any], Any]]:
    """Build one study-case transform for each loader's channel metadata."""
    return {
        split: build_study_case_transform(spec, config, loader)
        for split, loader in loaders.items()
    }


def build_study_case_transform(
    spec: ModelSpec,
    config: dict,
    loader,
) -> Callable[[Any], Any]:
    """Connect the configured channel condition or preserve EEG unchanged."""
    settings = config.get("study_case")
    if settings is None:
        return leave_eeg_unchanged
    if spec.study_case_module is None:
        raise ValueError(f"{spec.name} does not define a channel study case")

    mode = settings.get("channel_mode")
    if mode is None:
        raise ValueError("study_case.channel_mode is required when study_case is present")

    study_directory = Path(__file__).resolve().parents[1] / "StudyCase" / "Channel"
    if str(study_directory) not in sys.path:
        sys.path.insert(0, str(study_directory))

    study_module = importlib.import_module(spec.study_case_module)
    channel_names = getattr(loader.dataset, "channel_names", None)

    def apply_configured_channel_case(eeg):
        return study_module.apply_channel_case(eeg, channel_names, mode)

    return apply_configured_channel_case


def leave_eeg_unchanged(eeg):
    """Preserve model-ready EEG when no study case is configured."""
    return eeg


def build_epoch_contexts(
    model: nn.Module,
    device: torch.device,
    config: dict,
    transforms: dict[str, Callable[[Any], Any]],
) -> dict[str, EpochContext]:
    """Bind each split to its model, device, configuration, and transform."""
    return {
        split: EpochContext(model, device, config, transform)
        for split, transform in transforms.items()
    }


class LoRALinear(nn.Module):
    """Add a low-rank trainable update while preserving a frozen linear layer."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)

        device = base.weight.device
        dtype = base.weight.dtype
        self.lora_a = nn.Linear(
            base.in_features,
            rank,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.lora_b = nn.Linear(
            rank,
            base.out_features,
            bias=False,
            device=device,
            dtype=dtype,
        )
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, inputs):
        low_rank_update = self.lora_b(self.lora_a(self.dropout(inputs)))
        return self.base(inputs) + low_rank_update * self.scale


def configure_fine_tuning(model: nn.Module, config: dict) -> dict:
    """Direct the model to the selected, explicitly named adaptation strategy."""
    strategy = config["fine_tuning"]["strategy"]

    if strategy == "full_finetune":
        lora_targets = configure_full_fine_tuning(model)
    elif strategy == "linear_probe":
        lora_targets = configure_linear_probe(model)
    elif strategy == "lora":
        lora_targets = configure_lora(model, config["fine_tuning"]["lora"])
    else:
        raise ValueError(f"Unsupported fine-tuning strategy: {strategy}")

    audit = audit_fine_tuning_strategy(model, strategy, lora_targets)
    return {
        "strategy": strategy,
        **audit,
        "lora_targets": lora_targets,
    }


def configure_full_fine_tuning(model: nn.Module) -> list[str]:
    """Allow every pretrained model parameter to update."""
    set_all_parameters_trainable(model, True)
    return []


def configure_linear_probe(model: nn.Module) -> list[str]:
    """Freeze the backbone and train only recognized task-head parameters."""
    set_all_parameters_trainable(model, False)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = is_head_parameter(name)
    return []


def configure_lora(model: nn.Module, settings: dict) -> list[str]:
    """Freeze the backbone and add trainable low-rank adapters to safe layers."""
    set_all_parameters_trainable(model, False)
    candidates = find_lora_target_modules(model)
    if not candidates:
        raise RuntimeError("LoRA selected, but no safe backbone nn.Linear targets were found")

    targets = []
    for name, module in candidates:
        replacement = LoRALinear(
            module,
            rank=int(settings["rank"]),
            alpha=float(settings["alpha"]),
            dropout=float(settings["dropout"]),
        )
        replace_module(model, name, replacement)
        targets.append(name)

    for name, parameter in model.named_parameters():
        parameter.requires_grad = "lora_" in name or is_head_parameter(name)
    return targets


def set_all_parameters_trainable(model: nn.Module, trainable: bool) -> None:
    """Apply one trainability state before a strategy selects exceptions."""
    for parameter in model.parameters():
        parameter.requires_grad = trainable


def is_head_parameter(name: str) -> bool:
    """Recognize task heads across the six author model naming conventions."""
    lowered = name.lower()
    head_names = ("classifier", "head", "final_layer", "fc_out", "linear_probe")
    return any(token in lowered for token in head_names)


def find_lora_target_modules(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    """Find safe backbone linear layers without asking readers for module names."""
    targets = []
    for name, module in model.named_modules():
        if not name or not isinstance(module, nn.Linear):
            continue
        if is_head_parameter(name) or ".base" in name:
            continue
        if name.endswith("out_proj") or name.endswith("qkv"):
            continue
        targets.append((name, module))
    return targets


def replace_module(root: nn.Module, name: str, replacement: nn.Module) -> None:
    """Replace a dotted child module without editing the author repository."""
    parent = root
    parts = name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], replacement)


def count_trainable_parameters(model: nn.Module) -> int:
    """Count parameters that the selected strategy will update."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def audit_fine_tuning_strategy(
    model: nn.Module,
    strategy: str,
    lora_targets: list[str],
) -> dict:
    """Prove that the selected strategy made only its intended tensors trainable."""
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    frozen_names = [
        name for name, parameter in model.named_parameters() if not parameter.requires_grad
    ]
    if not trainable_names:
        raise RuntimeError(f"{strategy} selected zero trainable parameter tensors")

    if strategy == "full_finetune":
        if frozen_names:
            raise RuntimeError(
                "full_finetune left frozen parameter tensors: "
                + ", ".join(frozen_names[:10])
            )
    elif strategy == "linear_probe":
        unexpected = [
            name for name in trainable_names if not is_head_parameter(name)
        ]
        if unexpected:
            raise RuntimeError(
                "linear_probe made non-head tensors trainable: "
                + ", ".join(unexpected[:10])
            )
        if not frozen_names:
            raise RuntimeError("linear_probe did not freeze any backbone parameters")
    elif strategy == "lora":
        validate_lora_audit(model, trainable_names, lora_targets)

    return {
        "trainable_parameters": count_trainable_parameters(model),
        "frozen_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if not parameter.requires_grad
        ),
        "trainable_parameter_tensors": trainable_names,
    }


def validate_lora_audit(
    model: nn.Module,
    trainable_names: list[str],
    lora_targets: list[str],
) -> None:
    """Require LoRA adapters to train while their original linear layers stay frozen."""
    if not lora_targets:
        raise RuntimeError("lora did not replace any backbone linear layers")

    unexpected = [
        name
        for name in trainable_names
        if "lora_" not in name and not is_head_parameter(name)
    ]
    if unexpected:
        raise RuntimeError(
            "lora made unexpected backbone tensors trainable: "
            + ", ".join(unexpected[:10])
        )

    for target_name in lora_targets:
        adapter = find_module(model, target_name)
        if not isinstance(adapter, LoRALinear):
            raise RuntimeError(f"LoRA target was not replaced: {target_name}")
        if any(parameter.requires_grad for parameter in adapter.base.parameters()):
            raise RuntimeError(f"LoRA base layer is not frozen: {target_name}")
        if not all(parameter.requires_grad for parameter in adapter.lora_a.parameters()):
            raise RuntimeError(f"LoRA A projection is frozen: {target_name}")
        if not all(parameter.requires_grad for parameter in adapter.lora_b.parameters()):
            raise RuntimeError(f"LoRA B projection is frozen: {target_name}")


def find_module(model: nn.Module, dotted_name: str) -> nn.Module:
    """Resolve one recorded module name for post-configuration auditing."""
    module = model
    for part in dotted_name.split("."):
        module = getattr(module, part)
    return module


def build_optimization(model: nn.Module, config: dict, steps_per_epoch: int) -> Optimization:
    """Build the optimizer and optional scheduler after fine-tuning is configured."""
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, config, steps_per_epoch)
    return Optimization(optimizer=optimizer, scheduler=scheduler)


def make_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Create the configured optimizer over trainable parameters only."""
    settings = config["training"]
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    common_arguments = {
        "lr": float(settings["learning_rate"]),
        "weight_decay": float(settings["weight_decay"]),
    }
    optimizer_name = settings["optimizer"].lower()

    if optimizer_name == "adamw":
        return torch.optim.AdamW(parameters, **common_arguments)
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, **common_arguments)
    if optimizer_name == "sgd":
        return torch.optim.SGD(parameters, momentum=0.9, **common_arguments)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def make_scheduler(optimizer, config: dict, steps_per_epoch: int):
    """Create the configured epoch- or batch-stepped learning-rate schedule."""
    settings = config["training"]
    scheduler_name = str(settings["scheduler"]).lower()
    if scheduler_name in {"none", "null"}:
        return None

    scheduler_step = settings.get("scheduler_step")
    steps = steps_per_epoch if scheduler_step == "batch" else 1
    total_steps = int(settings["epochs"]) * steps
    minimum_learning_rate = float(settings.get("minimum_learning_rate") or 0.0)

    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(total_steps, 1),
            eta_min=minimum_learning_rate,
        )
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def step_epoch_scheduler(optimization: Optimization, config: dict) -> None:
    """Advance schedules configured to change once after each epoch."""
    if optimization.scheduler is None:
        return
    if config["training"].get("scheduler_step") == "epoch":
        optimization.scheduler.step()


def train_one_epoch(
    context: EpochContext,
    loader,
    optimization: Optimization,
) -> dict[str, float]:
    """Update model parameters for one complete pass over the training split."""
    return _run_split(context, loader, optimization=optimization)


def evaluate_one_split(context: EpochContext, loader) -> dict[str, float]:
    """Measure one complete validation or test split without updating the model."""
    return _run_split(context, loader, optimization=None)


def _run_split(
    context: EpochContext,
    loader,
    optimization: Optimization | None,
) -> dict[str, float]:
    """Share batch mechanics after the caller has named training or evaluation."""
    training = optimization is not None
    context.model.train(training)

    losses = []
    labels = []
    scores = []
    use_mixed_precision = (
        bool(context.config["training"].get("mixed_precision", False))
        and context.device.type == "cuda"
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_mixed_precision)

    for batch in loader:
        inputs, targets = move_batch_to_device(batch, context)
        if training:
            optimization.optimizer.zero_grad(set_to_none=True)

        with gradient_and_precision_context(training, use_mixed_precision, context.device):
            logits = context.model(inputs)
            normalized_logits, loss = prepare_logits_and_loss(
                logits,
                targets,
                context.config["training"]["loss"],
            )

        if training:
            update_model(loss, context.model, optimization, context.config, scaler)

        losses.append(float(loss.detach()))
        labels.extend(targets.detach().cpu().reshape(-1).tolist())
        probabilities = probabilities_from_logits(
            normalized_logits.detach(),
            context.config["training"]["loss"],
        )
        scores.extend(probabilities.cpu().tolist())

    metrics = binary_metrics(
        labels,
        scores,
        float(context.config["evaluation"]["classification_threshold"]),
    )
    metrics["loss"] = float(np.mean(losses))
    return metrics


def move_batch_to_device(batch, context: EpochContext):
    """Apply the study case and move one EEG-label batch to the selected device."""
    inputs = context.transform(batch[0]).to(context.device, non_blocking=True)
    targets = batch[1].to(context.device, non_blocking=True)
    return inputs, targets


@contextmanager
def gradient_and_precision_context(
    training: bool,
    use_mixed_precision: bool,
    device: torch.device,
):
    """Enable gradients only for training and autocast only for CUDA AMP."""
    precision_context = (
        torch.amp.autocast("cuda", enabled=use_mixed_precision)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.set_grad_enabled(training), precision_context:
        yield


def update_model(
    loss,
    model: nn.Module,
    optimization: Optimization,
    config: dict,
    scaler,
) -> None:
    """Backpropagate, clip optional gradients, and advance batch-level state."""
    scaler.scale(loss).backward()

    clip_norm = config["training"].get("gradient_clip_norm")
    if clip_norm is not None:
        scaler.unscale_(optimization.optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), float(clip_norm))

    scaler.step(optimization.optimizer)
    scaler.update()
    if (
        optimization.scheduler is not None
        and config["training"].get("scheduler_step") == "batch"
    ):
        optimization.scheduler.step()


def prepare_logits_and_loss(logits, labels, loss_name: str):
    """Normalize two-logit and one-logit models into their configured loss."""
    labels = labels.to(logits.device)
    if loss_name == "cross_entropy":
        normalized = logits.reshape(labels.shape[0], -1)
        if normalized.shape[1] != 2:
            raise ValueError(
                f"cross_entropy requires [B,2] logits; got {list(normalized.shape)}"
            )
        loss = nn.functional.cross_entropy(normalized, labels.long())
        return normalized, loss

    normalized = logits.reshape(-1)
    if normalized.shape[0] != labels.shape[0]:
        raise ValueError(
            "bce_with_logits requires one logit per sample; "
            f"got {list(normalized.shape)}"
        )
    loss = nn.functional.binary_cross_entropy_with_logits(normalized, labels.float())
    return normalized, loss


def probabilities_from_logits(logits, loss_name: str) -> torch.Tensor:
    """Convert either output convention into class-one probabilities."""
    if loss_name == "cross_entropy":
        return torch.softmax(logits, dim=1)[:, 1]
    return torch.sigmoid(logits)


def binary_metrics(
    labels: list[float],
    scores: list[float],
    threshold: float,
) -> dict[str, float]:
    """Calculate balanced accuracy, AUROC, and AUPRC consistently."""
    try:
        from sklearn.metrics import (
            average_precision_score,
            balanced_accuracy_score,
            roc_auc_score,
        )
    except ImportError as exc:
        raise RuntimeError("Install scikit-learn to calculate evaluation metrics.") from exc

    truth = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(scores, dtype=np.float64)
    predictions = (probabilities >= threshold).astype(np.int64)
    if np.unique(truth).size < 2:
        raise ValueError("AUROC and balanced accuracy require both classes in this split")

    return {
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "auroc": float(roc_auc_score(truth, probabilities)),
        "auprc": float(average_precision_score(truth, probabilities)),
    }


def format_epoch_record(
    epoch: int,
    train_metrics: dict,
    validation_metrics: dict,
) -> dict:
    """Name train and validation values explicitly in the saved history."""
    record = {"epoch": epoch}
    record.update({f"train_{name}": value for name, value in train_metrics.items()})
    record.update(
        {f"validation_{name}": value for name, value in validation_metrics.items()}
    )
    return record


def save_selected_checkpoint(
    path: Path,
    model: nn.Module,
    epoch: int,
    validation_metrics: dict,
) -> None:
    """Save model state only when validation performance establishes a new best."""
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": epoch,
            "validation": validation_metrics,
        },
        path,
    )


def save_json(path: Path, value: Any) -> None:
    """Write an indented, portable reproducibility artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
