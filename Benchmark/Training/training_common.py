"""Run one auditable training protocol across all six EEG foundation models.

Model-specific files provide model construction and checkpoint-name rules. This
module owns the decisions that must remain identical across the benchmark:
configuration validation, deterministic seeds, loader construction, mandatory
pretrained initialization, fine-tuning, validation-based model selection,
LaBraM-compatible output logging, final test evaluation, and cross-seed
reporting.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import random
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
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
class SplitContext:
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
    contexts: dict[str, SplitContext]
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


COMMON_BACKBONE_CHECKPOINT_PREFIXES = (
    "module.backbone.",
    "model.backbone.",
    "backbone.",
    "module.",
)
BIOT_ENCODER_CHECKPOINT_PREFIXES = (
    "module.biot.",
    "model.biot.",
    "biot.",
    "module.",
    "model.",
)


def run_experiments(spec: ModelSpec, default_config: str) -> None:
    """Run every configured seed and summarize final test performance."""
    arguments = parse_arguments(default_config)
    config = load_config(arguments.config, arguments)
    if arguments.dry_run:
        print(json.dumps(config, indent=2, default=str))
        return
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

    fine_tuning_report = configure_fine_tuning(model, config, spec.name)
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
    """Train, log train/val/test each epoch, and select using validation only."""
    config = experiment.config
    selection_metric = config["evaluation"]["selection_metric"]
    output_format = load_evaluation_output_format()
    best_value = -math.inf
    best_epoch = 0
    history = []
    checkpoint_path = experiment.run_dir / "checkpoint-best.pth"
    n_parameters = output_format.trainable_parameter_count(experiment.model)

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
        test_metrics = evaluate_one_split(
            experiment.contexts["test"],
            experiment.loaders["test"],
        )
        step_epoch_scheduler(experiment.optimization, config)

        epoch_record = format_epoch_record(
            epoch,
            train_metrics,
            validation_metrics,
            test_metrics,
        )
        history.append(epoch_record)
        log_record = output_format.epoch_log_record(
            epoch,
            train_metrics,
            validation_metrics,
            test_metrics,
            n_parameters,
        )
        output_format.record_log_txt(experiment.run_dir, log_record)
        output_format.save_epoch_checkpoints(
            experiment.run_dir,
            experiment.model,
            experiment.optimization.optimizer,
            experiment.optimization.scheduler,
            epoch,
            log_record,
            config,
        )

        current_value = validation_metrics[selection_metric]
        if current_value > best_value:
            best_value = current_value
            best_epoch = epoch
            checkpoint_path = output_format.save_best_checkpoint(
                experiment.run_dir,
                experiment.model,
                experiment.optimization.optimizer,
                experiment.optimization.scheduler,
                epoch,
                log_record,
                config,
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
    """Expose benchmark controls plus machine-specific path overrides."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--model-repo")
    parser.add_argument("--checkpoint")
    parser.add_argument("--h5-file")
    parser.add_argument("--split-index")
    parser.add_argument("--original-data")
    parser.add_argument("--output")
    parser.add_argument("--data-source", choices=["tuab_unified60", "original"])
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--validation-samples", type=int)
    parser.add_argument("--test-samples", type=int)
    parser.add_argument(
        "--tuab-mode",
        choices=["subset_tuab", "full_tuab"],
        default=None,
        help="TUAB dataset study case. Overrides config study_case.tuab_mode.",
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--fine-tuning-strategy",
        choices=["full_finetune", "freeze_backbone", "lora"],
        help="Override config fine_tuning.strategy.",
    )
    parser.add_argument(
        "--evaluation-metrics",
        nargs="+",
        choices=["balanced_accuracy", "roc_auc", "pr_auc", "accuracy"],
    )
    parser.add_argument("--classification-threshold", type=float)
    parser.add_argument(
        "--selection-metric",
        choices=["balanced_accuracy", "roc_auc", "pr_auc", "accuracy"],
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(path: str | Path, arguments: argparse.Namespace | None = None) -> dict:
    """Read YAML, merge locked recipe defaults, and validate before setup."""
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
    validate_reader_facing_config(config)
    config = merge_locked_recipe(config)
    normalize_config_metric_names(config)
    if arguments is not None:
        apply_argument_overrides(config, arguments)
    apply_tuab_mode_case(config)
    validate_config(config)
    return config


def validate_reader_facing_config(config: dict) -> None:
    """Reject model-recipe knobs outside the locked recipe section."""
    allowed_sections = {
        "paths",
        "data",
        "loader",
        "training",
        "fine_tuning",
        "evaluation",
        "study_case",
        "fixed_recipe",
        "_config_path",
    }
    extra_sections = set(config) - allowed_sections
    if extra_sections:
        raise ValueError(
            "Only editable benchmark settings may appear at top level. Move locked "
            f"recipe sections under fixed_recipe: {sorted(extra_sections)}"
        )

    allowed_keys = {
        "paths": {
            "h5_file", "split_index", "original_data", "output",
        },
        "data": {
            "source", "train_samples", "validation_samples", "test_samples",
        },
        "loader": {"batch_size"},
        "training": {"seeds", "epochs"},
        "fine_tuning": {"strategy"},
        "evaluation": {
            "selection_metric", "classification_threshold", "report_metrics",
        },
        "study_case": {"tuab_mode", "channel_mode"},
    }
    for section, keys in allowed_keys.items():
        values = config.get(section, {})
        if not isinstance(values, dict):
            continue
        extra_keys = set(values) - keys
        if extra_keys:
            raise ValueError(
                f"{section} contains locked recipe fields {sorted(extra_keys)}. "
                "Keep model-specific settings under fixed_recipe."
            )


def merge_locked_recipe(config: dict) -> dict:
    """Copy fixed model-recipe values into the runtime config."""
    merged = copy.deepcopy(config)
    recipe = merged.get("fixed_recipe", {})
    if recipe is None:
        recipe = {}
    if not isinstance(recipe, dict):
        raise ValueError("fixed_recipe must be a YAML mapping")

    for section in (
        "paths",
        "data",
        "loader",
        "training",
        "fine_tuning",
        "model",
        "study_case",
    ):
        recipe_values = recipe.get(section)
        if recipe_values is None:
            continue
        if not isinstance(recipe_values, dict):
            raise ValueError(f"fixed_recipe.{section} must be a YAML mapping")
        section_values = merged.setdefault(section, {})
        if not isinstance(section_values, dict):
            raise ValueError(f"{section} must be a YAML mapping")
        for key, value in recipe_values.items():
            section_values.setdefault(key, value)
    return merged


def apply_argument_overrides(config: dict, arguments: argparse.Namespace) -> None:
    """Apply CLI overrides for approved controls and local path locations."""
    mappings = {
        "model_repo": ("paths", "model_repo"),
        "checkpoint": ("paths", "checkpoint"),
        "h5_file": ("paths", "h5_file"),
        "split_index": ("paths", "split_index"),
        "original_data": ("paths", "original_data"),
        "output": ("paths", "output"),
        "data_source": ("data", "source"),
        "train_samples": ("data", "train_samples"),
        "validation_samples": ("data", "validation_samples"),
        "test_samples": ("data", "test_samples"),
        "tuab_mode": ("study_case", "tuab_mode"),
        "seeds": ("training", "seeds"),
        "epochs": ("training", "epochs"),
        "batch_size": ("loader", "batch_size"),
        "fine_tuning_strategy": ("fine_tuning", "strategy"),
        "evaluation_metrics": ("evaluation", "report_metrics"),
        "classification_threshold": ("evaluation", "classification_threshold"),
        "selection_metric": ("evaluation", "selection_metric"),
    }
    for argument_name, (section, key) in mappings.items():
        value = getattr(arguments, argument_name)
        if value is not None:
            config.setdefault(section, {})[key] = normalize_metric_names(value)


def normalize_metric_names(value):
    """Accept reader-facing ROC/PR names and use stable internal names."""
    aliases = {"roc_auc": "auroc", "pr_auc": "auprc"}
    if isinstance(value, list):
        return [aliases.get(item, item) for item in value]
    return aliases.get(value, value)


def normalize_config_metric_names(config: dict) -> None:
    """Normalize metric spelling in the loaded YAML in-place."""
    evaluation = config.get("evaluation", {})
    if "selection_metric" in evaluation:
        evaluation["selection_metric"] = normalize_metric_names(
            evaluation["selection_metric"]
        )
    if "report_metrics" in evaluation:
        evaluation["report_metrics"] = normalize_metric_names(
            evaluation["report_metrics"]
        )


def validate_config(config: dict) -> None:
    """Validate required values, controlled choices, and numeric ranges."""
    required = {
        "paths": [
            "model_repo", "checkpoint", "h5_file", "split_index",
            "original_data", "output",
        ],
        "data": ["source"],
        "study_case": ["tuab_mode"],
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


def apply_tuab_mode_case(config: dict) -> None:
    """Let the TUAB study case own subset-vs-full sample-limit decisions."""
    study_directory = Path(__file__).resolve().parents[1] / "StudyCase" / "Dataset"
    if str(study_directory) not in sys.path:
        sys.path.insert(0, str(study_directory))

    tuab_case = importlib.import_module("TUAB_subset_vs_fullset")
    tuab_case.apply_tuab_mode(config)


def validate_config_choices(config: dict) -> None:
    """Reject misspelled experiment choices and show the accepted values."""
    choices = {
        "data.source": (config["data"]["source"], {"tuab_unified60", "original"}),
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
            {"full_finetune", "freeze_backbone", "lora"},
        ),
        "evaluation.selection_metric": (
            config["evaluation"]["selection_metric"],
            {"balanced_accuracy", "auroc", "auprc", "accuracy"},
        ),
    }

    for name, (value, allowed) in choices.items():
        if value not in allowed:
            raise ValueError(f"{name} must be one of {sorted(allowed)}, got {value!r}")

    supported_metrics = {"balanced_accuracy", "auroc", "auprc", "accuracy"}
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

    for key in ("train_samples", "validation_samples", "test_samples"):
        value = config["data"].get(key)
        if value is not None and int(value) < 1:
            raise ValueError(f"data.{key} must be positive or null for full_tuab")


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


def strip_first_matching_prefix(source_key: str, prefixes: tuple[str, ...]) -> str:
    """Remove the first known author-wrapper prefix from a checkpoint key."""
    for prefix in prefixes:
        if source_key.startswith(prefix):
            return source_key.removeprefix(prefix)
    return source_key


def make_checkpoint_key_mapper(prefixes: tuple[str, ...]) -> Callable[[str], str]:
    """Create a checkpoint key mapper from a shared prefix list."""
    return lambda source_key: strip_first_matching_prefix(source_key, prefixes)


def load_prefixed_checkpoint(
    target: nn.Module,
    checkpoint_path: Path,
    prefixes: tuple[str, ...],
) -> dict:
    """Load an EEG-FM checkpoint through an internal namespace bridge.

    The checkpoint path comes from the model's fixed EEG-FM dependency. Prefix
    stripping only maps author checkpoint names onto Benchmark wrapper names; it
    is not a user-facing recipe setting.
    """
    return load_matching_weights(
        target=target,
        checkpoint_path=checkpoint_path,
        key_mapper=make_checkpoint_key_mapper(prefixes),
    )


def namespace_from_config(settings: dict, field_map: dict[str, Any]) -> SimpleNamespace:
    """Build the small argparse-like namespace expected by author model code."""
    values = {}
    for author_name, spec in field_map.items():
        if isinstance(spec, tuple):
            config_name, converter = spec
        else:
            config_name, converter = spec, lambda item: item
        values[author_name] = converter(settings[config_name])
    return SimpleNamespace(**values)


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
    if config["data"]["source"] == "tuab_unified60":
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

    mode = settings.get("channel_mode")
    if mode is None:
        return leave_eeg_unchanged
    if spec.study_case_module is None:
        raise ValueError(f"{spec.name} does not define a channel study case")

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
) -> dict[str, SplitContext]:
    """Bind each split to its model, device, configuration, and transform."""
    return {
        split: SplitContext(model, device, config, transform)
        for split, transform in transforms.items()
    }


def configure_fine_tuning(model: nn.Module, config: dict, model_name: str) -> dict:
    """Apply the config-selected strategy through the shared StudyCase module."""
    study_directory = Path(__file__).resolve().parents[1] / "StudyCase" / "Finetuning"
    if str(study_directory) not in sys.path:
        sys.path.insert(0, str(study_directory))

    strategies = importlib.import_module("finetuning_strategies")
    return strategies.apply_finetuning_strategy(model, config, model_name)


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
    context: SplitContext,
    loader,
    optimization: Optimization,
) -> dict[str, float]:
    """Update model parameters for one complete pass over the training split."""
    return _run_split(context, loader, optimization=optimization)


def evaluate_one_split(context: SplitContext, loader) -> dict[str, float]:
    """Measure one complete validation or test split without updating the model."""
    return _run_split(context, loader, optimization=None)


def _run_split(
    context: SplitContext,
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


def move_batch_to_device(batch, context: SplitContext):
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
    """Calculate thresholded and ranking metrics consistently."""
    try:
        from sklearn.metrics import (
            accuracy_score,
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
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "auroc": float(roc_auc_score(truth, probabilities)),
        "auprc": float(average_precision_score(truth, probabilities)),
    }


def format_epoch_record(
    epoch: int,
    train_metrics: dict,
    validation_metrics: dict,
    test_metrics: dict,
) -> dict:
    """Name train, validation, and test values explicitly in saved history."""
    record = {"epoch": epoch}
    record.update({f"train_{name}": value for name, value in train_metrics.items()})
    record.update(
        {f"validation_{name}": value for name, value in validation_metrics.items()}
    )
    record.update({f"test_{name}": value for name, value in test_metrics.items()})
    return record


def load_evaluation_output_format():
    """Import the output-format helpers without requiring package installation."""
    evaluation_directory = Path(__file__).resolve().parents[1] / "Evaluation"
    if str(evaluation_directory) not in sys.path:
        sys.path.insert(0, str(evaluation_directory))
    return importlib.import_module("output_format")


def save_json(path: Path, value: Any) -> None:
    """Write an indented, portable reproducibility artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
