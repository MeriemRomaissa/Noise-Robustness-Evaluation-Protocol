import argparse
import datetime
import os
import random
import warnings

#newly added codes
import sys
#newly added codes
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')

#newly added codes
try:
    from Datasets import seedv_dataset as package_seedv_dataset
    _PACKAGE_DATASETS_IMPORT_ERROR = None
except ImportError as exc:
    package_seedv_dataset = None
    _PACKAGE_DATASETS_IMPORT_ERROR = exc

#newly added codes
try:
    from datasets import (faced_dataset, seedv_dataset, shu_dataset, isruc1_dataset, isruc3_dataset,
                          chb_dataset, speech_dataset, stress_dataset, tuev_dataset, tuab_dataset)
    _DATASETS_IMPORT_ERROR = None
except ImportError as exc:
    faced_dataset = seedv_dataset = shu_dataset = isruc1_dataset = isruc3_dataset = None
    chb_dataset = speech_dataset = stress_dataset = tuev_dataset = tuab_dataset = None
    _DATASETS_IMPORT_ERROR = exc

from Downstream.finetune_trainer import Trainer


#newly added codes
class CodeBrainTUABFlattenedForward(nn.Module):
    """Keep CodeBrain TUAB architecture but flatten backbone features for its head."""
    def __init__(self, author_model):
        super().__init__()
        self.author_model = author_model

    def forward(self, x):
        batch_size, channels, patches, patch_size = x.shape
        features = self.author_model.backbone(x)
        flattened = features.contiguous().view(batch_size, channels * patches * patch_size)
        out = self.author_model.classifier(flattened)
        return out.contiguous().view(batch_size)


#newly added codes
def _load_yaml_config(config_path):
    """Load optional Benchmark YAML defaults without replacing CLI overrides."""
    if not config_path:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("--config requires PyYAML to be installed") from exc

    path = Path(config_path).expanduser()
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


#newly added codes
def _first_seed(value):
    """Original CodeBrain trains one seed per process; YAML may list many."""
    if isinstance(value, (list, tuple)):
        return int(value[0]) if value else 42
    return int(value)


#newly added codes
def _apply_tuab_mode_defaults(config):
    """Mirror Benchmark subset/full study-case limits for this native script."""
    study_case = config.get("study_case", {})
    data = dict(config.get("data", {}))
    mode = study_case.get("tuab_mode")
    if mode == "subset_tuab":
        data.update({
            "train_samples": 8192,
            "validation_samples": 2048,
            "test_samples": 2048,
        })
    elif mode == "full_tuab":
        data.update({
            "train_samples": None,
            "validation_samples": None,
            "test_samples": None,
        })
    return data


#newly added codes
def _config_defaults(config):
    """Translate Benchmark YAML fields to native CodeBrain argparse defaults."""
    defaults = {}
    paths = config.get("paths", {})
    data = _apply_tuab_mode_defaults(config)
    loader = config.get("loader", {})
    training = config.get("training", {})
    fixed_recipe = config.get("fixed_recipe", {})
    fixed_model = fixed_recipe.get("model", {})
    fixed_training = fixed_recipe.get("training", {})

    if paths.get("original_data"):
        defaults["datasets_dir"] = paths["original_data"]
    if paths.get("h5_file"):
        defaults["h5_file"] = paths["h5_file"]
    if paths.get("split_index"):
        defaults["split_index"] = paths["split_index"]
    if paths.get("checkpoint"):
        defaults["foundation_dir"] = paths["checkpoint"]
    if paths.get("output"):
        defaults["model_dir"] = str(Path(paths["output"]) / "checkpoints")
        defaults["log_dir"] = str(Path(paths["output"]) / "logs")

    if data.get("source"):
        defaults["data_source"] = data["source"]
        #newly added codes
        if data["source"] == "tuab_unified60":
            defaults["downstream_dataset"] = "TUAB"
    if "train_samples" in data:
        defaults["train_samples"] = data["train_samples"]
    if "validation_samples" in data:
        defaults["validation_samples"] = data["validation_samples"]
    if "test_samples" in data:
        defaults["test_samples"] = data["test_samples"]

    if "batch_size" in loader:
        defaults["batch_size"] = int(loader["batch_size"])
    if "num_workers" in loader:
        defaults["num_workers"] = int(loader["num_workers"])
    if "pin_memory" in loader:
        defaults["pin_memory"] = bool(loader["pin_memory"])
    if "shuffle_train" in loader:
        defaults["shuffle_train"] = bool(loader["shuffle_train"])
    if "drop_last_train" in loader:
        defaults["drop_last_train"] = bool(loader["drop_last_train"])

    if "epochs" in training:
        defaults["epochs"] = int(training["epochs"])
    if "learning_rate" in training:
        defaults["lr"] = float(training["learning_rate"])
    elif "lr" in training:
        defaults["lr"] = float(training["lr"])
    if "seed" in training:
        defaults["seed"] = _first_seed(training["seed"])
    elif "seeds" in training:
        defaults["seed"] = _first_seed(training["seeds"])

    for key in ("downstream_dataset", "num_of_classes", "dropout", "n_layer",
                "use_pretrained_weights", "codebook_size_t", "codebook_size_f",
                "codebook_dim"):
        if key in fixed_model:
            defaults[key] = fixed_model[key]
    for key in ("weight_decay", "optimizer", "clip_value", "label_smoothing",
                "frozen", "multi_lr"):
        if key in fixed_training:
            defaults[key] = fixed_training[key]

    return defaults


#newly added codes
def _optional_int(value):
    """Allow YAML null/CLI none to mean no split cap."""
    if value is None or str(value).lower() == "none":
        return None
    return int(value)


#newly added codes
def _str_to_bool(value):
    """Parse Benchmark boolean overrides while accepting original bool values."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


#newly added codes
def _find_benchmark_root():
    """Find the Benchmark folder from this nested original script."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "Benchmark"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("could not locate Benchmark folder from CodeBrain finetune_main.py")


#newly added codes
def _add_benchmark_paths_to_syspath():
    """Let this native script reuse Benchmark loaders without moving files."""
    benchmark_root = _find_benchmark_root()
    for relative_path in ("Loader",):
        path = benchmark_root / relative_path
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


#newly added codes
def _benchmark_h5_loader_config(params, split):
    """Build the flat config expected by Benchmark/Loader/loader_common.py."""
    max_samples = {
        "train": params.train_samples,
        "val": params.validation_samples,
        "test": params.test_samples,
    }[split]
    return {
        "h5_path": params.h5_file,
        "split_index_path": params.split_index,
        "max_samples": max_samples,
        "seed": params.seed,
        "batch_size": params.batch_size,
        "num_workers": params.num_workers,
        "pin_memory": params.pin_memory,
        "shuffle": params.shuffle_train if split == "train" else False,
        "drop_last": params.drop_last_train if split == "train" else False,
    }


#newly added codes
def load_unified60_tuab_dataset(params):
    """Read unified60 H5 rows through the Benchmark CodeBrain loader."""
    _add_benchmark_paths_to_syspath()
    from loader_codebrain import build_unified60_loader

    if not params.h5_file:
        raise ValueError("--h5_file is required when --data_source tuab_unified60")
    if not params.split_index:
        raise ValueError("--split_index is required when --data_source tuab_unified60")
    data_loader = {
        "train": build_unified60_loader(_benchmark_h5_loader_config(params, "train"), "train"),
        "val": build_unified60_loader(_benchmark_h5_loader_config(params, "val"), "val"),
        "test": build_unified60_loader(_benchmark_h5_loader_config(params, "test"), "test"),
    }
    print(
        "Benchmark unified60 H5 dataset:",
        f"train={len(data_loader['train'].dataset)}",
        f"val={len(data_loader['val'].dataset)}",
        f"test={len(data_loader['test'].dataset)}",
        flush=True,
    )
    return data_loader


#newly added codes
def _require_dataset(module, dataset_name):
    """Raise a clear error when this repo copy lacks original dataset modules."""
    if module is None:
        raise ImportError(
            f"CodeBrain original dataset module for {dataset_name} is unavailable. "
            "Use --data_source tuab_unified60 for Benchmark H5 loading or restore "
            f"the original datasets package. Original import error: {_DATASETS_IMPORT_ERROR}; "
            f"package import error: {_PACKAGE_DATASETS_IMPORT_ERROR}"
        )
    return module


#newly added codes
def _load_model_module(module_name):
    """Import only the model file needed by the selected downstream branch."""
    import importlib

    try:
        return importlib.import_module(f"models.{module_name}")
    except ImportError:
        return importlib.import_module(f"Models.{module_name}")


def main():
    #newly added codes
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', type=str, default='',
                               help='Optional Benchmark YAML. YAML values become defaults; CLI args override them.')
    config_args, remaining_args = config_parser.parse_known_args()
    config_defaults = _config_defaults(_load_yaml_config(config_args.config))

    parser = argparse.ArgumentParser(description='CodeBrain Downstream')
    #newly added codes
    parser.add_argument('--config', type=str, default=config_args.config,
                        help='Optional Benchmark YAML. YAML values become defaults; CLI args override them.')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 0)')
    parser.add_argument('--cuda', type=int, default=4, help='cuda number (default: 1)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay (default: 1e-2)')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer (AdamW, SGD)')
    parser.add_argument('--clip_value', type=float, default=5, help='clip_value')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--n_layer', type=int, default=8, help='n_layer')

    parser.add_argument('--downstream_dataset', type=str, default='SEED-V',
                        help='[FACED, SEED-V, SHU-MI, ISRUC_S1, ISRUC_S3'
                             'CHB-MIT, BCIC2020-3, MentalArithmetic, TUEV, TUAB]')
    parser.add_argument('--datasets_dir', type=str,
                        default='',
                        help='datasets_dir')
    parser.add_argument('--num_of_classes', type=int, default=5, help='number of classes')
    parser.add_argument('--model_dir', type=str,
                        default='',
                        help='model_dir')
    parser.add_argument('--log_dir', type=str,
                        default='',
                        help='log_dir')

    parser.add_argument('--num_workers', type=int, default=16, help='num_workers')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='label_smoothing')
    parser.add_argument('--frozen', type=_str_to_bool,
                        default=False, help='frozen')
    parser.add_argument('--use_pretrained_weights', type=_str_to_bool,
                        default=True, help='use_pretrained_weights')
    parser.add_argument('--foundation_dir', type=str,
                        default='',
                        help='foundation_dir')

    parser.add_argument('--codebook_size_t', default=4096, type=int,
                        help='number of temporal codebook (default: 4096)')
    parser.add_argument('--codebook_size_f', default=4096, type=int,
                        help='number of frequency codebook (default: 4096)')
    parser.add_argument('--codebook_dim', default=32, type=int,
                        help='dimention of codebook (default: 32)')
    #newly added codes
    parser.add_argument('--multi_lr', type=_str_to_bool, default=False,
                        help='multi_lr used by the original trainer SGD path.')
    #newly added codes
    parser.add_argument('--data_source', type=str, default='original',
                        choices=['original', 'tuab_unified60'],
                        help='original uses CodeBrain processed data; tuab_unified60 uses Benchmark H5 loaders.')
    #newly added codes
    parser.add_argument('--h5_file', type=str, default='',
                        help='Benchmark unified60 H5 path when data_source=tuab_unified60.')
    #newly added codes
    parser.add_argument('--split_index', type=str, default='',
                        help='Benchmark split CSV path when data_source=tuab_unified60.')
    #newly added codes
    parser.add_argument('--tuab_mode', type=str, default='subset_tuab',
                        choices=['subset_tuab', 'full_tuab'],
                        help='Benchmark TUAB study case; controls split sample caps.')
    #newly added codes
    parser.add_argument('--train_samples', type=_optional_int, default=None,
                        help='Max unified H5 train rows; none means full split.')
    #newly added codes
    parser.add_argument('--validation_samples', type=_optional_int, default=None,
                        help='Max unified H5 validation rows; none means full split.')
    #newly added codes
    parser.add_argument('--test_samples', type=_optional_int, default=None,
                        help='Max unified H5 test rows; none means full split.')
    #newly added codes
    parser.add_argument('--shuffle_train', type=_str_to_bool, default=True,
                        help='Shuffle train split.')
    #newly added codes
    parser.add_argument('--drop_last_train', type=_str_to_bool, default=True,
                        help='Drop incomplete train batches.')
    #newly added codes
    parser.add_argument('--pin_memory', type=_str_to_bool, default=True,
                        help='Pin DataLoader memory.')

    #newly added codes
    parser.set_defaults(**config_defaults)
    #newly added codes
    params = parser.parse_args(remaining_args)
    params.model_dir = os.path.join(params.model_dir, params.downstream_dataset) + '/'
    params.log_dir = os.path.join(params.log_dir, params.downstream_dataset) + '/'
    os.makedirs(params.model_dir, exist_ok=True)
    os.makedirs(params.log_dir, exist_ok=True)
    current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    params.file_name = str(params.log_dir) + str(current_time) + "_" + str(params.cuda) + ".txt"
    print(params)
    with open(params.file_name, "a") as file:
        file.write(str(params) + "\n")

    setup_seed(params.seed)
    #newly added codes
    if torch.cuda.is_available():
        torch.cuda.set_device(params.cuda)
    print('The downstream dataset is {}'.format(params.downstream_dataset))
    with open(params.file_name, "a") as file:
        file.write('The downstream dataset is {}'.format(params.downstream_dataset))
        file.write("\n")
    if params.downstream_dataset == 'SEED-V':
        params.num_of_classes = 5
        dataset_module = _require_dataset(package_seedv_dataset or seedv_dataset, 'SEED-V')
        load_dataset = dataset_module.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model_for_seedv = _load_model_module("model_for_seedv")
        model = model_for_seedv.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'FACED':
        params.num_of_classes = 9
        dataset_module = _require_dataset(faced_dataset, 'FACED')
        load_dataset = dataset_module.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model_for_faced = _load_model_module("model_for_faced")
        model = model_for_faced.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'SHU-MI':
        params.num_of_classes = 2
        dataset_module = _require_dataset(shu_dataset, 'SHU-MI')
        load_dataset = dataset_module.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model_for_shu = _load_model_module("model_for_shu")
        model = model_for_shu.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'ISRUC_S1':
        params.num_of_classes = 5
        dataset_module = _require_dataset(isruc1_dataset, 'ISRUC_S1')
        load_dataset = dataset_module.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model_for_isruc = _load_model_module("model_for_isruc")
        model = model_for_isruc.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'ISRUC_S3':
        params.num_of_classes = 5
        dataset_module = _require_dataset(isruc3_dataset, 'ISRUC_S3')
        load_dataset = dataset_module.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model_for_isruc = _load_model_module("model_for_isruc")
        model = model_for_isruc.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'CHB-MIT':
        params.num_of_classes = 2
        dataset_module = _require_dataset(chb_dataset, 'CHB-MIT')
        load_dataset = dataset_module.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model_for_chb = _load_model_module("model_for_chb")
        model = model_for_chb.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'BCIC2020-3':
        params.num_of_classes = 5
        dataset_module = _require_dataset(speech_dataset, 'BCIC2020-3')
        load_dataset = dataset_module.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model_for_speech = _load_model_module("model_for_speech")
        model = model_for_speech.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'MentalArithmetic':
        params.num_of_classes = 2
        dataset_module = _require_dataset(stress_dataset, 'MentalArithmetic')
        load_dataset = dataset_module.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model_for_stress = _load_model_module("model_for_stress")
        model = model_for_stress.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'TUEV':
        params.num_of_classes = 6
        dataset_module = _require_dataset(tuev_dataset, 'TUEV')
        load_dataset = dataset_module.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model_for_tuev = _load_model_module("model_for_tuev")
        model = model_for_tuev.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'TUAB':
        params.num_of_classes = 2
        #newly added codes
        if params.data_source == 'tuab_unified60':
            data_loader = load_unified60_tuab_dataset(params)
        else:
            dataset_module = _require_dataset(tuab_dataset, 'TUAB')
            load_dataset = dataset_module.LoadDataset(params)
            data_loader = load_dataset.get_data_loader()
        #newly added codes
        model_for_tuab = _load_model_module("model_for_tuab")
        model = CodeBrainTUABFlattenedForward(model_for_tuab.Model(params))
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()


def setup_seed(seed):
    torch.manual_seed(seed)
    #newly added codes
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()
