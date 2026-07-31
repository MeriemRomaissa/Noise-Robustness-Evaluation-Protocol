import argparse
import random

#newly added codes
import sys
#newly added codes
from pathlib import Path

import numpy as np
import torch

#newly added codes
try:
    from datasets import faced_dataset, seedv_dataset, physio_dataset, shu_dataset, isruc_dataset, chb_dataset, \
        speech_dataset, mumtaz_dataset, seedvig_dataset, stress_dataset, tuev_dataset, tuab_dataset, bciciv2a_dataset, tusl_dataset
    from datasets import siena_dataset, hmc_dataset
    _DATASETS_IMPORT_ERROR = None
except ImportError as exc:
    faced_dataset = seedv_dataset = physio_dataset = shu_dataset = isruc_dataset = chb_dataset = None
    speech_dataset = mumtaz_dataset = seedvig_dataset = stress_dataset = tuev_dataset = tuab_dataset = None
    bciciv2a_dataset = tusl_dataset = siena_dataset = hmc_dataset = None
    _DATASETS_IMPORT_ERROR = exc

#newly added codes
Trainer = None


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
    """Original CSBrain trains one seed per process; YAML may list many."""
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
    """Translate Benchmark YAML fields to native CSBrain argparse defaults."""
    defaults = {}
    paths = config.get("paths", {})
    data = _apply_tuab_mode_defaults(config)
    study_case = config.get("study_case", {})
    loader = config.get("loader", {})
    training = config.get("training", {})

    if paths.get("original_data"):
        defaults["datasets_dir"] = paths["original_data"]
    if paths.get("h5_file"):
        defaults["h5_file"] = paths["h5_file"]
    if paths.get("split_index"):
        defaults["split_index"] = paths["split_index"]
    if paths.get("checkpoint"):
        defaults["foundation_dir"] = paths["checkpoint"]
        #newly added codes
        defaults["use_pretrained_weights"] = True
    if paths.get("output"):
        defaults["model_dir"] = paths["output"]

    if data.get("source"):
        defaults["data_source"] = data["source"]
        #newly added codes
        if data["source"] == "tuab_unified60":
            defaults["downstream_dataset"] = "TUAB"
    if "scale_mode" in data:
        defaults["scale_mode"] = data["scale_mode"]
    if "scale_mode" in study_case:
        defaults["scale_mode"] = study_case["scale_mode"]
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
    """Find the Benchmark folder from this original script."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "Benchmark"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("could not locate Benchmark folder from CSBrain finetune_main.py")


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
        "scale_mode": params.scale_mode,
    }


#newly added codes
def load_unified60_tuab_dataset(params):
    """Read unified60 H5 rows through the Benchmark CSBrain loader."""
    _add_benchmark_paths_to_syspath()
    from loader_csbrain import build_unified60_loader

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
            f"CSBrain original dataset module for {dataset_name} is unavailable. "
            "Use --data_source tuab_unified60 for Benchmark H5 loading or restore "
            f"the original datasets package. Original import error: {_DATASETS_IMPORT_ERROR}"
        )
    return module


#newly added codes
def _load_model_module(module_name):
    """Import only the model file needed by the selected downstream branch."""
    import importlib
    return importlib.import_module(f"models.{module_name}")


#newly added codes
def _load_trainer_class():
    """Delay trainer import so --help/config parsing does not load trainer-only deps."""
    global Trainer
    if Trainer is None:
        from finetune_trainer import Trainer as TrainerClass
        Trainer = TrainerClass
    return Trainer


def main():
    #newly added codes
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', type=str, default='',
                               help='Optional Benchmark YAML. YAML values become defaults; CLI args override them.')
    config_args, remaining_args = config_parser.parse_known_args()
    config_defaults = _config_defaults(_load_yaml_config(config_args.config))

    parser = argparse.ArgumentParser(description='Big model downstream')
    #newly added codes
    parser.add_argument('--config', type=str, default=config_args.config,
                        help='Optional Benchmark YAML. YAML values become defaults; CLI args override them.')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 0)') # 42
    parser.add_argument('--cuda', type=int, default=0, help='cuda number (default: 1)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='weight decay (default: 1e-2)')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer (AdamW, SGD)')
    parser.add_argument('--clip_value', type=float, default=1, help='clip_value')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')

    """############ Downstream dataset settings ############"""
    parser.add_argument('--downstream_dataset', type=str, default='FACED',
                        help='[FACED, SEED-V, PhysioNet-MI, SHU-MI, ISRUC, CHB-MIT, BCIC2020-3, Mumtaz2016, SEED-VIG, MentalArithmetic, TUEV, TUAB, BCIC-IV-2a]')
    parser.add_argument('--datasets_dir', type=str,
                        default='',
                        help='datasets_dir')
    parser.add_argument('--num_of_classes', type=int, default=9, help='number of classes')
    parser.add_argument('--model_dir', type=str, default='', help='model_dir')
    """############ Downstream dataset settings ############"""
    parser.add_argument('--num_workers', type=int, default=16, help='num_workers')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='label_smoothing')
    parser.add_argument('--multi_lr', type=_str_to_bool, default=False,
                        help='multi_lr')
    parser.add_argument('--frozen', type=_str_to_bool,
                        default=False, help='frozen')
    parser.add_argument('--use_pretrained_weights', action='store_true', help='Use pretrained weights')
    parser.add_argument('--foundation_dir', type=str, default='pth/CSBrain.pth', help='foundation_dir')
    parser.add_argument('--model', type=str, default='CSBrain', help='CBraMod CSBrain CSBrain_new CSBrain_I CSBrain_II')
    parser.add_argument('--use_CrossTemEmbed', type=_str_to_bool, default=False, help='CrossTemEmbedEEGLayer')
    parser.add_argument('--use_SmallerToken', type=_str_to_bool, default=False, help='SmallerToken->dataset.py')
    parser.add_argument('--CrossTemEmbed_kernel_sizes', type=str, default="[(1,), (3,), (5,),]")
    parser.add_argument('--use_CSBrainTF', action='store_true', default=False, help='use_CSBrainTF')
    parser.add_argument('--use_CSBrainTF_Tep_Spa', action='store_true', default=False, help='use_CSBrainTF_Tep_Spa')
    parser.add_argument('--use_CSBrainTF_Tep_Bra', action='store_true', default=False, help='use_CSBrainTF_Tep_Bra')
    parser.add_argument('--use_CSBrainTF_Tep_Bra_Tiny', action='store_true', default=False, help='use_CSBrainTF_Tep_Bra_Tiny')
    parser.add_argument('--use_CSBrainTF_Tep_Bra_Pal', action='store_true', default=False, help='use_CSBrainTF_Tep_Bra_Pal')
    parser.add_argument('--use_IntraBraEmbed', action='store_true', default=False, help='use_IntraBraEmbed')
    parser.add_argument('--n_layer', type=int, default=12, help='n_layer')
    parser.add_argument('--use_finetune_weights', type=_str_to_bool, default=False, help='use_finetune_weights')
    #newly added codes
    parser.add_argument('--data_source', type=str, default='original',
                        choices=['original', 'tuab_unified60'],
                        help='original uses CSBrain processed data; tuab_unified60 uses Benchmark H5 loaders.')
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
    parser.add_argument('--scale_mode', type=str, default='div100',
                        choices=['mul10000', 'mul1000', 'div100'],
                        help='Benchmark CSBrain amplitude condition.')
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
    print(params)

    setup_seed(params.seed)
    #newly added codes
    trainer_class = _load_trainer_class()
    #newly added codes
    if torch.cuda.is_available():
        torch.cuda.set_device(params.cuda)
    print('The downstream dataset is {}'.format(params.downstream_dataset))
    if params.downstream_dataset == 'FACED':
        dataset_module = _require_dataset(faced_dataset, 'FACED')
        load_dataset = dataset_module.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = _load_model_module("model_for_faced").Model(params)
        t = trainer_class(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'SEED-V':
        dataset_module = _require_dataset(seedv_dataset, 'SEED-V')
        load_dataset = dataset_module.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = _load_model_module("model_for_seedv").Model(params)
        t = trainer_class(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'PhysioNet-MI':
        dataset_module = _require_dataset(physio_dataset, 'PhysioNet-MI')
        load_dataset = dataset_module.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = _load_model_module("model_for_physio").Model(params)
        t = trainer_class(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'SHU-MI':
        dataset_module = _require_dataset(shu_dataset, 'SHU-MI')
        load_dataset = dataset_module.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = _load_model_module("model_for_shu").Model(params)
        t = trainer_class(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'ISRUC':
        dataset_module = _require_dataset(isruc_dataset, 'ISRUC')
        load_dataset = dataset_module.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = _load_model_module("model_for_isruc").Model(params)
        t = trainer_class(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'CHB-MIT':
        dataset_module = _require_dataset(chb_dataset, 'CHB-MIT')
        load_dataset = dataset_module.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = _load_model_module("model_for_chb").Model(params)
        t = trainer_class(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'BCIC2020-3':
        dataset_module = _require_dataset(speech_dataset, 'BCIC2020-3')
        load_dataset = dataset_module.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = _load_model_module("model_for_speech").Model(params)
        t = trainer_class(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'Mumtaz2016':
        dataset_module = _require_dataset(mumtaz_dataset, 'Mumtaz2016')
        load_dataset = dataset_module.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = _load_model_module("model_for_mumtaz").Model(params)
        t = trainer_class(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'SEED-VIG':
        dataset_module = _require_dataset(seedvig_dataset, 'SEED-VIG')
        load_dataset = dataset_module.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = _load_model_module("model_for_seedvig").Model(params)
        t = trainer_class(params, data_loader, model)
        t.train_for_regression()
    elif params.downstream_dataset == 'MentalArithmetic':
        dataset_module = _require_dataset(stress_dataset, 'MentalArithmetic')
        load_dataset = dataset_module.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = _load_model_module("model_for_stress").Model(params)
        t = trainer_class(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'TUEV':
        dataset_module = _require_dataset(tuev_dataset, 'TUEV')
        load_dataset = dataset_module.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = _load_model_module("model_for_tuev").Model(params)
        t = trainer_class(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'TUAB':
        #newly added codes
        if params.data_source == 'tuab_unified60':
            data_loader = load_unified60_tuab_dataset(params)
        else:
            dataset_module = _require_dataset(tuab_dataset, 'TUAB')
            load_dataset = dataset_module.LoadDataset(params)
            data_loader = load_dataset.get_data_loader()
        model = _load_model_module("model_for_tuab").Model(params)
        t = trainer_class(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'TUSL':
        dataset_module = _require_dataset(tusl_dataset, 'TUSL')
        data_loader = dataset_module.get_data_loader(params)
        model = _load_model_module("model_for_tusl").Model(params)
        t = trainer_class(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'BCIC-IV-2a':
        dataset_module = _require_dataset(bciciv2a_dataset, 'BCIC-IV-2a')
        load_dataset = dataset_module.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = _load_model_module("model_for_bciciv2a").Model(params)
        t = trainer_class(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'siena':
        dataset_module = _require_dataset(siena_dataset, 'siena')
        load_dataset = dataset_module.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = _load_model_module("model_for_siena").Model(params)
        t = trainer_class(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'HMC':
        dataset_module = _require_dataset(hmc_dataset, 'HMC')
        load_dataset = dataset_module.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = _load_model_module("model_for_hmc").Model(params)
        t = trainer_class(params, data_loader, model)
        t.train_for_multiclass()
    print("model:", params.model, "seed:", params.seed, "lr:", params.lr, "weight_decay:", params.weight_decay, "dropout:", params.dropout, "foundation_dir:", params.foundation_dir)
    print('Done!!!!!')


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
