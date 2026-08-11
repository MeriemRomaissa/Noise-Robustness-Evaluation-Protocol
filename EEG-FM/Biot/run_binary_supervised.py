import os
import argparse
import pickle
import json
#newly added codes
import sys
#newly added codes
from pathlib import Path

import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pyhealth.metrics import binary_metrics_fn

from model import (
    SPaRCNet,
    ContraWR,
    CNNTransformer,
    FFCL,
    STTransformer,
    BIOTClassifier,
)
from utils import TUABLoader, CHBMITLoader, PTBLoader, focal_loss, BCE


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
    """Original BIOT trains one seed per process; YAML may list many."""
    if isinstance(value, (list, tuple)):
        return int(value[0]) if value else 0
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
    """Translate Benchmark YAML fields to native BIOT argparse defaults."""
    defaults = {}
    paths = config.get("paths", {})
    data = _apply_tuab_mode_defaults(config)
    loader = config.get("loader", {})
    training = config.get("training", {})
    fixed_recipe = config.get("fixed_recipe", {})
    fixed_model = fixed_recipe.get("model", {})
    fixed_training = fixed_recipe.get("training", {})
    fixed_data = fixed_recipe.get("data", {})

    if paths.get("original_data"):
        defaults["data_path"] = paths["original_data"]
    if paths.get("h5_file"):
        defaults["h5_file"] = paths["h5_file"]
    if paths.get("split_index"):
        defaults["split_index"] = paths["split_index"]
    if paths.get("checkpoint"):
        defaults["pretrain_model_path"] = paths["checkpoint"]
    if paths.get("output"):
        defaults["output_dir"] = paths["output"]

    if data.get("source"):
        defaults["data_source"] = data["source"]
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
    if "weight_decay" in fixed_training:
        defaults["weight_decay"] = float(fixed_training["weight_decay"])

    for key in ("dataset", "model", "in_channels", "sample_length", "n_classes",
                "sampling_rate", "token_size", "hop_length"):
        if key in fixed_model:
            defaults[key] = fixed_model[key]
    if "normalization_epsilon" in fixed_data:
        defaults["normalization_epsilon"] = float(fixed_data["normalization_epsilon"])

    return defaults


#newly added codes
def _optional_int(value):
    """Allow YAML null/CLI none to mean no split cap."""
    if value is None or str(value).lower() == "none":
        return None
    return int(value)


#newly added codes
def _add_benchmark_paths_to_syspath():
    """Let this native script reuse Benchmark loaders without moving files."""
    benchmark_root = Path(__file__).resolve().parents[2] / "Benchmark"
    for relative_path in ("DataLoader",):
        path = benchmark_root / relative_path
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


#newly added codes
def _benchmark_h5_loader_config(args, split):
    """Build the flat config expected by Benchmark/DataLoader/loader_common.py."""
    max_samples = {
        "train": args.train_samples,
        "val": args.validation_samples,
        "test": args.test_samples,
    }[split]
    return {
        "h5_path": args.h5_file,
        "split_index_path": args.split_index,
        "max_samples": max_samples,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "shuffle": args.shuffle_train if split == "train" else False,
        "drop_last": args.drop_last_train if split == "train" else False,
        "normalization_epsilon": args.normalization_epsilon,
    }


#newly added codes
def prepare_unified60_dataloader(args):
    """Read unified60 H5 rows through the Benchmark BIOT loader."""
    _add_benchmark_paths_to_syspath()
    from loader_biot import build_unified60_loader

    if not args.h5_file:
        raise ValueError("--h5_file is required when --data_source tuab_unified60")
    if not args.split_index:
        raise ValueError("--split_index is required when --data_source tuab_unified60")
    train_loader = build_unified60_loader(_benchmark_h5_loader_config(args, "train"), "train")
    test_loader = build_unified60_loader(_benchmark_h5_loader_config(args, "test"), "test")
    val_loader = build_unified60_loader(_benchmark_h5_loader_config(args, "val"), "val")
    print(
        "Benchmark unified60 H5 dataset:",
        f"train={len(train_loader.dataset)}",
        f"val={len(val_loader.dataset)}",
        f"test={len(test_loader.dataset)}",
        flush=True,
    )
    return train_loader, test_loader, val_loader


class LitModel_finetune(pl.LightningModule):
    def __init__(self, args, model):
        super().__init__()
        self.model = model
        self.threshold = 0.5
        self.args = args
        self.n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self._train_probs = []
        self._train_targets = []
        self._train_losses = []
        self._latest_train_metrics = {}
        self.test_loader_for_epoch_logging = None

    def on_train_epoch_start(self):
        self._train_probs = []
        self._train_targets = []
        self._train_losses = []

    def training_step(self, batch, batch_idx):
        X, y = batch
        prob = self.model(X)
        loss = BCE(prob, y)  # focal_loss(prob, y)
        with torch.no_grad():
            self._train_probs.append(torch.sigmoid(prob.detach()).cpu())
            self._train_targets.append(y.detach().cpu())
            self._train_losses.append(float(loss.detach().cpu()))
        self.log("train_loss", loss)
        return loss

    def training_epoch_end(self, training_step_outputs):
        if not self._train_probs:
            return
        result = torch.cat([x.reshape(-1) for x in self._train_probs]).numpy()
        gt = torch.cat([x.reshape(-1) for x in self._train_targets]).numpy()
        metrics = self._binary_metrics(result, gt, threshold=0.5)
        self._latest_train_metrics = {
            "loss": float(np.mean(self._train_losses)) if self._train_losses else None,
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
        }
        self.log("train_acc", metrics["accuracy"], sync_dist=True)
        self.log("train_bacc", metrics["balanced_accuracy"], sync_dist=True)

    def _binary_metrics(self, result, gt, threshold):
        if sum(gt) * (len(gt) - sum(gt)) != 0:
            return binary_metrics_fn(
                gt,
                result,
                metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"],
                threshold=threshold,
            )
        return {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "pr_auc": 0.0,
            "roc_auc": 0.0,
        }

    def _evaluate_loader_for_binaryclass(self, data_loader):
        result = np.array([])
        gt = np.array([])
        losses = []
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for X, y in data_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                prob = self.model(X)
                losses.append(float(BCE(prob, y).detach().cpu()))
                result = np.append(result, torch.sigmoid(prob).detach().cpu().numpy())
                gt = np.append(gt, y.detach().cpu().numpy())
        if was_training:
            self.model.train()
        metrics = self._binary_metrics(result, gt, threshold=self.threshold)
        metrics["loss"] = float(np.mean(losses)) if losses else None
        return metrics

    def _write_epoch_log(self, val_metrics, test_metrics):
        if not self.args.output_dir:
            return
        if getattr(self.trainer, "sanity_checking", False):
            return
        os.makedirs(self.args.output_dir, exist_ok=True)
        optimizers = getattr(self.trainer, "optimizers", [])
        train_lr = optimizers[0].param_groups[0]["lr"] if optimizers else self.args.lr
        train_metrics = self._latest_train_metrics
        log_stats = {
            "train_loss": train_metrics.get("loss"),
            "train_lr": train_lr,
            "train_min_lr": None,
            "train_loss_scale": None,
            "train_weight_decay": float(self.args.weight_decay),
            "train_class_acc": train_metrics.get("accuracy"),
            "train_balanced_accuracy": train_metrics.get("balanced_accuracy"),
            "train_grad_norm": None,
            "val_pr_auc": val_metrics.get("pr_auc"),
            "val_roc_auc": val_metrics.get("roc_auc"),
            "val_accuracy": val_metrics.get("accuracy"),
            "val_balanced_accuracy": val_metrics.get("balanced_accuracy"),
            "val_loss": val_metrics.get("loss"),
            "test_pr_auc": test_metrics.get("pr_auc"),
            "test_roc_auc": test_metrics.get("roc_auc"),
            "test_accuracy": test_metrics.get("accuracy"),
            "test_balanced_accuracy": test_metrics.get("balanced_accuracy"),
            "test_loss": test_metrics.get("loss"),
            "epoch": int(self.current_epoch),
            "n_parameters": self.n_parameters,
        }
        with open(os.path.join(self.args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")

    def validation_step(self, batch, batch_idx):
        X, y = batch
        with torch.no_grad():
            prob = self.model(X)
            step_result = torch.sigmoid(prob).cpu().numpy()
            step_gt = y.cpu().numpy()
            step_loss = float(BCE(prob, y).detach().cpu())
        return step_result, step_gt, step_loss

    def validation_epoch_end(self, val_step_outputs):
        result = np.array([])
        gt = np.array([])
        losses = []
        for out in val_step_outputs:
            result = np.append(result, out[0])
            gt = np.append(gt, out[1])
            if len(out) > 2:
                losses.append(out[2])

        if (
            sum(gt) * (len(gt) - sum(gt)) != 0
        ):  # to prevent all 0 or all 1 and raise the AUROC error
            self.threshold = np.sort(result)[-int(np.sum(gt))]
            result = binary_metrics_fn(
                gt,
                result,
                metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"],
                threshold=self.threshold,
            )
        else:
            result = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
            }
        result["loss"] = float(np.mean(losses)) if losses else None
        self.log("val_acc", result["accuracy"], sync_dist=True)
        self.log("val_bacc", result["balanced_accuracy"], sync_dist=True)
        self.log("val_pr_auc", result["pr_auc"], sync_dist=True)
        self.log("val_auroc", result["roc_auc"], sync_dist=True)
        print(result)
        test_result = {}
        if self.test_loader_for_epoch_logging is not None:
            test_result = self._evaluate_loader_for_binaryclass(self.test_loader_for_epoch_logging)
        self._write_epoch_log(result, test_result)

    def test_step(self, batch, batch_idx):
        X, y = batch
        with torch.no_grad():
            convScore = self.model(X)
            step_result = torch.sigmoid(convScore).cpu().numpy()
            step_gt = y.cpu().numpy()
            step_loss = float(BCE(convScore, y).detach().cpu())
        return step_result, step_gt, step_loss

    def test_epoch_end(self, test_step_outputs):
        result = np.array([])
        gt = np.array([])
        losses = []
        for out in test_step_outputs:
            result = np.append(result, out[0])
            gt = np.append(gt, out[1])
            if len(out) > 2:
                losses.append(out[2])
        if (
            sum(gt) * (len(gt) - sum(gt)) != 0
        ):  # to prevent all 0 or all 1 and raise the AUROC error
            result = binary_metrics_fn(
                gt,
                result,
                metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"],
                threshold=self.threshold,
            )
        else:
            result = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
            }
        result["loss"] = float(np.mean(losses)) if losses else None
        self.log("test_acc", result["accuracy"], sync_dist=True)
        self.log("test_bacc", result["balanced_accuracy"], sync_dist=True)
        self.log("test_pr_auc", result["pr_auc"], sync_dist=True)
        self.log("test_auroc", result["roc_auc"], sync_dist=True)

        return result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

        return [optimizer]  # , [scheduler]


def prepare_TUAB_dataloader(args):
    #newly added codes
    if args.data_source == "tuab_unified60":
        return prepare_unified60_dataloader(args)

    # set random seed
    #newly added codes
    seed = args.seed
    torch.manual_seed(seed)
    #newly added codes
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    #newly added codes
    root = args.data_path

    train_files = os.listdir(os.path.join(root, "train"))
    np.random.shuffle(train_files)
    # train_files = train_files[:100000]
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        TUABLoader(os.path.join(root, "train"),
                   train_files, args.sampling_rate),
        batch_size=args.batch_size,
        #newly added codes
        shuffle=args.shuffle_train,
        #newly added codes
        drop_last=args.drop_last_train,
        num_workers=args.num_workers,
        #newly added codes
        persistent_workers=args.num_workers > 0,
        #newly added codes
        pin_memory=args.pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        TUABLoader(os.path.join(root, "test"), test_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        #newly added codes
        persistent_workers=args.num_workers > 0,
        #newly added codes
        pin_memory=args.pin_memory,
    )
    val_loader = torch.utils.data.DataLoader(
        TUABLoader(os.path.join(root, "val"), val_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        #newly added codes
        persistent_workers=args.num_workers > 0,
        #newly added codes
        pin_memory=args.pin_memory,
    )
    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader


def prepare_CHB_MIT_dataloader(args):
    # set random seed
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    root = "/srv/local/data/physionet.org/files/chbmit/1.0.0/clean_segments"

    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        CHBMITLoader(os.path.join(root, "train"),
                     train_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        CHBMITLoader(os.path.join(root, "test"),
                     test_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        CHBMITLoader(os.path.join(root, "val"), val_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader


def prepare_PTB_dataloader(args):
    # set random seed
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    root = "/srv/local/data/WFDB/processed2"

    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        PTBLoader(os.path.join(root, "train"),
                  train_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        PTBLoader(os.path.join(root, "test"), test_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        PTBLoader(os.path.join(root, "val"), val_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader


def supervised(args):
    # get data loaders
    if args.dataset == "TUAB":
        train_loader, test_loader, val_loader = prepare_TUAB_dataloader(args)

    else:
        raise NotImplementedError

    # define the model
    if args.model == "SPaRCNet":
        model = SPaRCNet(
            in_channels=args.in_channels,
            sample_length=int(args.sampling_rate * args.sample_length),
            n_classes=args.n_classes,
            block_layers=4,
            growth_rate=16,
            bn_size=16,
            drop_rate=0.5,
            conv_bias=True,
            batch_norm=True,
        )

    elif args.model == "ContraWR":
        model = ContraWR(
            in_channels=args.in_channels,
            n_classes=args.n_classes,
            fft=args.token_size,
            steps=args.hop_length // 5,
        )

    elif args.model == "CNNTransformer":
        model = CNNTransformer(
            in_channels=args.in_channels,
            n_classes=args.n_classes,
            fft=args.sampling_rate,
            steps=args.hop_length // 5,
            dropout=0.2,
            nhead=4,
            emb_size=256,
        )

    elif args.model == "FFCL":
        model = FFCL(
            in_channels=args.in_channels,
            n_classes=args.n_classes,
            fft=args.token_size,
            steps=args.hop_length // 5,
            sample_length=int(args.sampling_rate * args.sample_length),
            shrink_steps=20,
        )

    elif args.model == "STTransformer":
        model = STTransformer(
            emb_size=256,
            depth=4,
            n_classes=args.n_classes,
            channel_legnth=int(
                args.sampling_rate * args.sample_length
            ),  # (sampling_rate * duration)
            n_channels=args.in_channels,
        )

    elif args.model == "BIOT":
        model = BIOTClassifier(
            n_classes=args.n_classes,
            # set the n_channels according to the pretrained model if necessary
            n_channels=args.in_channels,
            n_fft=args.token_size,
            hop_length=args.hop_length,
        )
        if args.pretrain_model_path and (args.sampling_rate == 200):
            #newly added codes
            if not os.path.isfile(args.pretrain_model_path):
                raise FileNotFoundError(
                    "BIOT pretrained checkpoint is configured but missing: "
                    f"{args.pretrain_model_path}"
                )
            model.biot.load_state_dict(torch.load(args.pretrain_model_path))
            print(f"load pretrain model from {args.pretrain_model_path}")

    else:
        raise NotImplementedError
    lightning_model = LitModel_finetune(args, model)
    lightning_model.test_loader_for_epoch_logging = test_loader

    # logger and callbacks
    version = f"{args.dataset}-{args.model}-{args.lr}-{args.batch_size}-{args.sampling_rate}-{args.token_size}-{args.hop_length}"
    logger = TensorBoardLogger(
        #newly added codes
        save_dir=args.output_dir,
        version=version,
        name="log",
    )
    early_stop_callback = EarlyStopping(
        monitor="val_auroc", patience=5, verbose=False, mode="max"
    )

    trainer = pl.Trainer(
        devices=[0],
        #newly added codes
        accelerator="gpu" if args.device != "cpu" else "cpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        auto_select_gpus=True,
        benchmark=True,
        enable_checkpointing=True,
        logger=logger,
        max_epochs=args.epochs,
        callbacks=[early_stop_callback],
    )

    # train the model
    trainer.fit(
        lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )

    # test the model
    pretrain_result = trainer.test(
        model=lightning_model, ckpt_path="best", dataloaders=test_loader
    )[0]
    print(pretrain_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #newly added codes
    parser.add_argument("--config", type=str, default="",
                        help="Optional Benchmark YAML. YAML values become defaults; CLI args override them.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight_decay", type=float,
                        default=1e-5, help="weight decay")
    parser.add_argument("--batch_size", type=int,
                        default=512, help="batch size")
    parser.add_argument("--num_workers", type=int,
                        default=32, help="number of workers")
    #newly added codes
    parser.add_argument("--seed", type=int, default=12345,
                        help="random seed")
    #newly added codes
    parser.add_argument("--data_path", type=str,
                        default="/srv/local/data/TUH/tuh3/tuh_eeg_abnormal/v3.0.0/edf/processed",
                        help="processed TUAB data root. Benchmark YAML paths.original_data maps here.")
    #newly added codes
    parser.add_argument("--data_source", type=str, default="original",
                        choices=["original", "tuab_unified60"],
                        help="original uses BIOT PKLs; tuab_unified60 uses Benchmark H5 loaders.")
    #newly added codes
    parser.add_argument("--h5_file", type=str, default="",
                        help="Benchmark unified60 H5 path when data_source=tuab_unified60.")
    #newly added codes
    parser.add_argument("--split_index", type=str, default="",
                        help="Benchmark split CSV path when data_source=tuab_unified60.")
    #newly added codes
    parser.add_argument("--tuab_mode", type=str, default="subset_tuab",
                        choices=["subset_tuab", "full_tuab"],
                        help="Benchmark TUAB study case; controls split sample caps.")
    #newly added codes
    parser.add_argument("--train_samples", type=_optional_int, default=None,
                        help="Max unified H5 train rows; none means full split.")
    #newly added codes
    parser.add_argument("--validation_samples", type=_optional_int, default=None,
                        help="Max unified H5 validation rows; none means full split.")
    #newly added codes
    parser.add_argument("--test_samples", type=_optional_int, default=None,
                        help="Max unified H5 test rows; none means full split.")
    #newly added codes
    parser.add_argument("--shuffle_train", action="store_true")
    #newly added codes
    parser.add_argument("--no_shuffle_train", action="store_false", dest="shuffle_train")
    #newly added codes
    parser.set_defaults(shuffle_train=True)
    #newly added codes
    parser.add_argument("--drop_last_train", action="store_true")
    #newly added codes
    parser.add_argument("--no_drop_last_train", action="store_false", dest="drop_last_train")
    #newly added codes
    parser.set_defaults(drop_last_train=True)
    #newly added codes
    parser.add_argument("--pin_memory", action="store_true")
    #newly added codes
    parser.add_argument("--no_pin_memory", action="store_false", dest="pin_memory")
    #newly added codes
    parser.set_defaults(pin_memory=True)
    #newly added codes
    parser.add_argument("--normalization_epsilon", type=float, default=1e-8,
                        help="BIOT robust normalization denominator offset.")
    #newly added codes
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Benchmark output/log root.")
    #newly added codes
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda or cpu")
    parser.add_argument("--dataset", type=str, default="TUAB", help="dataset")
    parser.add_argument(
        "--model", type=str, default="SPaRCNet", help="which supervised model to use"
    )
    parser.add_argument(
        "--in_channels", type=int, default=16, help="number of input channels"
    )
    parser.add_argument(
        "--sample_length", type=float, default=10, help="length (s) of sample"
    )
    parser.add_argument(
        "--n_classes", type=int, default=1, help="number of output classes"
    )
    parser.add_argument(
        "--sampling_rate", type=int, default=200, help="sampling rate (r)"
    )
    parser.add_argument("--token_size", type=int,
                        default=200, help="token size (t)")
    parser.add_argument(
        "--hop_length", type=int, default=100, help="token hop length (t - p)"
    )
    parser.add_argument(
        "--pretrain_model_path", type=str, default="", help="pretrained model path"
    )
    #newly added codes
    known_args, _ = parser.parse_known_args()
    #newly added codes
    if known_args.config:
        parser.set_defaults(**_config_defaults(_load_yaml_config(known_args.config)))
    args = parser.parse_args()
    print(args)
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        open(os.path.join(args.output_dir, "log.txt"), mode="w", encoding="utf-8").close()

    supervised(args)
