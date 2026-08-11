import os
import argparse
import pickle
import json

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
    # set random seed
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    root = "/srv/local/data/TUH/tuh3/tuh_eeg_abnormal/v3.0.0/edf/processed"

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
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        TUABLoader(os.path.join(root, "test"), test_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        TUABLoader(os.path.join(root, "val"), val_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
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
            model.biot.load_state_dict(torch.load(args.pretrain_model_path))
            print(f"load pretrain model from {args.pretrain_model_path}")

    else:
        raise NotImplementedError
    lightning_model = LitModel_finetune(args, model)
    lightning_model.test_loader_for_epoch_logging = test_loader

    # logger and callbacks
    version = f"{args.dataset}-{args.model}-{args.lr}-{args.batch_size}-{args.sampling_rate}-{args.token_size}-{args.hop_length}"
    logger = TensorBoardLogger(
        save_dir="./",
        version=version,
        name="log",
    )
    early_stop_callback = EarlyStopping(
        monitor="val_auroc", patience=5, verbose=False, mode="max"
    )

    trainer = pl.Trainer(
        devices=[0],
        accelerator="gpu",
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
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight_decay", type=float,
                        default=1e-5, help="weight decay")
    parser.add_argument("--batch_size", type=int,
                        default=512, help="batch size")
    parser.add_argument("--num_workers", type=int,
                        default=32, help="number of workers")
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
    parser.add_argument("--output_dir", type=str, default="", help="directory for log.txt")
    args = parser.parse_args()
    print(args)
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        open(os.path.join(args.output_dir, "log.txt"), mode="w", encoding="utf-8").close()

    supervised(args)
