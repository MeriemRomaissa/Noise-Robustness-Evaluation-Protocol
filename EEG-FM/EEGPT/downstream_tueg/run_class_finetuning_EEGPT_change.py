import argparse
import datetime
import math
from pyexpat import model
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
#newly added codes
import sys

from pathlib import Path
from collections import OrderedDict
from timm.data.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner

from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
from Modules.models.EEGPT_mcae_finetune_change import EEGPTClassifier


#newly added codes
def train_class_batch(model, samples, target, criterion, ch_names=None):
    """Run EEGPT's native classifier path on [B,23,2000] windows."""
    outputs = model(samples)
    loss = criterion(outputs, target)
    return loss, outputs


#newly added codes
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: ModelEma = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, ch_names=None,
                    is_binary=True, classification_threshold=0.5):
    """Local EEGPT epoch loop because the EEGPT repo has no separate engine file."""
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    optimizer.zero_grad()
    all_targets = []
    all_preds = []

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step
        if (lr_schedule_values is not None or wd_schedule_values is not None) and data_iter_step % update_freq == 0:
            for param_group in optimizer.param_groups:
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.float().to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if is_binary:
            targets = targets.float().unsqueeze(-1)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            loss, output = train_class_batch(model, samples, targets, criterion, ch_names)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= update_freq
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=is_second_order,
            update_grad=(data_iter_step + 1) % update_freq == 0,
        )
        if (data_iter_step + 1) % update_freq == 0:
            optimizer.zero_grad()
            if model_ema is not None:
                model_ema.update(model)
        loss_scale_value = loss_scaler.state_dict()["scale"]
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if is_binary:
            probabilities = torch.sigmoid(output).detach().cpu().numpy()
            target_np = targets.detach().cpu().numpy()
            class_acc = utils.get_metrics(probabilities, target_np, ["accuracy"], is_binary, classification_threshold)["accuracy"]
            all_preds.append((probabilities >= classification_threshold).astype(int).flatten())
            all_targets.append(target_np.astype(int).flatten())
        else:
            preds = output.max(-1)[-1].detach().cpu()
            class_acc = (preds == targets.squeeze().detach().cpu()).float().mean()
            all_preds.append(preds.numpy().flatten())
            all_targets.append(targets.squeeze().detach().cpu().numpy().flatten())

        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if all_targets and all_preds:
        from sklearn.metrics import balanced_accuracy_score

        all_targets_np = np.concatenate(all_targets)
        all_preds_np = np.concatenate(all_preds)
        stats['balanced_accuracy'] = balanced_accuracy_score(all_targets_np, all_preds_np)
    return stats


#newly added codes
@torch.no_grad()
def evaluate(data_loader, model, device, header='Test:', ch_names=None,
             metrics=['acc'], is_binary=True, classification_threshold=0.5):
    """Local EEGPT evaluation loop because the EEGPT repo has no engine file."""
    criterion = torch.nn.BCEWithLogitsLoss() if is_binary else torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    model.eval()
    pred = []
    true = []
    for step, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        eeg = batch[0].float().to(device, non_blocking=True)
        target = batch[-1].to(device, non_blocking=True)
        if is_binary:
            target = target.float().unsqueeze(-1)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            output = model(eeg)
            loss = criterion(output, target)

        if is_binary:
            output = torch.sigmoid(output).cpu()
        else:
            output = output.cpu()
        target = target.cpu()

        results = utils.get_metrics(output.numpy(), target.numpy(), metrics, is_binary, classification_threshold)
        pred.append(output)
        true.append(target)
        batch_size = eeg.shape[0]
        metric_logger.update(loss=loss.item())
        for key, value in results.items():
            metric_logger.meters[key].update(value, n=batch_size)

    metric_logger.synchronize_between_processes()
    print('* loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))
    pred = torch.cat(pred, dim=0).numpy()
    true = torch.cat(true, dim=0).numpy()
    ret = utils.get_metrics(pred, true, metrics, is_binary, classification_threshold)
    ret['loss'] = metric_logger.loss.global_avg
    return ret


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
    """Original EEGPT trains one seed per process; YAML may list many."""
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
    """Translate Benchmark YAML fields to native EEGPT argparse defaults."""
    defaults = {}
    paths = config.get("paths", {})
    data = _apply_tuab_mode_defaults(config)
    study_case = config.get("study_case", {})
    loader = config.get("loader", {})
    training = config.get("training", {})
    evaluation = config.get("evaluation", {})

    if paths.get("original_data"):
        defaults["data_path"] = paths["original_data"]
    if paths.get("h5_file"):
        defaults["h5_file"] = paths["h5_file"]
    if paths.get("split_index"):
        defaults["split_index"] = paths["split_index"]
    if paths.get("checkpoint"):
        defaults["finetune"] = paths["checkpoint"]
    if paths.get("output"):
        defaults["output_dir"] = paths["output"]
        defaults["log_dir"] = str(Path(paths["output"]) / "tensorboard")

    if data.get("source"):
        defaults["data_source"] = data["source"]
    if "train_samples" in data:
        defaults["train_samples"] = data["train_samples"]
    if "validation_samples" in data:
        defaults["validation_samples"] = data["validation_samples"]
    if "test_samples" in data:
        defaults["test_samples"] = data["test_samples"]
    if study_case.get("tuab_mode"):
        defaults["tuab_mode"] = study_case["tuab_mode"]
    if study_case.get("channel_mode"):
        defaults["channel_mode"] = study_case["channel_mode"]

    if "batch_size" in loader:
        defaults["batch_size"] = int(loader["batch_size"])
    if "num_workers" in loader:
        defaults["num_workers"] = int(loader["num_workers"])
    if "pin_memory" in loader:
        defaults["pin_mem"] = bool(loader["pin_memory"])

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

    if evaluation.get("classification_threshold") is not None:
        defaults["classification_threshold"] = float(evaluation["classification_threshold"])
    if evaluation.get("report_metrics"):
        defaults["report_metrics"] = ",".join(str(item) for item in evaluation["report_metrics"])

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
    benchmark_root = Path(__file__).resolve().parents[3] / "Benchmark"
    for relative_path in (
        "Loader",
        "choose_StudyCase/Channel",
    ):
        path = benchmark_root / relative_path
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


#newly added codes
def _benchmark_unified_h5_transform(window, channel_names=None, config=None):
    """Apply Benchmark channel case and keep EEGPT input as [23,2000]."""
    from EEGPT_23ch_vs_16ch import apply_channel_case

    config = config or {}
    return apply_channel_case(
        window,
        channel_names,
        config.get("channel_mode", "23channels"),
    )


#newly added codes
def _benchmark_h5_dataset_config(args, split):
    """Build the flat config expected by Benchmark/Loader/loader_common.py."""
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
        "channel_mode": args.channel_mode,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_mem,
    }


#newly added codes
def _get_benchmark_unified_h5_dataset(args):
    """Read unified60 H5 rows while preserving EEGPT's original engine path."""
    _add_benchmark_paths_to_syspath()
    from loader_common import Unified60Dataset

    if not args.h5_file:
        raise ValueError("--h5_file is required when --data_source tuab_unified60")
    if not args.split_index:
        raise ValueError("--split_index is required when --data_source tuab_unified60")

    train_dataset = Unified60Dataset(
        _benchmark_h5_dataset_config(args, "train"),
        "train",
        _benchmark_unified_h5_transform,
    )
    val_dataset = Unified60Dataset(
        _benchmark_h5_dataset_config(args, "val"),
        "val",
        _benchmark_unified_h5_transform,
    )
    test_dataset = Unified60Dataset(
        _benchmark_h5_dataset_config(args, "test"),
        "test",
        _benchmark_unified_h5_transform,
    )
    ch_names = [name.upper() for name in train_dataset.channel_names]
    args.nb_classes = 1
    metrics = [item.strip() for item in args.report_metrics.split(",") if item.strip()]
    if not metrics:
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]
    print(
        "Benchmark unified60 H5 dataset:",
        f"train={len(train_dataset)}",
        f"val={len(val_dataset)}",
        f"test={len(test_dataset)}",
        f"channel_mode={args.channel_mode}",
        flush=True,
    )
    return train_dataset, test_dataset, val_dataset, ch_names, metrics


def get_args():
    parser = argparse.ArgumentParser('fine-tuning and evaluation script for EEG classification', add_help=False)
    #newly added codes
    parser.add_argument('--config', default='', type=str,
                        help='Optional Benchmark YAML. YAML values become defaults; CLI args override them.')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)

    # robust evaluation
    parser.add_argument('--robust_test', default=None, type=str,
                        help='robust evaluation dataset')
    
    # Model parameters
    parser.add_argument('--model', default='', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--qkv_bias', action='store_true')
    parser.add_argument('--disable_qkv_bias', action='store_false', dest='qkv_bias')
    parser.set_defaults(qkv_bias=True)
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=False)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float, 
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

    parser.add_argument('--input_size', default=200, type=int,
                        help='EEG input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)

    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--layer_decay', type=float, default=0.9)

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module|state_dict', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--model_filter_name', default='gzp', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')
    parser.add_argument('--disable_weight_decay_on_rel_pos_bias', action='store_true', default=False)

    # Dataset parameters
    parser.add_argument('--nb_classes', default=0, type=int,
                        help='number of the classification types')
    #newly added codes
    parser.add_argument('--data_path',
                        default='../datasets/downstream/tuh_eeg_abnormal/v3.0.1/edf/processed/',
                        help='processed TUAB/TUEV data root. Benchmark YAML paths.original_data maps here.')
    #newly added codes
    parser.add_argument('--data_source', default='original', type=str,
                        choices=['original', 'tuab_unified60'],
                        help='original uses EEGPT PKLs; tuab_unified60 uses Benchmark H5 loaders.')
    #newly added codes
    parser.add_argument('--h5_file', default='', type=str,
                        help='Benchmark unified60 H5 path when data_source=tuab_unified60.')
    #newly added codes
    parser.add_argument('--split_index', default='', type=str,
                        help='Benchmark split CSV path when data_source=tuab_unified60.')
    #newly added codes
    parser.add_argument('--tuab_mode', default='subset_tuab', type=str,
                        choices=['subset_tuab', 'full_tuab'],
                        help='Benchmark TUAB study case; controls split sample caps.')
    #newly added codes
    parser.add_argument('--channel_mode', default='23channels', type=str,
                        choices=['23channels', '16channels_zeropadded'],
                        help='Benchmark channel-count study case for unified H5.')
    #newly added codes
    parser.add_argument('--train_samples', default=None, type=_optional_int,
                        help='Max unified H5 train rows; none means full split.')
    #newly added codes
    parser.add_argument('--validation_samples', default=None, type=_optional_int,
                        help='Max unified H5 validation rows; none means full split.')
    #newly added codes
    parser.add_argument('--test_samples', default=None, type=_optional_int,
                        help='Max unified H5 test rows; none means full split.')
    #newly added codes
    parser.add_argument('--classification_threshold', default=0.5, type=float,
                        help='Benchmark evaluation threshold.')
    #newly added codes
    parser.add_argument('--report_metrics', default='pr_auc,roc_auc,accuracy,balanced_accuracy', type=str,
                        help='Comma-separated metrics for Benchmark unified H5 evaluation.')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--enable_deepspeed', action='store_true', default=False)
    parser.add_argument('--dataset', default='TUAB', type=str,
                        help='dataset: TUAB | TUEV')

    known_args, _ = parser.parse_known_args()
    #newly added codes
    if known_args.config:
        parser.set_defaults(**_config_defaults(_load_yaml_config(known_args.config)))

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed==0.4.0'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init

def get_models(args):
    # CHANNEL_DICT = {k.upper():v for v,k in enumerate(
    #                  [      'FP1', 'FPZ', 'FP2', 
    #                     "AF7", 'AF3', 'AF4', "AF8", 
    #         'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 
    #     'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 
    #         'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 
    #     'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
    #          'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 
    #                   'PO7', "PO5", 'PO3', 'POZ', 'PO4', "PO6", 'PO8', 
    #                            'O1', 'OZ', 'O2', ])}

    # use_channels_names =   ['FP1', 'FPZ', 'FP2',
    #         'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 
    #         'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 
    #                   'PO3', 'POZ', 'PO4',
    #                   'O1', 'O2']

    use_channels_names = [      
             'FP1','FPZ', 'FP2',
        'F7', 'F3', 'FZ', 'F4', 'F8',
        'T7', 'C3', 'CZ', 'C4', 'T8',
        'P7', 'P3', 'PZ', 'P4', 'P8',
                'O1', 'O2' ]
    
    ch_names = ['EEG FP1', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
    ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
    model = EEGPTClassifier(
        num_classes=args.nb_classes,
        in_channels=len(ch_names), 
        img_size=[len(use_channels_names),2000], 
        use_channels_names=use_channels_names, 
        use_chan_conv=True,
        use_mean_pooling=args.use_mean_pooling,)
    
    # model = create_model(
    #     args.model,
    #     drop_rate=args.drop,
    #     drop_path_rate=args.drop_path,
    #     attn_drop_rate=args.attn_drop_rate,
    #     use_mean_pooling=args.use_mean_pooling,
    #     qkv_bias=args.qkv_bias,
    # )

    return model


def get_dataset(args):
    #newly added codes
    if args.data_source == 'tuab_unified60':
        return _get_benchmark_unified_h5_dataset(args)
    if args.dataset == 'TUAB':
        train_dataset, test_dataset, val_dataset = utils.prepare_TUAB_dataset(args.data_path)
        ch_names = ['EEG FP1', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
        ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        args.nb_classes = 1
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]

    elif args.dataset == 'TUEV':
        train_dataset, test_dataset, val_dataset = utils.prepare_TUEV_dataset("../datasets/downstream/tuh_eeg_events/v2.0.1/edf/processed/")
        ch_names = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
        
        ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        args.nb_classes = 6
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
        
    elif args.dataset == 'TUSZ':
        train_dataset, test_dataset, val_dataset = utils.prepare_TUSZ_dataset(None)
        
        # ch_names = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
        #             'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
        # ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        ch_names = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']
        args.nb_classes = 8
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
    return train_dataset, test_dataset, val_dataset, ch_names, metrics


def main(args, ds_init):
    utils.init_distributed_mode(args)

    if ds_init is not None:
        utils.create_ds_config(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    # dataset_train, dataset_test, dataset_val: follows the standard format of torch.utils.data.Dataset.
    # ch_names: list of strings, channel names of the dataset. It should be in capital letters.
    # metrics: list of strings, the metrics you want to use. We utilize PyHealth to implement it.
    dataset_train, dataset_test, dataset_val, ch_names, metrics = get_dataset(args)

    if args.disable_eval_during_finetuning:
        dataset_val = None
        dataset_test = None

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
            if type(dataset_test) == list:
                sampler_test = [torch.utils.data.DistributedSampler(
                    dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False) for dataset in dataset_test]
            else:
                sampler_test = torch.utils.data.DistributedSampler(
                    dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        if type(dataset_test) == list:
            data_loader_test = [torch.utils.data.DataLoader(
                dataset, sampler=sampler,
                batch_size=int(1.5 * args.batch_size),
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            ) for dataset, sampler in zip(dataset_test, sampler_test)]
        else:
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, sampler=sampler_test,
                batch_size=int(1.5 * args.batch_size),
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            )
    else:
        data_loader_val = None
        data_loader_test = None

    model = get_models(args)

    patch_size = 200#model.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (1, args.input_size // patch_size)
    args.patch_size = patch_size

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load ckpt from %s" % args.finetune)
        # checkpoint_model = None
        # for model_key in args.model_key.split('|'):
        #     if model_key in checkpoint:
        #         checkpoint_model = checkpoint[model_key]
        #         print("Load state_dict by model_key = %s" % model_key)
        #         break
        # if checkpoint_model is None:
        #     checkpoint_model = checkpoint
        # if (checkpoint_model is not None) and (args.model_filter_name != ''):
        #     all_keys = list(checkpoint_model.keys())
        #     new_dict = OrderedDict()
        #     for key in all_keys:
        #         if key.startswith('student.'):
        #             new_dict[key[8:]] = checkpoint_model[key]
        #         else:
        #             pass
        #     checkpoint_model = new_dict

        # state_dict = model.state_dict()
        # for k in ['head.weight', 'head.bias']:
        #     if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        #         print(f"Removing key {k} from pretrained checkpoint")
        #         del checkpoint_model[k]

        # all_keys = list(checkpoint_model.keys())
        checkpoint_model = checkpoint['state_dict']
        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()
    if args.disable_weight_decay_on_rel_pos_bias:
        for i in range(num_layers):
            skip_weight_decay_list.add("blocks.%d.attn.relative_position_bias_table" % i)

    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, args.weight_decay, skip_weight_decay_list,
            assigner.get_layer_id if assigner is not None else None,
            assigner.get_scale if assigner is not None else None)
        model, optimizer, _, _ = ds_init(
            args=args, model=model, model_parameters=optimizer_params, dist_init_required=not args.distributed,
        )

        print("model.gradient_accumulation_steps() = %d" % model.gradient_accumulation_steps())
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module

        optimizer = create_optimizer(
            args, model_without_ddp, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None, 
            get_layer_scale=assigner.get_scale if assigner is not None else None)
        loss_scaler = NativeScaler()

    print("Use step level LR scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    if args.nb_classes == 1:
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)
            
    if args.eval:
        balanced_accuracy = []
        accuracy = []
        for data_loader in data_loader_test:
            test_stats = evaluate(data_loader, model, device, header='Test:', ch_names=ch_names, metrics=metrics, is_binary=(args.nb_classes == 1),
                                  #newly added codes
                                  classification_threshold=args.classification_threshold)
            accuracy.append(test_stats['accuracy'])
            balanced_accuracy.append(test_stats['balanced_accuracy'])
        print(f"======Accuracy: {np.mean(accuracy)} {np.std(accuracy)}, balanced accuracy: {np.mean(balanced_accuracy)} {np.std(balanced_accuracy)}")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    max_accuracy_test = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema,
            log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq, 
            ch_names=ch_names, is_binary=args.nb_classes == 1,
            #newly added codes
            classification_threshold=args.classification_threshold
        )
        
        if args.output_dir and args.save_ckpt:
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema, save_ckpt_freq=args.save_ckpt_freq)
            
        if data_loader_val is not None:
            val_stats = evaluate(data_loader_val, model, device, header='Val:', ch_names=ch_names, metrics=metrics, is_binary=args.nb_classes == 1,
                                 #newly added codes
                                 classification_threshold=args.classification_threshold)
            print(f"Accuracy of the network on the {len(dataset_val)} val EEG: {val_stats['accuracy']:.2f}%")
            test_stats = evaluate(data_loader_test, model, device, header='Test:', ch_names=ch_names, metrics=metrics, is_binary=args.nb_classes == 1,
                                  #newly added codes
                                  classification_threshold=args.classification_threshold)
            print(f"Accuracy of the network on the {len(dataset_test)} test EEG: {test_stats['accuracy']:.2f}%")
            
            if max_accuracy < val_stats["accuracy"]:
                max_accuracy = val_stats["accuracy"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
                max_accuracy_test = test_stats["accuracy"]

            print(f'Max accuracy val: {max_accuracy:.2f}%, max accuracy test: {max_accuracy_test:.2f}%')
            if log_writer is not None:
                for key, value in val_stats.items():
                    if key == 'accuracy':
                        log_writer.update(accuracy=value, head="val", step=epoch)
                    elif key == 'balanced_accuracy':
                        log_writer.update(balanced_accuracy=value, head="val", step=epoch)
                    elif key == 'f1_weighted':
                        log_writer.update(f1_weighted=value, head="val", step=epoch)
                    elif key == 'pr_auc':
                        log_writer.update(pr_auc=value, head="val", step=epoch)
                    elif key == 'roc_auc':
                        log_writer.update(roc_auc=value, head="val", step=epoch)
                    elif key == 'cohen_kappa':
                        log_writer.update(cohen_kappa=value, head="val", step=epoch)
                    elif key == 'loss':
                        log_writer.update(loss=value, head="val", step=epoch)
                for key, value in test_stats.items():
                    if key == 'accuracy':
                        log_writer.update(accuracy=value, head="test", step=epoch)
                    elif key == 'balanced_accuracy':
                        log_writer.update(balanced_accuracy=value, head="test", step=epoch)
                    elif key == 'f1_weighted':
                        log_writer.update(f1_weighted=value, head="test", step=epoch)
                    elif key == 'pr_auc':
                        log_writer.update(pr_auc=value, head="test", step=epoch)
                    elif key == 'roc_auc':
                        log_writer.update(roc_auc=value, head="test", step=epoch)
                    elif key == 'cohen_kappa':
                        log_writer.update(cohen_kappa=value, head="test", step=epoch)
                    elif key == 'loss':
                        log_writer.update(loss=value, head="test", step=epoch)
                
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in val_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)
