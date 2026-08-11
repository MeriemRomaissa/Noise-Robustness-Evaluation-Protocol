import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from finetune_evaluator import Evaluator
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib as mpl
import umap
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
import copy
import os
import json


class TensorboardLogger(object):
    def __init__(self, log_dir):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(logdir=log_dir)
        self.step = 0

    def update(self, head='scalar', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(head + "/" + k, v, self.step if step is None else step)

    def flush(self):
        self.writer.flush()


class Trainer(object):
    def __init__(self, params, data_loader, model):
        self.params = params
        self.data_loader = data_loader

        self.val_eval = Evaluator(params, self.data_loader['val'])
        self.test_eval = Evaluator(params, self.data_loader['test'])

        self.model = model.cuda()
        if self.params.downstream_dataset in ['FACED', 'SEED-V', 'PhysioNet-MI', 'ISRUC', 'BCIC2020-3', 'TUEV', 'BCIC-IV-2a', 'TUSL', 'HMC']:
            self.criterion = CrossEntropyLoss(label_smoothing=self.params.label_smoothing).cuda()
        elif self.params.downstream_dataset in ['SHU-MI', 'CHB-MIT', 'Mumtaz2016', 'MentalArithmetic', 'TUAB', 'siena']:
            self.criterion = BCEWithLogitsLoss().cuda()
        elif self.params.downstream_dataset == 'SEED-VIG':
            self.criterion = MSELoss().cuda()


        self.best_model_states = None
        self.n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        if hasattr(params, 'log_dir') and params.log_dir:
            os.makedirs(params.log_dir, exist_ok=True)
            self.log_writer = TensorboardLogger(log_dir=params.log_dir)
        else:
            self.log_writer = None
        self.output_dir = getattr(params, 'output_dir', '')

        backbone_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)

                if params.frozen:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                other_params.append(param)

        if self.params.optimizer == 'AdamW':
            if self.params.multi_lr: # set different learning rates for different modules
                self.optimizer = torch.optim.AdamW([
                    {'params': backbone_params, 'lr': self.params.lr},
                    {'params': other_params, 'lr': self.params.lr * 5}
                ], weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.lr,
                                                   weight_decay=self.params.weight_decay)
        else:
            if self.params.multi_lr:
                self.optimizer = torch.optim.SGD([
                    {'params': backbone_params, 'lr': self.params.lr},
                    {'params': other_params, 'lr': self.params.lr * 5}
                ],  momentum=0.9, weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params.lr, momentum=0.9,
                                                 weight_decay=self.params.weight_decay)

        self.data_length = len(self.data_loader['train'])
        self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.params.epochs * self.data_length, eta_min=1e-6
        )
        print(self.model)

    def _save_checkpoint(self, epoch, tag='checkpoint'):
        if not self.output_dir:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        to_save = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'args': self.params,
        }
        torch.save(to_save, os.path.join(self.output_dir, f'{tag}.pth'))
        save_ckpt_freq = getattr(self.params, 'save_ckpt_freq', 0)
        if tag == 'checkpoint' and save_ckpt_freq > 0 and (epoch + 1) % save_ckpt_freq == 0:
            torch.save(to_save, os.path.join(self.output_dir, f'checkpoint-{epoch}.pth'))

    def train_for_multiclass(self):
        f1_best = 0
        kappa_best = 0
        acc_best = 0
        cm_best = None
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                pred = self.model(x)
                if self.params.downstream_dataset == 'ISRUC':
                    loss = self.criterion(pred.transpose(1, 2), y)
                else:
                    loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                acc, kappa, f1, cm = self.val_eval.get_metrics_for_multiclass(self.model)
                print(
                    "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins | model: {}, seed: {}, lr: {}, weight_decay: {}, dropout: {}, foundation_dir: {}".format(
                        epoch + 1,
                        np.mean(losses),
                        acc,
                        kappa,
                        f1,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60,
                        self.params.model,
                        self.params.seed,
                        self.params.lr,
                        self.params.weight_decay,
                        self.params.dropout,
                        self.params.foundation_dir
                    )
                )
                print(cm)
                if acc > acc_best: # zhouyc
                    print("kappa increasing....saving weights !! ")
                    print("Val Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(
                        acc,
                        kappa,
                        f1,
                    ))
                    best_f1_epoch = epoch + 1
                    acc_best = acc
                    kappa_best = kappa
                    f1_best = f1
                    cm_best = cm
                    self.best_model_states = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            acc, kappa, f1, cm = self.test_eval.get_metrics_for_multiclass(self.model)
            print("***************************Test results************************")
            print(
                "Test Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(
                    acc,
                    kappa,
                    f1,
                )
            )
            print(cm)
            if not os.path.exists(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = self.params.model_dir + "/epoch{}_acc_{:.5f}_kappa_{:.5f}_f1_{:.5f}.pth".format(best_f1_epoch, acc, kappa, f1)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)


    def train_for_binaryclass(self):
        acc_best = 0
        roc_auc_best = 0
        pr_auc_best = 0
        cm_best = None
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            train_accs = []
            train_preds = []
            train_targets = []
            grad_norms = []
            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                pred = self.model(x)

                loss = self.criterion(pred, y.float())

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    grad_norms.append(float(grad_norm))
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
                with torch.no_grad():
                    score_y = torch.sigmoid(pred.detach())
                    pred_y = torch.gt(score_y, 0.5).long()
                    batch_acc = (pred_y.squeeze() == y.long().squeeze()).float().mean().item()
                    train_accs.append(batch_acc)
                    train_preds.extend(np.asarray(pred_y.detach().cpu()).reshape(-1).astype(int).tolist())
                    train_targets.extend(np.asarray(y.detach().cpu()).reshape(-1).astype(int).tolist())
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                vm = self.val_eval.get_metrics_for_binaryclass(self.model, self.criterion)
                acc, pr_auc, roc_auc, cm = vm['balanced_accuracy'], vm['pr_auc'], vm['roc_auc'], vm['cm']
                print(
                    "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins | model: {}, seed: {}, lr: {}, weight_decay: {}, dropout: {}".format(
                        epoch + 1,
                        np.mean(losses),
                        acc,
                        pr_auc,
                        roc_auc,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60,
                        self.params.model,
                        self.params.seed,
                        self.params.lr,
                        self.params.weight_decay,
                        self.params.dropout
                    )
                )
                print(cm)
                tm = self.test_eval.get_metrics_for_binaryclass(self.model, self.criterion)
                if self.log_writer is not None:
                    self.log_writer.update(loss=float(np.mean(losses)), class_acc=float(np.mean(train_accs)), lr=optim_state['param_groups'][0]['lr'], grad_norm=float(np.mean(grad_norms)) if grad_norms else 0.0, head="train", step=epoch)
                    self.log_writer.update(balanced_accuracy=vm['balanced_accuracy'], accuracy=vm['accuracy'], pr_auc=vm['pr_auc'], roc_auc=vm['roc_auc'], head="val", step=epoch)
                    self.log_writer.update(balanced_accuracy=tm['balanced_accuracy'], accuracy=tm['accuracy'], pr_auc=tm['pr_auc'], roc_auc=tm['roc_auc'], head="test", step=epoch)
                    self.log_writer.flush()
                if self.output_dir:
                    os.makedirs(self.output_dir, exist_ok=True)
                    log_stats = {
                        'train_loss': float(np.mean(losses)),
                        'train_lr': optim_state['param_groups'][0]['lr'],
                        'train_min_lr': None,
                        'train_loss_scale': None,
                        'train_weight_decay': float(self.params.weight_decay),
                        'train_class_acc': float(np.mean(train_accs)) if train_accs else None,
                        'train_balanced_accuracy': float(balanced_accuracy_score(train_targets, train_preds)) if train_targets else None,
                        'train_grad_norm': float(np.mean(grad_norms)) if grad_norms else None,
                        'val_pr_auc': vm['pr_auc'], 'val_roc_auc': vm['roc_auc'],
                        'val_accuracy': vm['accuracy'], 'val_balanced_accuracy': vm['balanced_accuracy'],
                        'val_loss': vm['loss'],
                        'test_pr_auc': tm['pr_auc'], 'test_roc_auc': tm['roc_auc'],
                        'test_accuracy': tm['accuracy'], 'test_balanced_accuracy': tm['balanced_accuracy'],
                        'test_loss': tm['loss'],
                        'epoch': epoch,
                        'n_parameters': self.n_parameters,
                    }
                    with open(os.path.join(self.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                        f.write(json.dumps(log_stats) + "\n")
                self._save_checkpoint(epoch)
                if roc_auc > roc_auc_best:
                    print("roc_auc increasing....saving weights !! ")
                    print("Val Evaluation: acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(
                        acc,
                        pr_auc,
                        roc_auc,
                    ))
                    best_f1_epoch = epoch + 1
                    acc_best = acc
                    pr_auc_best = pr_auc
                    roc_auc_best = roc_auc
                    cm_best = cm
                    self.best_model_states = copy.deepcopy(self.model.state_dict())
                    self._save_checkpoint(epoch, 'checkpoint-best')
        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            tm_final = self.test_eval.get_metrics_for_binaryclass(self.model, self.criterion)
            acc, pr_auc, roc_auc, cm = tm_final['balanced_accuracy'], tm_final['pr_auc'], tm_final['roc_auc'], tm_final['cm']
            print("***************************Test results************************")
            print(
                "Test Evaluation: acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(
                    acc,
                    pr_auc,
                    roc_auc,
                )
            )
            print(cm)
            if not self.output_dir:
                if not os.path.isdir(self.params.model_dir):
                    os.makedirs(self.params.model_dir)
                model_path = self.params.model_dir + "/epoch{}_acc_{:.5f}_pr_{:.5f}_roc_{:.5f}.pth".format(best_f1_epoch, acc, pr_auc, roc_auc)
                torch.save(self.model.state_dict(), model_path)
                print("model save in " + model_path)


    def train_for_regression(self):
        corrcoef_best = 0
        r2_best = 0
        rmse_best = 0
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                pred = self.model(x)
                loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                corrcoef, r2, rmse = self.val_eval.get_metrics_for_regression(self.model)
                print(
                    "Epoch {} : Training Loss: {:.5f}, corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        corrcoef,
                        r2,
                        rmse,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                if r2 > r2_best:
                    print("kappa increasing....saving weights !! ")
                    print("Val Evaluation: corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}".format(
                        corrcoef,
                        r2,
                        rmse,
                    ))
                    best_r2_epoch = epoch + 1
                    corrcoef_best = corrcoef
                    r2_best = r2
                    rmse_best = rmse
                    self.best_model_states = copy.deepcopy(self.model.state_dict())

        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            corrcoef, r2, rmse = self.test_eval.get_metrics_for_regression(self.model)
            print("***************************Test results************************")
            print(
                "Test Evaluation: corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}".format(
                    corrcoef,
                    r2,
                    rmse,
                )
            )

            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = self.params.model_dir + "/epoch{}_corrcoef_{:.5f}_r2_{:.5f}_rmse_{:.5f}.pth".format(best_r2_epoch, corrcoef, r2, rmse)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)
