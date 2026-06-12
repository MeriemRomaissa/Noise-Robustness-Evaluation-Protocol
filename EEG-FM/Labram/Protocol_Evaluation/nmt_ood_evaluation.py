"""NMT OOD evaluation: LaBraM-Original, LaBraM-LoRA, and EEGNet.

Compares "Clean" performance (best checkpoint on TUAB test set, hard-coded
from training logs) against "OOD" performance (inference on the NMT Scalp-EEG
dataset, a different hospital/population/device).

The NMT dataset was preprocessed to match the LaBraM/TUAB format:
  - 23 channels (21 real + 2 zero-padded: T1, T2)
  - 200 Hz, 10-second segments → shape (23, 2000)
  - Saved as pickles with keys 'X' (float32, μV) and 'y' (0=normal, 1=abnormal)

EEGNet uses a separate conda environment (eengnetp2).  Run the helper first:
    /home/meriem-ubuntu/miniconda3/envs/eengnetp2/bin/python \\
        /home/meriem-ubuntu/Projects/Foundation-models/nmt_scalp_eeg_labram_ood/run_eegnet_nmt_inference.py

That saves nmt_eegnet_results.json next to the script.  Then run this script
(from the Labram directory) in the labram environment:
    /home/meriem-ubuntu/miniconda3/envs/labram/bin/python \\
        Overfitting_Evaluation/nmt_ood_evaluation.py

Optional flags:
    --ckpt_ft           path to FT checkpoint
    --ckpt_lora         path to LoRA checkpoint
    --eegnet_json       path to pre-computed EEGNet NMT results JSON
    --nmt_dir           path to NMT test pickles
    --output_dir        where to save results  (default: Results/NMT_OOD)
    --batch_size        inference batch size   (default: 128)
    --device            cpu | cuda             (default: auto)
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from einops import rearrange
from types import SimpleNamespace

_LABRAM_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, _LABRAM_ROOT)
from run_class_finetuning import get_models
from utils import get_input_chans
from Overfitting_Evaluation.finetune_strategies import strategy_lora

# ─────────────────────────────────────────────────────────────────────────────
# Hard-coded TUAB clean performance (best-checkpoint-by-val-accuracy metrics
# on the TUAB test split, seed 42).  These were read directly from log.txt.
# ─────────────────────────────────────────────────────────────────────────────

CLEAN_TUAB = {
    "ft": {
        "accuracy":          0.8228,
        "balanced_accuracy": 0.8186,
        "auc":               0.9040,
        "epoch":             1,
    },
    "lora": {
        "accuracy":          0.8186,
        "balanced_accuracy": 0.8144,
        "auc":               0.8981,
        "epoch":             22,
    },
    "eegnet": {
        "accuracy":          0.7869,
        "balanced_accuracy": 0.7821,
        "auc":               0.8536,
        "epoch":             11,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class _NumpyCompatUnpickler(pickle.Unpickler):
    """Remap numpy._core → numpy.core so pickles from numpy>=2.0 load in numpy<2.0."""
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def _load_pickle(path):
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError:
            f.seek(0)
            return _NumpyCompatUnpickler(f).load()


class PickleDataset(Dataset):
    def __init__(self, root: str):
        self.root = root
        self.files = sorted(f for f in os.listdir(root) if f.endswith(".pkl"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = _load_pickle(os.path.join(self.root, self.files[idx]))
        X = torch.FloatTensor(sample["X"])
        y = int(sample["y"])
        return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _default_args(dataset="TUAB"):
    return SimpleNamespace(
        model="labram_base_patch200_200",
        nb_classes=1,
        input_size=200,
        qkv_bias=False,
        rel_pos_bias=True,
        abs_pos_emb=True,
        layer_scale_init_value=0.1,
        drop=0.0,
        attn_drop_rate=0.0,
        drop_path=0.1,
        use_mean_pooling=True,
        init_scale=0.001,
        dataset=dataset,
    )


def load_ft_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    model = get_models(_default_args())
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model


def load_lora_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    lora_args = SimpleNamespace(lora_rank=2, lora_alpha=8.0)
    model = get_models(_default_args())
    strategy_lora(model, lora_args)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model


def load_eegnet_nmt_results(json_path: str) -> dict:
    """Load pre-computed EEGNet NMT results produced by run_eegnet_nmt_inference.py."""
    with open(json_path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_labram_inference(model, dataset, device, batch_size, input_chans) -> dict:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=(device.type == "cuda"))
    all_labels, all_probs = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.float().to(device) / 100.0
            x = rearrange(x, 'B N (A T) -> B N A T', T=200)
            logits = model(x, input_chans=input_chans)
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
            all_labels.append(np.array(y))
            all_probs.append(probs)
    labels = np.concatenate(all_labels)
    probs  = np.concatenate(all_probs)
    preds  = (probs > 0.5).astype(int)
    return {
        "accuracy":          float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "auc":               float(roc_auc_score(labels, probs)),
        "n_samples":         int(len(labels)),
    }




# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(results: dict, output_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.size": 10, "pdf.fonttype": 42})

    x_labels = ["TUAB (Clean)", "NMT (OOD)"]
    x_pos    = [0, 1]

    models = [
        ("ft",     "LaBraM Full FT",  "#1f4e79", "-",  "o"),
        ("lora",   "LaBraM LoRA (r=2)", "#c0392b", "--", "s"),
        ("eegnet", "EEGNet",           "#27ae60", "-.", "^"),
    ]

    for metric in ("accuracy", "balanced_accuracy", "auc"):
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        for mkey, mlabel, color, ls, marker in models:
            clean_val = results[mkey]["tuab_clean"][metric]
            ood_val   = results[mkey]["nmt_ood"][metric]
            ax.plot(x_pos, [clean_val, ood_val],
                    marker=marker, linewidth=1.8,
                    color=color, linestyle=ls, label=mlabel)
            ax.annotate(f"{clean_val:.3f}", xy=(0, clean_val),
                        xytext=(-0.06, clean_val), fontsize=7.5,
                        ha="right", va="center", color=color)
            ax.annotate(f"{ood_val:.3f}", xy=(1, ood_val),
                        xytext=(1.06, ood_val), fontsize=7.5,
                        ha="left", va="center", color=color)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.set_xlim(-0.3, 1.5)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"OOD Robustness — {metric.replace('_',' ').title()}\n"
                     f"(TUAB best ckpt → NMT, seed 42)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            path = os.path.join(output_dir, f"nmt_ood_{metric}.{ext}")
            fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  {metric} plot saved.")

    # Relative robustness (clean = 1.0)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for mkey, mlabel, color, ls, marker in models:
        clean_acc = results[mkey]["tuab_clean"]["accuracy"]
        ood_acc   = results[mkey]["nmt_ood"]["accuracy"]
        rel       = [1.0, ood_acc / clean_acc]
        ax.plot(x_pos, rel, marker=marker, linewidth=1.8,
                color=color, linestyle=ls, label=mlabel)
        ax.annotate(f"{rel[1]:.3f}", xy=(1, rel[1]),
                    xytext=(1.06, rel[1]), fontsize=7.5,
                    ha="left", va="center", color=color)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlim(-0.3, 1.5)
    ax.set_ylabel("Relative accuracy  (TUAB clean = 1.0)")
    ax.set_title("Relative OOD Robustness (TUAB → NMT, seed 42)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(output_dir, f"nmt_ood_relative_robustness.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Relative robustness plot saved.")

    # OOD accuracy drop bar chart
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    bar_labels = [m[1] for m in models]
    drops = [
        results[mkey]["tuab_clean"]["accuracy"] - results[mkey]["nmt_ood"]["accuracy"]
        for mkey, *_ in models
    ]
    colors = ["#1f4e79", "#c0392b", "#27ae60"]
    bars = ax.bar(range(len(models)), drops, color=colors,
                  edgecolor="black", linewidth=0.6, width=0.5)
    for bar, drop in zip(bars, drops):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{drop:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(bar_labels, fontsize=9)
    ax.set_ylabel("Accuracy drop  (TUAB clean − NMT OOD)")
    ax.set_title("Accuracy drop under NMT domain shift (seed 42)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(output_dir, f"nmt_ood_accuracy_drop.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Accuracy drop bar chart saved.")

    # Combined 3-metric bar chart: grouped by model, paired Clean vs OOD
    metrics_to_plot = [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Bal. Accuracy"),
        ("auc", "ROC-AUC"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    bar_w = 0.28
    x = np.arange(len(models))
    for ax, (metric, metric_label) in zip(axes, metrics_to_plot):
        clean_vals = [results[mkey]["tuab_clean"][metric] for mkey, *_ in models]
        ood_vals   = [results[mkey]["nmt_ood"][metric]   for mkey, *_ in models]
        b1 = ax.bar(x - bar_w / 2, clean_vals, width=bar_w,
                    label="TUAB (Clean)", color="#5b9bd5", edgecolor="black", linewidth=0.5)
        b2 = ax.bar(x + bar_w / 2, ood_vals,   width=bar_w,
                    label="NMT (OOD)",   color="#ed7d31", edgecolor="black", linewidth=0.5)
        for bar in list(b1) + list(b2):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([m[1] for m in models], fontsize=8)
        ax.set_title(metric_label)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        if ax == axes[0]:
            ax.legend(fontsize=8)
    fig.suptitle("Clean (TUAB) vs OOD (NMT) performance by model and metric",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(output_dir, f"nmt_ood_grouped_bars.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Grouped bar chart saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

_NMT_BASE = "/home/meriem-ubuntu/Projects/Foundation-models/nmt_scalp_eeg_labram_ood"
_CKPT_BASE = os.path.join(_NMT_BASE, "Checkpoints")


def main():
    parser = argparse.ArgumentParser(description="NMT OOD evaluation: FT vs LoRA vs EEGNet")
    parser.add_argument("--ckpt_ft",
        default=os.path.join(_CKPT_BASE, "checkpoint-best-original-42-TUAB.pth"))
    parser.add_argument("--ckpt_lora",
        default=os.path.join(_CKPT_BASE, "checkpoint-best-lora_r2-42-TUAB.pth"))
    parser.add_argument("--eegnet_json",
        default=os.path.join(_NMT_BASE, "nmt_eegnet_results.json"),
        help="Pre-computed EEGNet NMT results from run_eegnet_nmt_inference.py")
    parser.add_argument("--nmt_dir",
        default=os.path.join(_NMT_BASE, "test"))
    parser.add_argument("--output_dir", default="Results/NMT_OOD")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print("NMT OOD EVALUATION: LaBraM-FT vs LaBraM-LoRA vs EEGNet")
    print(f"{'='*70}")
    print(f"FT checkpoint    : {args.ckpt_ft}")
    print(f"LoRA checkpoint  : {args.ckpt_lora}")
    print(f"EEGNet JSON      : {args.eegnet_json}")
    print(f"NMT test dir     : {args.nmt_dir}")
    print(f"Results dir      : {args.output_dir}")
    print(f"Device           : {device}")
    print(f"{'='*70}\n")

    # ── 1. NMT dataset ──────────────────────────────────────────────────────
    print("─── Step 1: Loading NMT dataset ───")
    nmt_ds = PickleDataset(args.nmt_dir)
    print(f"  NMT samples: {len(nmt_ds)}\n")

    # ── 2. Channel mapping (LaBraM input_chans) ──────────────────────────────
    # TUAB channel names, matching the order used during fine-tuning.
    # (Derived from get_dataset() in run_class_finetuning.py.)
    TUAB_CH = [
        "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
        "F7", "F8", "T3", "T4", "T5", "T6", "A1", "A2", "FZ", "CZ", "PZ", "T1", "T2",
    ]
    input_chans = get_input_chans(TUAB_CH)

    # ── 3. Load EEGNet pre-computed results ──────────────────────────────────
    print("─── Step 2: Loading EEGNet NMT results ───")
    if not os.path.isfile(args.eegnet_json):
        raise FileNotFoundError(
            f"EEGNet NMT results not found: {args.eegnet_json}\n"
            "Run the helper script first:\n"
            "  /home/meriem-ubuntu/miniconda3/envs/eengnetp2/bin/python \\\n"
            "      /home/meriem-ubuntu/Projects/Foundation-models/"
            "nmt_scalp_eeg_labram_ood/run_eegnet_nmt_inference.py"
        )
    eegnet_nmt = load_eegnet_nmt_results(args.eegnet_json)
    print(f"  EEGNet NMT: acc={eegnet_nmt['accuracy']:.4f}  "
          f"bal_acc={eegnet_nmt['balanced_accuracy']:.4f}  auc={eegnet_nmt['auc']:.4f}\n")

    # ── 4. Load LaBraM models ────────────────────────────────────────────────
    print("─── Step 3: Loading LaBraM models ───")
    print("  Loading FT model...")
    model_ft = load_ft_model(args.ckpt_ft, device)
    print("  Loading LoRA model...")
    model_lora = load_lora_model(args.ckpt_lora, device)
    print()

    # ── 5. Run LaBraM inference on NMT ──────────────────────────────────────
    print("─── Step 4: Running LaBraM NMT inference ───")

    print("  [FT] on NMT...")
    ft_nmt = run_labram_inference(model_ft, nmt_ds, device, args.batch_size, input_chans)
    print(f"    acc={ft_nmt['accuracy']:.4f}  bal_acc={ft_nmt['balanced_accuracy']:.4f}  auc={ft_nmt['auc']:.4f}")

    print("  [LoRA] on NMT...")
    lora_nmt = run_labram_inference(model_lora, nmt_ds, device, args.batch_size, input_chans)
    print(f"    acc={lora_nmt['accuracy']:.4f}  bal_acc={lora_nmt['balanced_accuracy']:.4f}  auc={lora_nmt['auc']:.4f}\n")

    del model_ft, model_lora
    torch.cuda.empty_cache()

    # ── 5. Assemble results ──────────────────────────────────────────────────
    results = {
        "ft": {
            "tuab_clean": CLEAN_TUAB["ft"],
            "nmt_ood":    ft_nmt,
        },
        "lora": {
            "tuab_clean": CLEAN_TUAB["lora"],
            "nmt_ood":    lora_nmt,
        },
        "eegnet": {
            "tuab_clean": CLEAN_TUAB["eegnet"],
            "nmt_ood":    eegnet_nmt,
        },
        "description": (
            "NMT OOD evaluation. 'tuab_clean' = TUAB test-set metrics at the "
            "best-by-val-accuracy checkpoint (seed 42, read from log.txt). "
            "'nmt_ood' = inference on NMT Scalp-EEG preprocessed to LaBraM/TUAB format."
        ),
    }

    json_path = os.path.join(args.output_dir, "nmt_ood_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved → {json_path}")

    # ── 6. Plot ──────────────────────────────────────────────────────────────
    print("\n─── Step 5: Plotting ───")
    try:
        plot_results(results, args.output_dir)
    except Exception as e:
        print(f"  Warning: plotting failed ({e}). JSON results are still saved.")
        raise

    # ── 7. Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Model':<22} {'TUAB acc':>10} {'NMT acc':>10} {'Drop':>10} {'TUAB auc':>10} {'NMT auc':>10}")
    print(f"  {'-'*72}")
    for mkey, mlabel in [("ft", "LaBraM Full FT"), ("lora", "LaBraM LoRA r=2"), ("eegnet", "EEGNet")]:
        ca = results[mkey]["tuab_clean"]["accuracy"]
        oa = results[mkey]["nmt_ood"]["accuracy"]
        cu = results[mkey]["tuab_clean"]["auc"]
        ou = results[mkey]["nmt_ood"]["auc"]
        print(f"  {mlabel:<22} {ca:>10.4f} {oa:>10.4f} {ca-oa:>+10.4f} {cu:>10.4f} {ou:>10.4f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
