"""OOD robustness evaluation: FT vs LoRA under Gaussian noise perturbations.

Paper narrative: if FT's early-peak dynamics reflect overfitting to the training
distribution, it should degrade faster under synthetic distribution shift than
LoRA's stable trajectory.  CKA already showed top layers (B11-B12) diverge
between the two strategies — those are the layers most likely to be sensitive
to input distribution changes.

Noise levels (sigma in raw μV, before the /100 normalisation applied at inference):
  Level 1: sigma =  2   → SNR ≈  18 dB  (barely perceptible)
  Level 2: sigma =  5   → SNR ≈  10 dB  (mild)
  Level 3: sigma = 10   → SNR ≈   4 dB  (moderate)
  Level 4: sigma = 20   → SNR ≈  −2 dB  (strong — noise ≈ signal std)
  Level 5: sigma = 40   → SNR ≈  −8 dB  (very strong — signal overwhelmed)
(SNR computed relative to median test-set signal std ≈ 16 μV)

Steps:
  1. Generate 5 perturbed copies of the TUAB test set → save as pickle files
     under Out_distribution/Level_{1..5}/ alongside the processed/ folder.
  2. Run inference with FT and LoRA on original + 5 noisy sets.
  3. Compute accuracy, balanced accuracy, AUC at each level.
  4. Plot degradation curves and relative robustness curves.
  5. Save JSON + PNG/PDF results.

Usage (from the Labram directory):
    /home/meriem-ubuntu/miniconda3/envs/labram/bin/python \\
        Overfitting_Evaluation/ood_robustness.py

Optional flags:
    --skip_generate   skip dataset generation if Out_distribution/ already exists
    --ckpt_ft         path to FT checkpoint   (default: ../../CKA/checkpoint-best-original-42-TUAB.pth)
    --ckpt_lora       path to LoRA checkpoint (default: ../../CKA/checkpoint-best-lora_r2-42-TUAB.pth)
    --output_dir      where to save results    (default: Results/OOD)
    --batch_size      inference batch size     (default: 64)
    --device          cpu | cuda               (default: auto)
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

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from run_class_finetuning import get_models, get_dataset
from utils import get_input_chans

# ─────────────────────────────────────────────────────────────────────────────
# Noise configuration
# ─────────────────────────────────────────────────────────────────────────────

NOISE_LEVELS = {
    1: 2.0,
    2: 5.0,
    3: 10.0,
    4: 20.0,
    5: 40.0,
}
NOISE_SEED = 42   # fixed seed so generated datasets are deterministic


# ─────────────────────────────────────────────────────────────────────────────
# Dataset generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_noisy_datasets(test_dir: str, out_base: str, skip_if_exists: bool = False):
    """
    For each noise level, copy every pickle from test_dir, add iid Gaussian
    noise to X, and save the result under out_base/Level_{k}/.

    The noise is added to the raw signal (before /100 normalisation), so sigma
    is in the same units as the original data (μV).
    """
    files = sorted(os.listdir(test_dir))
    print(f"  Source test files: {len(files)}")

    for level, sigma in NOISE_LEVELS.items():
        level_dir = os.path.join(out_base, f"Level_{level}")
        if skip_if_exists and os.path.isdir(level_dir) and len(os.listdir(level_dir)) == len(files):
            print(f"  Level {level} (σ={sigma:5.1f} μV) — already exists, skipping.")
            continue

        os.makedirs(level_dir, exist_ok=True)
        rng = np.random.default_rng(NOISE_SEED + level)   # deterministic per level

        for fname in files:
            src_path = os.path.join(test_dir, fname)
            dst_path = os.path.join(level_dir, fname)
            sample = pickle.load(open(src_path, "rb"))
            X = sample["X"].astype(np.float64)
            noise = rng.normal(0.0, sigma, size=X.shape)
            noisy_sample = {"X": X + noise, "y": sample["y"]}
            with open(dst_path, "wb") as f:
                pickle.dump(noisy_sample, f)

        print(f"  Level {level} (σ={sigma:5.1f} μV) → {level_dir}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loader (matches TUABLoader in utils.py)
# ─────────────────────────────────────────────────────────────────────────────

class PickleDataset(Dataset):
    def __init__(self, root: str):
        self.root = root
        self.files = sorted(os.listdir(root))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = pickle.load(open(os.path.join(self.root, self.files[idx]), "rb"))
        X = torch.FloatTensor(sample["X"])
        y = int(sample["y"])
        return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (mirrors cka_analysis.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_default_args(dataset="TUAB"):
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
    model = get_models(get_default_args())
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def load_lora_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    from Overfitting_Evaluation.finetune_strategies import strategy_lora
    lora_args = SimpleNamespace(lora_rank=2, lora_alpha=8.0)
    model = get_models(get_default_args())
    strategy_lora(model, lora_args)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(model: torch.nn.Module, dataset: Dataset,
                  device: torch.device, batch_size: int,
                  input_chans) -> dict:
    """Return dict with accuracy, balanced_accuracy, auc."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=(device.type == "cuda"))

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

    keys_all = ["original"] + [f"Level_{k}" for k in sorted(NOISE_LEVELS)]
    sigmas   = [0.0] + [NOISE_LEVELS[k] for k in sorted(NOISE_LEVELS)]
    x_labels = ["Clean"] + [f"σ={NOISE_LEVELS[k]:.0f}" for k in sorted(NOISE_LEVELS)]
    x_pos    = range(len(sigmas))

    models = [
        ("ft",   "Full FT",   "#1f4e79", "-",  "o"),
        ("lora", "LoRA (r=2)", "#c0392b", "--", "s"),
    ]

    for metric in ("accuracy", "balanced_accuracy", "auc"):
        fig, ax = plt.subplots(figsize=(7, 3.8))
        for mkey, mlabel, color, ls, marker in models:
            vals = [results[mkey][lk][metric] for lk in keys_all]
            ax.plot(x_pos, vals, marker=marker, linewidth=1.8,
                    color=color, linestyle=ls, label=mlabel)
        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Gaussian noise level (σ, μV)")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"Robustness to Gaussian Noise — "
                     f"{metric.replace('_',' ').title()} (FT vs LoRA, TUAB, seed 42)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            path = os.path.join(output_dir, f"ood_robustness_{metric}.{ext}")
            fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  {metric} plot saved.")

    # Relative robustness (normalised to each model's clean performance = 1.0)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for mkey, mlabel, color, ls, marker in models:
        accs = [results[mkey][lk]["accuracy"] for lk in keys_all]
        rel  = [v / accs[0] for v in accs]
        ax.plot(x_pos, rel, marker=marker, linewidth=1.8,
                color=color, linestyle=ls, label=mlabel)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Gaussian noise level (σ, μV)")
    ax.set_ylabel("Relative accuracy  (clean = 1.0)")
    ax.set_title("Relative Robustness: FT vs LoRA (TUAB, seed 42)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(output_dir, f"ood_relative_robustness.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Relative robustness plot saved.")

    # Accuracy gap bars: FT − LoRA at each noise level
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ft_accs = [results["ft"][lk]["accuracy"]   for lk in keys_all]
    lr_accs = [results["lora"][lk]["accuracy"] for lk in keys_all]
    gaps    = [f - l for f, l in zip(ft_accs, lr_accs)]
    colors  = ["#2166AC" if g >= 0 else "#c0392b" for g in gaps]
    ax.bar(x_pos, gaps, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Gaussian noise level (σ, μV)")
    ax.set_ylabel("Accuracy gap  (FT − LoRA)")
    ax.set_title("Accuracy gap between FT and LoRA under noise")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(output_dir, f"ood_accuracy_gap.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Accuracy gap plot saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OOD robustness: FT vs LoRA under Gaussian noise")
    _dir = os.path.dirname(__file__)
    parser.add_argument("--ckpt_ft",
        default=os.path.join(_dir, "../../CKA/checkpoint-best-original-42-TUAB.pth"))
    parser.add_argument("--ckpt_lora",
        default=os.path.join(_dir, "../../CKA/checkpoint-best-lora_r2-42-TUAB.pth"))
    parser.add_argument("--test_dir",
        default="/home/meriem-ubuntu/Projects/LaBraM/Datasets_FineTune/"
                "tuh_eeg_abnormal/v3.0.1/edf/processed/test")
    parser.add_argument("--out_distribution_dir",
        default="/home/meriem-ubuntu/Projects/LaBraM/Datasets_FineTune/"
                "tuh_eeg_abnormal/v3.0.1/edf/processed/Out_distribution")
    parser.add_argument("--output_dir", default="Results/OOD")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip_generate", action="store_true",
                        help="Skip dataset generation if Level_* dirs already exist")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.out_distribution_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print("OOD ROBUSTNESS: FT vs LoRA — Gaussian noise perturbations")
    print(f"{'='*70}")
    print(f"FT checkpoint  : {args.ckpt_ft}")
    print(f"LoRA checkpoint: {args.ckpt_lora}")
    print(f"Test dir       : {args.test_dir}")
    print(f"OOD output dir : {args.out_distribution_dir}")
    print(f"Results dir    : {args.output_dir}")
    print(f"Device         : {device}")
    print(f"{'='*70}\n")

    # ── 1. Generate noisy datasets ───────────────────────────────────────────
    print("─── Step 1: Generating noisy test sets ───")
    generate_noisy_datasets(
        test_dir=args.test_dir,
        out_base=args.out_distribution_dir,
        skip_if_exists=args.skip_generate,
    )

    # ── 2. Build dataset objects ─────────────────────────────────────────────
    print("─── Step 2: Building datasets ───")
    datasets = {"original": PickleDataset(args.test_dir)}
    for level in sorted(NOISE_LEVELS):
        level_dir = os.path.join(args.out_distribution_dir, f"Level_{level}")
        datasets[f"Level_{level}"] = PickleDataset(level_dir)
    print(f"  Datasets: {list(datasets.keys())}")
    print(f"  Samples per dataset: {len(datasets['original'])}\n")

    # ── 3. Get channel mapping (same for all datasets) ───────────────────────
    _, _, _, ch_names, _ = get_dataset(get_default_args())
    input_chans = get_input_chans(ch_names) if ch_names is not None else None

    # ── 4. Load models ───────────────────────────────────────────────────────
    print("─── Step 3: Loading models ───")
    print("  Loading FT model...")
    model_ft = load_ft_model(args.ckpt_ft, device)
    print("  Loading LoRA model...")
    model_lora = load_lora_model(args.ckpt_lora, device)
    print()

    # ── 5. Run inference on all datasets × both models ───────────────────────
    print("─── Step 4: Running inference ───")
    results = {"ft": {}, "lora": {}}

    for ds_name, ds in datasets.items():
        sigma_str = "clean" if ds_name == "original" else f"σ={NOISE_LEVELS[int(ds_name.split('_')[1])]:.0f} μV"
        print(f"  [{ds_name}] ({sigma_str})")

        metrics_ft = run_inference(model_ft, ds, device, args.batch_size, input_chans)
        results["ft"][ds_name] = metrics_ft
        print(f"    FT   → acc={metrics_ft['accuracy']:.4f}  bal_acc={metrics_ft['balanced_accuracy']:.4f}  auc={metrics_ft['auc']:.4f}")

        metrics_lora = run_inference(model_lora, ds, device, args.batch_size, input_chans)
        results["lora"][ds_name] = metrics_lora
        print(f"    LoRA → acc={metrics_lora['accuracy']:.4f}  bal_acc={metrics_lora['balanced_accuracy']:.4f}  auc={metrics_lora['auc']:.4f}")

    del model_ft, model_lora
    torch.cuda.empty_cache()

    # ── 6. Save JSON ─────────────────────────────────────────────────────────
    results["noise_levels_sigma_uv"] = NOISE_LEVELS
    results["noise_seed"] = NOISE_SEED
    results["description"] = (
        "OOD robustness: FT vs LoRA under Gaussian noise. "
        "Sigma values are in raw μV (before /100 normalisation). "
        "Median test-set signal std ≈ 16 μV."
    )
    json_path = os.path.join(args.output_dir, "ood_robustness_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {json_path}")

    # ── 7. Plots ─────────────────────────────────────────────────────────────
    print("\n─── Step 5: Plotting ───")
    try:
        plot_results(results, args.output_dir)
    except Exception as e:
        print(f"  Warning: plotting failed ({e}). JSON results are still saved.")

    # ── 8. Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("ROBUSTNESS SUMMARY")
    print(f"{'='*70}")
    keys     = ["original"] + [f"Level_{k}" for k in sorted(NOISE_LEVELS)]
    sigmas   = [0.0] + [NOISE_LEVELS[k] for k in sorted(NOISE_LEVELS)]
    ft_accs  = [results["ft"][k]["accuracy"]   for k in keys]
    lor_accs = [results["lora"][k]["accuracy"] for k in keys]

    print(f"  {'Level':<12} {'σ (μV)':<10} {'FT acc':<10} {'LoRA acc':<10} {'FT−LoRA gap':>12}")
    print(f"  {'-'*60}")
    for k, sig, fa, la in zip(keys, sigmas, ft_accs, lor_accs):
        label = "Clean" if k == "original" else k
        print(f"  {label:<12} {sig:<10.1f} {fa:<10.4f} {la:<10.4f} {fa-la:+12.4f}")

    ft_drop   = ft_accs[0]  - ft_accs[-1]
    lora_drop = lor_accs[0] - lor_accs[-1]
    print(f"\n  FT   accuracy drop (clean → Level 5): {ft_drop:+.4f}")
    print(f"  LoRA accuracy drop (clean → Level 5): {lora_drop:+.4f}")
    most_robust = "FT" if ft_drop <= lora_drop else "LoRA"
    print(f"  Most robust model: {most_robust} (smallest accuracy drop under max noise)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
