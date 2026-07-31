"""CKA (Centered Kernel Alignment) analysis: FT vs LoRA checkpoints.

Compares internal representations of LaBraM fine-tuned with two strategies:
  - Original full fine-tuning (FT)
  - LoRA (r=2)

Steps:
  1. Load both checkpoints via the same inference pipeline as layer_swap_inference.py
  2. Register forward hooks on all 12 transformer blocks to collect activations
  3. Run the TUAB test set through both models
  4. Compute:
     a) Layer-wise CKA: CKA(FT_layer_i, LoRA_layer_i) for i = 1..12
     b) Full 12×12 cross-layer heatmap: CKA(FT_layer_i, LoRA_layer_j)
  5. Save results as JSON + PNG plots

Usage (from /home/meriem-ubuntu/Projects/Foundation-models/Labram):
    conda run -n labram python Overfitting_Evaluation/cka_analysis.py

Optional args:
    --ckpt_ft    path to FT best checkpoint   (default: ../CKA/checkpoint-best-original-42-TUAB.pth)
    --ckpt_lora  path to LoRA best checkpoint (default: ../CKA/checkpoint-best-lora_r2-42-TUAB.pth)
    --device     cpu | cuda                   (default: auto)
    --output_dir directory for results        (default: Results/CKA)
    --batch_size batch size for inference     (default: 64)
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from einops import rearrange

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from run_class_finetuning import get_models, get_dataset
from utils import get_input_chans
from types import SimpleNamespace


# ─────────────────────────────────────────────────────────────────────────────
# CKA
# ─────────────────────────────────────────────────────────────────────────────

def linear_CKA(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA between two activation matrices (N x D)."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    XTX = X @ X.T
    YTY = Y @ Y.T
    numerator = np.sum(XTX * YTY)
    denom = np.sqrt(np.sum(XTX * XTX) * np.sum(YTY * YTY))
    if denom == 0:
        return 0.0
    return float(numerator / denom)


# ─────────────────────────────────────────────────────────────────────────────
# Model helpers (mirrors layer_swap_inference.py)
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


def load_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    model_args = get_default_args()
    model = get_models(model_args)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def load_lora_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    """Load a LoRA checkpoint: apply strategy first so monkey-patched forwards are live."""
    from Overfitting_Evaluation.finetune_strategies import strategy_lora
    lora_args = SimpleNamespace(lora_rank=2, lora_alpha=8.0)
    model_args = get_default_args()
    model = get_models(model_args)
    strategy_lora(model, lora_args)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Activation extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_block_activations(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    batch_size: int = 64,
    num_blocks: int = 12,
) -> dict:
    """
    Run full test set through model, collect CLS-token (or mean-pool)
    output of each transformer block via forward hooks.

    Returns: dict {block_idx: np.ndarray of shape (N, embed_dim)}
    """
    _, _, _, ch_names, _ = get_dataset(get_default_args())
    input_chans = get_input_chans(ch_names) if ch_names is not None else None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    # Register hooks on every block
    activations = {i: [] for i in range(num_blocks)}
    hooks = []

    def make_hook(block_idx):
        def hook_fn(module, input, output):
            # output shape: (B, seq_len, embed_dim) — take mean over tokens
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            # mean pool over the sequence dimension → (B, embed_dim)
            rep = out.mean(dim=1).detach().cpu().float().numpy()
            activations[block_idx].append(rep)
        return hook_fn

    for i in range(num_blocks):
        h = model.blocks[i].register_forward_hook(make_hook(i))
        hooks.append(h)

    with torch.no_grad():
        for x, _ in loader:
            x = x.float().to(device) / 100
            x = rearrange(x, 'B N (A T) -> B N A T', T=200)
            model(x, input_chans=input_chans)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Concatenate batches → (N, embed_dim) per block
    return {i: np.concatenate(activations[i], axis=0) for i in range(num_blocks)}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CKA analysis: FT vs LoRA")
    parser.add_argument("--ckpt_ft",
        default=os.path.join(os.path.dirname(__file__), "../../CKA/checkpoint-best-original-42-TUAB.pth"),
        help="Path to FT best checkpoint")
    parser.add_argument("--ckpt_lora",
        default=os.path.join(os.path.dirname(__file__), "../../CKA/checkpoint-best-lora_r2-42-TUAB.pth"),
        help="Path to LoRA best checkpoint")
    parser.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    parser.add_argument("--output_dir", default="Results/CKA", help="Where to save results")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=12)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print("CKA ANALYSIS: FT vs LoRA (TUAB test set, seed 42)")
    print(f"{'='*70}")
    print(f"FT checkpoint  : {args.ckpt_ft}")
    print(f"LoRA checkpoint: {args.ckpt_lora}")
    print(f"Device         : {device}")
    print(f"Output dir     : {args.output_dir}")
    print(f"{'='*70}\n")

    # ── Load dataset ────────────────────────────────────────────────────────
    print("Loading TUAB test set...")
    _, test_dataset, _, _, _ = get_dataset(get_default_args())
    print(f"  ✓ {len(test_dataset)} test samples\n")

    # ── Load models ─────────────────────────────────────────────────────────
    print("Loading FT model...")
    model_ft = load_model(args.ckpt_ft, device)
    print("  ✓ FT model loaded\n")

    print("Loading LoRA model...")
    model_lora = load_lora_model(args.ckpt_lora, device)
    print("  ✓ LoRA model loaded\n")

    # ── Extract activations ──────────────────────────────────────────────────
    print("Extracting activations from FT model (full test set)...")
    acts_ft = extract_block_activations(model_ft, test_dataset, device, args.batch_size, args.num_blocks)
    print("  ✓ FT activations extracted\n")
    del model_ft
    torch.cuda.empty_cache()

    print("Extracting activations from LoRA model (full test set)...")
    acts_lora = extract_block_activations(model_lora, test_dataset, device, args.batch_size, args.num_blocks)
    print("  ✓ LoRA activations extracted\n")
    del model_lora
    torch.cuda.empty_cache()

    # ── Layer-wise CKA (diagonal) ────────────────────────────────────────────
    print("Computing layer-wise CKA (FT_i vs LoRA_i)...")
    layerwise_cka = []
    for i in range(args.num_blocks):
        cka = linear_CKA(acts_ft[i], acts_lora[i])
        layerwise_cka.append(cka)
        print(f"  Block {i+1:2d}: CKA = {cka:.4f}")

    # ── Full 12×12 cross-layer heatmap ───────────────────────────────────────
    print("\nComputing full 12×12 cross-layer CKA heatmap...")
    heatmap = np.zeros((args.num_blocks, args.num_blocks))
    for i in range(args.num_blocks):
        for j in range(args.num_blocks):
            heatmap[i, j] = linear_CKA(acts_ft[i], acts_lora[j])
    print("  ✓ Heatmap computed\n")

    # ── Save JSON results ────────────────────────────────────────────────────
    results = {
        "description": "CKA analysis: FT vs LoRA, TUAB test set, seed 42",
        "num_blocks": args.num_blocks,
        "num_samples": len(test_dataset),
        "layerwise_cka": {f"block_{i+1}": v for i, v in enumerate(layerwise_cka)},
        "heatmap_FT_row_LoRA_col": heatmap.tolist(),
    }
    json_path = os.path.join(args.output_dir, "cka_ft_vs_lora_TUAB_seed42.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {json_path}")

    # ── Plots ────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 10,
            "pdf.fonttype": 42,
        })

        # (a) Layer-wise bar chart
        fig, ax = plt.subplots(figsize=(7, 3.5))
        blocks = [f"B{i+1}" for i in range(args.num_blocks)]
        colors = ["#2166AC" if v >= 0.9 else "#4DA0D0" if v >= 0.7 else "#B2522E"
                  for v in layerwise_cka]
        ax.bar(blocks, layerwise_cka, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Transformer Block")
        ax.set_ylabel("Linear CKA")
        ax.set_title("Layer-wise CKA: FT vs LoRA (TUAB, seed 42)")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        fig.tight_layout()
        bar_path = os.path.join(args.output_dir, "cka_layerwise.png")
        fig.savefig(bar_path, dpi=200, bbox_inches="tight")
        bar_path_pdf = bar_path.replace(".png", ".pdf")
        fig.savefig(bar_path_pdf, bbox_inches="tight")
        plt.close(fig)
        print(f"Layer-wise plot → {bar_path}")

        # (b) 12×12 heatmap
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(heatmap, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_xticks(range(args.num_blocks))
        ax.set_yticks(range(args.num_blocks))
        ax.set_xticklabels([f"L{i+1}" for i in range(args.num_blocks)], fontsize=8)
        ax.set_yticklabels([f"L{i+1}" for i in range(args.num_blocks)], fontsize=8)
        ax.set_xlabel("LoRA block")
        ax.set_ylabel("FT block")
        ax.set_title("CKA Heatmap: FT (rows) vs LoRA (cols)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        heatmap_path = os.path.join(args.output_dir, "cka_heatmap.png")
        fig.savefig(heatmap_path, dpi=200, bbox_inches="tight")
        heatmap_path_pdf = heatmap_path.replace(".png", ".pdf")
        fig.savefig(heatmap_path_pdf, bbox_inches="tight")
        plt.close(fig)
        print(f"Heatmap plot    → {heatmap_path}")

    except Exception as e:
        print(f"Warning: plotting failed ({e}). JSON results are still saved.")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Mean layer-wise CKA : {np.mean(layerwise_cka):.4f}")
    print(f"  Min  (most diverged): Block {np.argmin(layerwise_cka)+1} → {np.min(layerwise_cka):.4f}")
    print(f"  Max  (most similar) : Block {np.argmax(layerwise_cka)+1} → {np.max(layerwise_cka):.4f}")
    print(f"\n  Interpretation:")
    print(f"    CKA ≈ 1.0 → FT and LoRA learned nearly identical representations")
    print(f"    CKA ≈ 0.0 → representations diverged completely")
    print(f"    Lower CKA in later blocks → LoRA's low-rank constraint shapes")
    print(f"                                 task-specific layers differently")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
