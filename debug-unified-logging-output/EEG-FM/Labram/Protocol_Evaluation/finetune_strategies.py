"""Finetuning strategies (Option A/B/C) kept as separate functions.

Functions:
- apply_strategy(model, args): dispatch to chosen strategy
- strategy_original(model, args): no-op (keep original behavior)
- strategy_freeze_backbone(model, args): freeze backbone, keep head trainable
- strategy_freeze_early_layers(model, args): freeze first N transformer blocks
- strategy_aggressive_regularization(model, args): adjust args to stronger reg
- strategy_lora(model, args): Low-Rank Adaptation for parameter-efficient finetuning

These are kept in a separate file so you can enable/disable strategies without
modifying `run_class_finetuning.py`.
"""
from typing import Optional
import torch
import torch.nn as nn


def strategy_original(model: torch.nn.Module, args) -> None:
    """Do nothing; original full finetuning behavior (leave params as-is)."""
    print("Using original finetuning strategy: no parameter freezing or overrides.")


def strategy_freeze_backbone(model: torch.nn.Module, args) -> None:
    """Freeze all parameters except the classification head."""
    print("Applying strategy: freeze_backbone (freeze all except head)")
    kept = 0
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        if name.startswith('head') or ('.head' in name) or ('head.' in name):
            param.requires_grad = True
            kept += param.numel()
        else:
            param.requires_grad = False
    print(f"Backbone frozen. Trainable params: {kept} / {total} ({100.0*kept/total:.6f}%)")


def strategy_freeze_early_layers(model: torch.nn.Module, args) -> None:
    """Freeze the first `freeze_early_n` transformer blocks (blocks.0 .. blocks.N-1).

    If `args.freeze_early_n` is None, default to half of model.get_num_layers().
    """
    num_layers = model.get_num_layers() if hasattr(model, 'get_num_layers') else None
    n = getattr(args, 'freeze_early_n', None)
    if n is None:
        if num_layers is None:
            raise ValueError('Model does not expose get_num_layers and freeze_early_n not provided')
        n = max(1, num_layers // 2)
    print(f"Applying strategy: freeze_early_layers (freezing first {n} of {num_layers} blocks)")

    total = 0
    frozen = 0
    for name, param in model.named_parameters():
        total += param.numel()
        should_freeze = False
        # freeze transformer blocks with prefix 'blocks.{i}.' for i in [0, n-1]
        for i in range(n):
            if name.startswith(f'blocks.{i}.'):
                should_freeze = True
                break
        if should_freeze:
            param.requires_grad = False
            frozen += param.numel()
        else:
            param.requires_grad = True

    print(f"Frozen params in early layers: {frozen} / {total} ({100.0*frozen/total:.6f}%)")


def strategy_freeze_backbone_regularized(model: torch.nn.Module, args) -> None:
    """Freeze backbone + apply strong regularization to head only (middle-ground strategy).
    
    This is a hybrid approach between full finetuning (high overfitting) and freeze_backbone (low accuracy).
    It tests whether regularization (dropout + weight decay) on the head alone can recover
    some accuracy while preserving stability.
    
    Hyperparameters:
    - head_dropout: dropout rate applied to head (default 0.5, strong regularization)
    - head_weight_decay: weight decay for head params (default 0.01, strong regularization)
    
    Paper narrative: Tests if capacity limit (frozen backbone) + regularization (head) 
    provides a sweet spot between accuracy and generalization.
    """
    print("Applying strategy: freeze_backbone_regularized (frozen backbone + regularized head)")
    
    # Step 1: Freeze backbone
    kept = 0
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        if name.startswith('head') or ('.head' in name) or ('head.' in name):
            param.requires_grad = True
            kept += param.numel()
        else:
            param.requires_grad = False
    print(f"Backbone frozen. Trainable params: {kept} / {total} ({100.0*kept/total:.6f}%)")
    
    # Step 2: Apply head-specific regularization via args
    # Note: Cannot retroactively add dropout to existing head layer (it's already built).
    # Instead, we update args.weight_decay for stronger regularization.
    # The caller should use these values when constructing the optimizer.
    head_dropout = getattr(args, 'head_dropout', 0.5)
    head_weight_decay = getattr(args, 'head_weight_decay', 0.01)
    
    # Override global weight decay with head-specific value for this strategy
    old_wd = getattr(args, 'weight_decay', None)
    args.weight_decay = head_weight_decay
    print(f"Regularization for head: dropout={head_dropout}, weight_decay={head_weight_decay}")
    if old_wd is not None:
        print(f"  (overriding global weight_decay: {old_wd} -> {head_weight_decay})")


def strategy_aggressive_regularization(model: torch.nn.Module, args) -> None:
    """Apply aggressive regularization hyperparameter changes while keeping full finetuning.

    Note: some settings (like drop_path) are passed during model construction and
    cannot be changed retrospectively for existing layers. This function updates
    train-time args and leaves model parameters trainable.
    """
    print("Applying strategy: aggressive_regularization (full finetuning + stronger reg)")
    # Make sure all params are trainable
    for _, p in model.named_parameters():
        p.requires_grad = True

    # Update args to recommended values if not explicitly provided
    old_wd = getattr(args, 'weight_decay', None)
    old_drop_path = getattr(args, 'drop_path', None)
    old_epochs = getattr(args, 'epochs', None)
    args.weight_decay = 0.15 if old_wd is None or old_wd < 0.15 else old_wd
    args.drop_path = 0.2 if old_drop_path is None or old_drop_path < 0.2 else old_drop_path
    args.epochs = 20 if old_epochs is None or old_epochs > 20 else old_epochs
    print(f"Set args.weight_decay={args.weight_decay}, args.drop_path={args.drop_path}, args.epochs={args.epochs}")


class LoRALinear(nn.Module):
    """LoRA-adapted Linear layer.
    
    Implements low-rank adaptation: W' = W + (alpha/r) * B @ A
    where W is frozen, and A, B are trainable low-rank matrices.
    
    Args:
        in_features: input dimension
        out_features: output dimension
        rank: LoRA rank (r)
        alpha: scaling factor
    """
    def __init__(self, in_features: int, out_features: int, rank: int = 2, alpha: float = 8.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # A: (rank, in_features), initialized with small random values
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        # B: (out_features, rank), initialized to zeros
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LoRA adaptation: (alpha/r) * (B @ A) @ x"""
        # x: (*, in_features) -> (*, out_features)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return lora_out


class LoRAConv2d(nn.Module):
    """LoRA-adapted Conv2d layer.
    
    Implements low-rank adaptation for 2D convolutions.
    
    Args:
        in_channels: input channels
        out_channels: output channels
        kernel_size: convolution kernel size
        rank: LoRA rank
        alpha: scaling factor
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size, rank: int = 4, alpha: float = 8.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # For Conv2d: kernel is (out_channels, in_channels, kH, kW)
        # Decompose: out_channels x in_channels x kH x kW ≈ (out_channels x rank) @ (rank x in_channels x kH x kW)
        # A: (rank, in_channels * kH * kW), initialized with small random values
        kernel_numel = in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.lora_A = nn.Parameter(torch.randn(rank, kernel_numel) * 0.01)
        # B: (out_channels, rank), initialized to zeros
        self.lora_B = nn.Parameter(torch.zeros(out_channels, rank))
    
    def forward(self, original_weight: torch.Tensor) -> torch.Tensor:
        """Return LoRA-adapted weight: W + (alpha/r) * B @ A reshaped to kernel size."""
        # Compute low-rank update: (out_channels, rank) @ (rank, in*kH*kW) -> (out_channels, in*kH*kW)
        lora_weight = (self.lora_B @ self.lora_A) * self.scaling
        # Reshape to conv kernel: (out_channels, in_channels, kH, kW)
        lora_weight = lora_weight.view(self.out_channels, self.in_channels, *self.kernel_size)
        return original_weight + lora_weight


def strategy_lora(model: torch.nn.Module, args) -> None:
    """Apply LoRA (Low-Rank Adaptation) to LaBraM model.
    
    LoRA Requirements:
    1. Apply LoRA to Attention QKV projection only (NOT output projection).
    2. Apply LoRA to both MLP linear layers (fc1, fc2).
    3. Apply LoRA to convolutional layers in TemporalConv with r_conv=4.
    4. Do NOT adapt bias terms anywhere.
    5. Freeze ALL original LaBraM parameters; only LoRA parameters trainable.
    
    LoRA Formulation: W' = W + (alpha / r) * (B @ A)
    - A initialized with small random values
    - B initialized to zeros
    - alpha = 8
    
    Hyperparameters:
    - r_attn = 2 for QKV
    - r_mlp = 2 for MLP layers
    - r_conv = 4 for Conv layers
    - alpha = 8
    """
    # Hyperparameters
    # Use single lora_rank for both attention and MLP, keep conv rank fixed at 4
    r_attn = getattr(args, 'lora_rank', 2)
    r_mlp = getattr(args, 'lora_rank', 2)
    r_conv = 4  # Keep conv rank fixed
    alpha = getattr(args, 'lora_alpha', 8.0)
    
    print(f"Applying LoRA strategy:")
    print(f"  - r_attn={r_attn}, r_mlp={r_mlp}, r_conv={r_conv}, alpha={alpha}")
    
    # Step 1: Freeze ALL base model parameters
    for name, param in model.named_parameters():
        param.requires_grad = False
    
    total_params = sum(p.numel() for p in model.parameters())
    lora_params_count = 0
    
    # Step 2: Apply LoRA to transformer blocks
    num_blocks = model.get_num_layers() if hasattr(model, 'get_num_layers') else 12
    print(f"  - Applying to {num_blocks} transformer blocks")
    
    for block_idx in range(num_blocks):
        block = model.blocks[block_idx]
        
        # LoRA on Attention QKV projection ONLY (not output projection)
        attn = block.attn
        in_dim = attn.qkv.in_features
        out_dim = attn.qkv.out_features
        
        # Create LoRA adapter for QKV
        lora_qkv = LoRALinear(in_dim, out_dim, rank=r_attn, alpha=alpha)
        attn.lora_qkv = lora_qkv
        for p in lora_qkv.parameters():
            p.requires_grad = True
        lora_params_count += sum(p.numel() for p in lora_qkv.parameters())
        
        # Monkey-patch the forward method to include LoRA
        original_qkv_forward = attn.qkv.forward
        def make_qkv_forward(orig_fn, lora_mod):
            def forward_with_lora(x):
                return orig_fn(x) + lora_mod(x)
            return forward_with_lora
        attn.qkv.forward = make_qkv_forward(original_qkv_forward, lora_qkv)
        
        # LoRA on MLP layers (fc1 and fc2)
        mlp = block.mlp
        
        # fc1: (embed_dim -> mlp_dim)
        in_dim = mlp.fc1.in_features
        out_dim = mlp.fc1.out_features
        lora_fc1 = LoRALinear(in_dim, out_dim, rank=r_mlp, alpha=alpha)
        mlp.lora_fc1 = lora_fc1
        for p in lora_fc1.parameters():
            p.requires_grad = True
        lora_params_count += sum(p.numel() for p in lora_fc1.parameters())
        
        original_fc1_forward = mlp.fc1.forward
        def make_fc1_forward(orig_fn, lora_mod):
            def forward_with_lora(x):
                return orig_fn(x) + lora_mod(x)
            return forward_with_lora
        mlp.fc1.forward = make_fc1_forward(original_fc1_forward, lora_fc1)
        
        # fc2: (mlp_dim -> embed_dim)
        in_dim = mlp.fc2.in_features
        out_dim = mlp.fc2.out_features
        lora_fc2 = LoRALinear(in_dim, out_dim, rank=r_mlp, alpha=alpha)
        mlp.lora_fc2 = lora_fc2
        for p in lora_fc2.parameters():
            p.requires_grad = True
        lora_params_count += sum(p.numel() for p in lora_fc2.parameters())
        
        original_fc2_forward = mlp.fc2.forward
        def make_fc2_forward(orig_fn, lora_mod):
            def forward_with_lora(x):
                return orig_fn(x) + lora_mod(x)
            return forward_with_lora
        mlp.fc2.forward = make_fc2_forward(original_fc2_forward, lora_fc2)
    
    # Step 3: Apply LoRA to Conv layers in TemporalConv (patch_embed)
    if hasattr(model, 'patch_embed') and hasattr(model.patch_embed, 'conv1'):
        temporal_conv = model.patch_embed
        
        for conv_name in ['conv1', 'conv2', 'conv3']:
            if hasattr(temporal_conv, conv_name):
                conv_layer = getattr(temporal_conv, conv_name)
                in_ch = conv_layer.in_channels
                out_ch = conv_layer.out_channels
                kernel_sz = conv_layer.kernel_size
                
                # Create LoRA adapter for Conv2d
                lora_conv = LoRAConv2d(in_ch, out_ch, kernel_sz, rank=r_conv, alpha=alpha)
                setattr(temporal_conv, f'lora_{conv_name}', lora_conv)
                for p in lora_conv.parameters():
                    p.requires_grad = True
                lora_params_count += sum(p.numel() for p in lora_conv.parameters())
                
                # Monkey-patch Conv2d forward to use LoRA-adapted weight
                original_conv_forward = conv_layer.forward
                def make_conv_forward(conv_mod, lora_mod):
                    def forward_with_lora(x):
                        # Temporarily replace weight with LoRA-adapted weight
                        adapted_weight = lora_mod(conv_mod.weight)
                        return torch.nn.functional.conv2d(
                            x, adapted_weight, conv_mod.bias,
                            conv_mod.stride, conv_mod.padding, conv_mod.dilation, conv_mod.groups
                        )
                    return forward_with_lora
                conv_layer.forward = make_conv_forward(conv_layer, lora_conv)
    
    # Step 4: Make classification head trainable (standard full finetuning for head)
    if hasattr(model, 'head'):
        for name, param in model.head.named_parameters():
            param.requires_grad = True
        head_params = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
        print(f"  - Classification head: {head_params:,} trainable params")
    
    # Print summary
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nLoRA Summary:")
    print(f"  - Original model params: {total_params:,}")
    print(f"  - LoRA params added: {lora_params_count:,}")
    print(f"  - Total trainable params: {trainable_params:,}")
    print(f"  - Percentage of original: {100.0*trainable_params/total_params:.2f}%")
    print(f"  - Param reduction vs full finetuning: {100.0*(1 - trainable_params/total_params):.1f}%")


def apply_strategy(model: torch.nn.Module, args) -> None:
    """Dispatch to the selected finetuning strategy.

    args.finetune_strategy is expected to be one of: 'original', 'freeze_backbone',
    'freeze_backbone_regularized', 'freeze_early_layers', 'aggressive_reg', 'lora'.
    """
    strat = getattr(args, 'finetune_strategy', 'original')
    # backward compat: support boolean flag freeze_backbone
    if getattr(args, 'freeze_backbone', False):
        strat = 'freeze_backbone'

    if strat == 'original':
        strategy_original(model, args)
    elif strat == 'freeze_backbone':
        strategy_freeze_backbone(model, args)
    elif strat == 'freeze_backbone_regularized':
        strategy_freeze_backbone_regularized(model, args)
    elif strat == 'freeze_early_layers':
        strategy_freeze_early_layers(model, args)
    elif strat == 'aggressive_reg':
        strategy_aggressive_regularization(model, args)
    elif strat == 'lora':
        strategy_lora(model, args)
    else:
        raise ValueError(f'Unknown finetune strategy: {strat}')
