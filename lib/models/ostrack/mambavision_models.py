"""
MambaVision model variants for OSTrack.

Model configurations:
- T (Tiny):   dim=80,  depths=[1,3,8,4],  window_size=[8,8,14,7]
- S (Small): dim=96,  depths=[3,3,7,5],   window_size=[8,8,14,7]
- B (Base):  dim=128, depths=[3,3,10,5],  window_size=[8,8,14,7]
- L (Large): dim=196, depths=[3,3,10,5],  window_size=[8,8,14,7]
"""

import os
import torch

from .mambavision_tim import MambaVisionTrack


# Model configurations
MAMBAVISION_CONFIGS = {
    "tiny": {
        "dim": 80,
        "depths": [1, 3, 8, 4],
        "window_size": [8, 8, 16, 16],
        "num_heads": [2, 4, 8, 16],
        "in_dim": 32,
    },
    "small": {
        "dim": 96,
        "depths": [3, 3, 7, 5],
        "window_size": [8, 8, 16, 16],
        "num_heads": [2, 4, 8, 16],
        "in_dim": 32,
    },
    "base": {
        "dim": 128,
        "depths": [3, 3, 10, 5],
        "window_size": [8, 8, 16, 16],
        "num_heads": [2, 4, 8, 16],
        "in_dim": 64,
    },
    "large": {
        "dim": 196,
        "depths": [3, 3, 10, 5],
        "window_size": [8, 8, 16, 16],
        "num_heads": [2, 4, 8, 16],
        "in_dim": 64,
    },
}


def mamba_vision_tiny_patch16_224(pretrained=False, drop_path_rate=0.2, **kwargs):
    """MambaVision-Tiny without CE."""
    config = MAMBAVISION_CONFIGS["tiny"]
    model = MambaVisionTrack(
        embed_dim=config["dim"],
        depths=config["depths"],
        window_size=config["window_size"],
        num_heads=config["num_heads"],
        in_dim=config["in_dim"],
        drop_path_rate=drop_path_rate,
        layer_scale=1e-5,
        layer_scale_conv=None,
        ce_loc=None,
        ce_keep_ratio=None,
        **kwargs,
    )
    if pretrained:
        model = load_mambavision_pretrained(model, pretrained)
    return model


def mamba_vision_tiny_patch16_224_ce(
    pretrained=False, drop_path_rate=0.2, ce_loc=None, ce_keep_ratio=None, **kwargs
):
    """MambaVision-Tiny with CE."""
    config = MAMBAVISION_CONFIGS["tiny"]
    model = MambaVisionTrack(
        embed_dim=config["dim"],
        depths=config["depths"],
        window_size=config["window_size"],
        num_heads=config["num_heads"],
        in_dim=config["in_dim"],
        drop_path_rate=drop_path_rate,
        layer_scale=1e-5,
        layer_scale_conv=None,
        ce_loc=ce_loc,
        ce_keep_ratio=ce_keep_ratio,
        **kwargs,
    )
    if pretrained:
        model = load_mambavision_pretrained(model, pretrained)
    return model


def mamba_vision_small_patch16_224(pretrained=False, drop_path_rate=0.2, **kwargs):
    """MambaVision-Small without CE."""
    config = MAMBAVISION_CONFIGS["small"]
    model = MambaVisionTrack(
        embed_dim=config["dim"],
        depths=config["depths"],
        window_size=config["window_size"],
        num_heads=config["num_heads"],
        in_dim=config["in_dim"],
        drop_path_rate=drop_path_rate,
        layer_scale=1e-5,
        layer_scale_conv=None,
        ce_loc=None,
        ce_keep_ratio=None,
        **kwargs,
    )
    if pretrained:
        model = load_mambavision_pretrained(model, pretrained)
    return model


def mamba_vision_small_patch16_224_ce(
    pretrained=False, drop_path_rate=0.2, ce_loc=None, ce_keep_ratio=None, **kwargs
):
    """MambaVision-Small with CE."""
    config = MAMBAVISION_CONFIGS["small"]
    model = MambaVisionTrack(
        embed_dim=config["dim"],
        depths=config["depths"],
        window_size=config["window_size"],
        num_heads=config["num_heads"],
        in_dim=config["in_dim"],
        drop_path_rate=drop_path_rate,
        layer_scale=1e-5,
        layer_scale_conv=None,
        ce_loc=ce_loc,
        ce_keep_ratio=ce_keep_ratio,
        **kwargs,
    )
    if pretrained:
        model = load_mambavision_pretrained(model, pretrained)
    return model


def mamba_vision_base_patch16_224(pretrained=False, drop_path_rate=0.2, **kwargs):
    """MambaVision-Base without CE."""
    config = MAMBAVISION_CONFIGS["base"]
    model = MambaVisionTrack(
        embed_dim=config["dim"],
        depths=config["depths"],
        window_size=config["window_size"],
        num_heads=config["num_heads"],
        in_dim=config["in_dim"],
        drop_path_rate=drop_path_rate,
        layer_scale=1e-5,
        layer_scale_conv=None,
        ce_loc=None,
        ce_keep_ratio=None,
        **kwargs,
    )
    if pretrained:
        model = load_mambavision_pretrained(model, pretrained)
    return model


def mamba_vision_base_patch16_224_ce(
    pretrained=False, drop_path_rate=0.2, ce_loc=None, ce_keep_ratio=None, **kwargs
):
    """MambaVision-Base with CE."""
    config = MAMBAVISION_CONFIGS["base"]
    model = MambaVisionTrack(
        embed_dim=config["dim"],
        depths=config["depths"],
        window_size=config["window_size"],
        num_heads=config["num_heads"],
        in_dim=config["in_dim"],
        drop_path_rate=drop_path_rate,
        layer_scale=1e-5,
        layer_scale_conv=None,
        ce_loc=ce_loc,
        ce_keep_ratio=ce_keep_ratio,
        **kwargs,
    )
    if pretrained:
        model = load_mambavision_pretrained(model, pretrained)
    return model


def mamba_vision_large_patch16_224(pretrained=False, drop_path_rate=0.2, **kwargs):
    """MambaVision-Large without CE."""
    config = MAMBAVISION_CONFIGS["large"]
    model = MambaVisionTrack(
        embed_dim=config["dim"],
        depths=config["depths"],
        window_size=config["window_size"],
        num_heads=config["num_heads"],
        in_dim=config["in_dim"],
        drop_path_rate=drop_path_rate,
        layer_scale=1e-5,
        layer_scale_conv=None,
        ce_loc=None,
        ce_keep_ratio=None,
        **kwargs,
    )
    if pretrained:
        model = load_mambavision_pretrained(model, pretrained)
    return model


def mamba_vision_large_patch16_224_ce(
    pretrained=False, drop_path_rate=0.2, ce_loc=None, ce_keep_ratio=None, **kwargs
):
    """MambaVision-Large with CE."""
    config = MAMBAVISION_CONFIGS["large"]
    model = MambaVisionTrack(
        embed_dim=config["dim"],
        depths=config["depths"],
        window_size=config["window_size"],
        num_heads=config["num_heads"],
        in_dim=config["in_dim"],
        drop_path_rate=drop_path_rate,
        layer_scale=1e-5,
        layer_scale_conv=None,
        ce_loc=ce_loc,
        ce_keep_ratio=ce_keep_ratio,
        **kwargs,
    )
    if pretrained:
        model = load_mambavision_pretrained(model, pretrained)
    return model


def load_mambavision_pretrained(model, pretrained_path, map_location="cpu"):
    """Load MambaVision pretrained weights.

    Args:
        model: MambaVisionTrack model
        pretrained_path: path to pretrained weights
        map_location: device to load weights to

    Returns:
        model with loaded weights
    """
    if not os.path.exists(pretrained_path):
        print(f"Pretrained path {pretrained_path} does not exist, skipping pretrained loading")
        return model

    checkpoint = torch.load(pretrained_path, map_location=map_location)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Filter out keys that don't match
    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    skipped_keys = []
    loaded_keys = []

    for k, v in state_dict.items():
        if k in model_state_dict:
            if v.shape == model_state_dict[k].shape:
                filtered_state_dict[k] = v
                loaded_keys.append(k)
            else:
                skipped_keys.append((k, f'shape mismatch {v.shape} vs {model_state_dict[k].shape}'))
        else:
            skipped_keys.append((k, 'not in model'))

    model.load_state_dict(filtered_state_dict, strict=False)

    # Print statistics by stage
    total_pretrained = len(state_dict)
    print(f"Loaded pretrained weights from {pretrained_path}")
    print(f"  Total pretrained keys: {total_pretrained}")
    print(f"  Successfully loaded: {len(loaded_keys)} ({100*len(loaded_keys)/total_pretrained:.1f}%)")
    print(f"  Skipped: {len(skipped_keys)}")

    # Print loaded keys by stage
    for stage in range(4):
        stage_loaded = [k for k in loaded_keys if f'levels.{stage}.' in k]
        if stage_loaded:
            print(f"  Stage {stage} loaded: {len(stage_loaded)} keys")

    # Print skipped keys summary
    if skipped_keys:
        skipped_categories = {}
        for k, reason in skipped_keys:
            if 'head.' in k:
                cat = 'head'
            elif 'norm.' in k:
                cat = 'norm'
            else:
                cat = 'other'
            skipped_categories.setdefault(cat, []).append(k)
        print(f"  Skipped categories:")
        for cat, keys in skipped_categories.items():
            print(f"    {cat}: {len(keys)} keys")

    return model
