"""
MambaVision blocks adapted for OSTrack tracking framework.

This module contains MambaVision components adapted from:
/mnt/disk2/Shawalt/SOT/MambaVision/mambavision/models/mamba_vision.py

Key adaptations:
- Support for template+search token processing
- CE (Candidate Elimination) support in Attention blocks
- Segment embeddings for template/search distinction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import DropPath, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat


def candidate_elimination(attn: torch.Tensor, tokens: torch.Tensor, lens_t: int, keep_ratio: float, global_index: torch.Tensor, box_mask_z: torch.Tensor, prior_removed_mask: torch.Tensor = None):
    """
    Eliminate potential background candidates for computation reduction and noise cancellation.

    Args:
        attn (torch.Tensor): [B, num_heads, L_t + L_s, L_t + L_s], attention weights
        tokens (torch.Tensor): [B, L_t + L_s, C], template and search region tokens
        lens_t (int): length of template
        keep_ratio (float): keep ratio of search region tokens (candidates)
        global_index (torch.Tensor): global index of search region tokens
        box_mask_z (torch.Tensor): template mask used to accumulate attention weights
        prior_removed_mask (torch.Tensor): [B, L_s], mask of positions removed by previous CE stages (True = removed)

    Returns:
        tokens_new (torch.Tensor): tokens after candidate elimination
        keep_index (torch.Tensor): indices of kept search region tokens
        removed_index (torch.Tensor): indices of removed search region tokens
    """
    lens_s = attn.shape[-1] - lens_t
    bs, hn, _, _ = attn.shape

    lens_keep = math.ceil(keep_ratio * lens_s)
    effective_lens_s_min = lens_s
    if prior_removed_mask is not None:
        # Per-sample effective length (handle batch with different removal counts)
        removed_per_sample = prior_removed_mask.sum(dim=1)  # [B]
        effective_lens_s = lens_s - removed_per_sample  # [B]
        effective_lens_s = effective_lens_s.clamp(min=1)  # At least 1 token
        # Use the minimum effective length across batch to avoid out-of-bounds
        effective_lens_s_min = effective_lens_s.min().item()

    if lens_keep >= effective_lens_s_min:
        # If we want to keep more than available, cap at effective minimum
        lens_keep = max(1, int(effective_lens_s_min))

    # Extract template-to-search attention [B, heads, L_t, L_s]
    attn_t = attn[:, :, :lens_t, lens_t:]  # [B, H, L_t, L_s]

    # Simple average over heads and template positions to get [B, L_s]
    attn_t = attn_t.mean(dim=2).mean(dim=1)  # [B, L_s]

    # Apply prior_removed_mask (set removed positions to -inf so they won't be selected)
    if prior_removed_mask is not None:
        attn_t = attn_t.masked_fill(prior_removed_mask, float('-inf'))

    # If all positions are masked, return original tokens
    all_masked = (attn_t == float('-inf')).all()
    if all_masked:
        return tokens, global_index, None

    # Sort and select top-k tokens (same lens_keep for all samples in batch)
    sorted_attn, indices = torch.sort(attn_t, dim=1, descending=True)

    # Top-k indices and values [B, lens_keep]
    topk_idx = indices[:, :lens_keep]
    topk_attn = sorted_attn[:, :lens_keep]

    # Non-topk indices and values [B, lens_s - lens_keep]
    non_topk_idx = indices[:, lens_keep:]
    non_topk_attn = sorted_attn[:, lens_keep:]

    keep_index = global_index.gather(dim=1, index=topk_idx)
    removed_index = global_index.gather(dim=1, index=non_topk_idx)

    # Separate template and search tokens
    tokens_t = tokens[:, :lens_t]
    tokens_s = tokens[:, lens_t:]

    # Gather attentive tokens [B, lens_keep, C]
    B, L, C = tokens_s.shape
    attentive_tokens = tokens_s.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))

    # Concatenate: template + attentive tokens
    tokens_new = torch.cat([tokens_t, attentive_tokens], dim=1)

    return tokens_new, keep_index, removed_index


def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size: window size
    Returns:
        local window features (num_windows*B, window_size*window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size * window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: local window features (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, windows.shape[2], H, W)
    return x


class MambaVisionMixer(nn.Module):
    """Mamba SSM-based mixer adapted from MambaVision."""

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.x_proj = nn.Linear(
            self.d_inner // 2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias=True, **factory_kwargs)

        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner // 2, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner // 2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner // 2, device=device))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)

        A = -torch.exp(self.A_log.float())
        x = F.silu(
            F.conv1d(
                input=x,
                weight=self.conv1d_x.weight,
                bias=self.conv1d_x.bias,
                padding="same",
                groups=self.d_inner // 2,
            )
        )
        z = F.silu(
            F.conv1d(
                input=z,
                weight=self.conv1d_z.weight,
                bias=self.conv1d_z.bias,
                padding="same",
                groups=self.d_inner // 2,
            )
        )

        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=None,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=None,
        )

        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out


class Attention(nn.Module):
    """Standard multi-head attention from MambaVision."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attn=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn and hasattr(F, 'scaled_dot_product_attention'):
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
        else:
            # Fallback for PyTorch < 2.0
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attn:
            # Return attention weights for CE
            attn_weights = q @ k.transpose(-2, -1)
            return x, attn_weights
        return x


class Mlp(nn.Module):
    """MLP from timm."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """MambaVision Block that alternates between MambaVisionMixer and Attention."""

    def __init__(
        self,
        dim,
        num_heads,
        counter,
        transformer_blocks,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        Mlp_block=Mlp,
        layer_scale=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        if counter in transformer_blocks:
            self.mixer = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                norm_layer=norm_layer,
            )
            self.use_attn = True
        else:
            self.mixer = MambaVisionMixer(
                d_model=dim,
                d_state=8,
                d_conv=3,
                expand=1,
            )
            self.use_attn = False

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
        )

        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class CEBlock(nn.Module):
    """MambaVision Block with Candidate Elimination (CE) support.

    CE is only applied in Attention blocks, not in MambaVisionMixer.
    """

    def __init__(
        self,
        dim,
        num_heads,
        counter,
        transformer_blocks,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        Mlp_block=Mlp,
        layer_scale=None,
        keep_ratio_search=1.0,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        if counter in transformer_blocks:
            self.mixer = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                norm_layer=norm_layer,
            )
            self.use_attn = True
        else:
            self.mixer = MambaVisionMixer(
                d_model=dim,
                d_state=8,
                d_conv=3,
                expand=1,
            )
            self.use_attn = False

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
        )

        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1

        self.keep_ratio_search = keep_ratio_search

    def forward(self, x, global_index_t=None, global_index_s=None, ce_template_mask=None, ce_keep_rate=None, prior_removed_mask=None):
        """
        Forward with optional CE support.

        Args:
            x: input tensor [B, L, C]
            global_index_t: global index of template tokens [B, L_t]
            global_index_s: global index of search tokens [B, L_s]
            ce_template_mask: template mask for CE
            ce_keep_rate: keep ratio for this layer (overrides self.keep_ratio_search if provided)
            prior_removed_mask: mask of positions removed by previous CE stages [B, L_s]

        Returns:
            x: output tensor after CE
            global_index_t: updated template index
            global_index_s: updated search index (only kept tokens)
            removed_index_s: indices of removed search tokens
            attn: attention weights
        """
        # Compute attention with weights return for CE
        if self.use_attn:
            attn_out, attn = self.mixer(self.norm1(x), return_attn=True)
        else:
            attn_out = self.mixer(self.norm1(x))
            attn = None

        # Apply residual
        x = x + self.drop_path(self.gamma_1 * attn_out)

        lens_t = global_index_t.shape[1] if global_index_t is not None else 0
        removed_index_s = None

        # CE only applied in Attention layers when keep_ratio < 1
        effective_keep_ratio = ce_keep_rate if ce_keep_rate is not None else self.keep_ratio_search

        if self.use_attn and effective_keep_ratio < 1.0 and ce_template_mask is not None and global_index_s is not None:
            # Apply candidate elimination with prior_removed_mask
            x, global_index_s, removed_index_s = candidate_elimination(
                attn, x, lens_t, effective_keep_ratio, global_index_s, ce_template_mask, prior_removed_mask
            )

        # MLP
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))

        return x, global_index_t, global_index_s, removed_index_s, attn


class MambaVisionBlock(nn.Module):
    """MambaVision Block with optional CE support for OSTrack."""

    def __init__(
        self,
        dim,
        num_heads,
        counter,
        transformer_blocks,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        layer_scale=None,
        ce_keep_ratio=1.0,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.ce_keep_ratio = ce_keep_ratio

        if counter in transformer_blocks:
            self.mixer = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                norm_layer=norm_layer,
            )
            self.use_attn = True
        else:
            self.mixer = MambaVisionMixer(d_model=dim, d_state=8, d_conv=3, expand=1)
            self.use_attn = False

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
        )

        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1

    def forward(self, x, ce_template_mask=None, ce_keep_ratio=None):
        """
        Forward pass with optional CE support.
        CE is only applied in Attention blocks, not MambaVisionMixer.
        """
        if self.use_attn and self.ce_keep_ratio < 1.0 and ce_template_mask is not None:
            # CE path - for Phase 2 implementation
            return self._forward_with_ce(x, ce_template_mask, ce_keep_ratio)
        else:
            # Standard path
            x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            return x

    def _forward_with_ce(self, x, ce_template_mask, ce_keep_ratio):
        """Forward with Candidate Elimination (Phase 2)."""
        # TODO: Implement CE for MambaVision
        # This will use attention weights to prune search tokens
        raise NotImplementedError("CE not implemented yet in Phase 1")


class ConvBlock(nn.Module):
    """Convolutional block from MambaVision."""

    def __init__(self, dim, drop_path=0.0, layer_scale=None, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
        self.act1 = nn.GELU(approximate="tanh")
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)
        self.layer_scale = layer_scale

        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.layer_scale = True
        else:
            self.layer_scale = False

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1)
        x = input + self.drop_path(x)
        return x


class Downsample(nn.Module):
    """Downsampling block from MambaVision."""

    def __init__(self, dim, keep_dim=False):
        super().__init__()
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.reduction = nn.Sequential(nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False))

    def forward(self, x):
        x = self.reduction(x)
        return x
