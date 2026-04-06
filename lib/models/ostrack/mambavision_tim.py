"""
MambaVision backbone adapted for OSTrack tracking framework.

Architecture (matching original MambaVision with downsample=True for Stage 0,1,2):
- Stage 0: PatchEmbed (stride 4) → H/4
- Stage 1: ConvBlock (stride 2, downsample) → H/8
- Stage 2: MambaVisionMixer/Attention (stride 2, downsample) → H/32
- Stage 3: MambaVisionMixer/Attention (no downsample) → H/32

Tracking workflow:
- Template (128×128) → Stage 0,1,2 → 4×4 = 16 tokens → flatten
- Search (256×256)   → Stage 0,1,2 → 8×8 = 64 tokens → window_partition
- Concatenated: 16 + 64 = 80 tokens → Stage 3 → Head

Note: Current output is [B, 80, 128] = 16 template + 64 search tokens.
OSTrack expects [B, 320, 128] = 64 template + 256 search tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from functools import partial

from ..layers.mamba_blocks import (
    MambaVisionMixer,
    Attention,
    Mlp,
    Block,
    ConvBlock,
    Downsample,
    window_partition,
    window_reverse,
)


class PatchEmbed(nn.Module):
    """Patch embedding from MambaVision (stride 4)."""

    def __init__(self, in_chans=3, in_dim=64, dim=96):
        super().__init__()
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv2d(in_dim, dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        return x


class MambaVisionLayer(nn.Module):
    """MambaVision layer with optional window partitioning."""

    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size,
        conv=False,
        downsample=True,
        expand_channel=False,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        layer_scale=None,
        layer_scale_conv=None,
        transformer_blocks=[],
        use_window_partition=True,
        stage_ce_loc=None,
        stage_ce_keep_ratio=None,
    ):
        super().__init__()
        self.conv = conv
        self.transformer_block = False

        # CE configuration for this stage
        # stage_ce_loc: indices within transformer_blocks where CE is applied
        # stage_ce_keep_ratio: keep ratios corresponding to stage_ce_loc
        self.stage_ce_loc = stage_ce_loc or []
        self.stage_ce_keep_ratio = stage_ce_keep_ratio or []

        if conv:
            self.blocks = nn.ModuleList(
                [
                    ConvBlock(
                        dim=dim,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        layer_scale=layer_scale_conv,
                    )
                    for i in range(depth)
                ]
            )
            self.transformer_block = False
        else:
            self.blocks = nn.ModuleList()
            for i in range(depth):
                # Check if this is an Attention layer (counter in transformer_blocks)
                is_attn_layer = i in transformer_blocks

                # Calculate local CE index within this stage's attention layers
                local_attn_idx = None
                if is_attn_layer and transformer_blocks:
                    # Find the index within transformer_blocks
                    attn_layer_indices = [j for j in transformer_blocks if j < depth]
                    if i in attn_layer_indices:
                        local_attn_idx = attn_layer_indices.index(i)

                # Determine if CE should be applied to this layer
                apply_ce = is_attn_layer and local_attn_idx is not None and local_attn_idx in self.stage_ce_loc

                if apply_ce:
                    # Get the keep_ratio for this layer
                    ce_idx = self.stage_ce_loc.index(local_attn_idx)
                    keep_ratio = self.stage_ce_keep_ratio[ce_idx] if ce_idx < len(self.stage_ce_keep_ratio) else 1.0
                    from ..layers.mamba_blocks import CEBlock
                    block = CEBlock(
                        dim=dim,
                        counter=i,
                        transformer_blocks=transformer_blocks,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        layer_scale=layer_scale,
                        keep_ratio_search=keep_ratio,
                    )
                else:
                    from ..layers.mamba_blocks import Block
                    block = Block(
                        dim=dim,
                        counter=i,
                        transformer_blocks=transformer_blocks,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        layer_scale=layer_scale,
                    )
                self.blocks.append(block)
            self.transformer_block = True

        self.downsample = None if not downsample else Downsample(dim=dim)
        # Channel expansion without spatial downsample (e.g., for Stage 2 to preserve resolution)
        self.channel_expand = None
        if expand_channel and not downsample:
            self.channel_expand = nn.Sequential(
                nn.Conv2d(dim, dim * 2, 1, bias=False),
                nn.BatchNorm2d(dim * 2, eps=1e-4),
            )
        self.window_size = window_size
        self.use_window_partition = use_window_partition

    def forward(self, x, global_index_t=None, global_index_s=None, ce_template_mask=None, ce_keep_rate=None, prior_removed_mask=None):
        """
        Forward pass with optional CE support.

        Args:
            x: input tensor [B, L, C] or [B, C, H, W]
            global_index_t: global index of template tokens [B, L_t]
            global_index_s: global index of search tokens [B, L_s]
            ce_template_mask: template mask for CE
            ce_keep_rate: keep ratio for CE (can be None to use block's default)
            prior_removed_mask: mask of positions removed by previous CE stages [B, L_s]

        Returns:
            x: output tensor
            global_index_t: updated template index
            global_index_s: updated search index
            removed_index_s: indices of removed search tokens (list)
            attn: last attention weights
        """
        # Check if input is already sequence format (B, L, C)
        if len(x.shape) == 3:
            # Input is already flattened sequence
            B, L, C = x.shape
            H, W = None, None  # Unknown spatial dims
            input_is_sequence = True
        else:
            # Input is spatial format (B, C, H, W)
            _, _, H, W = x.shape
            B = x.shape[0]
            input_is_sequence = False

        if self.transformer_block and self.use_window_partition and not input_is_sequence:
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            if pad_r > 0 or pad_b > 0:
                x = torch.nn.functional.pad(x, (0, pad_r, 0, pad_b))
                _, _, Hp, Wp = x.shape
            else:
                Hp, Wp = H, W
            x = window_partition(x, self.window_size)
        elif self.transformer_block and not self.use_window_partition and not input_is_sequence:
            # Flatten spatial dimensions to sequence: (B, C, H, W) -> (B, H*W, C)
            x = x.flatten(2).transpose(1, 2)

        removed_indexes_s = []
        last_attn = None

        for _, blk in enumerate(self.blocks):
            # Pass CE parameters if the block supports them
            if hasattr(blk, 'keep_ratio_search'):
                x, global_index_t, global_index_s, removed_idx_s, attn = blk(
                    x, global_index_t, global_index_s, ce_template_mask, ce_keep_rate, prior_removed_mask
                )
                if removed_idx_s is not None:
                    removed_indexes_s.append(removed_idx_s)
                if attn is not None:
                    last_attn = attn
            else:
                x = blk(x)

        if self.transformer_block and self.use_window_partition and not input_is_sequence:
            x = window_reverse(x, self.window_size, Hp, Wp)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()
        elif self.transformer_block and not self.use_window_partition and not input_is_sequence:
            # Reshape back to spatial: (B, L, C) -> (B, C, H, W)
            x = x.transpose(1, 2).view(B, -1, H, W)
            if self.downsample is not None:
                x = self.downsample(x)
            elif self.channel_expand is not None:
                # CE was applied in sequence format, pad back to original spatial shape with zeros
                # Original sequence length was H * W before flattening
                orig_L = H * W
                curr_L = x.shape[1]
                if curr_L < orig_L:
                    # Pad with zeros to restore original length
                    pad_len = orig_L - curr_L
                    pad_zeros = torch.zeros(B, pad_len, x.shape[2], device=x.device, dtype=x.dtype)
                    x = torch.cat([x, pad_zeros], dim=1)
                x = self.channel_expand(x)
            # Stage 0,1,2 output in spatial format
            return x
        elif self.downsample is None or input_is_sequence:
            # Only return tuple when operating in sequence format (Stage 3)
            return x, global_index_t, global_index_s, removed_indexes_s, last_attn
        else:
            # Spatial format output with downsample - Stage 0,1
            return self.downsample(x)


class MambaVisionTrack(nn.Module):
    """MambaVision backbone adapted for OSTrack tracking.

    Processes template and search regions separately through early stages,
    then concatenates and processes through transformer stages.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=128,
        depths=[3, 3, 10, 5],
        window_size=[8, 8, 14, 7],
        num_heads=[2, 4, 8, 16],
        mlp_ratio=4.0,
        drop_path_rate=0.2,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        layer_scale=None,
        layer_scale_conv=None,
        in_dim=64,
        ce_loc=None,
        ce_keep_ratio=None,
        **kwargs,
    ):
        super().__init__()

        self.num_features = int(embed_dim * 2 ** (len(depths) - 1))
        self.embed_dim = embed_dim
        self.depths = depths
        self.window_size = window_size
        self.num_heads = num_heads
        self.in_dim = in_dim

        # Patch embedding for early stages
        self.patch_embed = PatchEmbed(in_chans=in_chans, in_dim=in_dim, dim=embed_dim)

        # Drop path rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Create stages
        self.levels = nn.ModuleList()

        # Calculate CE configuration for Stage 2 and 3
        # ce_loc specifies which Attention layers (globally across Stage 2 and 3) use CE
        # For MambaVision-Base with depths=[3,3,10,5]:
        #   Stage 2 has 5 Attention layers (local indices 5-9)
        #   Stage 3 has 2 Attention layers (local indices 3-4)
        #   Combined: 7 Attention layers (global indices 0-6)
        stage_ce_loc = [ce_loc, ce_keep_ratio] if ce_loc else [None, None]

        for i in range(len(depths)):
            conv = True if (i == 0 or i == 1) else False
            use_win_partition = False if i >= 2 else True  # Skip window partition for Stage 2,3

            # Calculate transformer_blocks for this stage
            if depths[i] % 2 != 0:
                transformer_blocks = list(range(depths[i] // 2 + 1, depths[i]))
            else:
                transformer_blocks = list(range(depths[i] // 2, depths[i]))

            # For Stage 2 and 3, calculate per-stage CE config
            stage_ce_loc_i = None
            stage_ce_keep_ratio_i = None
            if ce_loc is not None and i >= 2:
                # Check if ce_loc is per-stage (nested list) or global (flat list)
                # Per-stage format: ce_loc = [[stage2_locs], [stage3_locs]]
                # Global format: ce_loc = [global_indices]
                if isinstance(ce_loc[0], (list, tuple)):
                    # Per-stage format: use directly
                    stage_idx = i - 2  # Stage 2 -> index 0, Stage 3 -> index 1
                    if stage_idx < len(ce_loc) and ce_loc[stage_idx]:
                        stage_ce_loc_i = list(ce_loc[stage_idx])
                        # Expand ce_keep_ratio if needed (can be single value or per-stage)
                        if isinstance(ce_keep_ratio[0], (list, tuple)):
                            stage_ce_keep_ratio_i = list(ce_keep_ratio[stage_idx])
                        else:
                            # Same keep_ratio for all stages/layers
                            stage_ce_keep_ratio_i = list(ce_keep_ratio)
                else:
                    # Global format: map global indices to local indices
                    # Count how many Attention layers are in stages before this one
                    prev_attn_count = 0
                    for j in range(i):
                        prev_depth = depths[j]
                        if prev_depth % 2 != 0:
                            tb = list(range(prev_depth // 2 + 1, prev_depth))
                        else:
                            tb = list(range(prev_depth // 2, prev_depth))
                        # Count attention layers (half of transformer blocks roughly)
                        prev_attn_count += len(tb)

                    # Map global ce_loc to local indices within this stage's attention layers
                    stage_local_ce_loc = []
                    stage_local_keep_ratio = []
                    for idx, (loc, kr) in enumerate(zip(ce_loc, ce_keep_ratio)):
                        # Global attention layer index relative to all stages
                        global_attn_idx = loc
                        # Check if this CE location falls within this stage
                        attn_in_stage = 0
                        for j in range(i + 1):
                            if j == 0 or j == 1:
                                continue  # Stage 0,1 are ConvBlock
                            tb_j_depth = depths[j]
                            if tb_j_depth % 2 != 0:
                                tb_j = list(range(tb_j_depth // 2 + 1, tb_j_depth))
                            else:
                                tb_j = list(range(tb_j_depth // 2, tb_j_depth))
                            attn_in_stage += len(tb_j)

                        # Calculate local index for this stage
                        local_idx = global_attn_idx - (prev_attn_count)
                        if 0 <= local_idx < len(transformer_blocks):
                            stage_local_ce_loc.append(local_idx)
                            stage_local_keep_ratio.append(kr)

                    if stage_local_ce_loc:
                        stage_ce_loc_i = stage_local_ce_loc
                        stage_ce_keep_ratio_i = stage_local_keep_ratio

            level = MambaVisionLayer(
                dim=int(embed_dim * 2**i),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                conv=conv,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                downsample=(i < 2),  # Stage 0,1 downsample; Stage 2,3 keep spatial resolution
                expand_channel=(i == 2),  # Stage 2: expand channels without spatial downsample
                layer_scale=layer_scale,
                layer_scale_conv=layer_scale_conv,
                transformer_blocks=transformer_blocks,
                use_window_partition=use_win_partition,
                stage_ce_loc=stage_ce_loc_i,
                stage_ce_keep_ratio=stage_ce_keep_ratio_i,
            )
            self.levels.append(level)

        # Projection layers for FPN fusion
        # Stage 2 output: embed_dim*4=512 channels
        self.stage2_proj = nn.Linear(int(embed_dim * 4), embed_dim)
        # Stage 3 output: embed_dim*8=1024 channels
        self.stage3_proj = nn.Linear(int(embed_dim * 8), embed_dim)

        # Channel projection for Stage 2 -> Stage 3
        # Stage 2 output: 512 channels, Stage 3 expects: embed_dim * 8 = 1024 channels
        self.stage2_to_stage3 = nn.Conv2d(int(embed_dim * 4), int(embed_dim * 8), kernel_size=1)

        # Projection for template tokens after Stage 3 (1024 -> 512 -> 128)
        self.template_proj = nn.Linear(int(embed_dim * 8), int(embed_dim * 4))

        # Sequence projection for Stage 2 -> Stage 3 (512 -> 1024)
        self.seq_proj_512_to_1024 = nn.Linear(int(embed_dim * 4), int(embed_dim * 8))

        self.apply(self._init_weights)

        # Projection to match head expectations
        # After Stage 0,1, channels are embed_dim * 4 = 512 (for base)
        # OSTrack head expects embed_dim = 128
        self.proj = nn.Linear(embed_dim * 4, embed_dim)

        # Store CE config
        self.ce_loc = ce_loc
        self.ce_keep_ratio = ce_keep_ratio

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def finetune_track(self, cfg=None, patch_start_index=1):
        """Adapt pre-trained MambaVision for tracking input sizes.

        Note: MambaVision doesn't use position embeddings like ViT,
        so this mainly handles the segment embeddings.
        """
        pass

    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None, **kwargs):
        """
        Forward pass for tracking with FPN-style feature fusion and CE support.

        Architecture (with channel expansion in Stage 2, no spatial downsample):
        - Template 128: PatchEmbed → Stage 0,1 → 8×8 → Stage 2 (channel expand) → 8×8 (64 tokens)
        - Search 256: PatchEmbed → Stage 0,1 → 16×16 → Stage 2 (channel expand) → 16×16 (256 tokens)
        - Stage 2 output: Template flattened, Search uses window_partition
        - Concatenate and process through Stage 3
        - Use Stage 3 search output for Head

        Args:
            z: template tensor [B, C, H_z, W_z] (typically 128x128)
            x: search tensor [B, C, H_x, W_x] (typically 256x256)
            ce_template_mask: template mask for CE
            ce_keep_rate: keep ratio for CE (can be a float or None)

        Returns:
            cat_feature: FPN-fused search features projected to embed_dim
            aux_dict: auxiliary dictionary
        """
        B = z.shape[0]

        # ===== Stage 0,1: Template path =====
        z = self.patch_embed(z)
        z = self.levels[0](z)
        z = self.levels[1](z)
        # z: [B, 256, 8, 8] after Stage 1

        # ===== Stage 0,1: Search path =====
        x = self.patch_embed(x)
        x = self.levels[0](x)
        x = self.levels[1](x)
        # x: [B, 256, 16, 16] after Stage 1

        # ===== Stage 2 preparation: concatenate for CE =====
        # Flatten both to sequences
        z_seq = z.flatten(2).transpose(1, 2)  # [B, 64, 256]
        x_seq = x.flatten(2).transpose(1, 2)   # [B, 256, 256]

        lens_t = z_seq.shape[1]  # 64 for template
        lens_s = x_seq.shape[1]  # 256 for search

        # Concatenate: [B, 64+256=320, 256]
        cat_seq = torch.cat([z_seq, x_seq], dim=1)  # [B, 320, 256]

        # Initialize global indices for CE
        global_index_t = torch.linspace(0, lens_t - 1, lens_t).to(cat_seq.device)
        global_index_t = global_index_t.repeat(B, 1)  # [B, 64]
        global_index_s = torch.linspace(0, lens_s - 1, lens_s).to(cat_seq.device)
        global_index_s = global_index_s.repeat(B, 1)  # [B, 256]

        # ===== Stage 2 with CE =====
        cat_seq, global_index_t, global_index_s, removed_indexes_s_stage2, last_attn_stage2 = self.levels[2](
            cat_seq, global_index_t, global_index_s, ce_template_mask, ce_keep_rate
        )

        # After CE, cat_seq contains lens_t_new + lens_s_new tokens
        # We need to pad search back to original lens_s, template back to original lens_t
        lens_t_new = global_index_t.shape[1]
        lens_s_new = global_index_s.shape[1]

        # Extract template and search from cat_seq
        template_seq = cat_seq[:, :lens_t_new, :]  # [B, lens_t_new, C]
        search_seq = cat_seq[:, lens_t_new:, :]    # [B, lens_s_new, C]

        # Create full template and search with original lengths
        # Vectorized scatter using index_put (much faster than Python loops)
        template_out = torch.zeros([B, lens_t, cat_seq.shape[2]], device=cat_seq.device)
        search_out = torch.zeros([B, lens_s, cat_seq.shape[2]], device=cat_seq.device)

        # Scatter template tokens to original positions [B, lens_t, C] <- [B, lens_t_new, C]
        # Use index_put for vectorized scatter
        idx_t = global_index_t.long().unsqueeze(-1).expand(-1, -1, template_seq.shape[-1])  # [B, lens_t_new, C]
        template_out.scatter_(dim=1, index=idx_t, src=template_seq)

        # Scatter search tokens to original positions [B, lens_s, C] <- [B, lens_s_new, C]
        idx_s = global_index_s.long().unsqueeze(-1).expand(-1, -1, search_seq.shape[-1])  # [B, lens_s_new, C]
        search_out.scatter_(dim=1, index=idx_s, src=search_seq)

        # Reshape to spatial format for Stage 3 input
        H_t, W_t = z.shape[2], z.shape[3]  # 8, 8
        H_s, W_s = x.shape[2], x.shape[3]  # 16, 16

        template_spatial = template_out.transpose(1, 2).view(B, -1, H_t, W_t)  # [B, 256, 8, 8]
        search_spatial = search_out.transpose(1, 2).view(B, -1, H_s, W_s)      # [B, 256, 16, 16]

        # Apply channel expansion (same as original downsample path)
        if self.levels[2].channel_expand is not None:
            template_spatial = self.levels[2].channel_expand(template_spatial)
            search_spatial = self.levels[2].channel_expand(search_spatial)

        # Stage 2 outputs in spatial format for Stage 3
        z = template_spatial  # [B, 512, 8, 8]
        x = search_spatial    # [B, 512, 16, 16]

        # ===== Stage 3 preparation =====
        # Flatten and window partition for Stage 3
        z_seq = z.flatten(2).transpose(1, 2)  # [B, 64, 512]
        x_seq = x.flatten(2).transpose(1, 2)  # [B, 256, 512]

        # Stage 3 only uses tokens from Stage 2 that were kept (126 tokens)
        # This ensures Stage 3 operates on Stage 2's remaining tokens, not all 256
        lens_t_s3 = z_seq.shape[1]
        lens_s_s3 = global_index_s.shape[1]  # Number of kept tokens from Stage 2 (126)

        # Select only the kept tokens from Stage 2 output using vectorized gather
        # x_seq: [B, 256, C], global_index_s: [B, 126]
        # x_seq_kept: [B, 126, C]
        idx_s = global_index_s.long().unsqueeze(-1).expand(-1, -1, x_seq.shape[-1])  # [B, 126, C]
        x_seq_kept = torch.gather(x_seq, dim=1, index=idx_s)

        # Concatenate for Stage 3: template (64) + kept search tokens from Stage 2 (126)
        cat_tokens = torch.cat([z_seq, x_seq_kept], dim=1)  # [B, 190, 512]

        # Re-initialize indices for Stage 3
        global_index_t = torch.linspace(0, lens_t_s3 - 1, lens_t_s3).to(cat_tokens.device)
        global_index_t = global_index_t.repeat(B, 1)

        # Keep Stage 2's original indices for final mapping back to 256
        global_index_s_original = global_index_s.clone()  # [B, 126], indices into original 256 positions

        # Stage 3's global_index_s will be indices into the 126 tokens (0-125)
        # Initialize fresh for Stage 3
        global_index_s = torch.linspace(0, lens_s_s3 - 1, lens_s_s3).to(cat_tokens.device)
        global_index_s = global_index_s.repeat(B, 1)  # [B, 126]

        # Stage 3 CE doesn't need prior_removed_mask since we're only processing kept tokens
        prior_removed_mask = None

        # ===== Stage 3 with CE =====
        cat_tokens, global_index_t, global_index_s, removed_indexes_s, last_attn = self.levels[3](
            cat_tokens, global_index_t, global_index_s, ce_template_mask, ce_keep_rate, prior_removed_mask
        )

        # ===== Recover original token order and pad back =====
        lens_t_new = global_index_t.shape[1] if global_index_t is not None else lens_t_s3
        lens_s_new = global_index_s.shape[1] if global_index_s is not None else lens_s_s3

        if removed_indexes_s and removed_indexes_s[0] is not None and len(removed_indexes_s) > 0:
            # Stage 3 CE was applied, use the reduced token count
            lens_t_final = lens_t_new  # 64
            lens_s_final = lens_s_new  # 89 after Stage 3 CE
        else:
            # No CE applied, use original token count
            lens_t_final = lens_t_s3
            lens_s_final = lens_s_s3

        # ===== Split and project =====
        # Template: first lens_t_final tokens
        # Search: last lens_s_final tokens
        template_out = cat_tokens[:, :lens_t_final, :]  # [B, lens_t_final, 512]
        search_out = cat_tokens[:, lens_t_final:, :]   # [B, lens_s_final, 512]

        # Project to embed_dim
        template_out = self.template_proj(template_out)   # [B, lens_t_final, 512] -> [B, lens_t_final, 512]
        template_out = self.proj(template_out)   # [B, lens_t_final, 128]
        search_out = self.stage3_proj(search_out)  # [B, lens_s_final, 128]

        # Map search tokens back to original 256 positions using global_index_s
        if removed_indexes_s and removed_indexes_s[0] is not None and len(removed_indexes_s) > 0:
            # global_index_s_original: original indices into 256 positions (from Stage 2)
            # global_index_s: indices into 0-125 range (which of the 126 tokens were kept by Stage 3)
            # Final kept indices = global_index_s_original[:, global_index_s[b, i]] for kept i

            lens_s_original = x_seq.shape[1]  # 256

            # Vectorized: Map indices to original positions
            # original_positions[b, i] = global_index_s_original[b, global_index_s[b, i]]
            original_positions = torch.gather(
                global_index_s_original.long().unsqueeze(-1).expand(-1, -1, search_out.shape[-1]),
                dim=1,
                index=global_index_s.long().unsqueeze(-1).expand(-1, -1, search_out.shape[-1])
            )  # [B, lens_s_new, C]

            # Scatter search_out to original positions
            search_full = torch.zeros([B, lens_s_original, search_out.shape[2]], device=search_out.device, dtype=search_out.dtype)
            search_full.scatter_(dim=1, index=original_positions, src=search_out)

            search_out = search_full  # [B, 256, 128]

        # Final output: template + search
        cat_out = torch.cat([template_out, search_out], dim=1)  # [B, lens_t_final+256, 128]

        aux_dict = {"attn": last_attn, "removed_indexes_s": removed_indexes_s, "removed_indexes_s_stage2": removed_indexes_s_stage2}
        return cat_out, aux_dict

    def forward_features(self, z, x):
        """Alias for forward for compatibility with OSTrack."""
        return self.forward(z, x)