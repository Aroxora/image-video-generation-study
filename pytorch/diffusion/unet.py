"""The classic diffusion U-Net denoiser (the Stable Diffusion-family backbone).

website: /denoiser  (U-Net -> DiT evolution)

WHAT this network does: given a NOISY latent x_t [B, Cin, H, W], the timestep t, and an
optional text `context`, predict the noise epsilon that was added (same shape as x). The
training loop (ddpm.py) regresses this prediction against the true noise.

WHY a U-Net (and how it differs from the DiT in dit.py):
  * A U-Net is a CONVOLUTIONAL encoder/decoder. It downsamples the feature map a few
    times (capturing coarse, global structure), then upsamples back, with SKIP
    CONNECTIONS that re-inject the high-resolution detail lost on the way down. This
    "see globally, reconstruct locally" shape is a great inductive bias for images and
    is why SD 1.x/2.x/SDXL all use U-Nets.
  * Conditioning enters two ways:
        - TIME: a TimestepEmbedding vector is projected and ADDED inside every ResBlock
          (FiLM-style scale on the features), so every layer "knows" how noisy x_t is.
        - TEXT: at the lower resolutions we insert a SpatialTransformer (self-attn +
          cross-attn to the prompt). Cross-attn is literally how the words steer pixels.
  * Compare dit.py: the DiT throws away the conv/U-shape entirely and treats the latent
    as a sequence of patches fed to a plain Transformer, conditioning via adaLN-Zero.
    DiTs scale better with compute and underlie Stable Diffusion 3 / Sora-style models;
    the U-Net is the older, more sample-efficient-at-small-scale design. Both predict eps
    and obey the SAME forward(x, t, context, mask) contract so they are interchangeable.

Tensor conventions: images/latents [B, C, H, W]; t LongTensor [B]; context [B, L, dim];
mask bool [B, L] (True == real token). Always returns [B, Cout, H, W].
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pytorch.diffusion.cross_attention import (
    CrossAttention,
    SpatialTransformer,
    TimestepEmbedding,
    _num_groups,
)


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    """Residual conv block that injects the timestep embedding.

    Structure (GroupNorm -> SiLU -> Conv) x2 with the time embedding added between the
    two convs, plus a skip (1x1 conv if the channel count changes). Adding the projected
    time vector to the feature map is FiLM-lite: it shifts the activations based on how
    noisy the input is, which is essential -- the network must denoise very differently
    at t=10 vs t=900.
    """

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        self.in_layers = nn.Sequential(
            nn.GroupNorm(_num_groups(in_ch), in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        )
        # Project the time embedding to out_ch and broadcast over H, W.
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.out_layers = nn.Sequential(
            nn.GroupNorm(_num_groups(out_ch), out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        )
        # Match channels on the skip path when in != out.
        self.skip = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        h = self.in_layers(x)
        # add time embedding: [B, out_ch] -> [B, out_ch, 1, 1]
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.out_layers(h)
        return h + self.skip(x)


class SelfAttentionBlock(nn.Module):
    """A pure spatial self-attention block (no text) used at the bottleneck.

    At the lowest resolution the feature map is small (cheap to attend over) but each
    position summarizes a large receptive field, so global self-attention here lets
    distant regions of the image become consistent (e.g. symmetry, global lighting).
    """

    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(_num_groups(channels), channels)
        self.attn = CrossAttention(
            channels, None, heads=heads, dim_head=max(channels // heads, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        h_ = self.norm(x).reshape(b, c, h * w).transpose(1, 2)  # [B, HW, C]
        h_ = self.attn(h_)
        h_ = h_.transpose(1, 2).reshape(b, c, h, w)
        return x + h_  # residual


class Downsample(nn.Module):
    """Halve H, W with a strided 3x3 conv (learnable, better than plain pooling)."""

    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.op(x)


class Upsample(nn.Module):
    """Double H, W with nearest-neighbour upsample + a smoothing 3x3 conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


# ---------------------------------------------------------------------------
# the U-Net
# ---------------------------------------------------------------------------
class UNet(nn.Module):
    """Encoder/decoder denoiser with time + (optional) text cross-attention.

    Args:
        in_channels / out_channels: latent channels in/out (4 for an SD-style VAE latent).
        model_channels: width at the top resolution; deeper levels multiply by channel_mult.
        channel_mult: per-resolution width multipliers; len-1 == number of downsamples.
        num_res_blocks: ResBlocks per resolution level.
        context_dim: if set, SpatialTransformers (cross-attn to text) are inserted; if
                     None the net is fully unconditional and forward ignores `context`.
        num_heads: attention heads for the transformer/self-attn blocks.
        dropout: dropout inside ResBlocks.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        model_channels: int = 64,
        channel_mult: Sequence[int] = (1, 2, 4),
        num_res_blocks: int = 2,
        context_dim: Optional[int] = None,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.context_dim = context_dim
        self.num_heads = num_heads

        # --- time embedding (shared by all ResBlocks) ---
        time_dim = model_channels * 4
        self.time_embed = TimestepEmbedding(time_dim)

        # Apply cross-attention only at the deeper (downsampled) levels, like SD: the top
        # level is high-res & expensive, and coarse semantics live deeper. Here we attend
        # at every level *except* the first when context is used.
        def use_attn(level: int) -> bool:
            return context_dim is not None and level > 0

        # ================= ENCODER (downsampling path) =================
        self.input_conv = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        # Track the channel count of every tensor we will later concat via skip connections.
        skip_channels = [model_channels]
        ch = model_channels
        num_levels = len(channel_mult)

        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                block = nn.ModuleList([ResBlock(ch, out_ch, time_dim, dropout)])
                ch = out_ch
                if use_attn(level):
                    block.append(SpatialTransformer(ch, context_dim, heads=num_heads))
                self.down_blocks.append(block)
                skip_channels.append(ch)
            # Downsample between levels (not after the last level).
            if level != num_levels - 1:
                self.down_blocks.append(nn.ModuleList([Downsample(ch)]))
                skip_channels.append(ch)

        # ================= BOTTLENECK (lowest resolution) =================
        # ResBlock -> self-attn (always) -> optional cross-attn -> ResBlock.
        self.mid_block1 = ResBlock(ch, ch, time_dim, dropout)
        self.mid_attn = SelfAttentionBlock(ch, heads=num_heads)
        self.mid_cross = (
            SpatialTransformer(ch, context_dim, heads=num_heads)
            if context_dim is not None
            else None
        )
        self.mid_block2 = ResBlock(ch, ch, time_dim, dropout)

        # ================= DECODER (upsampling path) =================
        self.up_blocks = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult
            # One extra ResBlock per level than the encoder: it consumes the skip taken
            # right before the corresponding downsample (standard U-Net bookkeeping).
            for _ in range(num_res_blocks + 1):
                skip = skip_channels.pop()
                block = nn.ModuleList([ResBlock(ch + skip, out_ch, time_dim, dropout)])
                ch = out_ch
                if use_attn(level):
                    block.append(SpatialTransformer(ch, context_dim, heads=num_heads))
                self.up_blocks.append(block)
            # Upsample between levels (not after the top level).
            if level != 0:
                self.up_blocks.append(nn.ModuleList([Upsample(ch)]))

        # ================= OUTPUT head =================
        self.out_norm = nn.GroupNorm(_num_groups(ch), ch)
        self.out_conv = nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)
        # Zero-init the final conv: the model starts by predicting eps == 0, which is a
        # calm, stable starting point for diffusion training.
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    # -- helpers to run a heterogeneous (ResBlock / Transformer / sample) block --
    @staticmethod
    def _run_block(
        block: nn.ModuleList,
        x: Tensor,
        t_emb: Tensor,
        context: Optional[Tensor],
        mask: Optional[Tensor],
    ) -> Tensor:
        for layer in block:
            if isinstance(layer, ResBlock):
                x = layer(x, t_emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context=context, mask=mask)
            else:  # Downsample / Upsample
                x = layer(x)
        return x

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        context: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        # If the net is unconditional, ignore any context that gets passed in.
        if self.context_dim is None:
            context = None

        t_emb = self.time_embed(t)  # [B, time_dim]

        # --- encoder ---
        h = self.input_conv(x)
        skips = [h]
        for block in self.down_blocks:
            h = self._run_block(block, h, t_emb, context, mask)
            skips.append(h)

        # --- bottleneck ---
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        if self.mid_cross is not None:
            h = self.mid_cross(h, context=context, mask=mask)
        h = self.mid_block2(h, t_emb)

        # --- decoder (concatenate the matching skip before each ResBlock) ---
        for block in self.up_blocks:
            first = block[0]
            if isinstance(first, ResBlock):
                h = torch.cat([h, skips.pop()], dim=1)  # skip connection
            h = self._run_block(block, h, t_emb, context, mask)

        # --- output head ---
        h = self.out_norm(h)
        h = F.silu(h)
        return self.out_conv(h)
