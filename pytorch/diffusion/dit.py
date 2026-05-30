"""The Diffusion Transformer (DiT) denoiser -- the U-Net's successor.

website: /denoiser  (U-Net -> DiT evolution)

WHAT and WHY (read this next to unet.py):
  The U-Net is convolutional with an encoder/decoder shape. A DiT throws all of that
  away. It treats the latent image as a SEQUENCE OF PATCHES (like ViT for classification)
  and runs a plain stack of Transformer blocks over them. No downsampling, no skips, no
  convs in the trunk -- just patch-embed -> N x TransformerBlock -> unpatchify. This is the
  architecture behind Stable Diffusion 3 (MM-DiT), PixArt, and Sora-style video models,
  because Transformers scale predictably with compute/data.

How conditioning works here (the key DiT idea: **adaLN-Zero**):
  Instead of adding a time vector inside conv blocks, the timestep (plus a pooled text
  vector if provided) is turned, per-block, into six modulation signals:
      shift_1, scale_1, gate_1   for the attention sub-layer
      shift_2, scale_2, gate_2   for the MLP sub-layer
  Each sub-layer does:  x = x + gate * sublayer( (1+scale) * LayerNorm(x) + shift ).
  The "-Zero" part: the linear that produces these signals is ZERO-INITIALIZED, so every
  block starts as the identity (gate=0) and the final projection starts at 0 -- the whole
  DiT initially predicts eps==0, giving very stable early training. This replaces FiLM-in-
  ResBlocks (U-Net) and the cross-attn-only conditioning of older nets with one clean
  mechanism.

Text conditioning two ways (both optional, controlled by context_dim):
  * POOLED context is mean-pooled over the L tokens and added to the time vector, so it
    feeds adaLN (a global "what is this image about" signal).
  * If context_dim is set we ALSO add a cross-attention sub-layer per block so image
    tokens can attend to the *individual* word tokens (fine-grained grounding), exactly
    like the U-Net's SpatialTransformer.

Tensor conventions: latents [B, C, H, W]; t LongTensor [B]; context [B, L, dim];
mask bool [B, L] (True == real token). Returns eps [B, C, H, W] (same shape as x).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pytorch.diffusion.cross_attention import (
    CrossAttention,
    SinusoidalPosEmb,  # re-exported convenience (toy/ and others may import from here)
    TimestepEmbedding,
)

__all__ = ["patchify", "unpatchify", "DiTBlock", "DiT", "SinusoidalPosEmb"]


# ---------------------------------------------------------------------------
# patch <-> token conversion
# ---------------------------------------------------------------------------
def patchify(x: Tensor, p: int) -> Tuple[Tensor, Tuple[int, int]]:
    """[B, C, H, W] -> (tokens [B, N, C*p*p], grid (gh, gw)) with N = gh*gw.

    Cut the image into a (gh x gw) grid of p x p patches and flatten each patch (channels
    fastest, then patch rows/cols) into a single vector. This linearizes a 2D image into
    a 1D token sequence -- the form a Transformer consumes.
    """
    b, c, h, w = x.shape
    if h % p != 0 or w % p != 0:
        raise ValueError(f"image size ({h},{w}) not divisible by patch_size {p}")
    gh, gw = h // p, w // p
    # [B,C,gh,p,gw,p] -> [B,gh,gw,C,p,p] -> [B, gh*gw, C*p*p]
    x = x.reshape(b, c, gh, p, gw, p)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    tokens = x.reshape(b, gh * gw, c * p * p)
    return tokens, (gh, gw)


def unpatchify(tokens: Tensor, p: int, grid: Tuple[int, int], C: int) -> Tensor:
    """Inverse of patchify: (tokens [B, N, C*p*p], grid) -> [B, C, gh*p, gw*p]."""
    gh, gw = grid
    b = tokens.shape[0]
    x = tokens.reshape(b, gh, gw, C, p, p)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()  # [B,C,gh,p,gw,p]
    return x.reshape(b, C, gh * p, gw * p)


# ---------------------------------------------------------------------------
# DiT block (adaLN-Zero)
# ---------------------------------------------------------------------------
def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    # FiLM on the token dim: per-block, per-sample (broadcast over the N tokens).
    # shift/scale: [B, D] -> [B, 1, D].
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """One DiT transformer block: adaLN-Zero self-attn (+ optional cross-attn) + MLP.

    cond [B, D] (time (+ pooled text)) -> 6 (or 9 with cross-attn) modulation vectors.
    """

    def __init__(
        self,
        hidden: int,
        heads: int,
        context_dim: Optional[int] = None,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.has_cross = context_dim is not None
        dim_head = max(hidden // heads, 1)

        # LayerNorms have NO learnable affine -- adaLN provides scale/shift instead.
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.attn = CrossAttention(hidden, None, heads=heads, dim_head=dim_head)  # self

        if self.has_cross:
            self.norm_cross = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
            self.cross_attn = CrossAttention(
                hidden, context_dim, heads=heads, dim_head=dim_head
            )

        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        inner = int(hidden * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, inner),
            nn.GELU(approximate="tanh"),
            nn.Linear(inner, hidden),
        )

        # adaLN modulator: cond -> [shift1,scale1,gate1, shift2,scale2,gate2 (+cross gate)].
        # We always emit the 6 self-attn/MLP signals; if cross-attn is present we also emit
        # a single gate for it (its norm uses no scale/shift to keep it simple).
        n_signals = 6 + (1 if self.has_cross else 0)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden, n_signals * hidden))
        # ZERO-INIT (the "-Zero"): blocks start as identity, gates start at 0.
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(
        self,
        x: Tensor,
        cond: Tensor,
        context: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        signals = self.ada(cond)
        if self.has_cross:
            shift1, scale1, gate1, shift2, scale2, gate2, gate_c = signals.chunk(7, dim=-1)
        else:
            shift1, scale1, gate1, shift2, scale2, gate2 = signals.chunk(6, dim=-1)

        # --- self-attention sub-layer (gated residual, adaLN-modulated input) ---
        x = x + gate1.unsqueeze(1) * self.attn(_modulate(self.norm1(x), shift1, scale1))

        # --- optional cross-attention to text tokens ---
        if self.has_cross and context is not None:
            x = x + gate_c.unsqueeze(1) * self.cross_attn(
                self.norm_cross(x), context=context, mask=mask
            )

        # --- MLP sub-layer ---
        x = x + gate2.unsqueeze(1) * self.mlp(_modulate(self.norm2(x), shift2, scale2))
        return x


# ---------------------------------------------------------------------------
# the DiT
# ---------------------------------------------------------------------------
class DiT(nn.Module):
    """Diffusion Transformer denoiser. Predicts eps with the same forward contract as UNet.

    Args:
        in_channels: latent channels (4 for an SD VAE latent).
        input_size: spatial side length H==W of the latent (e.g. 32).
        patch_size: side of each square patch; N = (input_size/patch_size)**2 tokens.
        hidden: transformer width.
        depth: number of DiTBlocks.
        heads: attention heads.
        context_dim: if set, enable pooled-text adaLN + per-block cross-attention.
    """

    def __init__(
        self,
        in_channels: int = 4,
        input_size: int = 32,
        patch_size: int = 2,
        hidden: int = 384,
        depth: int = 12,
        heads: int = 6,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.input_size = input_size
        self.patch_size = patch_size
        self.hidden = hidden
        self.context_dim = context_dim

        patch_dim = in_channels * patch_size * patch_size
        # default grid (used to size positional embeddings); forward re-derives the real
        # grid from x so non-default H/W still work as long as positions fit.
        gh = gw = input_size // patch_size
        self.num_patches = gh * gw

        # patch embedding: flat patch vector -> hidden.
        self.patch_embed = nn.Linear(patch_dim, hidden)
        # Learned positional embedding over the patch grid (ViT/DiT standard).
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden))
        nn.init.normal_(self.pos_embed, std=0.02)

        # timestep -> hidden conditioning vector (feeds adaLN of every block).
        self.t_embed = TimestepEmbedding(hidden)
        # pooled-text projection (only if conditioning); added to the time vector.
        self.context_pool = (
            nn.Linear(context_dim, hidden) if context_dim is not None else None
        )

        self.blocks = nn.ModuleList(
            [DiTBlock(hidden, heads, context_dim) for _ in range(depth)]
        )

        # Final adaLN + linear back to patch_dim. Zero-init (adaLN-Zero) so the model
        # starts predicting eps == 0.
        self.norm_out = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))
        self.proj_out = nn.Linear(hidden, patch_dim)
        nn.init.zeros_(self.ada_out[-1].weight)
        nn.init.zeros_(self.ada_out[-1].bias)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        context: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        if self.context_dim is None:
            context = None  # unconditional: ignore any passed context

        b, c, h, w = x.shape
        p = self.patch_size

        # 1) image -> patch tokens -> hidden, add positional embedding.
        tokens, grid = patchify(x, p)            # [B, N, C*p*p]
        tokens = self.patch_embed(tokens)        # [B, N, hidden]
        n = tokens.shape[1]
        # Slice/broadcast positional embeddings to the actual number of tokens.
        tokens = tokens + self.pos_embed[:, :n, :]

        # 2) conditioning vector: time (+ pooled text) -> [B, hidden].
        cond = self.t_embed(t)
        if self.context_pool is not None and context is not None:
            if mask is not None:
                # masked mean over real tokens only (avoid averaging in padding).
                m = mask.to(context.dtype).unsqueeze(-1)          # [B, L, 1]
                pooled = (context * m).sum(1) / m.sum(1).clamp(min=1e-6)
            else:
                pooled = context.mean(dim=1)
            cond = cond + self.context_pool(pooled)

        # 3) transformer trunk.
        for block in self.blocks:
            tokens = block(tokens, cond, context=context, mask=mask)

        # 4) final adaLN-Zero head -> patch vectors -> image.
        shift, scale = self.ada_out(cond).chunk(2, dim=-1)
        tokens = self.norm_out(tokens) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        tokens = self.proj_out(tokens)           # [B, N, C*p*p]
        return unpatchify(tokens, p, grid, c)    # [B, C, H, W]
