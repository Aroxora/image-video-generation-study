"""Factorized spatial + temporal attention -- the cheap way to get spacetime mixing.

website: /video  (factorized attention: look within a frame, then across frames)

Full 3D attention over every space-time token is O(N^2) with N = T*H*W tokens, which
blows up fast. The standard trick (ViViT, video DiT, AnimateDiff-style temporal layers)
is to FACTORIZE attention into two cheaper passes per block:

    1. SPATIAL  attention: within each frame, tokens attend to other tokens of the
       same frame (mix appearance / layout). Time index is held fixed.
    2. TEMPORAL attention: for each spatial location, tokens attend across frames
       (mix motion / dynamics). Space index is held fixed.

Stacking these two gives every token an indirect path to every other token, at a tiny
fraction of the cost of dense 3D attention. The TEMPORAL pass is what stitches frames
into a coherent clip -- "whole clip at once -> temporal coherence" (no per-frame flicker).

Token layout used here: x [B, T, S, D]
  T = number of (temporal) frames/patches along time,
  S = number of spatial tokens per frame (gh*gw),
  D = model width. We reshape to [B*T, S, D] for spatial and [B*S, T, D] for temporal.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Plain multi-head self-attention over a token sequence [B, N, D]
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    """Standard multi-head self-attention over [B, N, D] -> [B, N, D].

    Used as the inner kernel of both the spatial and temporal passes: we just feed it
    the appropriate (frames-as-batch or pixels-as-batch) reshaping of the video tokens.
    """

    def __init__(self, dim: int, heads: int = 6):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim {dim} must be divisible by heads {heads}")
        self.dim = dim
        self.heads = heads
        self.dim_head = dim // heads
        self.scale = self.dim_head ** -0.5
        # fused qkv projection, then output projection (GPT/ViT style).
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x)  # [B, N, 3D]
        q, k, v = qkv.chunk(3, dim=-1)
        # [B, N, D] -> [B, heads, N, dim_head]
        q = q.view(b, n, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(b, n, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(b, n, self.heads, self.dim_head).transpose(1, 2)

        # Optional key-padding mask: bool [B, N], True == real token.
        attn_mask = None
        if mask is not None:
            keep = mask[:, None, None, :].to(q.dtype)  # [B,1,1,N]
            attn_mask = (1.0 - keep) * torch.finfo(q.dtype).min

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)  # [B,H,N,d]
        out = out.transpose(1, 2).reshape(b, n, d)
        return self.proj(out)


# ---------------------------------------------------------------------------
# Temporal-only attention (attend across the T axis)
# ---------------------------------------------------------------------------
class TemporalAttention(nn.Module):
    """Self-attention across frames, applied independently at each spatial location.

    forward(x: [B, T, S, D]) -> [B, T, S, D]

    We treat each of the S spatial positions as its own little sequence of length T and
    attend along time. Implementation-wise: move S into the batch dimension so the inner
    Attention sees [B*S, T, D]. This is the pass that enforces motion consistency.
    """

    def __init__(self, dim: int, heads: int = 6):
        super().__init__()
        self.attn = Attention(dim, heads=heads)

    def forward(self, x: Tensor) -> Tensor:
        b, t, s, d = x.shape
        # [B, T, S, D] -> [B, S, T, D] -> [B*S, T, D]  (each spatial loc = one T-sequence)
        x = x.permute(0, 2, 1, 3).reshape(b * s, t, d)
        x = self.attn(x)
        # back to [B, T, S, D]
        x = x.reshape(b, s, t, d).permute(0, 2, 1, 3).contiguous()
        return x


# ---------------------------------------------------------------------------
# adaLN-conditioned factorized space-time block
# ---------------------------------------------------------------------------
def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """adaLN modulation: x * (1 + scale) + shift.

    `x` is [..., D] post-LayerNorm; shift/scale are [B, D] broadcast over tokens. The
    (1 + scale) form means "no modulation" is the zero vector, which pairs with the
    zero-initialized gate so each block starts as the identity (adaLN-Zero, from DiT).
    """
    # x: [B, T, S, D]; shift/scale: [B, D] -> [B, 1, 1, D]
    while shift.dim() < x.dim():
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
    return x * (1 + scale) + shift


class FactorizedSpacetimeBlock(nn.Module):
    """One transformer block: spatial self-attn -> temporal self-attn -> MLP.

    Each sub-layer is wrapped in adaLN-Zero conditioning driven by a per-sample vector
    `cond` [B, D] (typically time embedding [+ pooled text]). For every sub-layer we
    regress (shift, scale, gate) from `cond`; the gate is zero-initialized so the block
    is identity at init -> stable training (this is the DiT recipe, extended to video).

    forward(x: [B, T, S, D], cond: [B, D]) -> [B, T, S, D]
      * spatial pass  : attend within each frame      (reshape to [B*T, S, D])
      * temporal pass : attend across frames per pixel (TemporalAttention)
      * MLP           : per-token feed-forward
    """

    def __init__(self, dim: int, heads: int = 6, mlp_mult: int = 4):
        super().__init__()
        self.dim = dim

        # Pre-norms (no affine -- the affine part comes from adaLN modulation).
        self.norm_spatial = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm_temporal = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm_mlp = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        self.spatial_attn = Attention(dim, heads=heads)
        self.temporal_attn = TemporalAttention(dim, heads=heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_mult),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * mlp_mult, dim),
        )

        # adaLN modulation MLP: cond [B,D] -> 9 vectors of size D, i.e.
        # (shift, scale, gate) for each of the 3 sub-layers (spatial/temporal/mlp).
        self.ada = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 9 * dim),
        )
        # Zero-init the adaLN projection so gates start at 0 -> block == identity at init.
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        b, t, s, d = x.shape
        # Produce all nine modulation vectors at once, then split.
        params = self.ada(cond)  # [B, 9D]
        (
            shift_sa, scale_sa, gate_sa,
            shift_ta, scale_ta, gate_ta,
            shift_mlp, scale_mlp, gate_mlp,
        ) = params.chunk(9, dim=-1)

        # ---- spatial self-attention (within each frame) ----
        h = _modulate(self.norm_spatial(x), shift_sa, scale_sa)  # [B,T,S,D]
        h = h.reshape(b * t, s, d)            # frames into batch
        h = self.spatial_attn(h)
        h = h.reshape(b, t, s, d)
        x = x + gate_sa.unsqueeze(1).unsqueeze(1) * h

        # ---- temporal self-attention (across frames) ----
        h = _modulate(self.norm_temporal(x), shift_ta, scale_ta)
        h = self.temporal_attn(h)             # [B,T,S,D]
        x = x + gate_ta.unsqueeze(1).unsqueeze(1) * h

        # ---- per-token MLP ----
        h = _modulate(self.norm_mlp(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        x = x + gate_mlp.unsqueeze(1).unsqueeze(1) * h
        return x
