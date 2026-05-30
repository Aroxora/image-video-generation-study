"""Time embeddings + attention primitives shared by every denoiser backbone.

website: /denoiser  (how a network turns noisy x_t + t + text into an eps prediction)

This file is the "vocabulary" the U-Net and DiT are built from:

  * SinusoidalPosEmb / TimestepEmbedding  -- turn the integer timestep t into a
    continuous vector the network can condition on. The SAME idea (sin/cos of many
    frequencies) that positional encodings use, but here the "position" is the
    diffusion step. Other modules import these two from here (toy/, dit.py, video/).
  * CrossAttention   -- the workhorse: queries come from the image tokens, keys/values
    come EITHER from the image itself (self-attention) OR from the text context
    (cross-attention). Text conditioning in Stable Diffusion / DiT is literally this.
  * SpatialTransformer -- the block the U-Net drops into a conv feature map: it flattens
    [B,C,H,W] -> tokens, does (self-attn -> cross-attn -> feed-forward) with residuals,
    then reshapes back. This is exactly the "transformer block at each resolution" that
    SD's U-Net uses to inject the prompt.

Conventions (repo-wide):
  images/latents [B, C, H, W]; tokens [B, N, D]; timesteps t LongTensor [B];
  context [B, L, context_dim]; mask bool [B, L] where True == REAL token (False == pad).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Timestep conditioning
# ---------------------------------------------------------------------------
class SinusoidalPosEmb(nn.Module):
    """Map an integer timestep t -> a [B, dim] sinusoidal feature vector.

    WHY sinusoids: the network needs a *smooth* representation of t so that nearby
    steps (e.g. t=500 and t=501) get nearby embeddings, while far-apart steps differ.
    Using a geometric range of frequencies (the classic Transformer trick) lets a
    single vector encode the step at many scales. We do NOT learn anything here --
    it is a fixed featurization; the learned part lives in TimestepEmbedding's MLP.
    """

    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            # We split the dim in half for sin and half for cos; require even.
            raise ValueError(f"SinusoidalPosEmb dim must be even, got {dim}")
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:  # t: [B] (long or float) -> [B, dim]
        device = t.device
        half = self.dim // 2
        # frequencies geometrically spaced between 1 and 1/10000 (DDPM/Transformer std).
        # emb_freqs[k] = exp(-k * log(10000) / (half-1))
        emb = math.log(10000.0) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * -emb)
        # outer product: [B,1] * [1,half] -> [B, half]
        emb = t.float()[:, None] * emb[None, :]
        # interleave sin/cos -> [B, dim]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class TimestepEmbedding(nn.Module):
    """SinusoidalPosEmb -> small MLP, producing the conditioning vector used everywhere.

    The raw sinusoid is fixed; the MLP gives the network a *learnable* projection of t
    into whatever space the ResBlocks / adaLN modulators expect. Output dim == `dim`.
    """

    def __init__(self, dim: int, hidden: Optional[int] = None):
        super().__init__()
        hidden = hidden or dim * 4
        self.dim = dim
        self.sinusoidal = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),  # SiLU/Swish is the standard activation in diffusion nets
            nn.Linear(hidden, dim),
        )

    def forward(self, t: Tensor) -> Tensor:  # [B] -> [B, dim]
        return self.mlp(self.sinusoidal(t))


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------
class CrossAttention(nn.Module):
    """Multi-head (cross-)attention used for both self- and text-conditioning.

    forward(x, context=None, mask=None):
        x       : [B, N, query_dim]   -- the image/latent tokens (queries)
        context : [B, L, context_dim] -- text tokens (keys/values). If None -> SELF-attn
                                          (context := x), i.e. tokens attend to each other.
        mask    : bool [B, L]         -- True == real context token. Pad positions get
                                          -inf logits so they contribute nothing.

    This single module is *the* mechanism by which a prompt influences the image:
    each image token asks "which words are relevant to me?" and pulls their values in.
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
    ):
        super().__init__()
        inner_dim = heads * dim_head
        context_dim = context_dim if context_dim is not None else query_dim
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5  # 1/sqrt(d) scaling so logits don't explode

        # No bias on QKV is the common choice in SD/DiT attention.
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def _split_heads(self, x: Tensor) -> Tensor:
        # [B, N, heads*dim_head] -> [B, heads, N, dim_head]
        b, n, _ = x.shape
        return x.view(b, n, self.heads, self.dim_head).transpose(1, 2)

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        # If no context, this is self-attention: tokens attend among themselves.
        ctx = context if context is not None else x

        q = self._split_heads(self.to_q(x))      # [B, H, N, d]
        k = self._split_heads(self.to_k(ctx))    # [B, H, L, d]
        v = self._split_heads(self.to_v(ctx))    # [B, H, L, d]

        # Build an additive attention-bias from the key-padding mask, if given.
        # mask is True for REAL tokens; we must MASK OUT the False (pad) positions.
        attn_mask = None
        if mask is not None and context is not None:
            # mask: [B, L] -> [B, 1, 1, L] broadcast over heads and query positions.
            # Use a large negative bias (not literal -inf to keep softmax numerically safe).
            keep = mask[:, None, None, :].to(q.dtype)          # 1.0 real, 0.0 pad
            attn_mask = (1.0 - keep) * torch.finfo(q.dtype).min

        # Prefer PyTorch's fused scaled-dot-product attention (flash/mem-efficient on
        # GPU, correct on CPU). It applies softmax(QK^T/sqrt(d) + attn_mask) @ V.
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)  # [B,H,N,d]

        # merge heads -> [B, N, heads*dim_head] -> project back to query_dim
        b, h, n, d = out.shape
        out = out.transpose(1, 2).reshape(b, n, h * d)
        return self.to_out(out)


class FeedForward(nn.Module):
    """Transformer MLP with a GEGLU gate (SD uses GEGLU in its transformer blocks).

    GEGLU = (Linear_a(x) * gelu(Linear_b(x))) -- the gate lets the block learn to
    suppress channels, which empirically helps over a plain GELU MLP.
    """

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner = dim * mult
        self.proj_in = nn.Linear(dim, inner * 2)  # produces [val | gate]
        self.dropout = nn.Dropout(dropout)
        self.proj_out = nn.Linear(inner, dim)

    def forward(self, x: Tensor) -> Tensor:
        val, gate = self.proj_in(x).chunk(2, dim=-1)
        x = val * F.gelu(gate)
        return self.proj_out(self.dropout(x))


class SpatialTransformer(nn.Module):
    """Transformer block operating on a conv feature map [B, C, H, W].

    Pipeline (this is the SD U-Net's "Transformer2D" block):
        x -> GroupNorm -> 1x1 conv (proj_in) -> flatten to tokens [B, H*W, C]
          -> [ self-attn (residual) -> cross-attn to context (residual) -> FF (residual) ]
          -> reshape back -> 1x1 conv (proj_out) -> + input (outer residual)

    Self-attention mixes spatial information (long-range, unlike convs); cross-attention
    is where the PROMPT enters. With context=None it degrades gracefully to self-attn
    only, so the same U-Net works unconditionally.
    """

    def __init__(self, channels: int, context_dim: int, heads: int = 8):
        super().__init__()
        self.channels = channels
        # dim_head chosen so heads*dim_head == channels (keeps token width == C).
        dim_head = max(channels // heads, 1)

        self.norm = nn.GroupNorm(num_groups=_num_groups(channels), num_channels=channels)
        self.proj_in = nn.Conv2d(channels, channels, kernel_size=1)

        # Pre-norm sub-blocks operating on tokens of width `channels`.
        self.norm1 = nn.LayerNorm(channels)
        self.attn1 = CrossAttention(channels, None, heads=heads, dim_head=dim_head)  # self
        self.norm2 = nn.LayerNorm(channels)
        self.attn2 = CrossAttention(channels, context_dim, heads=heads, dim_head=dim_head)  # cross
        self.norm3 = nn.LayerNorm(channels)
        self.ff = FeedForward(channels)

        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)
        # Zero-init the output projection so the block starts as identity (stable training).
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        b, c, h, w = x.shape
        residual = x

        x = self.norm(x)
        x = self.proj_in(x)
        # [B, C, H, W] -> [B, H*W, C] tokens
        x = x.reshape(b, c, h * w).transpose(1, 2)

        # Pre-norm residual transformer block.
        x = x + self.attn1(self.norm1(x))                          # self-attention
        x = x + self.attn2(self.norm2(x), context=context, mask=mask)  # cross-attention
        x = x + self.ff(self.norm3(x))                             # feed-forward

        # tokens -> [B, C, H, W]
        x = x.transpose(1, 2).reshape(b, c, h, w)
        x = self.proj_out(x)
        return x + residual


# ---------------------------------------------------------------------------
# small helper
# ---------------------------------------------------------------------------
def _num_groups(channels: int, target: int = 32) -> int:
    """Pick a GroupNorm group count that divides `channels` (<= target).

    GroupNorm needs num_channels % num_groups == 0. With tiny laptop models the
    channel count can be small, so we fall back to the largest divisor <= target.
    """
    g = min(target, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return g
