"""VideoDiT: a DiT-style diffusion backbone that denoises a WHOLE video clip at once.

website: /video  (a transformer over space-time patches predicts the noise for the clip)

This is the video analogue of pytorch/diffusion/dit.py. The recipe:

    latent video [B,C,T,H,W]
        --SpacetimePatchEmbed (strided Conv3d)-->  tokens [B, N, hidden], grid (gt,gh,gw)
        + learned position embedding (per spatial-time slot)
        --[ FactorizedSpacetimeBlock x depth ]-->  conditioned on time (+ pooled text)
        --adaLN-Zero final layer-->                per-token eps
        --unembed + fold back-->                   eps [B,C,T,H,W]

WHY process the whole clip together: because the FactorizedSpacetimeBlock's TEMPORAL
attention lets a patch in frame 0 talk to a patch in frame 7, the model denoises all
frames jointly. Predicting every frame in one shot, with cross-frame attention, is what
gives temporal coherence -- "whole clip at once -> temporal coherence" (no flicker, motion
stays consistent) rather than denoising each frame independently and hoping they match.

Conditioning:
  * time t  [B]          -> TimestepEmbedding -> cond vector (always present).
  * context [B,L,Cdim]   -> (a) mean-pooled and added into the adaLN cond vector, and
                            (b) optionally cross-attended token-by-token if context_dim set.
  * cond_frames [B,C,Tc,H,W] (image-to-video): a few CLEAN, known frames. We encode them
    with the same patch embed and ADD a small "this token is known/clean" embedding, then
    splice their tokens into the sequence at the matching time slots, plus a binary mask
    channel telling the network which time-steps are observed. This is the simplest robust
    way to do I2V: the model sees the conditioning frames as already-denoised context and
    must produce frames consistent with them. (Documented choice; see _inject_cond_frames.)

forward signature matches the repo contract for video backbones:
    forward(x, t, context=None, cond_frames=None) -> eps  (same shape as x)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .spacetime import SpacetimePatchEmbed
from .temporal_attention import Attention, FactorizedSpacetimeBlock, _modulate

# website: /denoiser -- the timestep embedding is shared with the image backbones.
# We import the canonical one, but keep a local fallback so this file is runnable even
# if the diffusion package is unavailable (e.g. when extracted standalone for a demo).
try:  # pragma: no cover - import shim
    from pytorch.diffusion.cross_attention import TimestepEmbedding
except Exception:  # pragma: no cover - fallback path

    class TimestepEmbedding(nn.Module):  # type: ignore[no-redef]
        """Fallback sinusoidal-timestep -> MLP embedding (matches cross_attention)."""

        def __init__(self, dim: int, hidden: Optional[int] = None):
            super().__init__()
            hidden = hidden or dim * 4
            self.dim = dim
            self.mlp = nn.Sequential(
                nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, dim)
            )

        def _sinusoid(self, t: Tensor) -> Tensor:
            half = self.dim // 2
            freqs = torch.exp(
                -math.log(10000.0)
                * torch.arange(half, device=t.device, dtype=torch.float32)
                / max(half - 1, 1)
            )
            args = t.float()[:, None] * freqs[None, :]
            return torch.cat([args.sin(), args.cos()], dim=-1)

        def forward(self, t: Tensor) -> Tensor:
            return self.mlp(self._sinusoid(t))


class _CrossAttention(nn.Module):
    """Minimal cross-attention from video tokens (queries) to text tokens (k/v).

    Kept local (rather than reusing the image CrossAttention) because here both query
    and context already live at model `dim`, and we want a tiny, mask-aware version.
    """

    def __init__(self, dim: int, context_dim: int, heads: int = 6):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim {dim} must be divisible by heads {heads}")
        self.heads = heads
        self.dim_head = dim // heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(context_dim, dim, bias=False)
        self.to_v = nn.Linear(context_dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x: Tensor, context: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        b, n, d = x.shape
        l = context.shape[1]
        q = self.to_q(x).view(b, n, self.heads, self.dim_head).transpose(1, 2)
        k = self.to_k(context).view(b, l, self.heads, self.dim_head).transpose(1, 2)
        v = self.to_v(context).view(b, l, self.heads, self.dim_head).transpose(1, 2)

        attn_mask = None
        if mask is not None:
            keep = mask[:, None, None, :].to(q.dtype)  # [B,1,1,L], True == real
            attn_mask = (1.0 - keep) * torch.finfo(q.dtype).min

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(b, n, d)
        return self.to_out(out)


class VideoDiT(nn.Module):
    """Diffusion transformer over space-time patches; predicts epsilon for a clip.

    __init__(in_channels=4, input_size=16, num_frames=8, hidden=384, depth=8, heads=6,
             patch=(1,2,2), context_dim=None)

    The patch grid is fixed at construction (input_size, num_frames, patch) so we can
    allocate a learned positional embedding of the right length. forward asserts the
    input matches.
    """

    def __init__(
        self,
        in_channels: int = 4,
        input_size: int = 16,
        num_frames: int = 8,
        hidden: int = 384,
        depth: int = 8,
        heads: int = 6,
        patch: Tuple[int, int, int] = (1, 2, 2),
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.input_size = input_size
        self.num_frames = num_frames
        self.hidden = hidden
        self.patch = patch
        self.context_dim = context_dim

        pt, ph, pw = patch
        if num_frames % pt or input_size % ph or input_size % pw:
            raise ValueError("num_frames/input_size must be divisible by patch")
        self.gt = num_frames // pt
        self.gh = input_size // ph
        self.gw = input_size // pw
        self.num_spatial = self.gh * self.gw            # S tokens per frame
        self.num_tokens = self.gt * self.num_spatial    # N total

        # --- patch embedding ---
        # We embed the noisy clip AND (for I2V) a binary "known frame" mask channel.
        # Giving the patch embed in_channels+1 lets it read whether each token is an
        # observed/clean frame. cond_frames also OVERWRITE the token content (below).
        self.patch_embed = SpacetimePatchEmbed(in_channels + 1, hidden, patch=patch)

        # learned positional embedding over the (gt, gh, gw) grid, flattened to N.
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, hidden))
        nn.init.normal_(self.pos_embed, std=0.02)

        # a learned vector added to tokens belonging to KNOWN (conditioning) frames,
        # so the network can tell "denoise me" tokens from "trust me" tokens.
        self.cond_token_embed = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.normal_(self.cond_token_embed, std=0.02)

        # --- conditioning ---
        self.t_embed = TimestepEmbedding(hidden)
        if context_dim is not None:
            # pooled-context -> cond vector, plus per-block cross-attention layers.
            self.context_pool = nn.Linear(context_dim, hidden)
            self.cross_norms = nn.ModuleList(
                [nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6) for _ in range(depth)]
            )
            self.cross_attns = nn.ModuleList(
                [_CrossAttention(hidden, context_dim, heads=heads) for _ in range(depth)]
            )
            self.cross_gates = nn.ParameterList(
                [nn.Parameter(torch.zeros(hidden)) for _ in range(depth)]  # zero-init gate
            )
        else:
            self.context_pool = None
            self.cross_norms = None
            self.cross_attns = None
            self.cross_gates = None

        # --- transformer trunk ---
        self.blocks = nn.ModuleList(
            [FactorizedSpacetimeBlock(hidden, heads=heads) for _ in range(depth)]
        )

        # --- adaLN-Zero output head ---
        self.final_norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))
        self.out_dim = in_channels * pt * ph * pw
        self.final_linear = nn.Linear(hidden, self.out_dim)

        self._zero_init_output()

    def _zero_init_output(self) -> None:
        # adaLN-Zero: zero the final modulation + projection so the model outputs ~0 eps
        # at init, i.e. it starts as the identity diffusion step (stable, DiT recipe).
        nn.init.zeros_(self.final_ada[-1].weight)
        nn.init.zeros_(self.final_ada[-1].bias)
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)

    # -- helpers ----------------------------------------------------------------
    def _unpatch_tokens(self, tokens: Tensor) -> Tensor:
        """tokens [B, N, out_dim] -> eps video [B, C, T, H, W] (folds patches back)."""
        b = tokens.shape[0]
        pt, ph, pw = self.patch
        c = self.in_channels
        # [B, gt, gh, gw, C, pt, ph, pw]
        x = tokens.reshape(b, self.gt, self.gh, self.gw, c, pt, ph, pw)
        # -> [B, C, gt, pt, gh, ph, gw, pw]
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        x = x.reshape(b, c, self.gt * pt, self.gh * ph, self.gw * pw)
        return x

    def _inject_cond_frames(
        self, x: Tensor, cond_frames: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """Build the (in_channels+1)-channel input for the patch embed.

        Returns (x_aug, known_time) where:
          x_aug      : [B, C+1, T, H, W] -- noisy clip with conditioning frames spliced
                        in as clean content, plus a final binary mask channel marking
                        which frames are observed.
          known_time : bool [B, T]       -- per-frame "is this an observed frame" flag,
                        used to add cond_token_embed to the matching tokens.

        Choice (documented): image-to-video conditioning = "replace + mask". The first
        Tc time-steps are overwritten by the clean conditioning frames and flagged via a
        mask channel; the rest stay noisy and get denoised. This mirrors how inpainting /
        I2V is usually wired into a single forward pass.
        """
        b, c, t, h, w = x.shape
        mask_ch = torch.zeros(b, 1, t, h, w, device=x.device, dtype=x.dtype)
        known_time = torch.zeros(b, t, device=x.device, dtype=torch.bool)

        if cond_frames is not None:
            tc = cond_frames.shape[2]
            if tc > t:
                raise ValueError(f"cond_frames T={tc} exceeds clip T={t}")
            if cond_frames.shape[1] != c or cond_frames.shape[3:] != x.shape[3:]:
                raise ValueError("cond_frames must match x in C,H,W")
            # Overwrite the leading frames with the clean conditioning content...
            x = x.clone()
            x[:, :, :tc] = cond_frames
            # ...and flag them in the mask channel + per-frame known flag.
            mask_ch[:, :, :tc] = 1.0
            known_time[:, :tc] = True

        x_aug = torch.cat([x, mask_ch], dim=1)  # [B, C+1, T, H, W]
        return x_aug, known_time

    # -- forward ----------------------------------------------------------------
    def forward(
        self,
        x: Tensor,
        t: Tensor,
        context: Optional[Tensor] = None,
        cond_frames: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """x [B,C,T,H,W], t [B] -> eps [B,C,T,H,W].

        `mask` is the optional context key-padding mask (bool [B,L], True == real token).
        `cond_frames` [B,C,Tc,H,W] enables image-to-video conditioning.
        """
        b, c, t_frames, h, w = x.shape
        if (c, t_frames, h, w) != (
            self.in_channels,
            self.num_frames,
            self.input_size,
            self.input_size,
        ):
            raise ValueError(
                f"VideoDiT expected [B,{self.in_channels},{self.num_frames},"
                f"{self.input_size},{self.input_size}], got {tuple(x.shape)}"
            )

        # 1) augment with cond-frame content + mask channel, then patch-embed.
        x_aug, known_time = self._inject_cond_frames(x, cond_frames)
        tokens, grid = self.patch_embed(x_aug)  # [B, N, hidden]
        tokens = tokens + self.pos_embed

        # mark known-frame tokens (expand per-frame flag to its spatial tokens).
        if known_time.any():
            # known_time [B, gt] (pt==... -> we collapse over pt by 'any' below)
            pt = self.patch[0]
            # reduce the T flag down to the gt temporal grid (a frame-patch is "known"
            # if any of its pt frames are known; with pt==1 this is a no-op).
            kt = known_time.reshape(b, self.gt, pt).any(dim=-1)  # [B, gt]
            # broadcast each temporal slot over its S spatial tokens -> [B, N]
            kt = kt[:, :, None].expand(b, self.gt, self.num_spatial).reshape(b, self.num_tokens)
            tokens = tokens + kt[..., None].to(tokens.dtype) * self.cond_token_embed

        # 2) conditioning vector: time (+ pooled context).
        cond = self.t_embed(t)  # [B, hidden]
        if context is not None and self.context_pool is not None:
            if mask is not None:
                m = mask[..., None].to(context.dtype)        # [B,L,1]
                pooled = (context * m).sum(1) / m.sum(1).clamp(min=1.0)
            else:
                pooled = context.mean(dim=1)
            cond = cond + self.context_pool(pooled)

        # 3) reshape flat tokens to [B, T(grid), S, D] for the factorized blocks.
        gt, gh, gw = grid
        s = gh * gw
        h_tok = tokens.view(b, gt, s, self.hidden)

        for i, block in enumerate(self.blocks):
            h_tok = block(h_tok, cond)  # spatial + temporal self-attn + MLP
            # optional cross-attention to text tokens (zero-gated at init).
            if self.cross_attns is not None and context is not None:
                flat = h_tok.reshape(b, gt * s, self.hidden)
                ca = self.cross_attns[i](self.cross_norms[i](flat), context, mask=mask)
                flat = flat + self.cross_gates[i] * ca
                h_tok = flat.view(b, gt, s, self.hidden)

        # 4) adaLN-Zero output head -> per-token eps patches -> fold back to a clip.
        tokens = h_tok.reshape(b, self.num_tokens, self.hidden)
        shift, scale = self.final_ada(cond).chunk(2, dim=-1)
        tokens = _modulate(self.final_norm(tokens), shift, scale)
        tokens = self.final_linear(tokens)  # [B, N, out_dim]
        eps = self._unpatch_tokens(tokens)  # [B, C, T, H, W]
        return eps
