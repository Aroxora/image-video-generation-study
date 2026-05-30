"""A small VQ-VAE: the image <-> *discrete token* bridge for autoregressive models.

website: /autoregressive  (how an image becomes a grid of integer "codes" that a
language-model-style Transformer can predict one token at a time)

--------------------------------------------------------------------------------
WHY DISCRETE TOKENS?  (the whole point of the autoregressive family)
--------------------------------------------------------------------------------
Diffusion (see ../diffusion/) treats an image as a *continuous* tensor and denoises
it. The autoregressive family does something completely different: it first turns an
image into a small grid of INTEGER codes -- exactly like words in a sentence -- and
then models images the way GPT models text: predict the next token given all previous
tokens. DALL-E 1, VQGAN+Transformer, Parti, MUSE and (for video) MagViT / VideoPoet /
Sora-adjacent token models all live here.

The translator that makes this possible is the VQ-VAE (van den Oord et al., 2017):

  image x ──Encoder──> z_e (continuous feature grid [B, D, H', W'])
          ──VectorQuantizer──> z_q (each cell SNAPPED to its nearest of K codebook
                                    vectors) + indices [B, H', W'] (the integer codes)
          ──Decoder──> x_recon

After training the VQ-VAE is FROZEN. To model images we only ever look at the
`indices`: a HxW grid of integers in [0, K). Flatten that grid row-major and you have
a *sentence of image tokens* a Transformer can autoregress over (see transformer.py).
To turn predicted tokens back into pixels we look the codes up in the codebook and run
the decoder (`decode_indices`).

--------------------------------------------------------------------------------
THE VECTOR-QUANTIZER (the only really new idea here)
--------------------------------------------------------------------------------
A codebook is K learnable vectors of dimension D: an nn.Embedding(K, D). For each
spatial cell of z_e we find the nearest codebook vector (in L2) and replace the cell
with it -> z_q. Two problems and their standard fixes:

  1. argmin/nearest-neighbour is NOT differentiable. Fix: the *straight-through
     estimator*. On the forward pass z_q is the snapped vector; on the backward pass
     we copy the gradient straight from z_q to z_e by writing
         z_q = z_e + (z_q - z_e).detach()
     so d z_q / d z_e == 1 and the encoder still learns.

  2. Nothing yet ties z_e to the codebook. Fix: the VQ loss has two terms,
         codebook loss   ||sg[z_e] - e||^2   moves the chosen code e toward z_e,
         commitment loss  beta * ||z_e - sg[e]||^2   moves z_e toward its code,
     where sg[] = stop-gradient (.detach()). beta (~0.25) weights how strongly the
     encoder is asked to "commit" to a code rather than drift.

Tensor conventions (repo-wide): images [B, C, H, W]; here the code grid is
indices [B, H', W'] long, with H' = W' = H / 2**(len(ch_mult)-1).
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["VectorQuantizer", "VQVAE"]


# ---------------------------------------------------------------------------
# small conv building blocks (a trimmed-down version of the VAE's blocks)
# ---------------------------------------------------------------------------
def _group_norm(channels: int, groups: int = 32) -> nn.GroupNorm:
    """GroupNorm with a group count that always divides `channels`.

    GroupNorm (not BatchNorm) so the net behaves identically at batch size 1 and
    with the tiny batches a laptop uses.
    """
    g = math.gcd(groups, channels)
    return nn.GroupNorm(num_groups=max(1, g), num_channels=channels, eps=1e-6, affine=True)


class ResnetBlock(nn.Module):
    """Pre-norm residual block: GN -> SiLU -> conv, twice, plus a (maybe projected) skip."""

    def __init__(self, in_channels: int, out_channels: int | None = None) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        self.norm1 = _group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = _group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


# ===========================================================================
# The vector quantizer: continuous feature grid -> nearest codebook vectors.
# ===========================================================================
class VectorQuantizer(nn.Module):
    """Snap each spatial cell of z_e to its nearest of `num_codes` codebook vectors.

    website: /autoregressive (the step that makes images *discrete*)

    forward(z_e: [B, D, H, W]) -> (z_q: [B, D, H, W], indices: [B, H, W] long, vq_loss)
      * z_q      : the quantized grid, differentiable via the straight-through trick.
      * indices  : the chosen codebook index per cell -- THIS is the "image token" grid.
      * vq_loss  : codebook + beta*commitment loss (a scalar) to add to the recon loss.
    """

    def __init__(self, num_codes: int = 512, dim: int = 64, beta: float = 0.25):
        super().__init__()
        self.num_codes = num_codes
        self.dim = dim
        self.beta = beta
        # The codebook: K vectors of width D. nn.Embedding is just a (K, D) table.
        self.codebook = nn.Embedding(num_codes, dim)
        # Uniform init in [-1/K, 1/K] is the classic VQ-VAE choice; keeps codes small
        # and spread out so early nearest-neighbour lookups aren't degenerate.
        self.codebook.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)

    def forward(self, z_e: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        b, d, h, w = z_e.shape
        assert d == self.dim, f"VectorQuantizer expected dim={self.dim}, got {d}"

        # Flatten to a list of vectors: [B, D, H, W] -> [B*H*W, D].
        # We move D to the last axis first so each row is one spatial cell's vector.
        z_flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, d)  # [N, D], N = B*H*W

        # Squared L2 distance from every cell to every code, without forming the
        # full outer product: ||a-b||^2 = ||a||^2 - 2 a.b + ||b||^2.
        codebook = self.codebook.weight  # [K, D]
        dist = (
            z_flat.pow(2).sum(dim=1, keepdim=True)              # [N, 1]
            - 2.0 * z_flat @ codebook.t()                       # [N, K]
            + codebook.pow(2).sum(dim=1)[None, :]               # [1, K]
        )

        # Nearest code per cell -> the integer tokens.
        flat_indices = dist.argmin(dim=1)                       # [N]
        indices = flat_indices.view(b, h, w)                    # [B, H, W]

        # Gather the chosen code vectors and reshape back to a feature grid.
        z_q = self.codebook(flat_indices).view(b, h, w, d).permute(0, 3, 1, 2).contiguous()

        # --- VQ loss (two terms; see module docstring) -----------------------
        #   codebook loss    : pull the chosen code toward the encoder output
        #   commitment loss  : pull the encoder output toward its code (weighted by beta)
        # .detach() implements the stop-gradient sg[].
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + self.beta * commitment_loss

        # --- straight-through estimator --------------------------------------
        # Forward: z_q (snapped). Backward: gradient flows as if z_q == z_e, because
        # (z_q - z_e).detach() has zero gradient, so d z_q / d z_e == 1.
        z_q = z_e + (z_q - z_e).detach()

        return z_q, indices, vq_loss

    def embed_indices(self, indices: Tensor) -> Tensor:
        """[B, H, W] long codes -> [B, D, H, W] code vectors (the decode-time lookup)."""
        b, h, w = indices.shape
        z_q = self.codebook(indices)                            # [B, H, W, D]
        return z_q.permute(0, 3, 1, 2).contiguous()             # [B, D, H, W]


# ===========================================================================
# Encoder / Decoder: small conv stacks around the quantizer.
# ===========================================================================
class VQEncoder(nn.Module):
    """Image -> continuous feature grid z_e of width `dim` at the downsampled res."""

    def __init__(
        self,
        in_channels: int,
        dim: int,
        base_channels: int,
        ch_mult: Tuple[int, ...],
        num_res_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        cur_ch = base_channels
        for level, mult in enumerate(ch_mult):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResnetBlock(cur_ch, out_ch))
                cur_ch = out_ch
            self.down_blocks.append(blocks)
            # Halve resolution after every level except the last (a strided conv).
            if level != len(ch_mult) - 1:
                self.downsamplers.append(
                    nn.Conv2d(cur_ch, cur_ch, kernel_size=3, stride=2, padding=1)
                )
            else:
                self.downsamplers.append(nn.Identity())

        self.mid_block = ResnetBlock(cur_ch, cur_ch)
        self.norm_out = _group_norm(cur_ch)
        # Project to exactly `dim` so the quantizer's codebook width lines up.
        self.conv_out = nn.Conv2d(cur_ch, dim, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv_in(x)
        for level, blocks in enumerate(self.down_blocks):
            for block in blocks:
                h = block(h)
            h = self.downsamplers[level](h)
        h = self.mid_block(h)
        return self.conv_out(F.silu(self.norm_out(h)))


class VQDecoder(nn.Module):
    """Quantized feature grid z_q -> reconstructed image (mirror of the encoder)."""

    def __init__(
        self,
        out_channels: int,
        dim: int,
        base_channels: int,
        ch_mult: Tuple[int, ...],
        num_res_blocks: int = 2,
    ) -> None:
        super().__init__()
        cur_ch = base_channels * ch_mult[-1]
        self.conv_in = nn.Conv2d(dim, cur_ch, kernel_size=3, padding=1)
        self.mid_block = ResnetBlock(cur_ch, cur_ch)

        self.up_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        for level, mult in enumerate(reversed(ch_mult)):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                blocks.append(ResnetBlock(cur_ch, out_ch))
                cur_ch = out_ch
            self.up_blocks.append(blocks)
            # Double resolution after every level except the last (nearest + conv).
            if level != len(ch_mult) - 1:
                self.upsamplers.append(
                    nn.Conv2d(cur_ch, cur_ch, kernel_size=3, padding=1)
                )
                self._needs_upsample = True
            else:
                self.upsamplers.append(nn.Identity())

        self.norm_out = _group_norm(cur_ch)
        self.conv_out = nn.Conv2d(cur_ch, out_channels, kernel_size=3, padding=1)
        self.num_resolutions = len(ch_mult)

    def forward(self, z_q: Tensor) -> Tensor:
        h = self.conv_in(z_q)
        h = self.mid_block(h)
        for level, blocks in enumerate(self.up_blocks):
            for block in blocks:
                h = block(h)
            if level != self.num_resolutions - 1:
                # nearest-neighbour upsample, then the level's smoothing conv.
                h = F.interpolate(h, scale_factor=2.0, mode="nearest")
                h = self.upsamplers[level](h)
        return self.conv_out(F.silu(self.norm_out(h)))


# ===========================================================================
# The full VQ-VAE.
# ===========================================================================
class VQVAE(nn.Module):
    """Image <-> discrete-code translator for the autoregressive image-token pipeline.

    website: /autoregressive

    Downsample factor = 2 ** (len(ch_mult) - 1).  Defaults give factor 2:
        x:[B,3,32,32]  --encode-->  indices:[B,16,16]  --decode_indices-->  x_recon:[B,3,32,32]

    Public API (used by transformer.py / sample.py):
        encode(x)            -> indices [B, H', W'] long       (the image "sentence")
        decode_indices(idx)  -> x_recon [B, C, H, W]           (codes -> pixels)
        forward(x)           -> (x_recon, vq_loss, indices)    (for VQ-VAE training)
    """

    def __init__(
        self,
        in_channels: int = 3,
        dim: int = 64,
        num_codes: int = 512,
        ch_mult: Tuple[int, ...] = (1, 2),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.dim = dim
        self.num_codes = num_codes
        self.ch_mult = tuple(ch_mult)
        # Spatial reduction the encoder applies (and the decoder undoes).
        self.downsample_factor = 2 ** (len(ch_mult) - 1)

        base_channels = dim  # keep the conv trunk width tied to the code width
        self.encoder = VQEncoder(in_channels, dim, base_channels, self.ch_mult)
        self.quantizer = VectorQuantizer(num_codes=num_codes, dim=dim)
        self.decoder = VQDecoder(in_channels, dim, base_channels, self.ch_mult)

    # ----- the pieces the rest of the codebase calls ----------------------- #
    @torch.no_grad()
    def encode(self, x: Tensor) -> Tensor:
        """Image -> integer code grid [B, H', W'] (no gradients; inference helper).

        This is what you feed (flattened row-major) into the Transformer as the
        image half of the [BOS]text[BOI]codes[EOI] sequence.
        """
        z_e = self.encoder(x)
        _, indices, _ = self.quantizer(z_e)
        return indices

    def decode_indices(self, indices: Tensor) -> Tensor:
        """Integer code grid [B, H', W'] -> reconstructed image [B, C, H, W].

        Looks the codes up in the codebook (no nearest-neighbour search needed) and
        runs the decoder. This is the final step of autoregressive generation: the
        Transformer predicts a grid of indices, we turn them back into pixels here.
        """
        z_q = self.quantizer.embed_indices(indices)
        return self.decoder(z_q)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Full pass for TRAINING the VQ-VAE.

        Returns (x_recon, vq_loss, indices). The trainer minimizes
            reconstruction_loss(x_recon, x) + vq_loss
        (a real VQGAN adds perceptual + adversarial terms; we keep it to recon+VQ).
        """
        z_e = self.encoder(x)
        z_q, indices, vq_loss = self.quantizer(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss, indices


# ----------------------------------------------------------------------------- #
# Tiny self-test / demo: run `python -m pytorch.autoregressive.vqvae`.
# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)
    vqvae = VQVAE()  # ch_mult=(1,2) -> downsample factor 2
    x = torch.randn(2, 3, 32, 32)

    x_recon, vq_loss, indices = vqvae(x)
    print("indices shape:", tuple(indices.shape))     # expect (2, 16, 16)
    print("recon shape  :", tuple(x_recon.shape))     # expect (2, 3, 32, 32)
    print("vq_loss      :", float(vq_loss.detach()), "| scalar:", vq_loss.dim() == 0)
    assert tuple(x_recon.shape) == x.shape
    assert indices.dtype == torch.long and indices.min() >= 0 and indices.max() < vqvae.num_codes
    assert vq_loss.dim() == 0

    # encode -> decode_indices round-trip keeps the shape.
    idx = vqvae.encode(x)
    rec = vqvae.decode_indices(idx)
    assert tuple(rec.shape) == x.shape
    print("downsample factor:", vqvae.downsample_factor)
    print("OK")
