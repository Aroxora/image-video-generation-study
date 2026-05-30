"""Variational Autoencoder for *latent* diffusion (a small, faithful SD-style VAE).

website: /latent  (why we run diffusion in latent space)

--------------------------------------------------------------------------------
WHY A VAE AT ALL?  (the whole point of "latent diffusion")
--------------------------------------------------------------------------------
Running diffusion directly on pixels is brutally expensive: a 512x512 RGB image
is 512*512*3 ~= 786k numbers, and every denoising step is a full forward pass of
a big U-Net/DiT over ALL of them. Compute and memory scale with the spatial area,
so high resolution = quadratically painful.

Latent diffusion (Rombach et al., "Stable Diffusion") splits the job in two:

  1. A VAE *perceptual compressor* learns, once, to map an image into a small
     spatial latent z and back. With a downsample factor f (here f=4) a 32x32
     image becomes an 8x8 latent. Area shrinks by f*f = 16x; a real SD VAE uses
     f=8, shrinking 512x512 -> 64x64, a 64x reduction in tokens.

  2. The diffusion model then denoises in that tiny latent space instead of in
     pixels. Same architecture, far fewer "pixels" to process per step -> high-res
     generation becomes cheap enough to train and sample on modest hardware.

The VAE is trained SEPARATELY (reconstruction + KL, usually + a small adversarial
/ perceptual loss in the real thing) and then FROZEN. Diffusion never sees pixels;
it only ever sees latents. At the end we decode the final latent to an image.

--------------------------------------------------------------------------------
WHY VARIATIONAL (the KL term)?
--------------------------------------------------------------------------------
A plain autoencoder can carve out a wildly irregular, "spiky" latent space — fine
for reconstruction, terrible to put a smooth Gaussian diffusion prior on. The KL
penalty pulls the per-image posterior q(z|x) = N(mean, var) gently toward N(0, I),
keeping the latent space well-scaled and roughly unit-variance. That regularity is
exactly what lets the diffusion model treat z as "just some Gaussian-ish tensor."
The KL weight is kept TINY (we don't apply it here; the trainer scales it) so the
VAE stays reconstruction-dominated — it is a compressor first, a generator second.

--------------------------------------------------------------------------------
THE scale_factor (LATENT_SCALE = 0.18215)
--------------------------------------------------------------------------------
Even after the KL, the raw VAE latent has some empirical std != 1. Stable Diffusion
measured it and multiplies the latent by a constant so the data fed to diffusion has
~unit variance (this is the famous 0.18215). The convention is:

    z_for_diffusion = encode(x).sample() * scale_factor      # std ~= 1
    x_recon         = decode(z_for_diffusion)                # we divide by scale_factor inside

So `encode`/`decode` here operate in *diffusion space*: encode multiplies by
scale_factor, decode divides by it. That way the rest of the codebase can pretend
the latent is a clean unit-variance tensor and never juggle the constant.

Tensor conventions: images/latents are [B, C, H, W].
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ----------------------------------------------------------------------------- #
# Stable-Diffusion's empirical latent rescale. See the module docstring.
# ----------------------------------------------------------------------------- #
LATENT_SCALE: float = 0.18215


# ============================================================================= #
# The variational latent: a per-pixel diagonal Gaussian N(mean, diag(exp(logvar)))
# ============================================================================= #
class DiagonalGaussian:
    """A diagonal Gaussian posterior q(z|x) over a latent tensor.

    The encoder outputs 2*latent_channels feature maps; we split them into a
    `mean` and a `logvar` (log-variance is numerically friendlier than variance
    or std: it's unconstrained, so the network can output any real number, and
    exp() keeps the variance strictly positive).

    Shapes: mean, logvar are [B, C, H, W]; sampling/mode return the same shape.
    """

    def __init__(self, mean: Tensor, logvar: Tensor) -> None:
        self.mean = mean
        # Clamp logvar to a sane range so var = exp(logvar) never under/overflows.
        # (-30 => var ~ 1e-13, +20 => var ~ 5e8; well inside float32.)
        self.logvar = torch.clamp(logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self, generator: torch.Generator | None = None) -> Tensor:
        """Reparameterized sample: z = mean + std * eps, eps ~ N(0, I).

        The reparameterization trick keeps the randomness (`eps`) OUTSIDE the
        path that carries gradients, so we can backprop through `mean`/`std`.
        """
        eps = torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.mean.device,
            dtype=self.mean.dtype,
        )
        return self.mean + self.std * eps

    def mode(self) -> Tensor:
        """The distribution's mode == its mean (deterministic, noise-free latent).

        Used at inference when we want a stable, reproducible latent instead of
        a random draw.
        """
        return self.mean

    def kl(self) -> Tensor:
        """KL( N(mean, var) || N(0, I) ), summed over latent dims, mean over batch.

        Closed form for a diagonal Gaussian vs the unit Gaussian:
            KL = 0.5 * sum( mean^2 + var - 1 - logvar )
        Summed over (C, H, W) per image, then averaged over the batch B, giving a
        scalar suitable to add to the loss.
        """
        # Per-element KL contribution.
        per_elem = 0.5 * (self.mean.pow(2) + self.var - 1.0 - self.logvar)
        # Sum over channel + spatial dims, average over the batch.
        return per_elem.flatten(start_dim=1).sum(dim=1).mean()


# ============================================================================= #
# Building blocks: a GroupNorm + SiLU ResNet block and simple up/down sampling.
# (Mirrors the structure of the real SD VAE, just smaller.)
# ============================================================================= #
def _group_norm(channels: int, groups: int = 32) -> nn.GroupNorm:
    """GroupNorm with a group count that always divides `channels`.

    GroupNorm (not BatchNorm) because the VAE is trained with small batches and
    must behave identically at batch size 1 during inference.
    """
    g = math.gcd(groups, channels)
    return nn.GroupNorm(num_groups=max(1, g), num_channels=channels, eps=1e-6, affine=True)


class ResnetBlock(nn.Module):
    """Pre-norm residual block: GN -> SiLU -> conv, twice, plus a skip.

    A 1x1 conv adapts the skip connection when in/out channel counts differ.
    """

    def __init__(self, in_channels: int, out_channels: int | None = None) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        self.norm1 = _group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = _group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # Identity skip if channels match, else a learned 1x1 projection.
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    """Halve H and W with a strided 3x3 conv (learned anti-aliased downsample)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        # padding handled manually (asymmetric) to match the SD VAE's behavior.
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0.0)
        return self.conv(x)


class Upsample(nn.Module):
    """Double H and W with nearest-neighbor upsampling + a 3x3 conv to smooth it."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


# ============================================================================= #
# Encoder: image  ->  (mean, logvar) feature maps of the latent Gaussian.
# ============================================================================= #
class Encoder(nn.Module):
    """Conv encoder producing 2*latent_channels maps (mean ++ logvar).

    Downsamples by 2 once per *additional* ch_mult level, so the total factor is
    2**(len(ch_mult) - 1). With ch_mult=(1,2,4) -> factor 4.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 64,
        ch_mult: tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.num_resolutions = len(ch_mult)

        # Stem: lift RGB to the base feature width at full resolution.
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Down path: blocks at each resolution, downsample between resolutions.
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
            # Downsample after every level except the last (the last is the latent res).
            if level != self.num_resolutions - 1:
                self.downsamplers.append(Downsample(cur_ch))
            else:
                self.downsamplers.append(nn.Identity())

        # Bottleneck: two res blocks at the lowest resolution.
        self.mid_block1 = ResnetBlock(cur_ch, cur_ch)
        self.mid_block2 = ResnetBlock(cur_ch, cur_ch)

        # Head: GN -> SiLU -> conv to 2*latent_channels (mean and logvar stacked).
        self.norm_out = _group_norm(cur_ch)
        self.conv_out = nn.Conv2d(cur_ch, 2 * latent_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv_in(x)
        for level, blocks in enumerate(self.down_blocks):
            for block in blocks:
                h = block(h)
            h = self.downsamplers[level](h)
        h = self.mid_block1(h)
        h = self.mid_block2(h)
        h = self.conv_out(F.silu(self.norm_out(h)))
        return h  # [B, 2*latent_channels, H/f, W/f]


# ============================================================================= #
# Decoder: latent  ->  reconstructed image.  (Mirror image of the encoder.)
# ============================================================================= #
class Decoder(nn.Module):
    """Conv decoder mapping a latent back up to an RGB image.

    Upsamples by 2 once per *additional* ch_mult level (mirror of the encoder),
    so it exactly undoes the encoder's spatial reduction.
    """

    def __init__(
        self,
        out_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 64,
        ch_mult: tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.num_resolutions = len(ch_mult)

        # Start at the deepest (widest) channel width, matching the encoder's end.
        cur_ch = base_channels * ch_mult[-1]

        # Lift the latent into feature space at the latent resolution.
        self.conv_in = nn.Conv2d(latent_channels, cur_ch, kernel_size=3, padding=1)

        # Bottleneck mirror.
        self.mid_block1 = ResnetBlock(cur_ch, cur_ch)
        self.mid_block2 = ResnetBlock(cur_ch, cur_ch)

        # Up path: walk ch_mult in reverse. We use (num_res_blocks + 1) blocks per
        # level, the common SD VAE choice for a slightly heavier decoder.
        self.up_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        for level, mult in enumerate(reversed(ch_mult)):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                blocks.append(ResnetBlock(cur_ch, out_ch))
                cur_ch = out_ch
            self.up_blocks.append(blocks)
            # Upsample after every level except the last (the last is full res).
            if level != self.num_resolutions - 1:
                self.upsamplers.append(Upsample(cur_ch))
            else:
                self.upsamplers.append(nn.Identity())

        # Head: GN -> SiLU -> conv back to image channels.
        self.norm_out = _group_norm(cur_ch)
        self.conv_out = nn.Conv2d(cur_ch, out_channels, kernel_size=3, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        h = self.conv_in(z)
        h = self.mid_block1(h)
        h = self.mid_block2(h)
        for level, blocks in enumerate(self.up_blocks):
            for block in blocks:
                h = block(h)
            h = self.upsamplers[level](h)
        h = self.conv_out(F.silu(self.norm_out(h)))
        return h  # [B, out_channels, H, W]


# ============================================================================= #
# The full VAE: encode -> sample/mode latent -> decode.
# ============================================================================= #
@dataclass
class VAEConfig:
    """Plain config record (handy for logging / reconstructing the model)."""

    in_channels: int = 3
    latent_channels: int = 4
    base_channels: int = 64
    ch_mult: tuple[int, ...] = (1, 2, 4)
    scale_factor: float = LATENT_SCALE


class VAE(nn.Module):
    """SD-style latent VAE: the frozen image<->latent bridge for latent diffusion.

    Downsample factor = 2 ** (len(ch_mult) - 1).  Defaults give factor 4:
        x:[B,3,32,32]  --encode-->  z:[B,4,8,8]  --decode-->  x_recon:[B,3,32,32]

    IMPORTANT scale_factor convention (see module docstring):
      * `encode(x).sample()` returns a latent already MULTIPLIED by scale_factor,
        i.e. it lives in the ~unit-variance "diffusion space" the rest of the
        codebase expects.
      * `decode(z)` therefore DIVIDES by scale_factor before running the decoder,
        undoing that rescale. So encode/decode are exact inverses w.r.t. the
        constant and callers never have to think about 0.18215.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 64,
        ch_mult: tuple[int, ...] = (1, 2, 4),
        scale_factor: float = LATENT_SCALE,
    ) -> None:
        super().__init__()
        self.config = VAEConfig(
            in_channels=in_channels,
            latent_channels=latent_channels,
            base_channels=base_channels,
            ch_mult=tuple(ch_mult),
            scale_factor=scale_factor,
        )
        self.latent_channels = latent_channels
        self.scale_factor = scale_factor
        # Spatial reduction applied by encode (and undone by decode).
        self.downsample_factor = 2 ** (len(ch_mult) - 1)

        self.encoder = Encoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            base_channels=base_channels,
            ch_mult=tuple(ch_mult),
        )
        self.decoder = Decoder(
            out_channels=in_channels,
            latent_channels=latent_channels,
            base_channels=base_channels,
            ch_mult=tuple(ch_mult),
        )

    # ----- encode / decode -------------------------------------------------- #
    def encode(self, x: Tensor) -> DiagonalGaussian:
        """Image -> posterior q(z|x) in *diffusion space*.

        The encoder emits 2*latent_channels maps; we split them into mean/logvar
        and FOLD scale_factor into the mean & logvar so that anything drawn from
        the returned distribution (`.sample()` / `.mode()`) is already scaled.

        Scaling the Gaussian by a constant s: mean -> s*mean, var -> s^2*var,
        hence logvar -> logvar + 2*log(s). Doing it on the distribution (not just
        on a sample) keeps `.sample()`, `.mode()`, and `.kl()` mutually consistent.
        """
        h = self.encoder(x)
        mean, logvar = torch.chunk(h, 2, dim=1)
        s = self.scale_factor
        mean = mean * s
        logvar = logvar + 2.0 * math.log(s)
        return DiagonalGaussian(mean, logvar)

    def decode(self, z: Tensor) -> Tensor:
        """Latent (in diffusion space) -> reconstructed image.

        Divides by scale_factor first to return to the decoder's native latent
        space, then runs the conv decoder.
        """
        z = z / self.scale_factor
        return self.decoder(z)

    # ----- full forward (for VAE training) ---------------------------------- #
    def forward(self, x: Tensor) -> tuple[Tensor, DiagonalGaussian]:
        """Encode, sample a latent (reparameterized), decode.

        Returns (x_recon, posterior). The trainer combines a reconstruction loss
        on x_recon with a small `posterior.kl()` regularizer.
        """
        posterior = self.encode(x)
        z = posterior.sample()
        x_recon = self.decode(z)
        return x_recon, posterior


__all__ = [
    "LATENT_SCALE",
    "DiagonalGaussian",
    "Encoder",
    "Decoder",
    "VAE",
    "VAEConfig",
]


# ----------------------------------------------------------------------------- #
# Tiny self-test / demo: run `python -m pytorch.diffusion.vae` from the repo root.
# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)
    vae = VAE()  # ch_mult=(1,2,4) -> downsample factor 4
    x = torch.randn(2, 3, 32, 32)

    posterior = vae.encode(x)
    z = posterior.sample()
    print("latent (z) shape:", tuple(z.shape))          # expect (2, 4, 8, 8)
    assert tuple(z.shape) == (2, 4, 8, 8)

    x_recon = vae.decode(z)
    print("reconstruction shape:", tuple(x_recon.shape))  # expect (2, 3, 32, 32)
    assert tuple(x_recon.shape) == x.shape

    recon2, post2 = vae(x)
    assert recon2.shape == x.shape

    kl = posterior.kl()
    print("kl() scalar:", float(kl), "| is scalar:", kl.dim() == 0)
    assert kl.dim() == 0

    print("downsample factor:", vae.downsample_factor)
    print("OK")
