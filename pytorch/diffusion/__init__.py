"""Diffusion subpackage: schedules, DDPM/DDIM, denoiser backbones, VAE, text, guidance.

website: /diffusion, /latent, /denoiser, /text, /guidance

Re-exports the public API so callers can write::

    from pytorch.diffusion import NoiseSchedule, GaussianDiffusion, UNet, DiT, VAE

instead of reaching into each module.
"""

from pytorch.diffusion.schedule import (
    make_beta_schedule,
    extract,
    NoiseSchedule,
)
from pytorch.diffusion.ddpm import GaussianDiffusion
from pytorch.diffusion.guidance import (
    drop_context,
    classifier_free_guidance,
    make_null_context,
)
from pytorch.diffusion.cross_attention import (
    SinusoidalPosEmb,
    TimestepEmbedding,
    CrossAttention,
    SpatialTransformer,
    FeedForward,
)
from pytorch.diffusion.unet import UNet
from pytorch.diffusion.dit import DiT, patchify, unpatchify, DiTBlock
from pytorch.diffusion.vae import (
    LATENT_SCALE,
    DiagonalGaussian,
    Encoder,
    Decoder,
    VAE,
    VAEConfig,
)
from pytorch.diffusion.text_encoder import (
    CharTokenizer,
    TinyTextEncoder,
    build_text_encoder,
)

__all__ = [
    # schedule / process
    "make_beta_schedule",
    "extract",
    "NoiseSchedule",
    "GaussianDiffusion",
    # guidance
    "drop_context",
    "classifier_free_guidance",
    "make_null_context",
    # attention / conditioning primitives
    "SinusoidalPosEmb",
    "TimestepEmbedding",
    "CrossAttention",
    "SpatialTransformer",
    "FeedForward",
    # backbones
    "UNet",
    "DiT",
    "patchify",
    "unpatchify",
    "DiTBlock",
    # vae
    "LATENT_SCALE",
    "DiagonalGaussian",
    "Encoder",
    "Decoder",
    "VAE",
    "VAEConfig",
    # text
    "CharTokenizer",
    "TinyTextEncoder",
    "build_text_encoder",
]
