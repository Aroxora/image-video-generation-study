"""gen·lab — "how generation works", the build-it-yourself PyTorch companion.

website / repo: image-video-generation-study (GitHub: Aroxora/image-video-generation-study)

This package is the reference implementation each page of the gen·lab website links
to. It is intentionally small (laptop / CPU friendly) but faithful to how real image
and video generators are built: latent diffusion (Stable Diffusion), Diffusion
Transformers (DiT / SD3 / Sora-style), classifier-free guidance, a tiny text encoder,
factorized space-time video transformers, and a VQ-token autoregressive pipeline.

Import the subpackages directly, e.g.::

    from pytorch.diffusion import NoiseSchedule, GaussianDiffusion, UNet, DiT
    from pytorch.video import VideoDiT
    from pytorch.autoregressive import VQVAE, TokenTransformer

The training entry points live at the top level::

    python -m pytorch.train  --epochs 1 --batch 8 --dataset shapes
    python -m pytorch.sample  --ckpt runs/model.pt --prompt "a red circle"
    python -m pytorch.tests.smoke

Tensor conventions used everywhere:
    images / latents : [B, C, H, W]
    video            : [B, C, T, H, W]
    timesteps t      : LongTensor [B], values in [0, timesteps)
    context (text)   : [B, L, context_dim] float; mask bool [B, L] (True = REAL token)
    every backbone   : forward(x, t, context=None, mask=None) -> eps (same shape as x)
"""

__all__ = ["diffusion", "video", "autoregressive", "toy"]
