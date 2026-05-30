"""Load a checkpoint from ``pytorch.train`` and sample images from it.

website: /build, /guidance  (the inference side: noise -> images, optionally with CFG)

Given a checkpoint written by ``pytorch.train``, this script:
  1. reconstructs the EXACT backbone / VAE / text encoder from the stored config,
  2. builds the conditioning context from a --prompt (or --label class id),
  3. runs the reverse diffusion (DDPM ancestral OR DDIM few-step) with classifier-free
     guidance, and
  4. (if the model is latent) VAE-decodes, then saves an N-up grid PNG (matplotlib Agg).

Run after train.py has produced a checkpoint, e.g.::

    python -m pytorch.train  --epochs 1 --batch 8 --dataset shapes --out /tmp/m.pt
    python -m pytorch.sample --ckpt /tmp/m.pt --steps 10 --n 2 --out /tmp/s.png

With a text-conditioned checkpoint add ``--prompt "a red circle" --guidance 4.0``.
"""

from __future__ import annotations

import argparse
from typing import Optional

import torch
from torch import Tensor

from pytorch.diffusion.schedule import NoiseSchedule
from pytorch.diffusion.ddpm import GaussianDiffusion
from pytorch.diffusion.unet import UNet
from pytorch.diffusion.dit import DiT
from pytorch.diffusion.vae import VAE
from pytorch.diffusion.text_encoder import build_text_encoder
from pytorch.diffusion.guidance import make_null_context
from pytorch.train import TrainConfig, _safe_channel_mult, CLASS_CAPTIONS


# ---------------------------------------------------------------------------
# Rebuild the model stack from a saved checkpoint.
# ---------------------------------------------------------------------------
def load_checkpoint(path: str, device: torch.device):
    """Reconstruct (cfg, backbone, vae_or_None, text_encoder_or_None) from a ckpt file.

    We trust the stored config to size every module exactly as it was trained, then
    load the saved weights. Everything is put in eval() mode for sampling.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = TrainConfig(**ckpt["config"])

    context_dim: Optional[int] = cfg.context_dim if cfg.text else None

    # --- backbone (must match build_models in train.py) ---
    if cfg.backbone == "unet":
        channel_mult = _safe_channel_mult(cfg.latent_size)
        backbone = UNet(
            in_channels=cfg.in_channels,
            out_channels=cfg.in_channels,
            model_channels=cfg.model_channels,
            channel_mult=channel_mult,
            num_res_blocks=2,
            context_dim=context_dim,
            num_heads=4,
        )
    elif cfg.backbone == "dit":
        patch = 2 if cfg.latent_size % 2 == 0 else 1
        backbone = DiT(
            in_channels=cfg.in_channels,
            input_size=cfg.latent_size,
            patch_size=patch,
            hidden=cfg.dit_hidden,
            depth=cfg.dit_depth,
            heads=6,
            context_dim=context_dim,
        )
    else:
        raise ValueError(f"unknown backbone {cfg.backbone!r}")
    backbone.load_state_dict(ckpt["model"])
    backbone = backbone.to(device).eval()

    # --- optional VAE ---
    vae = None
    if cfg.latent:
        vae = VAE(in_channels=3, latent_channels=4).to(device).eval()
        if ckpt.get("vae") is not None:
            vae.load_state_dict(ckpt["vae"])

    # --- optional text encoder ---
    text_encoder = None
    if cfg.text:
        text_encoder = build_text_encoder(dim=cfg.context_dim).to(device).eval()
        if ckpt.get("text_encoder") is not None:
            text_encoder.load_state_dict(ckpt["text_encoder"])

    return cfg, backbone, vae, text_encoder


# ---------------------------------------------------------------------------
# Build the (context, mask, uncond_context) for sampling with CFG.
# ---------------------------------------------------------------------------
@torch.no_grad()
def build_conditioning(cfg, text_encoder, prompt, label, n, device):
    """Return (context, mask, uncond_context) for ``n`` samples, or all-None if uncond.

    - If the model has no text encoder -> unconditional (all None).
    - With --prompt we encode the same prompt for every sample.
    - With --label we look up that class id's caption from CLASS_CAPTIONS.
    The unconditional context is the all-zeros "empty prompt" (make_null_context),
    which is exactly what condition-dropout trained the model to handle.
    """
    if text_encoder is None:
        return None, None, None

    if prompt is None:
        # Fall back to a class label -> caption (default class 0 if not given).
        cls = 0 if label is None else int(label) % len(CLASS_CAPTIONS)
        prompt = CLASS_CAPTIONS[cls]

    context, mask = text_encoder.encode_text([prompt] * n)  # [n, L, dim], [n, L]
    uncond = make_null_context(n, context.shape[1], context.shape[2], device=device)
    return context.to(device), mask.to(device), uncond


# ---------------------------------------------------------------------------
# Sampling.
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample_images(
    cfg,
    backbone,
    vae,
    text_encoder,
    n: int,
    sampler: str,
    steps: int,
    guidance: float,
    prompt: Optional[str],
    label: Optional[int],
    device: torch.device,
) -> Tensor:
    """Run reverse diffusion and return decoded images [n, 3, H, W] in [0, 1] for display."""
    schedule = NoiseSchedule(timesteps=cfg.timesteps, kind="cosine").to(device)
    diffusion = GaussianDiffusion(schedule, predict="eps")

    context, mask, uncond = build_conditioning(
        cfg, text_encoder, prompt, label, n, device
    )
    shape = (n, cfg.in_channels, cfg.latent_size, cfg.latent_size)

    # Guidance only matters when we actually have a condition to push toward.
    g = guidance if context is not None else 1.0

    if sampler == "ddim":
        x0 = diffusion.ddim_sample(
            backbone, shape, steps=steps, eta=0.0,
            context=context, mask=mask, guidance_scale=g, uncond_context=uncond,
            device=device,
        )
    else:  # full ancestral DDPM (uses all cfg.timesteps steps)
        x0 = diffusion.sample(
            backbone, shape,
            context=context, mask=mask, guidance_scale=g, uncond_context=uncond,
            device=device, progress=True,
        )

    # latent -> pixels if needed.
    if vae is not None:
        x0 = vae.decode(x0)

    # diffusion works in [-1, 1]; map to [0, 1] for display and clamp.
    return ((x0 + 1.0) * 0.5).clamp(0.0, 1.0)


def save_grid(images: Tensor, out: str, title: str = "") -> None:
    """Save [N, 3, H, W] images in [0,1] as a single grid PNG (headless matplotlib)."""
    import math as _math

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = images.shape[0]
    cols = int(_math.ceil(_math.sqrt(n)))
    rows = int(_math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = [axes] if n == 1 else (axes.flatten() if hasattr(axes, "flatten") else axes)

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            img = images[i].detach().cpu().permute(1, 2, 0).numpy()  # HWC
            ax.imshow(img)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[sample] wrote {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sample images from a trained checkpoint.")
    p.add_argument("--ckpt", type=str, required=True, help="path to a train.py checkpoint.")
    p.add_argument("--prompt", type=str, default=None, help="text prompt (text models).")
    p.add_argument("--label", type=int, default=None, help="class id (alternative to --prompt).")
    p.add_argument("--guidance", type=float, default=4.0, help="CFG scale (text models).")
    p.add_argument("--steps", type=int, default=50, help="DDIM steps (ignored for ddpm).")
    p.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddim")
    p.add_argument("--n", type=int, default=4, help="number of images to generate.")
    p.add_argument("--out", type=str, default="samples.png")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv: Optional[list] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    cfg, backbone, vae, text_encoder = load_checkpoint(args.ckpt, device)
    print(
        f"[sample] loaded {args.ckpt}: backbone={cfg.backbone} latent={cfg.latent} "
        f"text={cfg.text} | sampling {args.n} via {args.sampler}"
    )

    images = sample_images(
        cfg, backbone, vae, text_encoder,
        n=args.n, sampler=args.sampler, steps=args.steps,
        guidance=args.guidance, prompt=args.prompt, label=args.label, device=device,
    )

    title = args.prompt or (f"class {args.label}" if args.label is not None else "samples")
    save_grid(images, args.out, title=title)


if __name__ == "__main__":
    main()
