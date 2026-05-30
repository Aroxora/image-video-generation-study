"""End-to-end trainer for the image diffusion pipeline (the /build page in code).

website: /build  (wire the pieces together: data -> [VAE] -> diffusion -> denoiser)

This single script ties together everything in ``pytorch/diffusion``:

    dataset  ->  (optional) VAE.encode to a latent  ->  NoiseSchedule + GaussianDiffusion
             ->  a denoiser backbone (UNet or DiT) with optional text cross-attention
             ->  classifier-free-guidance dropout on the text  ->  AdamW loop  ->  save.

It is deliberately tiny so ``python -m pytorch.train --epochs 1 --batch 8 --dataset
shapes`` finishes in seconds on a CPU with NO downloads (the default "shapes" dataset is
generated synthetically). The checkpoint it writes is consumed by ``pytorch.sample``.

Defaults chosen for speed/clarity:
  --backbone unet     U-Net denoiser (use --backbone dit for the transformer).
  --dataset shapes    synthetic colored shapes (circle/square/triangle), no downloads.
  --no-latent         diffuse on raw pixels (faster to demo than training a VAE too).
  --no-text           condition on a few class labels mapped to short strings, OR run
                      fully unconditional. With --text we build a TinyTextEncoder and
                      cross-attend to per-class captions.

Latent vs pixel (website: /latent):
  With --latent we encode 32x32 images to a 4-channel 8x8 latent with the (here
  randomly-initialized, for-demo) VAE and run diffusion there. Real systems FREEZE a
  pretrained VAE; we keep it trainable-but-frozen-grad here only so the shapes line up.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from pytorch.diffusion.schedule import NoiseSchedule
from pytorch.diffusion.ddpm import GaussianDiffusion
from pytorch.diffusion.unet import UNet
from pytorch.diffusion.dit import DiT
from pytorch.diffusion.vae import VAE
from pytorch.diffusion.guidance import drop_context
from pytorch.diffusion.text_encoder import build_text_encoder, TinyTextEncoder


# ---------------------------------------------------------------------------
# A few class labels and the short captions we use when --text is on.
# (Index == class id; the synthetic dataset draws one of these shapes/colors.)
# ---------------------------------------------------------------------------
SHAPE_NAMES = ["circle", "square", "triangle"]
COLOR_NAMES = ["red", "green", "blue"]
# The label space is (shape, color); we flatten to a single class id.
CLASS_CAPTIONS: List[str] = [
    f"a {color} {shape}" for shape in SHAPE_NAMES for color in COLOR_NAMES
]
NUM_CLASSES = len(CLASS_CAPTIONS)  # 9


# ---------------------------------------------------------------------------
# Config (dataclass per repo style). Stored in the checkpoint so sample.py can
# faithfully reconstruct the exact same model.
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    backbone: str = "unet"          # "unet" | "dit"
    dataset: str = "shapes"         # "shapes" | "mnist"
    latent: bool = False            # diffuse in VAE latent space?
    text: bool = False              # condition on text captions (else class-label/uncond)
    image_size: int = 32
    timesteps: int = 1000
    guidance_dropout: float = 0.1   # CFG condition-dropout probability
    epochs: int = 1
    batch: int = 8
    steps_per_epoch: int = 50       # synthetic data is infinite; cap steps/epoch
    lr: float = 2e-4
    context_dim: int = 256          # must match TinyTextEncoder dim
    model_channels: int = 64        # UNet width
    dit_hidden: int = 192           # DiT width (small for laptop)
    dit_depth: int = 6
    seed: int = 0
    device: str = "cpu"
    out: str = "runs/model.pt"

    # Derived at build time (filled in by build_models); stored for sample.py.
    in_channels: int = 0            # channels the diffusion backbone sees (3 px / 4 latent)
    latent_size: int = 0            # spatial size the backbone sees (32 px / 8 latent)


# ---------------------------------------------------------------------------
# Synthetic "shapes" dataset: colored circle / square / triangle on a dark bg.
# Returns images in [-1, 1] (the range diffusion expects) plus class ids.
# ---------------------------------------------------------------------------
def _draw_shapes_batch(
    batch: int, image_size: int, generator: torch.Generator
) -> Tuple[Tensor, Tensor]:
    """Render a batch of [B, 3, H, W] images in [-1, 1] with their class ids [B].

    No PIL needed: we rasterize circles/squares/triangles with tensor masks. Each
    image has ONE shape in ONE color, so the class id encodes (shape, color).
    """
    H = W = image_size
    # Normalized pixel coordinate grids in [-1, 1].
    ys = torch.linspace(-1.0, 1.0, H).view(H, 1).expand(H, W)
    xs = torch.linspace(-1.0, 1.0, W).view(1, W).expand(H, W)

    imgs = torch.full((batch, 3, H, W), -1.0)  # dark background == -1 in all channels
    labels = torch.randint(0, NUM_CLASSES, (batch,), generator=generator)

    # Three base colors (RGB) at full intensity (+1) on their channel.
    palette = torch.tensor(
        [[1.0, -1.0, -1.0],   # red
         [-1.0, 1.0, -1.0],   # green
         [-1.0, -1.0, 1.0]],  # blue
    )

    for b in range(batch):
        cls = int(labels[b])
        shape_idx = cls // len(COLOR_NAMES)   # 0..2
        color_idx = cls % len(COLOR_NAMES)    # 0..2
        color = palette[color_idx]            # [3]

        # Random center jitter + radius so the model sees variety (not one fixed sprite).
        cx = (torch.rand(1, generator=generator).item() - 0.5) * 0.5
        cy = (torch.rand(1, generator=generator).item() - 0.5) * 0.5
        r = 0.35 + 0.25 * torch.rand(1, generator=generator).item()

        if shape_idx == 0:      # circle: inside a radius
            mask = ((xs - cx) ** 2 + (ys - cy) ** 2) <= r * r
        elif shape_idx == 1:    # square: inside an L-inf ball
            mask = (xs - cx).abs().clamp_min((ys - cy).abs()) <= r
        else:                   # triangle: below two slanted lines + a flat bottom
            up = (ys - cy) <= (r - 2.0 * (xs - cx).abs())
            bottom = (ys - cy) >= -r
            mask = up & bottom

        # Paint the masked pixels with the chosen color on all 3 channels.
        m = mask.unsqueeze(0)  # [1, H, W]
        imgs[b] = torch.where(m, color.view(3, 1, 1).expand(3, H, W), imgs[b])

    return imgs.clamp(-1.0, 1.0), labels


def _mnist_loader(image_size: int):
    """Lazy MNIST loader (only if --dataset mnist). Requires torchvision (optional dep).

    Returns a callable batch sampler with the SAME interface as the shapes generator:
    (batch, image_size, generator) -> (images [B,3,H,W] in [-1,1], labels [B]).
    MNIST is greyscale; we replicate to 3 channels so the rest of the pipeline is uniform.
    """
    try:
        import torchvision  # type: ignore
        from torchvision import transforms  # type: ignore
    except Exception as e:  # pragma: no cover - optional path
        raise RuntimeError(
            "--dataset mnist needs torchvision (pip install torchvision). "
            "The default --dataset shapes needs no extra packages."
        ) from e

    tfm = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),                 # -> [0, 1]
            transforms.Normalize((0.5,), (0.5,)),  # -> [-1, 1]
        ]
    )
    root = os.path.join(os.path.expanduser("~"), ".cache", "genlab-mnist")
    ds = torchvision.datasets.MNIST(root, train=True, download=True, transform=tfm)

    def sampler(batch: int, image_size: int, generator: torch.Generator):
        idx = torch.randint(0, len(ds), (batch,), generator=generator)
        xs, ys = [], []
        for i in idx.tolist():
            x, y = ds[i]
            xs.append(x.expand(3, image_size, image_size))  # grey -> RGB
            ys.append(y)
        return torch.stack(xs), torch.tensor(ys, dtype=torch.long)

    return sampler


def make_dataset(cfg: TrainConfig):
    """Return a ``sampler(batch, image_size, generator) -> (images, labels)``."""
    if cfg.dataset == "shapes":
        return _draw_shapes_batch
    if cfg.dataset == "mnist":
        return _mnist_loader(cfg.image_size)
    raise ValueError(f"unknown dataset {cfg.dataset!r} (expected 'shapes' or 'mnist')")


# ---------------------------------------------------------------------------
# Build the backbone (+ optional VAE / text encoder) consistent with cfg.
# ---------------------------------------------------------------------------
def build_models(cfg: TrainConfig, device: torch.device):
    """Construct (backbone, vae_or_None, text_encoder_or_None) and fill cfg-derived dims.

    Decides the diffusion-space channels/size from the latent flag, then sizes the
    backbone accordingly. Returns models already moved to ``device``.
    """
    # --- optional VAE: decides what spatial size / channel count diffusion runs on ---
    vae: Optional[VAE] = None
    if cfg.latent:
        vae = VAE(in_channels=3, latent_channels=4).to(device)
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)  # treat the VAE as FROZEN (as in real latent diffusion)
        in_channels = vae.latent_channels                       # 4
        latent_size = cfg.image_size // vae.downsample_factor    # 32 / 4 = 8
    else:
        in_channels = 3
        latent_size = cfg.image_size

    cfg.in_channels = in_channels
    cfg.latent_size = latent_size

    # --- optional text encoder (context) ---
    text_encoder: Optional[TinyTextEncoder] = None
    context_dim: Optional[int] = None
    if cfg.text:
        text_encoder = build_text_encoder(dim=cfg.context_dim).to(device)
        context_dim = cfg.context_dim

    # --- the denoiser backbone ---
    if cfg.backbone == "unet":
        # channel_mult (1,2,4) -> downsample factor 4; latent_size must be divisible by it.
        channel_mult = _safe_channel_mult(latent_size)
        backbone: nn.Module = UNet(
            in_channels=in_channels,
            out_channels=in_channels,
            model_channels=cfg.model_channels,
            channel_mult=channel_mult,
            num_res_blocks=2,
            context_dim=context_dim,
            num_heads=4,
        )
    elif cfg.backbone == "dit":
        patch = 2 if latent_size % 2 == 0 else 1
        backbone = DiT(
            in_channels=in_channels,
            input_size=latent_size,
            patch_size=patch,
            hidden=cfg.dit_hidden,
            depth=cfg.dit_depth,
            heads=6,
            context_dim=context_dim,
        )
    else:
        raise ValueError(f"unknown backbone {cfg.backbone!r} (expected 'unet' or 'dit')")

    backbone = backbone.to(device)
    return backbone, vae, text_encoder


def _safe_channel_mult(latent_size: int) -> Tuple[int, ...]:
    """Pick a U-Net channel_mult whose downsample factor divides ``latent_size``.

    The U-Net downsamples by 2 per extra level. We want every downsample to keep the
    spatial size an integer, so the number of downsamples = log2 of the largest power
    of two dividing latent_size, capped at 2 extra levels (factor 4) for small images.
    """
    levels = 0
    s = latent_size
    while s % 2 == 0 and levels < 2:
        s //= 2
        levels += 1
    # levels extra downsamples -> channel_mult of length (levels + 1).
    return tuple(2 ** i for i in range(levels + 1))  # e.g. (1,2,4), (1,2), or (1,)


# ---------------------------------------------------------------------------
# Turn a batch of labels into the (context, mask) the backbone cross-attends to.
# ---------------------------------------------------------------------------
@torch.no_grad()
def labels_to_context(
    labels: Tensor, text_encoder: TinyTextEncoder
) -> Tuple[Tensor, Tensor]:
    """Map class ids -> captions -> text embeddings (context, mask)."""
    captions = [CLASS_CAPTIONS[int(c) % NUM_CLASSES] for c in labels]
    emb, mask = text_encoder.encode_text(captions)  # [B, L, dim], [B, L]
    return emb, mask


# ---------------------------------------------------------------------------
# The training loop.
# ---------------------------------------------------------------------------
def train(cfg: TrainConfig) -> dict:
    """Run the full trainer and return the checkpoint dict (also saved to cfg.out)."""
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    gen = torch.Generator().manual_seed(cfg.seed)  # CPU generator for the data sampler

    sampler = make_dataset(cfg)
    backbone, vae, text_encoder = build_models(cfg, device)

    schedule = NoiseSchedule(timesteps=cfg.timesteps, kind="cosine").to(device)
    diffusion = GaussianDiffusion(schedule, predict="eps")

    # Only the backbone (and text encoder, if any) train; the VAE is frozen.
    params = list(backbone.parameters())
    if text_encoder is not None:
        params += list(text_encoder.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr)

    backbone.train()
    print(
        f"[train] backbone={cfg.backbone} dataset={cfg.dataset} "
        f"latent={cfg.latent} text={cfg.text} | diffusion shape "
        f"[B,{cfg.in_channels},{cfg.latent_size},{cfg.latent_size}] "
        f"T={cfg.timesteps} device={device}"
    )

    step = 0
    for epoch in range(cfg.epochs):
        for _ in range(cfg.steps_per_epoch):
            images, labels = sampler(cfg.batch, cfg.image_size, gen)
            images = images.to(device)
            labels = labels.to(device)

            # --- to diffusion space (pixels, or VAE latent) ---
            if vae is not None:
                with torch.no_grad():
                    x0 = vae.encode(images).sample()  # [B, 4, 8, 8], already scaled
            else:
                x0 = images                            # [B, 3, 32, 32]

            # --- build the text context and apply CFG condition-dropout ---
            context = mask = None
            if text_encoder is not None:
                context, mask = labels_to_context(labels, text_encoder)
                context, mask = drop_context(
                    context, mask, p=cfg.guidance_dropout, generator=gen
                )

            # --- the diffusion training objective (random t, predict eps, MSE) ---
            loss = diffusion.training_loss(backbone, x0, context=context, mask=mask)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if step % 10 == 0:
                print(f"  epoch {epoch} step {step:4d}  loss {loss.item():.4f}")
            step += 1

    # --- assemble + save the checkpoint -----------------------------------
    ckpt = {
        "config": asdict(cfg),
        "model": backbone.state_dict(),
        "vae": vae.state_dict() if vae is not None else None,
        "text_encoder": text_encoder.state_dict() if text_encoder is not None else None,
        "class_captions": CLASS_CAPTIONS,
    }
    out_dir = os.path.dirname(cfg.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(ckpt, cfg.out)
    print(f"[train] saved checkpoint -> {cfg.out}")
    return ckpt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train the image diffusion pipeline (UNet/DiT, pixels or VAE latents)."
    )
    p.add_argument("--backbone", choices=["unet", "dit"], default="unet")
    p.add_argument("--dataset", choices=["shapes", "mnist"], default="shapes")
    # --latent / --no-latent and --text / --no-text BooleanOptionalAction pairs.
    p.add_argument("--latent", action=argparse.BooleanOptionalAction, default=False,
                   help="diffuse in VAE latent space (default: --no-latent, pixels).")
    p.add_argument("--text", action=argparse.BooleanOptionalAction, default=False,
                   help="condition on text captions (default: --no-text).")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--steps-per-epoch", type=int, default=50, dest="steps_per_epoch")
    p.add_argument("--image-size", type=int, default=32, dest="image_size")
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--guidance-dropout", type=float, default=0.1, dest="guidance_dropout")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out", type=str, default="runs/model.pt")
    return p


def main(argv: Optional[list] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = TrainConfig(
        backbone=args.backbone,
        dataset=args.dataset,
        latent=args.latent,
        text=args.text,
        image_size=args.image_size,
        timesteps=args.timesteps,
        guidance_dropout=args.guidance_dropout,
        epochs=args.epochs,
        batch=args.batch,
        steps_per_epoch=args.steps_per_epoch,
        lr=args.lr,
        seed=args.seed,
        device=_resolve_device(args.device),
        out=args.out,
    )
    train(cfg)


if __name__ == "__main__":
    main()
