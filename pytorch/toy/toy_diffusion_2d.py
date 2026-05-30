"""Toy 2-D diffusion: the *whole* DDPM idea on points you can plot.

website: /playground  (this is the reference the live JS Playground mirrors)

Everything that makes image diffusion intimidating -- U-Nets, latents, text
encoders, attention -- is stripped away here. We diffuse 2-D points (x, y)
instead of images, so the data, the model, and the result all fit on a scatter
plot. The ALGORITHM, though, is exactly the same one used by Stable Diffusion /
DiT, and it reuses the very same ``NoiseSchedule`` + ``GaussianDiffusion`` code
that the big models use. If you understand this file, you understand the engine.

The loop:

  data x0 in R^2  --FORWARD (fixed math)-->  noisy x_t  --train a net to predict
  the noise eps-->  then SAMPLE by starting from pure noise and walking back.

A point cloud is a 2-D probability distribution. The reverse diffusion process
learns to turn a Gaussian blob (pure noise) back into that distribution. Watch
the output PNG: dots that started as a featureless cloud reassemble into moons /
a spiral / blobs.

Run (tiny, for a smoke test):
    python -m pytorch.toy.toy_diffusion_2d --target moons --epochs 50 --steps 50 --out /tmp/toy.png

Run (a real, still-laptop-fast, fit):
    python -m pytorch.toy.toy_diffusion_2d --target spiral --epochs 3000 --steps 200 --out spiral.png
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

# We reuse the *exact same* diffusion machinery the image models use. The toy
# model is the only thing that's special; the schedule and the training/sampling
# loops are shared. (website: /diffusion)
from pytorch.diffusion.schedule import NoiseSchedule
from pytorch.diffusion.ddpm import GaussianDiffusion
from pytorch.diffusion.cross_attention import SinusoidalPosEmb


# ---------------------------------------------------------------------------
# The denoiser: a tiny MLP. (Replaces the U-Net / DiT for 2-D data.)
# ---------------------------------------------------------------------------
class MLPDenoiser(nn.Module):
    """Predict the noise eps added to a 2-D point, given the point and timestep t.

    This is the laptop-sized stand-in for a U-Net. The contract is identical to
    every other backbone in the repo: it predicts ``eps`` and returns the SAME
    shape as its ``x`` input ([B, 2] here).

    How t enters: we featurize the integer timestep with the SAME
    ``SinusoidalPosEmb`` the big models use (so nearby steps get nearby vectors),
    push it through a small MLP, and concatenate that time-embedding with the
    point coordinates before the main network. Conditioning a denoiser on t is
    *essential* -- the right amount of noise to remove depends entirely on how
    far along the diffusion chain we are.
    """

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.hidden = hidden
        time_dim = hidden  # width of the timestep feature vector

        # Fixed sinusoidal featurization of t -> learned projection (same pattern
        # as TimestepEmbedding, kept inline so the toy file is self-explanatory).
        self.time_sinusoid = SinusoidalPosEmb(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Main network: takes [x (2) ; time_embedding (time_dim)] -> eps (2).
        self.net = nn.Sequential(
            nn.Linear(2 + time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: Tensor, t: Tensor, context=None, mask=None) -> Tensor:
        """x: [B, 2], t: [B] long -> predicted eps: [B, 2].

        ``context`` / ``mask`` are accepted (and ignored) only to match the
        repo-wide backbone signature ``model(x, t, context=None, mask=None)``,
        so ``GaussianDiffusion`` can call this model exactly like a U-Net.
        """
        t_emb = self.time_mlp(self.time_sinusoid(t))  # [B, time_dim]
        h = torch.cat([x, t_emb], dim=-1)             # [B, 2 + time_dim]
        return self.net(h)


# ---------------------------------------------------------------------------
# Target distributions: each returns a cloud of 2-D points to learn.
# ---------------------------------------------------------------------------
def make_target(name: str, n: int) -> Tensor:
    """Return ``[N, 2]`` points sampled from a named 2-D distribution.

    Supported: "moons", "spiral", "gaussians". Points are roughly centered and
    scaled to live in about [-2, 2] x [-2, 2], which sits comfortably in the
    ~[-3, 3] range the sampler clamps to (so the data isn't fighting the clamp).
    """
    rng = np.random.default_rng()  # fresh RNG; seeding is handled in main()

    if name == "moons":
        # Two interleaving half-circles (the classic sklearn "two moons").
        n_out = n // 2
        n_in = n - n_out
        # Outer moon: upper half-circle.
        theta_out = rng.uniform(0, math.pi, n_out)
        outer = np.stack([np.cos(theta_out), np.sin(theta_out)], axis=1)
        # Inner moon: lower half-circle, shifted right and down so they interlock.
        theta_in = rng.uniform(0, math.pi, n_in)
        inner = np.stack([1.0 - np.cos(theta_in), 0.5 - np.sin(theta_in)], axis=1)
        pts = np.concatenate([outer, inner], axis=0)
        pts = pts + rng.normal(0, 0.05, pts.shape)  # a little blur
        # Center and scale to ~[-2, 2].
        pts = (pts - pts.mean(0)) * 1.6

    elif name == "spiral":
        # A single Archimedean spiral arm with noise.
        t = np.sqrt(rng.uniform(0, 1, n)) * 3.0 * math.pi  # denser near center
        r = t / (3.0 * math.pi)                            # radius grows with angle
        pts = np.stack([r * np.cos(t), r * np.sin(t)], axis=1)
        pts = pts + rng.normal(0, 0.02, pts.shape)
        pts = pts * 2.0  # scale to ~[-2, 2]

    elif name == "gaussians":
        # 8 Gaussian blobs arranged on a ring ("8 gaussians", a common GAN/diffusion toy).
        n_modes = 8
        centers = np.stack(
            [
                [math.cos(2 * math.pi * k / n_modes), math.sin(2 * math.pi * k / n_modes)]
                for k in range(n_modes)
            ]
        ) * 1.6
        which = rng.integers(0, n_modes, n)
        pts = centers[which] + rng.normal(0, 0.1, (n, 2))

    else:
        raise ValueError(
            f"unknown target {name!r}; expected one of 'moons', 'spiral', 'gaussians'"
        )

    return torch.from_numpy(pts.astype(np.float32))


# ---------------------------------------------------------------------------
# Config (a dataclass, per repo style) -- thin wrapper over argparse values.
# ---------------------------------------------------------------------------
@dataclass
class ToyConfig:
    target: str = "moons"
    steps: int = 200          # diffusion timesteps T
    epochs: int = 2000        # number of optimizer steps
    batch_size: int = 512
    n_points: int = 4096      # size of the training point cloud
    hidden: int = 128
    lr: float = 2e-3
    schedule: str = "cosine"  # "cosine" or "linear"
    n_samples: int = 2000     # points to draw when sampling for the plot
    seed: int = 0
    out: str = "toy.png"
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Train + sample + plot
# ---------------------------------------------------------------------------
def train_toy(cfg: ToyConfig):
    """Train an ``MLPDenoiser`` to denoise points from ``cfg.target``.

    Returns ``(model, diffusion, data)`` so a caller (or main) can sample/plot.
    This is the canonical DDPM training loop, just on 2-D data:
      - draw a minibatch of real points x0,
      - ``GaussianDiffusion.training_loss`` picks a random t, forward-diffuses to
        x_t, asks the model for eps, and returns the MSE -- the SAME L_simple the
        image models train on.
    """
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Fixed forward-process math, shared with the image models.
    schedule = NoiseSchedule(timesteps=cfg.steps, kind=cfg.schedule).to(device)
    diffusion = GaussianDiffusion(schedule, predict="eps")

    model = MLPDenoiser(hidden=cfg.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    data = make_target(cfg.target, cfg.n_points).to(device)  # [N, 2]

    model.train()
    for epoch in range(cfg.epochs):
        # Random minibatch of real points (sampling with replacement is fine here).
        idx = torch.randint(0, data.shape[0], (cfg.batch_size,), device=device)
        x0 = data[idx]

        loss = diffusion.training_loss(model, x0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if epoch % max(cfg.epochs // 10, 1) == 0 or epoch == cfg.epochs - 1:
            print(f"epoch {epoch:5d}/{cfg.epochs}  loss {loss.item():.4f}")

    return model, diffusion, data


@torch.no_grad()
def sample_toy(model: MLPDenoiser, diffusion: GaussianDiffusion, n: int, device="cpu") -> Tensor:
    """Draw ``n`` points by running the full reverse diffusion (DDPM ancestral).

    Start from pure Gaussian noise of shape [n, 2] and let ``GaussianDiffusion``
    walk it back to the data distribution. This is the same ``sample`` call the
    image models use -- only the ``shape`` differs.
    """
    model.eval()
    samples = diffusion.sample(model, shape=(n, 2), device=device, progress=False)
    return samples


def plot_results(data: Tensor, samples: Tensor, target: str, out: str) -> None:
    """Save a side-by-side scatter of real data vs. generated samples.

    Uses matplotlib's headless 'Agg' backend so it works on a server / in CI with
    no display. (We import + set the backend here, lazily, so importing this
    module never requires matplotlib unless you actually plot.)
    """
    import matplotlib
    matplotlib.use("Agg")  # headless: render to a file, never a window
    import matplotlib.pyplot as plt

    data_np = data.detach().cpu().numpy()
    samp_np = samples.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].scatter(data_np[:, 0], data_np[:, 1], s=4, alpha=0.5, color="#1f77b4")
    axes[0].set_title(f"target data: {target}")
    axes[1].scatter(samp_np[:, 0], samp_np[:, 1], s=4, alpha=0.5, color="#d62728")
    axes[1].set_title("diffusion samples")

    for ax in axes:
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("toy 2-D diffusion (same algorithm as Stable Diffusion / DiT)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Toy 2-D diffusion: train a tiny denoiser and plot samples."
    )
    parser.add_argument(
        "--target", type=str, default="moons",
        choices=["moons", "spiral", "gaussians"],
        help="which 2-D distribution to learn",
    )
    parser.add_argument("--steps", type=int, default=200, help="diffusion timesteps T")
    parser.add_argument("--epochs", type=int, default=2000, help="optimizer steps")
    parser.add_argument("--batch-size", type=int, default=512, dest="batch_size")
    parser.add_argument("--n-points", type=int, default=4096, dest="n_points",
                        help="size of the training point cloud")
    parser.add_argument("--hidden", type=int, default=128, help="MLP width")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--schedule", type=str, default="cosine",
                        choices=["cosine", "linear"])
    parser.add_argument("--n-samples", type=int, default=2000, dest="n_samples",
                        help="points to draw for the plot")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="toy.png", help="output PNG path")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args(argv)

    cfg = ToyConfig(
        target=args.target,
        steps=args.steps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_points=args.n_points,
        hidden=args.hidden,
        lr=args.lr,
        schedule=args.schedule,
        n_samples=args.n_samples,
        seed=args.seed,
        out=args.out,
        device=args.device,
    )

    model, diffusion, data = train_toy(cfg)
    samples = sample_toy(model, diffusion, cfg.n_samples, device=cfg.device)
    plot_results(data, samples, cfg.target, cfg.out)


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
