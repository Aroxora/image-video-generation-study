"""Toy subpackage: 2-D diffusion you can plot — the same algorithm, stripped bare.

website: /playground

``toy_diffusion_2d`` diffuses 2-D points (moons / spiral / gaussians) using the exact
same NoiseSchedule + GaussianDiffusion the image models use. It is the reference the
live JS Playground mirrors.
"""

from pytorch.toy.toy_diffusion_2d import (
    MLPDenoiser,
    make_target,
    ToyConfig,
    train_toy,
    sample_toy,
    plot_results,
    main,
)

__all__ = [
    "MLPDenoiser",
    "make_target",
    "ToyConfig",
    "train_toy",
    "sample_toy",
    "plot_results",
    "main",
]
