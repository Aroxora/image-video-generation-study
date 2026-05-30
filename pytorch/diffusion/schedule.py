"""Noise schedules and the FORWARD diffusion process.

website: /diffusion (forward & reverse)

This module is the mathematical heart of DDPM (Ho et al. 2020) and the
improved cosine schedule (Nichol & Dhariwal 2021). It is deliberately
*self-contained*: it only depends on ``torch`` and contains NO learnable
parameters. Everything here is fixed math.

The big conceptual split this repo keeps hammering on:

  FORWARD process  q(x_t | x_0):  pure math, no learning. We gradually add
                   Gaussian noise to a clean sample x_0 until, at t = T-1, the
                   sample is (approximately) standard Gaussian noise. Because
                   the noise is Gaussian and the schedule is fixed, we have a
                   closed form: we can jump directly from x_0 to x_t in one step
                   (``q_sample``) without simulating every intermediate step.

  REVERSE process  p_theta(x_{t-1} | x_t):  the LEARNED part. A neural network
                   predicts the noise that was added; that lives in ``ddpm.py``.
                   This file only provides the *targets* and the *posterior*
                   q(x_{t-1} | x_t, x_0) that the learned reverse step matches.

Key identity used everywhere (the "nice property" from Ho et al., eq. 4):

    x_t = sqrt(alphas_cumprod[t]) * x_0  +  sqrt(1 - alphas_cumprod[t]) * eps,
    eps ~ N(0, I).

So given a clean x_0 and a sampled eps, producing x_t is a single affine
combination. Training then asks a network to recover eps from x_t and t.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Beta schedules
# ---------------------------------------------------------------------------
def make_beta_schedule(
    timesteps: int,
    kind: str = "cosine",
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
) -> Tensor:
    """Return the per-step variances ``betas`` of shape ``[T]``.

    The forward process adds noise step by step:

        q(x_t | x_{t-1}) = N(x_t; sqrt(1 - beta_t) x_{t-1}, beta_t I).

    ``beta_t`` controls how much fresh noise is injected at step t. Two choices:

    - "linear" (Ho et al. 2020): betas increase linearly from ``beta_start`` to
      ``beta_end``. Simple, but it destroys information quite fast at high
      resolution -- the last ~20% of steps are nearly pure noise and contribute
      little. Good enough and historically the original DDPM schedule.

    - "cosine" (Nichol & Dhariwal 2021, "Improved DDPM"): we instead define the
      cumulative product ``alphas_cumprod`` (the total signal retained at step t)
      with a cosine curve, then *derive* betas from it. This keeps more signal in
      the middle/late steps and noticeably improves sample quality and
      log-likelihood. This is the default and what modern systems lean toward.

    We clamp betas to a safe (0, 0.999] range for numerical stability.
    """
    if kind == "linear":
        # Plain linear interpolation of betas. (DDPM original.)
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64).float()

    if kind == "cosine":
        # Nichol & Dhariwal define a smooth f(t) and set
        #   alphas_cumprod(t) = f(t) / f(0),   f(t) = cos^2( ((t/T + s) / (1+s)) * pi/2 ).
        # The small offset s prevents beta_t from being too small near t=0.
        # We compute alphas_cumprod at the T+1 grid points, then betas from ratios.
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        f = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = f / f[0]  # normalize so alphas_cumprod[0] == 1
        # beta_t = 1 - alphas_cumprod[t] / alphas_cumprod[t-1]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        # Clamp: the upper bound 0.999 avoids a degenerate (zero-signal) final step.
        return betas.clamp(min=1e-8, max=0.999).float()

    raise ValueError(f"unknown beta schedule kind: {kind!r} (expected 'cosine' or 'linear')")


# ---------------------------------------------------------------------------
# Indexing helper
# ---------------------------------------------------------------------------
def extract(a: Tensor, t: Tensor, x_shape) -> Tensor:
    """Gather ``a[t]`` and reshape so it broadcasts over a tensor of ``x_shape``.

    ``a`` is a 1-D buffer of length T (e.g. ``sqrt_alphas_cumprod``).
    ``t`` is a LongTensor ``[B]`` of timesteps (one per batch element).
    ``x_shape`` is the shape of the data tensor, e.g. ``[B, C, H, W]`` for images
    or ``[B, C, T, H, W]`` for video.

    We return shape ``[B, 1, 1, ...]`` (a leading B, then 1s) so that elementwise
    ops against ``x`` broadcast correctly regardless of how many spatial/temporal
    dims ``x`` has. This is the standard trick that lets one schedule serve images,
    latents, and video without special-casing rank.
    """
    b = t.shape[0]
    # Move the schedule onto the same device as the (integer) timesteps, then gather.
    out = a.to(t.device).gather(0, t)
    # View as [B, 1, 1, ...] with (len(x_shape) - 1) trailing singleton dims.
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# ---------------------------------------------------------------------------
# NoiseSchedule: precomputes every coefficient the forward/reverse math needs
# ---------------------------------------------------------------------------
class NoiseSchedule:
    """Holds all fixed schedule tensors. NOT an ``nn.Module`` (no parameters).

    We precompute, once, every coefficient that ``q_sample``, ``predict_x0``,
    and the posterior need, so the hot training/sampling loops only do indexing
    and arithmetic. All tensors are kept in float32 on CPU by default; call
    ``.to(device)`` to move them.

    Naming follows the DDPM paper / reference implementations so it is easy to
    cross-check against the math:

      betas                          : per-step noise variance, [T]
      alphas              = 1 - betas
      alphas_cumprod      = prod_{s<=t} alpha_s   (total signal retained at t)
      alphas_cumprod_prev = alphas_cumprod shifted right by 1 (value 1.0 at t=0)
      sqrt_alphas_cumprod              : coefficient on x_0 in q_sample
      sqrt_one_minus_alphas_cumprod    : coefficient on eps in q_sample
      sqrt_recip_alphas   = 1/sqrt(alpha_t)       : used in the reverse mean
      posterior_variance               : Var of q(x_{t-1}|x_t,x_0)
    """

    def __init__(self, timesteps: int = 1000, kind: str = "cosine") -> None:
        self.timesteps = int(timesteps)
        self.kind = kind

        # ----- forward-process coefficients (pure math) -----
        betas = make_beta_schedule(self.timesteps, kind=kind)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        # Prepend 1.0 (the "signal" before any noise) and drop the last entry so
        # that alphas_cumprod_prev[t] == alphas_cumprod[t-1].
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        # Coefficients for q_sample: x_t = sqrt(acp) x0 + sqrt(1 - acp) eps.
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # 1/sqrt(alpha_t): used to recompute the reverse-process mean efficiently.
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # For predict_x0_from_eps: x0 = sqrt(1/acp) x_t - sqrt(1/acp - 1) eps.
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1.0)

        # ----- posterior q(x_{t-1} | x_t, x_0) coefficients -----
        # Posterior variance (Ho et al. eq. 7):
        #   beta_tilde_t = beta_t * (1 - acp_{t-1}) / (1 - acp_t)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_variance = posterior_variance
        # Log-variance, clipped: at t=0 the posterior variance is 0, and log(0) is
        # -inf, which breaks downstream code. We clamp to the variance at t=1.
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([posterior_variance[1:2], posterior_variance[1:]], dim=0)
        )
        # Posterior mean coefficients (Ho et al. eq. 7):
        #   mean = coef1 * x_0 + coef2 * x_t
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

        # Track current device for convenience.
        self.device = torch.device("cpu")

    # -- bookkeeping -------------------------------------------------------
    def to(self, device) -> "NoiseSchedule":
        """Move every cached tensor to ``device`` in place and return self."""
        device = torch.device(device)
        for name, val in vars(self).items():
            if isinstance(val, Tensor):
                setattr(self, name, val.to(device))
        self.device = device
        return self

    def __len__(self) -> int:
        return self.timesteps

    # -- FORWARD process (pure math, no learning) --------------------------
    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor | None = None) -> Tensor:
        """Sample x_t ~ q(x_t | x_0): the forward diffusion in ONE step.

        Implements the closed form
            x_t = sqrt(alphas_cumprod[t]) * x0 + sqrt(1 - alphas_cumprod[t]) * eps.

        This is *the* reason diffusion training is cheap: we never simulate the
        T-step Markov chain during training. For a random t we jump straight to
        x_t, ask the model to predict ``eps``, and backprop. No learning happens
        here -- ``noise`` is the regression target, returned implicitly via the
        caller who sampled it.
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_acp = extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_1m_acp = extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_acp * x0 + sqrt_1m_acp * noise

    def predict_x0_from_eps(self, x_t: Tensor, t: Tensor, eps: Tensor) -> Tensor:
        """Invert ``q_sample``: recover the implied clean x_0 from x_t and eps.

            x_0 = sqrt(1/acp_t) * x_t - sqrt(1/acp_t - 1) * eps.

        The reverse sampler uses this: the network predicts eps, we back out the
        predicted x_0, and from x_0 we form the posterior mean for x_{t-1}. This
        "predict eps -> derive x0" indirection is what makes eps-parameterization
        numerically nicer than predicting x_0 directly.
        """
        sqrt_recip = extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1 = extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return sqrt_recip * x_t - sqrt_recipm1 * eps

    def posterior(self, x0: Tensor, x_t: Tensor, t: Tensor):
        """Return (mean, var, log_var_clipped) of q(x_{t-1} | x_t, x_0).

        This is the TRUE reverse step *given the clean x_0* -- still pure math,
        no learning. It is the distribution the learned p_theta(x_{t-1}|x_t) is
        trained to approximate. In sampling we substitute the model's predicted
        x_0 (via ``predict_x0_from_eps``) for the true one we don't have.

            mean = coef1(t) * x_0 + coef2(t) * x_t
            var  = beta_tilde_t      (independent of x; depends only on t)
        """
        mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x0
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(self.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped


# ---------------------------------------------------------------------------
# Self-test (run: python -m pytorch.diffusion.schedule)
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover - manual smoke test
    torch.manual_seed(0)

    for kind in ("cosine", "linear"):
        sched = NoiseSchedule(timesteps=1000, kind=kind)
        assert len(sched) == 1000
        # betas in (0, 1), alphas_cumprod monotonically decreasing toward ~0.
        assert (sched.betas > 0).all() and (sched.betas < 1).all()
        acp = sched.alphas_cumprod
        assert (acp[1:] <= acp[:-1] + 1e-6).all(), "alphas_cumprod should be non-increasing"
        assert acp[0] <= 1.0 + 1e-5 and acp[-1] < 0.1, "schedule should approach noise"
        print(f"[{kind}] betas[0]={sched.betas[0]:.5f} "
              f"acp[0]={acp[0]:.4f} acp[-1]={acp[-1]:.4f}")

    # q_sample a random [2,3,16,16] image at random t; check shapes.
    sched = NoiseSchedule(timesteps=1000, kind="cosine")
    x0 = torch.randn(2, 3, 16, 16)
    t = torch.randint(0, len(sched), (2,))
    noise = torch.randn_like(x0)
    x_t = sched.q_sample(x0, t, noise)
    assert x_t.shape == x0.shape, x_t.shape

    # predict_x0_from_eps must invert q_sample exactly (within float error).
    x0_rec = sched.predict_x0_from_eps(x_t, t, noise)
    err = (x0_rec - x0).abs().max().item()
    assert err < 1e-4, f"x0 reconstruction error too large: {err}"
    print(f"q_sample shape OK {tuple(x_t.shape)}; x0 round-trip max err={err:.2e}")

    # posterior() runs and returns broadcastable shapes.
    mean, var, log_var = sched.posterior(x0, x_t, t)
    assert mean.shape == x0.shape
    assert var.shape == (2, 1, 1, 1) and log_var.shape == (2, 1, 1, 1)
    assert torch.isfinite(log_var).all(), "log-variance must be finite (clipping)"
    print(f"posterior OK: mean{tuple(mean.shape)} var{tuple(var.shape)} "
          f"log_var finite={torch.isfinite(log_var).all().item()}")

    # .to() and video-rank broadcasting sanity (5-D tensor).
    vid = torch.randn(2, 4, 3, 8, 8)
    tv = torch.randint(0, len(sched), (2,))
    xv = sched.q_sample(vid, tv)
    assert xv.shape == vid.shape
    sched.to("cpu")
    print("video-rank q_sample OK", tuple(xv.shape), "-- all schedule self-tests passed")
