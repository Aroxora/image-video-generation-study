"""DDPM training loss and samplers (the REVERSE process wrapper).

website: /diffusion (forward & reverse)

``schedule.py`` gave us the fixed FORWARD math. This module wires it to a
*learned* denoiser and implements:

  - the training objective (predict the noise -> simple MSE),
  - ancestral DDPM sampling (the original, stochastic, T-step reverse chain),
  - DDIM sampling (deterministic / few-step, Song et al. 2021),
  - classifier-free guidance (via ``pytorch.diffusion.guidance``).

THE FORWARD vs REVERSE distinction (read this once):

  FORWARD  q(x_t | x_0)   -- pure math, NO parameters. We add Gaussian noise on
           a fixed schedule. We use it only to *make training targets*: pick a
           random t, jump to x_t with ``schedule.q_sample``, and remember the
           noise ``eps`` we added.

  REVERSE  p_theta(x_{t-1} | x_t) -- the LEARNED part. A neural net ``model``
           looks at the noisy x_t and timestep t and predicts the noise
           ``eps_theta``. Training minimizes  || eps - eps_theta ||^2  (Ho et
           al.'s "L_simple"). At sampling time we start from pure noise x_T and
           repeatedly subtract the predicted noise (plus a little fresh noise,
           for DDPM) to walk back to a clean x_0.

The ``model`` is any backbone in this repo (UNet, DiT, VideoDiT, the toy MLP).
Per the repo contract every backbone has the signature
    model(x, t, context=None, mask=None) -> eps   (same shape as x).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from pytorch.diffusion.schedule import NoiseSchedule, extract
from pytorch.diffusion.guidance import classifier_free_guidance


class GaussianDiffusion:
    """Ties a fixed ``NoiseSchedule`` to a learned denoiser.

    ``predict`` selects the parameterization of the network output. We default
    to ``"eps"`` (predict the noise), which is the standard and most stable
    choice and what every backbone in this repo returns. ``"x0"`` (predict the
    clean sample) is supported for completeness/teaching.
    """

    def __init__(self, schedule: NoiseSchedule, predict: str = "eps") -> None:
        if predict not in ("eps", "x0"):
            raise ValueError(f"predict must be 'eps' or 'x0', got {predict!r}")
        self.schedule = schedule
        self.predict = predict

    # ------------------------------------------------------------------
    # Internal: turn a raw model output into (predicted eps, predicted x0).
    # ------------------------------------------------------------------
    def _eps_and_x0(self, model_out: Tensor, x_t: Tensor, t: Tensor):
        """Normalize the network output to both eps and x0, whatever it predicts.

        Keeping both lets the samplers below derive the posterior mean uniformly.
        """
        sched = self.schedule
        if self.predict == "eps":
            eps = model_out
            x0 = sched.predict_x0_from_eps(x_t, t, eps)
        else:  # predict == "x0"
            x0 = model_out
            # Invert q_sample for eps: eps = (sqrt(1/acp) x_t - x0) / sqrt(1/acp - 1)
            sqrt_recip = extract(sched.sqrt_recip_alphas_cumprod, t, x_t.shape)
            sqrt_recipm1 = extract(sched.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
            eps = (sqrt_recip * x_t - x0) / sqrt_recipm1
        return eps, x0

    # ==================================================================
    # TRAINING (uses the FORWARD process to build targets)
    # ==================================================================
    def p_losses(
        self,
        model,
        x0: Tensor,
        t: Tensor,
        context: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """L_simple at a *given* batch of timesteps ``t``.

        Steps (all of this is the forward process feeding a learned reverse):
          1. Sample noise eps ~ N(0, I).
          2. Forward-diffuse: x_t = q_sample(x0, t, eps).   (pure math)
          3. Ask the model to predict eps from (x_t, t, context).  (learned)
          4. Return MSE between the true eps and the predicted eps.

        When ``predict == "x0"`` the regression target is x0 itself instead.
        """
        noise = torch.randn_like(x0)
        x_t = self.schedule.q_sample(x0, t, noise)
        model_out = model(x_t, t, context, mask)
        target = noise if self.predict == "eps" else x0
        return torch.nn.functional.mse_loss(model_out, target)

    def training_loss(
        self,
        model,
        x0: Tensor,
        context: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Sample a random timestep per batch element, then ``p_losses``.

        Each example gets its own uniformly-random t in [0, T). Averaging this
        objective over training is the simple, unweighted variant of the ELBO
        that Ho et al. found works best in practice.
        """
        b = x0.shape[0]
        t = torch.randint(0, len(self.schedule), (b,), device=x0.device, dtype=torch.long)
        return self.p_losses(model, x0, t, context=context, mask=mask)

    # ==================================================================
    # SAMPLING (the LEARNED reverse process)
    # ==================================================================
    def _model_eps(
        self,
        model,
        x_t: Tensor,
        t: Tensor,
        context: Optional[Tensor],
        mask: Optional[Tensor],
        guidance_scale: float,
        uncond_context: Optional[Tensor],
    ) -> Tensor:
        """Predict eps, applying classifier-free guidance when requested.

        Guidance is only meaningful for conditional models. We apply it iff
        ``guidance_scale != 1`` AND an unconditional context is supplied; both
        passes (conditional + unconditional) are delegated to ``guidance.py``.
        """
        if guidance_scale != 1.0 and uncond_context is not None:
            # CFG returns the guided eps directly:
            #   eps = eps_uncond + scale * (eps_cond - eps_uncond)
            return classifier_free_guidance(
                model, x_t, t, context, uncond_context, guidance_scale, mask=mask
            )
        # Plain (conditional or unconditional) forward pass.
        model_out = model(x_t, t, context, mask)
        eps, _ = self._eps_and_x0(model_out, x_t, t)
        return eps

    @torch.no_grad()
    def p_sample(
        self,
        model,
        x_t: Tensor,
        t: Tensor,
        context: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        guidance_scale: float = 1.0,
        uncond_context: Optional[Tensor] = None,
    ) -> Tensor:
        """One ancestral DDPM reverse step: sample x_{t-1} ~ p_theta(x_{t-1}|x_t).

        Recipe (Ho et al. Algorithm 2):
          1. eps_theta = model(x_t, t)        (with optional CFG)
          2. x0_hat    = predict_x0_from_eps  (back out the clean estimate)
          3. (mean, _, log_var) = posterior(x0_hat, x_t, t)  (true reverse given x0)
          4. x_{t-1}   = mean + exp(0.5 log_var) * z,  z ~ N(0,I) -- EXCEPT at
             t == 0 where we return the mean (no noise) so the final sample is clean.
        """
        sched = self.schedule
        eps = self._model_eps(model, x_t, t, context, mask, guidance_scale, uncond_context)
        x0_hat = sched.predict_x0_from_eps(x_t, t, eps)
        # Clamp the predicted x0 to a sane range; latents/images live ~[-1, 1] in
        # this repo, but a slightly looser bound avoids over-restricting early steps.
        x0_hat = x0_hat.clamp(-3.0, 3.0)

        mean, _var, log_var = sched.posterior(x0_hat, x_t, t)
        noise = torch.randn_like(x_t)
        # No noise at the last step (t == 0): mask out the noise term per batch row.
        nonzero = (t != 0).float().reshape(-1, *((1,) * (x_t.dim() - 1)))
        return mean + nonzero * (0.5 * log_var).exp() * noise

    @torch.no_grad()
    def sample(
        self,
        model,
        shape,
        context: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        guidance_scale: float = 1.0,
        uncond_context: Optional[Tensor] = None,
        device="cpu",
        progress: bool = False,
    ) -> Tensor:
        """Full ancestral DDPM sampling loop: start at pure noise, walk to x_0.

        We initialize x_T ~ N(0, I) and apply ``p_sample`` for t = T-1 ... 0.
        This is the original DDPM sampler: stochastic and uses ALL T steps, so
        it is faithful but slow. For fast sampling use ``ddim_sample``.
        """
        device = torch.device(device)
        x_t = torch.randn(shape, device=device)
        timesteps = list(range(len(self.schedule) - 1, -1, -1))

        iterator = timesteps
        if progress:
            try:  # tqdm is optional; never a hard import-time dependency.
                from tqdm.auto import tqdm  # type: ignore

                iterator = tqdm(timesteps, desc="DDPM sampling")
            except Exception:
                iterator = timesteps

        b = shape[0]
        for i in iterator:
            t = torch.full((b,), i, device=device, dtype=torch.long)
            x_t = self.p_sample(
                model, x_t, t,
                context=context, mask=mask,
                guidance_scale=guidance_scale, uncond_context=uncond_context,
            )
        return x_t

    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        shape,
        steps: int = 50,
        eta: float = 0.0,
        context: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        guidance_scale: float = 1.0,
        uncond_context: Optional[Tensor] = None,
        device="cpu",
    ) -> Tensor:
        """DDIM sampling (Song et al. 2021): few-step, optionally deterministic.

        DDIM keeps the SAME trained eps-network but uses a non-Markovian reverse
        update that lets us skip timesteps. We pick a sub-sequence of ``steps``
        timesteps out of T and, at each, use the predicted x0 + predicted eps to
        jump to the next (earlier) timestep:

            x_{t_prev} = sqrt(acp_prev) * x0_hat
                       + sqrt(1 - acp_prev - sigma^2) * eps_theta
                       + sigma * z

        ``eta`` interpolates between deterministic (eta=0, no z term -> an ODE,
        very few steps suffice) and fully stochastic DDPM-like (eta=1). With
        eta=0 you typically get good samples in 20-50 steps instead of 1000.
        """
        sched = self.schedule
        device = torch.device(device)
        b = shape[0]

        T = len(sched)
        steps = min(steps, T)
        # Evenly spaced timesteps from T-1 down to 0, e.g. [999, 979, ..., 0].
        seq = torch.linspace(0, T - 1, steps, dtype=torch.long).flip(0).tolist()

        x_t = torch.randn(shape, device=device)
        acp = sched.alphas_cumprod.to(device)

        for idx, ti in enumerate(seq):
            t = torch.full((b,), ti, device=device, dtype=torch.long)
            eps = self._model_eps(
                model, x_t, t, context, mask, guidance_scale, uncond_context
            )
            x0_hat = sched.predict_x0_from_eps(x_t, t, eps).clamp(-3.0, 3.0)

            a_t = acp[ti]
            # The "previous" (earlier, less-noisy) timestep in our sub-sequence;
            # at the final step we target alphas_cumprod_prev = 1 (clean x0).
            t_prev = seq[idx + 1] if idx + 1 < len(seq) else -1
            a_prev = acp[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)

            # DDIM stochasticity term sigma (eq. 16). eta=0 -> sigma=0 -> ODE.
            sigma = (
                eta
                * torch.sqrt((1 - a_prev) / (1 - a_t))
                * torch.sqrt(1 - a_t / a_prev)
            )
            # Direction pointing to x_t, with the remaining (non-sigma) variance.
            dir_xt = torch.sqrt((1 - a_prev - sigma ** 2).clamp(min=0.0)) * eps
            noise = sigma * torch.randn_like(x_t) if eta > 0 else 0.0
            x_t = torch.sqrt(a_prev) * x0_hat + dir_xt + noise

        return x_t
