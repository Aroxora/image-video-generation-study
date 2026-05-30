"""Classifier-free guidance (CFG) — the trick that makes prompts "stronger".

website: /guidance  (run the denoiser twice — once WITH the prompt, once
WITHOUT — and push the prediction toward the difference)
website: /text      (how a text prompt actually steers an image via
cross-attention inside the backbone)

The big idea
------------
A conditional diffusion model learns ``eps_theta(x_t, t, c)`` — given a noisy
image ``x_t`` and a condition ``c`` (e.g. text embeddings), predict the noise.
If we ALSO teach the same network to run with NO condition (the "null"/empty
prompt), then at sampling time we can compute two noise predictions:

    eps_cond   = eps_theta(x_t, t, c)        # "what the prompt wants"
    eps_uncond = eps_theta(x_t, t, null)     # "what any image looks like"

and extrapolate AWAY from the unconditional prediction, toward the conditional
one, by a guidance ``scale`` ``w``:

    eps = eps_uncond + w * (eps_cond - eps_uncond)

- ``w = 0`` -> ignore the prompt entirely (pure unconditional).
- ``w = 1`` -> ordinary conditional sampling (no extra push).
- ``w > 1`` -> *over*-emphasise the prompt: sharper, more on-prompt images at
  the cost of diversity (Stable Diffusion typically uses w in 5..12).

Why this needs no extra classifier: Ho & Salimans (2022) showed the score of an
implicit classifier ``p(c|x_t)`` is exactly ``eps_cond - eps_uncond`` (up to a
constant), so we get classifier guidance *for free* from a single model — hence
"classifier-FREE". That single model is trained to handle both cases by randomly
dropping the condition on a fraction ``p`` of training examples (see
``drop_context`` below).

Shapes (see repo TENSOR CONVENTIONS)
------------------------------------
- x_t / eps : images/latents ``[B, C, H, W]`` (or video ``[B, C, T, H, W]``).
- t         : LongTensor ``[B]`` in ``[0, timesteps)``.
- context   : text/condition tokens ``[B, L, context_dim]`` (float), or ``None``.
- mask      : bool ``[B, L]``, ``True`` = REAL token, ``False`` = padding.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

__all__ = ["drop_context", "classifier_free_guidance", "make_null_context"]


def make_null_context(batch: int, length: int, dim: int, device) -> Tensor:
    """Return the "no prompt" / unconditional condition: zeros ``[B, L, dim]``.

    website: /guidance (the "without the prompt" run)

    A real system usually trains a *learned* null embedding (a single
    parameter vector repeated over the sequence) so the network has a concrete,
    optimisable token to lean on when no prompt is given. Here we use the simple
    educational stand-in: an all-zeros context. Because our text encoder maps
    padding to (near) zero anyway, zeros are a faithful "empty prompt".

    To upgrade to a learned null, store ``self.null = nn.Parameter(torch.randn(dim))``
    in your model and broadcast it to ``[B, L, dim]`` instead of zeros.
    """
    return torch.zeros(batch, length, dim, device=device)


def drop_context(
    context: Optional[Tensor],
    mask: Optional[Tensor],
    p: float,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """Randomly blank out the condition on a fraction ``p`` of the batch rows.

    website: /guidance (this is what makes CFG *possible* — the model must have
    seen empty prompts during training, otherwise the "without the prompt" pass
    is undefined behaviour at sampling time)

    This is "condition dropout": for each example in the batch we flip a coin
    with probability ``p`` and, if it comes up heads, we replace that row's
    context with the null (zeros) condition and mark every token as padding in
    the mask. Training on these blanked rows teaches the *same* weights to also
    produce a sensible *unconditional* ``eps_uncond`` — which CFG then needs.

    Typical ``p`` is ~0.1 (drop 10% of prompts). ``p = 0`` is a no-op;
    ``p = 1`` blanks everything (purely unconditional training).

    Args:
        context: ``[B, L, dim]`` float, or ``None`` (nothing to drop).
        mask: ``[B, L]`` bool (True = real token), or ``None``.
        p: probability in ``[0, 1]`` of dropping each row independently.
        generator: optional ``torch.Generator`` for reproducible coin flips.

    Returns:
        ``(context, mask)`` with the same shapes/dtypes, ``p``-fraction of rows
        zeroed (context) and masked-out (mask). When ``context is None`` the
        inputs are returned unchanged.
    """
    if context is None or p <= 0.0:
        return context, mask

    B = context.shape[0]
    # One Bernoulli draw per batch row. Sample on CPU then move to the context
    # device so a CPU `generator` works regardless of where the tensors live.
    coin = torch.rand(B, generator=generator).to(context.device)
    drop = coin < p  # [B] bool: True where we wipe the prompt

    if not bool(drop.any()):
        return context, mask

    # Broadcast the per-row decision over (L, dim) and zero those rows.
    keep = (~drop).to(context.dtype).view(B, *([1] * (context.dim() - 1)))
    context = context * keep

    if mask is not None:
        # Where we dropped the prompt, ALL tokens become padding (False).
        mask = mask & (~drop).view(B, *([1] * (mask.dim() - 1)))

    return context, mask


@torch.no_grad()
def classifier_free_guidance(
    model,
    x_t: Tensor,
    t: Tensor,
    context: Optional[Tensor],
    uncond_context: Optional[Tensor],
    scale: float,
    mask: Optional[Tensor] = None,
    uncond_mask: Optional[Tensor] = None,
) -> Tensor:
    """Guided noise prediction: ``eps_uncond + scale * (eps_cond - eps_uncond)``.

    website: /guidance (run with & without the prompt, push toward the difference)
    website: /text     (the conditional pass is where cross-attention lets each
    image patch "read" the prompt tokens and pull itself on-prompt)

    We batch the conditional and unconditional passes into a SINGLE forward call
    by stacking them along the batch dimension. This is exactly how production
    samplers (Stable Diffusion, DiT) keep CFG cheap: one doubled-batch forward
    instead of two separate forwards, so cross-attention / convolutions run once.

    Args:
        model: a denoiser with signature
            ``model(x, t, context=None, mask=None) -> eps`` returning the same
            shape as ``x`` (it predicts epsilon).
        x_t: noisy sample ``[B, C, H, W]`` (or ``[B, C, T, H, W]`` for video).
        t: LongTensor ``[B]``.
        context: conditional context ``[B, L, dim]``. If ``None`` we fall back to
            a plain conditional (== unconditional) forward and ``scale`` is moot.
        uncond_context: the "empty prompt" context ``[B, L, dim]``. If ``None``
            we synthesise zeros (``make_null_context``) matching ``context``.
        scale: guidance weight ``w``. ``w == 1`` returns the conditional eps
            unchanged (no guidance), so we short-circuit and do one forward.
        mask / uncond_mask: bool ``[B, L]`` key-padding masks for each pass.

    Returns:
        eps ``[B, C, H, W]`` (same shape as ``x_t``): the guided noise estimate.
    """
    # No condition at all -> there is nothing to guide toward; just denoise.
    if context is None:
        return model(x_t, t, context=None, mask=mask)

    # scale == 1 is ordinary conditional sampling: the (eps_cond - eps_uncond)
    # term is multiplied by 1 and added back to eps_uncond, but the algebra
    # collapses to eps_cond only when scale == 1 *and* we'd otherwise pay for a
    # second forward. Short-circuit to save that compute.
    if scale == 1.0:
        return model(x_t, t, context=context, mask=mask)

    B = x_t.shape[0]
    L = context.shape[1]
    dim = context.shape[2]

    # Build the unconditional ("empty prompt") context if the caller didn't.
    if uncond_context is None:
        uncond_context = make_null_context(B, L, dim, device=context.device)

    # --- Try the cheap path: one doubled-batch forward. ---------------------
    # This works when the conditional and unconditional contexts share the same
    # sequence length L so we can concatenate along the batch axis.
    can_batch = uncond_context.shape[1] == L
    if can_batch:
        x_in = torch.cat([x_t, x_t], dim=0)              # [2B, ...]
        t_in = torch.cat([t, t], dim=0)                  # [2B]
        c_in = torch.cat([uncond_context, context], 0)   # [2B, L, dim]

        if mask is None and uncond_mask is None:
            m_in: Optional[Tensor] = None
        else:
            # A missing mask means "all tokens real" -> all-True.
            m_cond = mask if mask is not None else torch.ones(
                B, L, dtype=torch.bool, device=x_t.device
            )
            m_uncond = uncond_mask if uncond_mask is not None else torch.ones(
                B, uncond_context.shape[1], dtype=torch.bool, device=x_t.device
            )
            m_in = torch.cat([m_uncond, m_cond], dim=0)

        eps = model(x_in, t_in, context=c_in, mask=m_in)
        eps_uncond, eps_cond = eps[:B], eps[B:]
    else:
        # Fallback: differing sequence lengths -> two separate forwards.
        eps_uncond = model(x_t, t, context=uncond_context, mask=uncond_mask)
        eps_cond = model(x_t, t, context=context, mask=mask)

    # The guidance step: extrapolate from "any image" toward "this prompt".
    return eps_uncond + scale * (eps_cond - eps_uncond)
