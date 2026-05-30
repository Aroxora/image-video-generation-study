"""Space-time patchification: turn a video clip into a flat sequence of tokens.

website: /video  (how a video model sees a clip as one big sequence of patches)

A picture/DiT model chops an image [B,C,H,W] into PxP patches and treats them as a
sequence of tokens. A video model does the SAME thing, but the patch is a little
*brick* of pixels spanning (pt frames) x (ph rows) x (pw cols). Flattening all those
bricks gives ONE sequence that mixes space AND time:

    video [B, C, T, H, W]  ->  tokens [B, N, C*pt*ph*pw]   with N = (T/pt)*(H/ph)*(W/pw)

WHY do it this way (and why it matters): if the whole clip becomes one token sequence,
attention can relate a patch in frame 0 to a patch in frame 7. The model processes the
WHOLE CLIP AT ONCE -> temporal coherence (no per-frame flicker, motion stays consistent).
This is the Sora/DiT-for-video idea: "spacetime patches" are the unit of computation.

Conventions (repo-wide):
  video  [B, C, T, H, W];  tokens [B, N, D];  grid = (gt, gh, gw) patch counts per axis.
  We require T % pt == 0, H % ph == 0, W % pw == 0 (pad upstream if needed).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Pure tensor reshape patchify / unpatchify (no learned weights -> exact inverse)
# ---------------------------------------------------------------------------
def patchify_video(
    x: Tensor, pt: int, ph: int, pw: int
) -> Tuple[Tensor, Tuple[int, int, int]]:
    """Flatten a video into space-time patch tokens.

    x : [B, C, T, H, W]
    returns:
        tokens : [B, N, C*pt*ph*pw]   N = (T//pt)*(H//ph)*(W//pw)
        grid   : (gt, gh, gw)         patch counts along time/height/width

    The token ordering is row-major over (gt, gh, gw): time outermost, then height,
    then width. unpatchify_video reverses exactly this, so the round-trip is lossless.

    Implementation is a pure `reshape`/`permute` (no convolution), which guarantees an
    EXACT inverse -- handy for sanity checks and for the toy/educational setting. Real
    systems usually fold this into a strided conv (see SpacetimePatchEmbed below).
    """
    b, c, t, h, w = x.shape
    if t % pt or h % ph or w % pw:
        raise ValueError(
            f"video dims (T={t},H={h},W={w}) must be divisible by patch (pt={pt},ph={ph},pw={pw})"
        )
    gt, gh, gw = t // pt, h // ph, w // pw

    # Split each axis into (grid, patch): [B, C, gt, pt, gh, ph, gw, pw]
    x = x.reshape(b, c, gt, pt, gh, ph, gw, pw)
    # Reorder so all the grid axes come first (token index) and all the per-patch
    # content axes (C, pt, ph, pw) come last (token feature vector):
    #   [B, gt, gh, gw, C, pt, ph, pw]
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    # Merge grid -> N, and content -> D = C*pt*ph*pw.
    tokens = x.reshape(b, gt * gh * gw, c * pt * ph * pw)
    return tokens, (gt, gh, gw)


def unpatchify_video(
    tokens: Tensor,
    pt: int,
    ph: int,
    pw: int,
    grid: Tuple[int, int, int],
    C: int,
) -> Tensor:
    """Inverse of patchify_video. tokens [B,N,C*pt*ph*pw] -> video [B,C,T,H,W].

    `grid` and `C` tell us how to fold the flat token sequence back into a clip.
    """
    b, n, d = tokens.shape
    gt, gh, gw = grid
    if n != gt * gh * gw:
        raise ValueError(f"token count {n} != grid product {gt*gh*gw}")
    if d != C * pt * ph * pw:
        raise ValueError(f"token dim {d} != C*pt*ph*pw = {C * pt * ph * pw}")

    # [B, gt, gh, gw, C, pt, ph, pw]
    x = tokens.reshape(b, gt, gh, gw, C, pt, ph, pw)
    # back to [B, C, gt, pt, gh, ph, gw, pw]
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
    # merge grid*patch on each axis -> [B, C, T, H, W]
    x = x.reshape(b, C, gt * pt, gh * ph, gw * pw)
    return x


# ---------------------------------------------------------------------------
# Learned patch embedding (the version a real model uses)
# ---------------------------------------------------------------------------
class SpacetimePatchEmbed(nn.Module):
    """Embed space-time patches into model width via a single strided 3D conv.

    A Conv3d with kernel == stride == patch size is *exactly* a per-patch linear
    projection: each output position sees one non-overlapping brick of input and
    maps it to a `hidden`-dim token. This is the standard ViT/DiT "patch embed",
    extended to time. We then flatten the (T',H',W') output grid into a sequence.

        x [B, C, T, H, W]  --Conv3d(patch)-->  [B, hidden, gt, gh, gw]
                           --flatten-->         tokens [B, N, hidden],  grid=(gt,gh,gw)

    Unlike the pure reshape above, this is learnable (and not exactly invertible),
    which is what you want when feeding a transformer.
    """

    def __init__(self, in_channels: int, hidden: int, patch: Tuple[int, int, int] = (1, 2, 2)):
        super().__init__()
        self.in_channels = in_channels
        self.hidden = hidden
        self.patch = patch  # (pt, ph, pw)
        # kernel == stride == patch -> non-overlapping bricks, one token each.
        self.proj = nn.Conv3d(in_channels, hidden, kernel_size=patch, stride=patch)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int, int]]:
        b, c, t, h, w = x.shape
        pt, ph, pw = self.patch
        if t % pt or h % ph or w % pw:
            raise ValueError(
                f"video dims (T={t},H={h},W={w}) must be divisible by patch {self.patch}"
            )
        x = self.proj(x)  # [B, hidden, gt, gh, gw]
        gt, gh, gw = x.shape[2], x.shape[3], x.shape[4]
        # [B, hidden, gt, gh, gw] -> [B, N, hidden] (row-major over gt,gh,gw)
        tokens = x.flatten(2).transpose(1, 2)
        return tokens, (gt, gh, gw)
