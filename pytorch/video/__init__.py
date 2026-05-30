"""Video subpackage: space-time patchification, factorized attention, VideoDiT.

website: /video

A video DiT is the image DiT extended to a clip: chop the clip into space-time
patches, run a transformer that attends within frames (spatial) AND across frames
(temporal), and predict the noise for the WHOLE clip at once -> temporal coherence.
"""

from pytorch.video.spacetime import (
    patchify_video,
    unpatchify_video,
    SpacetimePatchEmbed,
)
from pytorch.video.temporal_attention import (
    Attention,
    TemporalAttention,
    FactorizedSpacetimeBlock,
)
from pytorch.video.video_dit import VideoDiT

__all__ = [
    "patchify_video",
    "unpatchify_video",
    "SpacetimePatchEmbed",
    "Attention",
    "TemporalAttention",
    "FactorizedSpacetimeBlock",
    "VideoDiT",
]
