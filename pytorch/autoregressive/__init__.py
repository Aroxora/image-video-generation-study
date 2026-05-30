"""Autoregressive subpackage: VQ-VAE tokens + a GPT over the shared token stream.

website: /autoregressive

The "next token == next piece of image" family. A VQ-VAE turns an image into a grid
of integer codes; a decoder-only Transformer then models text and image codes as ONE
interleaved sequence  [BOS] <text> [BOI] <image codes> [EOI]  and generates images by
"continuing the sentence". sample.py ties the two together.
"""

from pytorch.autoregressive.vqvae import VectorQuantizer, VQVAE
from pytorch.autoregressive.transformer import (
    CausalSelfAttention,
    Block,
    TokenTransformer,
)
from pytorch.autoregressive.sample import (
    TokenVocab,
    build_prompt_sequence,
    autoregressive_generate,
)

__all__ = [
    "VectorQuantizer",
    "VQVAE",
    "CausalSelfAttention",
    "Block",
    "TokenTransformer",
    "TokenVocab",
    "build_prompt_sequence",
    "autoregressive_generate",
]
