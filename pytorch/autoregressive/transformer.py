"""A GPT-style decoder-only Transformer over a SINGLE stream of discrete tokens.

website: /autoregressive  (the "next-token == next-piece-of-image" family)

--------------------------------------------------------------------------------
THE BIG IDEA: text and image tokens live in ONE sequence
--------------------------------------------------------------------------------
The autoregressive image/video models (DALL-E 1, Parti, VideoPoet, Sora-adjacent
token models) do NOT have separate "text branch" and "image branch". They flatten
*everything* into one sequence of integer tokens and run a plain GPT over it,
predicting the next token given all previous ones. The magic is entirely in the
LAYOUT of that one sequence and in a shared vocabulary:

    vocab = [ text tokens ... | special tokens | image-code tokens ... ]
              (e.g. chars)      (BOS, BOI, EOI)   (the VQ-VAE's K codes)

A single training/inference example is the interleaved sequence:

    [BOS]  t0 t1 t2 ... tk   [BOI]  c0 c1 c2 ............ c_{HW-1}  [EOI]
     ^start  ^the text prompt  ^"image    ^the VQ-VAE code grid,        ^"image
             (conditioning)     begins"    flattened ROW-MAJOR           ends"

Because attention is CAUSAL (each position sees only earlier positions), predicting
c0 conditions on the whole prompt; predicting c1 conditions on the prompt AND c0;
and so on. "Generate an image from text" is then literally "continue the sentence
after [BOI]" -- the same `generate()` loop GPT uses for text. For VIDEO you just keep
appending more frames' code grids to the same stream, which is why this family is
described as "next token == next patch == next frame": one sequence, one objective.

This module is the generic engine. It does NOT know about text vs codes -- it just
sees integer ids in `[0, vocab_size)`. The *caller* (sample.py) builds the
[BOS]text[BOI]codes[EOI] layout and chooses the vocab offsets. Keeping the
Transformer vocab-agnostic is exactly how the real systems share one GPT across
modalities.

Tensor conventions: token ids `idx` are LongTensor [B, L]; logits are [B, L, vocab].
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["CausalSelfAttention", "Block", "TokenTransformer"]


# ---------------------------------------------------------------------------
# Causal (masked) multi-head self-attention.
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where each position may only attend to the PAST.

    The causal mask (lower-triangular) is what makes the model autoregressive:
    the prediction at position i is a function of tokens 0..i only, so we can train
    on the whole sequence in parallel yet still sample one token at a time.

    We use PyTorch's fused `scaled_dot_product_attention` with `is_causal=True`,
    which builds the triangular mask internally (correct on CPU, flash on GPU).
    """

    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} must be divisible by heads {heads}"
        self.heads = heads
        self.dim_head = dim // heads
        # One projection for Q, K, V together (the standard GPT fusion), then output.
        self.to_qkv = nn.Linear(dim, 3 * dim)
        self.to_out = nn.Linear(dim, dim)
        self.attn_dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:  # [B, L, D] -> [B, L, D]
        b, l, d = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)  # each [B, L, D]
        # [B, L, D] -> [B, heads, L, dim_head]
        q = q.view(b, l, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(b, l, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(b, l, self.heads, self.dim_head).transpose(1, 2)

        # is_causal=True applies the lower-triangular mask: position i sees 0..i.
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )  # [B, heads, L, dim_head]

        out = out.transpose(1, 2).reshape(b, l, d)
        return self.resid_dropout(self.to_out(out))


class MLP(nn.Module):
    """The position-wise feed-forward: Linear -> GELU -> Linear (4x inner width)."""

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, mult * dim)
        self.fc2 = nn.Linear(mult * dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    """A pre-norm GPT block: x = x + attn(LN(x)); x = x + mlp(LN(x)).

    Pre-norm (LayerNorm *before* each sublayer, residual *around* it) is the modern
    GPT-2+/standard recipe -- it trains far more stably than the original post-norm.
    """

    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, heads=heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# The full decoder-only Transformer (a small GPT).
# ---------------------------------------------------------------------------
class TokenTransformer(nn.Module):
    """A small GPT over one interleaved [BOS]text[BOI]codes[EOI] token stream.

    website: /autoregressive

    forward(idx [B, L]) -> logits [B, L, vocab_size]
        logits[:, i, :] is the model's distribution over the token at position i+1
        (next-token prediction). Training minimizes cross-entropy of logits[:, :-1]
        vs idx[:, 1:].

    generate(idx, max_new, ...) -> idx extended by `max_new` sampled tokens
        the same loop text GPTs use; sample.py wraps it to emit an image-code grid.

    The vocabulary is SHARED across modalities: ids below the image-code offset are
    text/special tokens, ids at/above it are VQ-VAE codes. The Transformer itself is
    modality-agnostic -- see sample.py for how the offsets are assigned.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        depth: int = 6,
        heads: int = 8,
        max_len: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.dim = dim

        # Token embedding (shared text+image vocab) and LEARNED absolute position
        # embeddings. Learned positions are GPT-2's choice and are plenty for the
        # short fixed-length sequences here.
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [Block(dim, heads=heads, dropout=dropout) for _ in range(depth)]
        )
        self.ln_f = nn.LayerNorm(dim)
        # The output head maps each position's features to a distribution over the vocab.
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # Weight tying (GPT-2 / "Using the Output Embedding..."): share the input
        # embedding table with the output projection -> fewer params, better fit.
        self.head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: Tensor) -> Tensor:  # [B, L] long -> [B, L, vocab]
        b, l = idx.shape
        assert l <= self.max_len, f"sequence length {l} exceeds max_len {self.max_len}"

        positions = torch.arange(l, device=idx.device)          # [L]
        x = self.token_emb(idx) + self.pos_emb(positions)[None]  # [B, L, D]
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)                                      # [B, L, vocab]

    @torch.no_grad()
    def generate(
        self,
        idx: Tensor,
        max_new: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Tensor:
        """Autoregressively extend `idx` by `max_new` tokens (greedy/sampled).

        website: /autoregressive (this loop is the heart of the family)

        At each step: run the model on the current sequence, take the LAST position's
        logits (the prediction for the next token), optionally temperature-scale and
        top-k filter, sample, and append. Because attention is causal, re-running on
        the growing sequence is correct (just not KV-cached -- we favour clarity).
        """
        self.eval()
        for _ in range(max_new):
            # Crop to the last max_len tokens so positions stay in range.
            idx_cond = idx if idx.size(1) <= self.max_len else idx[:, -self.max_len :]
            logits = self(idx_cond)[:, -1, :]  # [B, vocab] -- next-token logits

            if temperature != 1.0:
                logits = logits / max(temperature, 1e-8)

            if top_k is not None:
                # Keep only the top_k logits; set the rest to -inf so they can't be drawn.
                k = min(top_k, logits.size(-1))
                kth = torch.topk(logits, k, dim=-1).values[:, -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))

            probs = F.softmax(logits, dim=-1)              # [B, vocab]
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
            idx = torch.cat([idx, next_token], dim=1)
        return idx


# ----------------------------------------------------------------------------- #
# Tiny self-test / demo: run `python -m pytorch.autoregressive.transformer`.
# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)
    model = TokenTransformer(vocab_size=600, dim=128, depth=4, heads=4, max_len=64)

    idx = torch.randint(0, 600, (2, 20))
    logits = model(idx)
    print("logits shape:", tuple(logits.shape))   # expect (2, 20, 600)
    assert tuple(logits.shape) == (2, 20, 600)

    out = model.generate(idx, max_new=8, temperature=1.0, top_k=50)
    print("generated shape:", tuple(out.shape))    # expect (2, 28)
    assert out.shape == (2, 28)
    assert out.max() < 600 and out.min() >= 0
    print("OK")
