"""A tiny text encoder + tokenizer — turns prompts into condition tokens.

website: /text  (a prompt becomes a sequence of embeddings ``[B, L, dim]`` that
the image backbone *cross-attends* to: every image patch gets to "read" the
prompt and pull itself toward the words that describe it)

In real systems the text encoder is a big FROZEN model (CLIP for Stable
Diffusion 1.x/2.x, OpenCLIP for SDXL, T5-XXL for Imagen / DeepFloyd / many DiT
setups). It is NOT trained with the diffusion model — it is pretrained on
text (and image-text) and then its per-token hidden states are fed in as
``context``. See the "swap in real CLIP / T5" block at the bottom of this file.

Here we build a *miniature* trainable stand-in so the whole repo runs on a
laptop with no downloads:

  text str ──CharTokenizer──> ids [B, L] (+ pad mask)
            ──TinyTextEncoder──> emb [B, L, dim]

The contract that matters to the rest of the codebase is the OUTPUT shape:
``[B, L, context_dim]`` plus a bool key-padding ``mask`` ``[B, L]`` where
``True`` marks a REAL token and ``False`` marks padding. Any encoder that
honours that contract (including frozen CLIP/T5) is a drop-in replacement.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["CharTokenizer", "TinyTextEncoder", "build_text_encoder"]


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
class CharTokenizer:
    """Character-level tokenizer with PAD / BOS specials and a fixed ``max_len``.

    website: /text (the first step: chop the prompt into discrete tokens)

    We use *characters* rather than BPE/word-pieces purely for simplicity and
    zero dependencies — the vocabulary is just the printable ASCII range plus
    two special tokens. A production tokenizer (CLIP's BPE, T5's SentencePiece)
    has a much larger vocab and subword merges, but the interface is identical:
    text -> integer ids + a mask telling the model which positions are padding.

    Layout of an encoded sequence (length ``max_len``):

        [BOS] c0 c1 c2 ... ck [PAD] [PAD] ... [PAD]
         ^real ^real .......^real   ^padding -> mask False

    ``encode`` always returns fixed-width ``[B, max_len]`` so a batch of
    different-length prompts stacks into one tensor.
    """

    PAD = "<pad>"
    BOS = "<bos>"

    def __init__(self, max_len: int = 32):
        self.max_len = max_len
        # Special tokens first so their ids are stable (PAD must be id 0 so that
        # an all-zeros / freshly-allocated id tensor reads as "all padding").
        specials = [self.PAD, self.BOS]
        # Printable ASCII 32..126 (space through '~') covers normal prompts.
        chars = [chr(c) for c in range(32, 127)]
        self._itos: List[str] = specials + chars
        self._stoi = {ch: i for i, ch in enumerate(self._itos)}
        self.pad_id = self._stoi[self.PAD]
        self.bos_id = self._stoi[self.BOS]

    @property
    def vocab_size(self) -> int:
        return len(self._itos)

    def encode(self, texts: List[str]) -> Tuple[Tensor, Tensor]:
        """Encode a list of strings -> ``(ids [B, L] long, mask [B, L] bool)``.

        - Prepends BOS, truncates to ``max_len``, right-pads with PAD.
        - ``mask[b, i] == True`` iff position ``i`` is a REAL token (BOS or a
          character), ``False`` for padding — matching the repo convention
          (True = REAL token).
        """
        B = len(texts)
        L = self.max_len
        ids = torch.full((B, L), self.pad_id, dtype=torch.long)
        mask = torch.zeros((B, L), dtype=torch.bool)

        for b, text in enumerate(texts):
            # BOS marks "start of prompt"; the model can use it as a summary slot.
            toks = [self.bos_id]
            for ch in text:
                # Unknown chars (outside printable ASCII) fall back to PAD-as-unk;
                # they simply contribute nothing, which is fine for a teaching demo.
                toks.append(self._stoi.get(ch, self.pad_id))
            toks = toks[:L]  # truncate over-long prompts to the window
            n = len(toks)
            ids[b, :n] = torch.tensor(toks, dtype=torch.long)
            mask[b, :n] = True
        return ids, mask

    def decode(self, ids: Tensor) -> List[str]:
        """Inverse of ``encode`` (handy for debugging); drops specials."""
        out: List[str] = []
        rows = ids.tolist() if ids.dim() == 2 else [ids.tolist()]
        for row in rows:
            chars = [
                self._itos[i]
                for i in row
                if i not in (self.pad_id, self.bos_id)
            ]
            out.append("".join(chars))
        return out


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class _Block(nn.Module):
    """One pre-norm Transformer encoder block (bidirectional self-attention).

    Bidirectional (NOT causal): a text encoder may look at the whole prompt at
    once — "red" should see "cube" and vice-versa. This is the standard
    encoder-style attention used by CLIP's text tower and T5's encoder.
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        # nn.MultiheadAttention expects key_padding_mask where True = IGNORE.
        # Our repo mask is True = REAL token, so we invert it at the boundary.
        h = self.norm1(x)
        attn_out, _ = self.attn(
            h, h, h,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class TinyTextEncoder(nn.Module):
    """Embedding + sinusoidal positions + a few Transformer blocks -> ``[B, L, dim]``.

    website: /text (these per-token vectors are the ``context`` the image
    backbone cross-attends to)

    The output is one ``dim``-vector PER token (not a single pooled vector), so
    cross-attention can attend to "red" and "cube" separately. ``dim`` MUST match
    the backbone's ``context_dim`` (default 256 throughout this repo).
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        depth: int = 2,
        heads: int = 4,
        max_len: int = 32,
    ):
        super().__init__()
        self.dim = dim
        self.max_len = max_len

        self.token_emb = nn.Embedding(vocab_size, dim)
        # Learned absolute positions (CLIP-style). Simple and adequate at L<=32.
        self.pos_emb = nn.Parameter(self._sinusoidal_init(max_len, dim))

        self.blocks = nn.ModuleList(
            [_Block(dim, heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

        # The tokenizer is attached here so `encode_text` is fully self-contained.
        self.tokenizer = CharTokenizer(max_len=max_len)

    @staticmethod
    def _sinusoidal_init(length: int, dim: int) -> Tensor:
        """Classic sinusoidal table; used to *initialise* the learned positions
        so even an untrained encoder has sensible position structure (helps the
        no-checkpoint smoke tests look reasonable)."""
        pos = torch.arange(length).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe = torch.zeros(length, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        return pe

    def forward(self, ids: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """ids ``[B, L]`` (long) -> embeddings ``[B, L, dim]`` (float).

        ``mask`` ``[B, L]`` bool (True = REAL token) is used as a key-padding
        mask so attention ignores PAD positions. Padded output rows are zeroed
        so a downstream all-zeros context reads cleanly as "empty prompt".
        """
        B, L = ids.shape
        x = self.token_emb(ids) + self.pos_emb[:L].unsqueeze(0)

        # nn.MultiheadAttention wants True = IGNORE; invert our True = REAL mask.
        kpm = (~mask) if mask is not None else None
        for block in self.blocks:
            x = block(x, key_padding_mask=kpm)
        x = self.norm(x)

        if mask is not None:
            # Hard-zero padded positions: keeps the contract that padding carries
            # no signal (and matches make_null_context's all-zeros null prompt).
            x = x * mask.unsqueeze(-1).to(x.dtype)
        return x

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> Tuple[Tensor, Tensor]:
        """Convenience: raw strings -> ``(emb [B, L, dim], mask [B, L] bool)``.

        Runs the internal tokenizer then the encoder, on the module's own device.
        This is the method samplers / training scripts call to turn a prompt list
        into ``context``.
        """
        device = self.pos_emb.device
        ids, mask = self.tokenizer.encode(texts)
        ids, mask = ids.to(device), mask.to(device)
        emb = self.forward(ids, mask)
        return emb, mask


def build_text_encoder(dim: int = 256) -> TinyTextEncoder:
    """Construct a ready-to-use ``TinyTextEncoder`` (tokenizer included).

    website: /text

    ``dim`` is the context dimension that must match the diffusion backbone's
    ``context_dim`` argument (UNet / DiT / VideoDiT). Default 256 across the repo.
    """
    tokenizer = CharTokenizer(max_len=32)
    encoder = TinyTextEncoder(
        vocab_size=tokenizer.vocab_size,
        dim=dim,
        depth=2,
        heads=4,
        max_len=tokenizer.max_len,
    )
    return encoder


# ===========================================================================
# Swapping in a REAL frozen text encoder (CLIP / T5)
# ===========================================================================
# The diffusion code only depends on the OUTPUT contract:
#     emb : float [B, L, context_dim]
#     mask: bool  [B, L]   (True = real token, False = pad)
# Any encoder that returns that pair is a drop-in replacement. Two recipes:
#
# --- (A) Frozen CLIP via open_clip (this is Stable Diffusion 1.x/2.x style) ---
#
#   import open_clip, torch
#   model, _, _ = open_clip.create_model_and_transforms(
#       "ViT-B-32", pretrained="laion2b_s34b_b79k")
#   tok = open_clip.get_tokenizer("ViT-B-32")
#   model = model.eval().requires_grad_(False)          # FROZEN
#
#   @torch.no_grad()
#   def encode_text_clip(prompts):
#       ids = tok(prompts)                              # [B, 77] long
#       x = model.token_embedding(ids)                  # [B, 77, D]
#       x = x + model.positional_embedding
#       x = model.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
#       x = model.ln_final(x)                           # per-token hidden states
#       mask = ids != 0                                 # True = real token
#       return x, mask                                  # [B, 77, D], [B, 77]
#   # NOTE: SD uses the per-token hidden states (above), NOT the pooled vector.
#   # Set the backbone's context_dim == D (512 for ViT-B/32, 768 for ViT-L/14).
#
# --- (B) Frozen T5 encoder via transformers (Imagen / DeepFloyd / many DiTs) ---
#
#   from transformers import T5Tokenizer, T5EncoderModel
#   import torch
#   tok = T5Tokenizer.from_pretrained("t5-base")
#   enc = T5EncoderModel.from_pretrained("t5-base").eval().requires_grad_(False)
#
#   @torch.no_grad()
#   def encode_text_t5(prompts, max_len=77):
#       batch = tok(prompts, padding="max_length", truncation=True,
#                   max_length=max_len, return_tensors="pt")
#       out = enc(input_ids=batch.input_ids,
#                 attention_mask=batch.attention_mask).last_hidden_state
#       mask = batch.attention_mask.bool()              # True = real token
#       return out, mask                                # [B, L, 768], [B, L]
#
# In both cases keep the encoder FROZEN (requires_grad_(False)) and only train
# the diffusion backbone + its cross-attention layers. Match context_dim to the
# encoder's hidden size.
