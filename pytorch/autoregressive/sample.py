"""Text -> image-token -> pixels: tying the VQ-VAE and the GPT together.

website: /autoregressive  (the full "next-token == next-piece-of-image" pipeline)

--------------------------------------------------------------------------------
THE ONE INTERLEAVED SEQUENCE
--------------------------------------------------------------------------------
This file is the payoff of vqvae.py + transformer.py. Autoregressive image models
generate by *continuing a sentence* whose vocabulary mixes text and image codes:

    [BOS]  <text token ids...>  [BOI]  <image code tokens, row-major>  [EOI]
     |________ the prompt ______|        |____ the VQ-VAE code grid ____|
        (given, conditioning)               (predicted one token at a time)

A SHARED vocabulary stacks the two modalities into one integer range:

    id range                       meaning
    --------                       -------
    [0, n_text)                    text/prompt tokens (here: char-tokenizer ids)
    BOS = n_text + 0               "beginning of sequence"
    BOI = n_text + 1               "beginning of image" (decode starts after this)
    EOI = n_text + 2               "end of image"
    [code_offset, code_offset+K)   the VQ-VAE's K image codes   (code_offset = n_text+3)

Generation = "given everything up to and including [BOI], sample H*W image-code
tokens, then [EOI]." We strip the offset back off, reshape row-major into the HxW
grid the VQ-VAE expects, and `decode_indices` -> pixels. For VIDEO you would just
keep sampling more frames' grids into the SAME stream (next token == next frame).

This `main()` runs the SHAPE-CORRECT loop on RANDOMLY-INITIALIZED weights (no
checkpoint, no download): the produced image will be noise, but every shape and the
text->codes->image plumbing is exactly what a trained model uses.

Tensor conventions: token ids [B, L] long; image code grid [B, H, W] long;
image [B, C, H, W].
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor

from pytorch.autoregressive.transformer import TokenTransformer
from pytorch.autoregressive.vqvae import VQVAE


# ===========================================================================
# Vocabulary layout: where text ids, specials, and image codes live.
# ===========================================================================
@dataclass
class TokenVocab:
    """Bookkeeping for the shared text+image vocabulary (see module docstring).

    Given `n_text` text-token ids and `num_codes` VQ-VAE codes, this lays out the
    special tokens and the image-code block and exposes the total `vocab_size` the
    TokenTransformer must be built with.
    """

    n_text: int          # number of distinct text/prompt token ids
    num_codes: int       # number of VQ-VAE codes (K)

    def __post_init__(self) -> None:
        self.bos = self.n_text + 0          # beginning of sequence
        self.boi = self.n_text + 1          # beginning of image
        self.eoi = self.n_text + 2          # end of image
        self.code_offset = self.n_text + 3  # first image-code id
        self.vocab_size = self.code_offset + self.num_codes

    def code_to_token(self, codes: Tensor) -> Tensor:
        """VQ-VAE code ids [0, K) -> shared-vocab ids by adding the offset."""
        return codes + self.code_offset

    def token_to_code(self, tokens: Tensor) -> Tensor:
        """Shared-vocab image-code ids -> VQ-VAE code ids [0, K) (subtract offset).

        We clamp into the valid code range so that, with random/untrained weights,
        a stray non-code token still maps to a usable index (keeps the demo robust).
        """
        codes = tokens - self.code_offset
        return codes.clamp_(0, self.num_codes - 1)


def build_prompt_sequence(prompt_ids: Tensor, vocab: TokenVocab) -> Tensor:
    """Build the conditioning prefix [BOS] <text...> [BOI] for a batch of prompts.

    prompt_ids : [B, T] long, text-token ids already in [0, n_text).
    returns    : [B, T+2] long, ready to hand to `generate` -- decoding continues
                 right after the trailing [BOI].
    """
    b = prompt_ids.size(0)
    device = prompt_ids.device
    bos = torch.full((b, 1), vocab.bos, dtype=torch.long, device=device)
    boi = torch.full((b, 1), vocab.boi, dtype=torch.long, device=device)
    return torch.cat([bos, prompt_ids, boi], dim=1)


# ===========================================================================
# The generation loop: prompt ids -> image-code grid.
# ===========================================================================
@torch.no_grad()
def autoregressive_generate(
    transformer: TokenTransformer,
    vqvae: VQVAE,
    prompt_ids: Tensor,
    grid_hw: Tuple[int, int],
    vocab: Optional[TokenVocab] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """Generate an image from text by sampling its VQ-VAE code grid token by token.

    website: /autoregressive

    Args:
        transformer : the GPT over the shared text+image vocabulary.
        vqvae       : the (frozen) image<->code translator; we use decode_indices.
        prompt_ids  : [B, T] long text-token ids in [0, n_text).
        grid_hw     : (H, W) of the VQ-VAE code grid to produce (== image_size /
                      vqvae.downsample_factor).
        vocab       : the TokenVocab layout; if None we infer n_text from
                      transformer.vocab_size and vqvae.num_codes.
        temperature, top_k : standard sampling controls forwarded to generate().

    Returns:
        (code_grid [B, H, W] long in [0, K),  image [B, C, H, W]).

    Steps: assemble [BOS]text[BOI], autoregressively sample exactly H*W image-code
    tokens (the model continues the sentence), slice those out, strip the vocab
    offset, reshape row-major into the HxW grid, and decode to pixels.
    """
    transformer.eval()
    vqvae.eval()

    h, w = grid_hw
    n_codes = h * w

    if vocab is None:
        # Recover the layout: vocab_size = n_text + 3 specials + num_codes.
        n_text = transformer.vocab_size - 3 - vqvae.num_codes
        vocab = TokenVocab(n_text=n_text, num_codes=vqvae.num_codes)

    device = next(transformer.parameters()).device
    prompt_ids = prompt_ids.to(device)

    # Prefix: [BOS] <text> [BOI]. Decoding continues right after [BOI].
    seq = build_prompt_sequence(prompt_ids, vocab)        # [B, T+2]
    prefix_len = seq.size(1)

    # Sample exactly H*W image-code tokens (one per code-grid cell, row-major).
    seq = transformer.generate(seq, max_new=n_codes, temperature=temperature, top_k=top_k)

    # Slice out just the generated image tokens, strip the offset -> code ids [0, K).
    image_tokens = seq[:, prefix_len : prefix_len + n_codes]   # [B, H*W]
    codes = vocab.token_to_code(image_tokens)                  # [B, H*W] in [0, K)
    code_grid = codes.view(-1, h, w).contiguous()              # [B, H, W] row-major

    # Codes -> pixels via the VQ-VAE codebook + decoder.
    image = vqvae.decode_indices(code_grid)                    # [B, C, H, W]
    return code_grid, image


# ===========================================================================
# main(): a self-contained, checkpoint-free demonstration.
# ===========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autoregressive text->image-token->pixels demo (random weights)."
    )
    parser.add_argument("--prompt", type=str, default="a small red house",
                        help="text prompt (used only for its token ids in this demo).")
    parser.add_argument("--image-size", type=int, default=32,
                        help="output image H==W in pixels.")
    parser.add_argument("--num-codes", type=int, default=512,
                        help="VQ-VAE codebook size K.")
    parser.add_argument("--dim", type=int, default=256, help="transformer width.")
    parser.add_argument("--depth", type=int, default=6, help="transformer blocks.")
    parser.add_argument("--heads", type=int, default=8, help="attention heads.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # --- 1. the VQ-VAE: fixes the code grid size and codebook K -------------- #
    vqvae = VQVAE(in_channels=3, num_codes=args.num_codes).to(device)
    grid = args.image_size // vqvae.downsample_factor
    grid_hw = (grid, grid)
    print(f"VQ-VAE: image {args.image_size}x{args.image_size} -> code grid "
          f"{grid_hw[0]}x{grid_hw[1]}  (downsample x{vqvae.downsample_factor}, K={vqvae.num_codes})")

    # --- 2. tokenize the prompt (char-level, no downloads) ------------------- #
    # We reuse the diffusion repo's CharTokenizer so text ids are real, not random.
    from pytorch.diffusion.text_encoder import CharTokenizer

    tokenizer = CharTokenizer(max_len=32)
    n_text = tokenizer.vocab_size
    prompt_ids, _ = tokenizer.encode([args.prompt])     # [1, max_len] long in [0, n_text)
    prompt_ids = prompt_ids.to(device)

    # --- 3. lay out the shared vocab and build the GPT to match -------------- #
    vocab = TokenVocab(n_text=n_text, num_codes=vqvae.num_codes)
    transformer = TokenTransformer(
        vocab_size=vocab.vocab_size,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        # max_len must hold [BOS] + text + [BOI] + H*W codes + [EOI] with headroom.
        max_len=prompt_ids.size(1) + 3 + grid * grid + 8,
    ).to(device)
    print(f"Vocab: n_text={n_text}  BOS={vocab.bos} BOI={vocab.boi} EOI={vocab.eoi}  "
          f"code_offset={vocab.code_offset}  vocab_size={vocab.vocab_size}")
    print("Interleaved sequence: [BOS] <text ids> [BOI] <{} image codes> [EOI]".format(grid * grid))

    # --- 4. run the shape-correct text->image-token loop --------------------- #
    code_grid, image = autoregressive_generate(
        transformer, vqvae, prompt_ids, grid_hw,
        vocab=vocab, temperature=args.temperature, top_k=args.top_k,
    )

    print(f"\nProduced code grid {tuple(code_grid.shape)} (values in [0, {vqvae.num_codes})):")
    print(code_grid[0].cpu().numpy())
    print(f"\nDecoded image shape: {tuple(image.shape)}  "
          f"(range [{image.min():.3f}, {image.max():.3f}])")
    print("\nNote: weights are random, so the image is noise -- but the text->codes->"
          "pixels plumbing and every shape match a trained model. (website: /autoregressive)")


if __name__ == "__main__":
    main()
