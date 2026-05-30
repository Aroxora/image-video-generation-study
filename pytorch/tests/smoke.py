"""One script that imports and exercises EVERY module with tiny tensors.

Run from the repo root::

    python -m pytorch.tests.smoke

This is the fastest "is the whole repo wired up?" check. It does forward/backward on
miniature inputs (no training, no downloads, CPU) and asserts shapes/dtypes for:

  * schedule  : q_sample + x0 round-trip
  * UNet & DiT: forward with AND without text context
  * VAE       : encode -> decode round-trip (shapes)
  * text      : TinyTextEncoder.encode_text
  * diffusion : GaussianDiffusion.training_loss + a 5-step DDIM sample
  * guidance  : classifier_free_guidance (doubled-batch CFG)
  * video     : patchify/unpatchify round-trip + VideoDiT forward (incl. cond_frames)
  * autoreg   : VQVAE round-trip + TokenTransformer.generate + autoregressive_generate

Prints "SMOKE OK" on success; raises (non-zero exit) on the first failure.
"""

from __future__ import annotations

import torch

# Import via the package __init__ re-exports to also test those.
from pytorch.diffusion import (
    NoiseSchedule,
    GaussianDiffusion,
    UNet,
    DiT,
    VAE,
    build_text_encoder,
    classifier_free_guidance,
    make_null_context,
    drop_context,
)
from pytorch.video import (
    patchify_video,
    unpatchify_video,
    SpacetimePatchEmbed,
    FactorizedSpacetimeBlock,
    VideoDiT,
)
from pytorch.autoregressive import (
    VQVAE,
    TokenTransformer,
    TokenVocab,
    autoregressive_generate,
    build_prompt_sequence,
)
from pytorch.autoregressive.sample import autoregressive_generate as ar_gen  # alias check
from pytorch.toy import MLPDenoiser, make_target


def _ok(name: str) -> None:
    print(f"  [ok] {name}")


def test_schedule() -> None:
    sched = NoiseSchedule(timesteps=50, kind="cosine")
    x0 = torch.randn(2, 3, 8, 8)
    t = torch.randint(0, len(sched), (2,))
    noise = torch.randn_like(x0)
    x_t = sched.q_sample(x0, t, noise)
    assert x_t.shape == x0.shape
    x0_rec = sched.predict_x0_from_eps(x_t, t, noise)
    assert (x0_rec - x0).abs().max().item() < 1e-3
    mean, var, logvar = sched.posterior(x0, x_t, t)
    assert mean.shape == x0.shape and torch.isfinite(logvar).all()
    _ok("schedule q_sample + x0 round-trip + posterior")


def test_unet() -> None:
    # Unconditional U-Net.
    net = UNet(in_channels=3, out_channels=3, model_channels=16, channel_mult=(1, 2),
               num_res_blocks=1, context_dim=None, num_heads=2)
    x = torch.randn(2, 3, 16, 16)
    t = torch.randint(0, 50, (2,))
    eps = net(x, t)
    assert eps.shape == x.shape
    # Conditional U-Net (cross-attention to text context).
    cnet = UNet(in_channels=3, out_channels=3, model_channels=16, channel_mult=(1, 2),
                num_res_blocks=1, context_dim=32, num_heads=2)
    ctx = torch.randn(2, 5, 32)
    mask = torch.ones(2, 5, dtype=torch.bool)
    eps_c = cnet(x, t, context=ctx, mask=mask)
    assert eps_c.shape == x.shape
    _ok("UNet forward (uncond + conditional)")


def test_dit() -> None:
    net = DiT(in_channels=3, input_size=16, patch_size=2, hidden=48, depth=2, heads=4,
              context_dim=None)
    x = torch.randn(2, 3, 16, 16)
    t = torch.randint(0, 50, (2,))
    assert net(x, t).shape == x.shape
    cnet = DiT(in_channels=3, input_size=16, patch_size=2, hidden=48, depth=2, heads=4,
               context_dim=32)
    ctx = torch.randn(2, 5, 32)
    mask = torch.ones(2, 5, dtype=torch.bool)
    assert cnet(x, t, context=ctx, mask=mask).shape == x.shape
    _ok("DiT forward (uncond + conditional)")


def test_vae() -> None:
    vae = VAE(in_channels=3, latent_channels=4, base_channels=16, ch_mult=(1, 2, 4))
    x = torch.randn(2, 3, 32, 32)
    posterior = vae.encode(x)
    z = posterior.sample()
    assert z.shape == (2, 4, 8, 8), z.shape          # downsample factor 4
    x_rec = vae.decode(z)
    assert x_rec.shape == x.shape
    assert posterior.kl().dim() == 0
    _ok("VAE encode/decode round-trip + kl")


def test_text_encoder() -> None:
    enc = build_text_encoder(dim=32)
    emb, mask = enc.encode_text(["a red circle", "blue square"])
    assert emb.shape[0] == 2 and emb.shape[2] == 32
    assert mask.dtype == torch.bool and mask.shape == emb.shape[:2]
    _ok("TinyTextEncoder.encode_text")


def test_diffusion_loss_and_sample() -> None:
    sched = NoiseSchedule(timesteps=50, kind="cosine")
    diffusion = GaussianDiffusion(sched, predict="eps")
    net = UNet(in_channels=3, out_channels=3, model_channels=16, channel_mult=(1, 2),
               num_res_blocks=1, context_dim=None, num_heads=2)
    x0 = torch.randn(2, 3, 16, 16)
    loss = diffusion.training_loss(net, x0)
    assert loss.dim() == 0 and torch.isfinite(loss)
    loss.backward()  # exercise the backward pass too
    # 5-step DDIM sample.
    out = diffusion.ddim_sample(net, (2, 3, 16, 16), steps=5, device="cpu")
    assert out.shape == (2, 3, 16, 16)
    _ok("GaussianDiffusion training_loss + 5-step DDIM sample")


def test_guidance() -> None:
    cnet = DiT(in_channels=3, input_size=16, patch_size=2, hidden=48, depth=2, heads=4,
               context_dim=32)
    x = torch.randn(2, 3, 16, 16)
    t = torch.randint(0, 50, (2,))
    ctx = torch.randn(2, 5, 32)
    mask = torch.ones(2, 5, dtype=torch.bool)
    uncond = make_null_context(2, 5, 32, device=x.device)
    eps = classifier_free_guidance(cnet, x, t, ctx, uncond, scale=4.0, mask=mask)
    assert eps.shape == x.shape
    # drop_context sanity: dropping with p=1 blanks all rows.
    dctx, dmask = drop_context(ctx.clone(), mask.clone(), p=1.0)
    assert dctx.abs().sum().item() == 0.0 and (~dmask).all()
    _ok("classifier_free_guidance + drop_context")


def test_video() -> None:
    # patchify/unpatchify is a pure reshape -> EXACT round-trip.
    x = torch.randn(2, 4, 4, 8, 8)  # [B, C, T, H, W]
    tokens, grid = patchify_video(x, 1, 2, 2)
    x_rec = unpatchify_video(tokens, 1, 2, 2, grid, C=4)
    assert torch.allclose(x, x_rec, atol=1e-5)

    # learned space-time patch embed produces a token sequence.
    embed = SpacetimePatchEmbed(in_channels=4, hidden=24, patch=(1, 2, 2))
    tok2, grid2 = embed(x)
    assert tok2.shape == (2, grid2[0] * grid2[1] * grid2[2], 24)

    # one factorized block on [B, T, S, D].
    blk = FactorizedSpacetimeBlock(dim=24, heads=4)
    h = torch.randn(2, 4, 16, 24)
    cond = torch.randn(2, 24)
    assert blk(h, cond).shape == h.shape

    # VideoDiT forward (uncond, conditional, and image-to-video cond_frames).
    vdit = VideoDiT(in_channels=4, input_size=8, num_frames=4, hidden=48, depth=2,
                    heads=4, patch=(1, 2, 2), context_dim=None)
    v = torch.randn(2, 4, 4, 8, 8)
    t = torch.randint(0, 50, (2,))
    assert vdit(v, t).shape == v.shape
    cond_frames = torch.randn(2, 4, 1, 8, 8)  # one known leading frame (I2V)
    assert vdit(v, t, cond_frames=cond_frames).shape == v.shape

    cvdit = VideoDiT(in_channels=4, input_size=8, num_frames=4, hidden=48, depth=2,
                     heads=4, patch=(1, 2, 2), context_dim=32)
    ctx = torch.randn(2, 5, 32)
    mask = torch.ones(2, 5, dtype=torch.bool)
    assert cvdit(v, t, context=ctx, mask=mask).shape == v.shape
    _ok("video patchify round-trip + VideoDiT forward (uncond/text/I2V)")


def test_autoregressive() -> None:
    vqvae = VQVAE(in_channels=3, dim=16, num_codes=64, ch_mult=(1, 2))
    x = torch.randn(2, 3, 16, 16)
    x_rec, vq_loss, indices = vqvae(x)
    assert x_rec.shape == x.shape and vq_loss.dim() == 0
    grid = vqvae.encode(x)
    assert grid.dtype == torch.long
    assert vqvae.decode_indices(grid).shape == x.shape

    # GPT over a small interleaved vocab.
    n_text = 16
    vocab = TokenVocab(n_text=n_text, num_codes=vqvae.num_codes)
    h = w = grid.shape[1]
    transformer = TokenTransformer(
        vocab_size=vocab.vocab_size, dim=48, depth=2, heads=4,
        max_len=2 + 4 + 3 + h * w + 4,
    )
    idx = torch.randint(0, n_text, (1, 4))
    seq = build_prompt_sequence(idx, vocab)
    out = transformer.generate(seq, max_new=3, temperature=1.0, top_k=10)
    assert out.shape[1] == seq.shape[1] + 3

    # full text -> codes -> image shape loop (random weights, shape-correct).
    prompt_ids = torch.randint(0, n_text, (1, 4))
    code_grid, image = autoregressive_generate(
        transformer, vqvae, prompt_ids, grid_hw=(h, w), vocab=vocab, top_k=10
    )
    assert code_grid.shape == (1, h, w)
    assert image.shape == (1, 3, 16, 16)
    _ok("VQVAE round-trip + TokenTransformer.generate + autoregressive_generate")


def test_toy() -> None:
    # The toy MLP denoiser obeys the backbone contract and make_target works.
    model = MLPDenoiser(hidden=32)
    x = torch.randn(8, 2)
    t = torch.randint(0, 50, (8,))
    assert model(x, t).shape == (8, 2)
    for name in ("moons", "spiral", "gaussians"):
        pts = make_target(name, 64)
        assert pts.shape == (64, 2)
    _ok("toy MLPDenoiser + make_target")


def main() -> None:
    torch.manual_seed(0)
    print("running smoke tests...")
    test_schedule()
    test_unet()
    test_dit()
    test_vae()
    test_text_encoder()
    test_diffusion_loss_and_sample()
    test_guidance()
    test_video()
    test_autoregressive()
    test_toy()
    print("SMOKE OK")


if __name__ == "__main__":
    main()
