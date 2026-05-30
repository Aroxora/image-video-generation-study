# gen·lab — the PyTorch implementation

This is the **build-it-yourself** code that accompanies the **gen·lab** website
("how image & video generation works"). Every page of the site links to the file(s)
here that implement the idea it explains, so you can go from *"what is classifier-free
guidance?"* to the exact lines that do it.

It is a faithful, *teaching-scale* re-implementation of how modern systems actually
work — latent diffusion (Stable Diffusion), Diffusion Transformers (DiT / SD3 /
Sora-style video), classifier-free guidance, factorized space-time attention, and a
VQ-token autoregressive pipeline — but small enough to **train on a laptop CPU with no
downloads**.

Repo: **image-video-generation-study** (GitHub: `Aroxora/image-video-generation-study`).

---

## What's in here

```
pytorch/
├── diffusion/
│   ├── schedule.py        # beta schedules + the forward diffusion math (NoiseSchedule)
│   ├── ddpm.py            # training loss + DDPM/DDIM samplers (GaussianDiffusion)
│   ├── guidance.py        # classifier-free guidance + condition-dropout
│   ├── cross_attention.py # timestep embeddings + (self/cross) attention + SpatialTransformer
│   ├── unet.py            # the Stable-Diffusion-family U-Net denoiser
│   ├── dit.py             # the Diffusion Transformer denoiser (adaLN-Zero)
│   ├── vae.py             # SD-style latent VAE (the image <-> latent bridge)
│   └── text_encoder.py    # a tiny char-level text encoder (+ how to swap in CLIP/T5)
├── video/
│   ├── spacetime.py            # space-time patchify / unpatchify / patch-embed
│   ├── temporal_attention.py   # spatial + temporal (factorized) attention blocks
│   └── video_dit.py            # VideoDiT: denoise a whole clip at once (incl. image-to-video)
├── autoregressive/
│   ├── vqvae.py          # VQ-VAE: image <-> grid of discrete codes
│   ├── transformer.py    # GPT over one interleaved [BOS]text[BOI]codes[EOI] stream
│   └── sample.py         # ties VQVAE + GPT: text -> image-code tokens -> pixels
├── toy/
│   └── toy_diffusion_2d.py  # 2-D diffusion you can plot (mirrors the live Playground)
├── train.py              # end-to-end trainer (data -> [VAE] -> diffusion -> UNet/DiT)
├── sample.py             # load a checkpoint and generate a grid PNG
├── tests/smoke.py        # exercises EVERY module with tiny tensors
└── requirements.txt      # torch, numpy, matplotlib, tqdm, pillow (torchvision optional)
```

---

## Website page ↔ code

| Website page | What it explains | Files that implement it |
|---|---|---|
| `/` | The two families; why "next-frame" intuition is backwards | this README |
| `/diffusion` | Forward (add noise) & reverse (learned denoise) | `diffusion/schedule.py`, `diffusion/ddpm.py` |
| `/latent` | Why Stable Diffusion diffuses in a VAE latent, not pixels | `diffusion/vae.py` |
| `/denoiser` | The noise-predicting network: U-Net → DiT | `diffusion/unet.py`, `diffusion/dit.py`, `diffusion/cross_attention.py` |
| `/text` | Cross-attention: the prompt as a steering signal | `diffusion/text_encoder.py`, `diffusion/cross_attention.py` |
| `/guidance` | Classifier-free guidance, the prompt amplifier | `diffusion/guidance.py` |
| `/video` | Space-time latents, patches & temporal attention | `video/spacetime.py`, `video/temporal_attention.py`, `video/video_dit.py` |
| `/autoregressive` | VQ-VAE tokens + a Transformer, text & image interleaved | `autoregressive/vqvae.py`, `autoregressive/transformer.py`, `autoregressive/sample.py` |
| `/build` | How the pieces assemble into a training/sampling loop | `train.py`, `sample.py` |
| `/playground` | Real reverse diffusion, live — the same algorithm in JS | `toy/toy_diffusion_2d.py` |

---

## Quickstart

All commands run from the **repo root** (the package is imported as `pytorch.*`).

```bash
# 1. install (CPU torch is fine; torchvision is optional, only for --dataset mnist)
pip install -r pytorch/requirements.txt

# 2. smoke test — imports + forward/backward on every module with tiny tensors
python -m pytorch.tests.smoke
#    -> prints "SMOKE OK"

# 3. train on synthetic colored shapes (no downloads), 1 epoch, ~seconds on CPU
python -m pytorch.train --epochs 1 --batch 8 --dataset shapes --out runs/model.pt

# 4. sample a grid from the checkpoint (DDIM, 50 steps by default)
python -m pytorch.sample --ckpt runs/model.pt --steps 50 --n 4 --out samples.png

# 5. the 2-D toy behind the live Playground (saves a scatter PNG)
python -m pytorch.toy.toy_diffusion_2d --target moons --epochs 2000 --out toy.png
```

### Useful training flags

```bash
# Diffusion Transformer instead of the U-Net:
python -m pytorch.train --backbone dit --epochs 1 --dataset shapes

# Latent diffusion (run in a VAE latent space) + text-conditioned (class captions):
python -m pytorch.train --latent --text --epochs 1 --dataset shapes

# Then sample a text prompt with classifier-free guidance:
python -m pytorch.sample --ckpt runs/model.pt --prompt "a red circle" --guidance 4.0
```

`--dataset shapes` is generated on the fly (circles / squares / triangles in red /
green / blue) so nothing is downloaded. `--dataset mnist` exists too but needs the
optional `torchvision` package. `--device auto` picks CUDA / MPS / CPU automatically.

---

## The tensor-shape contract

Every module follows these conventions (so the backbones are interchangeable):

| thing | shape / type |
|---|---|
| images / latents | `[B, C, H, W]` |
| video | `[B, C, T, H, W]` |
| timesteps `t` | `LongTensor [B]`, values in `[0, timesteps)` |
| text/context | `[B, L, context_dim]` float |
| key-padding `mask` | `bool [B, L]`, **`True` = REAL token**, `False` = pad |

And every denoiser backbone predicts **epsilon** (the noise) and returns the **same
shape as its `x` input**:

```python
eps = model(x, t, context=None, mask=None)        # UNet / DiT / toy MLP
eps = video_model(x, t, context=None, cond_frames=None)   # VideoDiT (adds cond_frames)
```

This single contract is why `GaussianDiffusion`, classifier-free guidance, and the
training loop work unchanged across the U-Net, the DiT, the video DiT, and the 2-D toy.

---

## How the pieces fit (the `/build` story)

```
dataset image [B,3,32,32]
   │  (optional) VAE.encode().sample()         # /latent — diffuse in a small latent
   ▼
x0  [B,3,32,32]  or  [B,4,8,8]
   │  NoiseSchedule.q_sample(x0, t, eps)        # /diffusion — the FORWARD process (pure math)
   ▼
x_t (noisy)
   │  model(x_t, t, context, mask) -> eps_hat   # /denoiser — the LEARNED reverse step
   │  drop_context(...) during training         # /guidance — teaches the empty-prompt case
   ▼
MSE(eps_hat, eps)  ->  AdamW  ->  checkpoint
                                     │
                                     ▼  sample.py
   noise x_T ── DDPM/DDIM reverse loop (+ CFG) ──▶ x0 ── VAE.decode ──▶ image
```

---

## This is educational — real systems are bigger

This code optimizes for **clarity and faithfulness of the ideas**, not output quality:

- The models here have thousands–millions of parameters; Stable Diffusion / DiT / Sora
  have hundreds of millions to billions, trained on huge datasets for a long time.
- The text encoder is a tiny char-level stand-in. Real systems use a **frozen**
  pretrained CLIP or T5 (see the swap-in recipe at the bottom of `text_encoder.py`).
- The VAE here is small and (in the demo trainer) not separately pretrained; real latent
  diffusion uses a carefully pretrained, **frozen** VAE with a perceptual + adversarial
  objective.
- 1-epoch shapes training produces blobs, not art — it exists to prove the *plumbing*.
  Crank `--epochs` / `--steps-per-epoch` and the shapes do emerge.

But architecturally these are the same pieces, wired the same way. If you can read and
run this repo, you understand how the big systems work. Issues / PRs that make an
explanation clearer or a module more faithful are very welcome.
