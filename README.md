# gen·lab — how image & video generation works

An **interactive explainer** of how modern image and video models actually generate pixels —
paired with a **small, runnable PyTorch implementation** of every idea it describes.

The website teaches the concepts by *showing* them (live diffusion, cross-attention heatmaps,
spacetime patches, token-by-token generation). Every page links directly to the PyTorch file in
this repo that implements the idea, so you can go from "what is classifier-free guidance" to the
exact lines of code in one click.

> **The one-sentence version:** diffusion denoises the whole image (or the whole clip) at once,
> with text steering it via cross-attention at every step; autoregression predicts discrete tokens
> in sequence, where text and frames truly interleave. The "next-frame prediction" mental model
> fits the *second* family — but most of the impressive results you've seen come from the *first*.

---

## Repository layout

```
.
├── web/                 # Angular 21 site (Firebase Hosting), zoneless + standalone components
│   └── src/app/
│       ├── features/    # one interactive chapter per page
│       ├── shared/      # <app-chapter>, <app-code-ref>, <app-math>
│       └── core/        # firebase init + analytics, repo links, the chapter registry
├── pytorch/             # the runnable implementation each page links to
│   ├── diffusion/       # schedule · vae · unet · dit · cross_attention · text_encoder · guidance · ddpm
│   ├── video/           # spacetime patches · temporal attention · video DiT
│   ├── autoregressive/  # VQ-VAE tokenizer · GPT-style token transformer · sampler
│   ├── toy/             # a tiny 2D diffusion you can train in seconds (mirrors the live playground)
│   ├── train.py         # end-to-end training entrypoint
│   └── sample.py        # generate from a checkpoint
├── firebase.json        # Hosting config (serves web/dist/web/browser as an SPA)
└── .firebaserc          # Firebase project: image-video-gen-models
```

## Chapters ↔ code

| Page | What it explains | PyTorch |
|------|------------------|---------|
| `/` | The two families; why "next-frame" is backwards | `pytorch/README.md` |
| `/diffusion` | Forward (add noise) & reverse (learned denoise) | `diffusion/schedule.py`, `diffusion/ddpm.py` |
| `/latent` | Why Stable Diffusion diffuses in a VAE latent | `diffusion/vae.py` |
| `/denoiser` | The noise-predicting network: U-Net → DiT | `diffusion/unet.py`, `diffusion/dit.py` |
| `/text` | Cross-attention: the prompt as a steering signal | `diffusion/text_encoder.py`, `diffusion/cross_attention.py` |
| `/guidance` | Classifier-free guidance, the prompt amplifier | `diffusion/guidance.py` |
| `/video` | Spacetime latents, patches & temporal attention | `video/*.py` |
| `/autoregressive` | VQ-VAE tokens + a Transformer, interleaved | `autoregressive/*.py` |
| `/build` | How the pieces assemble into a training loop | `train.py`, `sample.py` |
| `/playground` | Real reverse diffusion, live in your browser | `toy/toy_diffusion_2d.py` |

---

## Run the website locally

```bash
cd web
npm install
npm start            # http://localhost:4200
npm run build        # production build → web/dist/web/browser
```

Built with Angular 21 (zoneless change detection, standalone components, lazy routes), KaTeX for
math, and highlight.js for code. Firebase Analytics is initialized from `core/firebase.service.ts`.

## Run the PyTorch implementation

```bash
cd pytorch                       # or run module-style from the repo root
pip install -r requirements.txt

python -m pytorch.tests.smoke    # exercises every module with tiny tensors
python -m pytorch.train --epochs 1 --dataset shapes      # trains on synthetic data, no downloads
python -m pytorch.sample --ckpt runs/model.pt --steps 50 --guidance 4
python -m pytorch.toy.toy_diffusion_2d --target moons    # the 2D toy behind the live playground
```

Everything is CPU-friendly and intentionally small — it's a *teaching-scale* model. Production
systems are vastly larger, but architecturally they are the same pieces you'll read here.

## Deploy (Firebase Hosting)

```bash
firebase deploy --only hosting   # builds web/ via predeploy, serves the SPA
```

---

## License & intent

Open and educational. The goal is intuition you can trust, backed by code you can run.
Issues and PRs that make an explanation clearer or a module more faithful are very welcome.
