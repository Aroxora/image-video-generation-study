# Dataset layout

Put your training data in a `dataset/` folder at the repo root (it is uploaded to
the box at `~/lambda_lab/dataset` by the `sync_up` step). It is git-ignored.

## Image LoRA (FLUX / SDXL)

15–40 images is plenty for a subject or style LoRA. Each image gets a `.txt`
caption with the **same base name**:

```
dataset/
  01.png      01.txt   # "p3rs0n, a photo of a woman with red hair, smiling"
  02.png      02.txt
  ...
```

Tips that save money (fewer steps to converge):
- Caption consistently; include your `trigger_word` from the config.
- Variety of pose / background / lighting beats near-duplicate shots.
- `cache_latents_to_disk: true` (already set) encodes each image once.

## Video LoRA (diffusion-pipe)

diffusion-pipe reads a `dataset.toml` that points at clip folders. Keep clips
short (2–5 s) and captioned. See `diffusion-pipe/examples/` on the box and the
[diffusion-pipe docs](https://github.com/tdrussell/diffusion-pipe) for the exact
`dataset.toml` schema, then drop yours at `dataset/dataset.toml`.

## Responsible use

Only train on data you have the rights to. Respect each base model's license
(FLUX.1-dev is non-commercial; SDXL/Wan are more permissive). Don't fine-tune on
a real person without consent. You own what you generate and how you use it.
