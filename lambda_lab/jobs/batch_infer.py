#!/usr/bin/env python3
"""
Minimal, dependency-light batch image generation with 🤗 diffusers — the cheapest
way to turn a rented GPU into N finished images and then get out. Works for FLUX
and SDXL checkpoints; optionally stacks a trained LoRA.

  python batch_infer.py --model black-forest-labs/FLUX.1-schnell \
      --prompt "a fox in snow, cinematic" --n 16 --out ./output [--lora ./output/my_lora.safetensors]

FLUX.1-schnell is Apache-2.0 and needs only ~4 steps, so it is the most economical
"just make images" model. Swap --model for SDXL or a FLUX.1-dev finetune as needed.
"""

from __future__ import annotations

import argparse
import os
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--steps", type=int, default=0, help="0 = model default (4 for schnell, ~28 for dev/SDXL)")
    ap.add_argument("--guidance", type=float, default=0.0, help="0 = model default")
    ap.add_argument("--out", default="./output")
    ap.add_argument("--lora", default="", help="optional LoRA .safetensors to fuse")
    ap.add_argument("--batch", type=int, default=4, help="images per forward batch")
    args = ap.parse_args()

    import torch
    from diffusers import AutoPipelineForText2Image

    os.makedirs(args.out, exist_ok=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    pipe = AutoPipelineForText2Image.from_pretrained(args.model, torch_dtype=dtype)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    # Offload if the model is large relative to VRAM — keeps big models on small cards.
    try:
        pipe.enable_model_cpu_offload()
    except Exception:
        pass
    if args.lora:
        pipe.load_lora_weights(args.lora)

    is_schnell = "schnell" in args.model.lower()
    steps = args.steps or (4 if is_schnell else 28)
    guidance = args.guidance or (0.0 if is_schnell else 3.5)

    made, t0 = 0, time.time()
    while made < args.n:
        k = min(args.batch, args.n - made)
        out = pipe(
            [args.prompt] * k,
            num_inference_steps=steps,
            guidance_scale=guidance,
        )
        for img in out.images:
            img.save(os.path.join(args.out, f"img_{made:04d}.png"))
            made += 1
        print(f"  {made}/{args.n} images ({(time.time()-t0):.1f}s elapsed)", flush=True)

    dt = time.time() - t0
    print(f"done: {made} images in {dt:.1f}s = {made/dt*60:.1f} img/min -> {args.out}")


if __name__ == "__main__":
    main()
