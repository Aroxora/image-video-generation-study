#!/usr/bin/env python3
"""
Minimal SDXL inference HTTP server for the gen·lab Studio web client.

  POST /generate  {prompt, negative?, steps?, width?, height?, seed?, guidance?,
                   init_image?, strength?}
                  -> {image: "data:image/png;base64,...", model, seed, mode}
  GET  /health    -> {ready, model, device, img2img}

If `init_image` (a data-URL or bare base64 PNG/JPEG) is present it runs
image-to-image: the upload is the starting point and `strength` (0..1) sets how
far the prompt is allowed to push it (higher = more change). Otherwise it runs
text-to-image. The img2img pipeline is built with `from_pipe`, so it SHARES the
text-to-image weights — no extra VRAM, no second download.

Binds to 127.0.0.1 only (reached via the SSH tunnel; never public). CORS + the
Chrome Private-Network-Access header are sent so the static web app can call it.
SDXL ships no prompt classifier; the SD1.x output checker is disabled if present.
You are responsible for lawful use — no sexual content involving minors, no
non-consensual imagery of real people.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE = {
    "pipe": None,        # text-to-image
    "img2img": None,     # image-to-image (shares pipe's weights)
    "model": None,
    "ready": False,
    "lock": threading.Lock(),
}


def load(model: str) -> None:
    import torch
    from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    pipe = AutoPipelineForText2Image.from_pretrained(model, torch_dtype=dtype)
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    else:
        pipe = pipe.to("cpu")
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    # Reuse the SAME weights for image-to-image — no extra VRAM / download.
    img2img = AutoPipelineForImage2Image.from_pipe(pipe)
    if hasattr(img2img, "safety_checker"):
        img2img.safety_checker = None
    STATE.update(pipe=pipe, img2img=img2img, model=model, ready=True)
    print(f"model ready: {model} (text2img + img2img)", flush=True)


def _decode_image(data_url: str):
    """data-URL or bare base64 -> RGB PIL image."""
    from PIL import Image

    payload = data_url.split(",", 1)[1] if "," in data_url else data_url
    raw = base64.b64decode(payload)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _round8(n: int) -> int:
    return max(256, (int(n) // 8) * 8)


class Handler(BaseHTTPRequestHandler):
    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Private-Network", "true")

    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:
        if self.path.startswith("/health"):
            import torch
            self._json(200, {
                "ready": STATE["ready"],
                "model": STATE["model"],
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "img2img": STATE["img2img"] is not None,
            })
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not self.path.startswith("/generate"):
            self._json(404, {"error": "not found"})
            return
        if not STATE["ready"]:
            self._json(503, {"error": "model still loading — try again in a moment"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            self._json(400, {"error": "bad json"})
            return
        prompt = (req.get("prompt") or "").strip()
        if not prompt:
            self._json(400, {"error": "empty prompt"})
            return

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        seed = req.get("seed")
        gen = None
        if seed not in (None, "", -1, "-1"):
            gen = torch.Generator(device=device).manual_seed(int(seed))

        width = _round8(req.get("width", 1024))
        height = _round8(req.get("height", 1024))
        steps = int(req.get("steps", 30))
        guidance = float(req.get("guidance", 6.0))
        negative = req.get("negative") or None
        init_url = req.get("init_image")

        try:
            with STATE["lock"]:  # one GPU -> serialize requests
                if init_url:
                    # ---- image-to-image: transform the upload ----
                    init = _decode_image(init_url).resize((width, height))
                    strength = float(req.get("strength", 0.6))
                    strength = min(0.99, max(0.05, strength))
                    image = STATE["img2img"](
                        prompt=prompt,
                        image=init,
                        strength=strength,
                        negative_prompt=negative,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        generator=gen,
                    ).images[0]
                    mode = "img2img"
                else:
                    # ---- text-to-image ----
                    image = STATE["pipe"](
                        prompt=prompt,
                        negative_prompt=negative,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        width=width,
                        height=height,
                        generator=gen,
                    ).images[0]
                    mode = "text2img"
        except Exception as e:  # noqa: BLE001
            self._json(500, {"error": f"generation failed: {e}"})
            return

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        data = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        self._json(200, {"image": data, "model": STATE["model"], "seed": seed, "mode": mode})

    def log_message(self, *args) -> None:
        return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    print(f"loading {args.model} …", flush=True)
    load(args.model)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"serving SDXL on 127.0.0.1:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
