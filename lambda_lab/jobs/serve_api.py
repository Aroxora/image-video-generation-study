#!/usr/bin/env python3
"""
Minimal SDXL inference HTTP server for the gen·lab Studio web client.

  POST /generate  {prompt, negative?, steps?, width?, height?, seed?, guidance?}
                  -> {image: "data:image/png;base64,...", model, seed}
  GET  /health    -> {ready, model, device}

Binds to 127.0.0.1 only, so it is reachable solely through the SSH tunnel the
orchestrator prints — it is NEVER exposed to the public internet. CORS (incl.
Chrome's Private-Network-Access preflight) is enabled so the static web app can
call it across the tunnel. Stdlib only besides torch/diffusers.

SDXL ships no prompt classifier; the optional SD1.x output checker (which
false-positives constantly) is disabled if present. You are responsible for
lawful use — no sexual content involving minors, no non-consensual imagery of
real people.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE = {"pipe": None, "model": None, "ready": False, "lock": threading.Lock()}


def load(model: str) -> None:
    import torch
    from diffusers import AutoPipelineForText2Image

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
    # SD1.x carries a safety_checker that mostly false-positives; SDXL has none.
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    STATE.update(pipe=pipe, model=model, ready=True)
    print(f"model ready: {model}", flush=True)


class Handler(BaseHTTPRequestHandler):
    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        # let an https page reach this localhost endpoint (Chrome PNA preflight)
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
            self._json(200, {"ready": STATE["ready"], "model": STATE["model"],
                             "device": "cuda" if torch.cuda.is_available() else "cpu"})
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
        kwargs = dict(
            prompt=prompt,
            negative_prompt=(req.get("negative") or None),
            num_inference_steps=int(req.get("steps", 30)),
            guidance_scale=float(req.get("guidance", 6.0)),
            width=int(req.get("width", 1024)),
            height=int(req.get("height", 1024)),
        )
        if gen is not None:
            kwargs["generator"] = gen
        try:
            with STATE["lock"]:  # one GPU -> serialize requests
                image = STATE["pipe"](**kwargs).images[0]
        except Exception as e:  # noqa: BLE001
            self._json(500, {"error": f"generation failed: {e}"})
            return
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        data = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        self._json(200, {"image": data, "model": STATE["model"], "seed": seed})

    def log_message(self, *args) -> None:  # quiet
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
