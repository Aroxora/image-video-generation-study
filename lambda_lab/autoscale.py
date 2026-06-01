"""
Local autoscaling controller — scale-to-zero GPU.

Run once on your machine (the Studio's endpoint is unchanged):

    python -m lambda_lab.run autoscale --idle 15 --budget 10

It listens on 127.0.0.1:8000 (the Studio default). The GPU box is OFF by default
($0). On the first /generate it lazily launches the SDXL box (reusing a
run-id-tagged instance + an optional persistent filesystem), opens an SSH tunnel,
and proxies to it. After --idle minutes with no requests it terminates the box.

So GPU cost ≈ (time you actually generate + a short idle grace), not a 24/7 burn.
Cold start is ~2-5 min the first time after idle; pass --filesystem to keep weights
on a persistent disk so restarts skip the multi-GB download. The controller itself
is a tiny local CPU process and costs nothing. Ctrl-C tears the box down.
"""

from __future__ import annotations

import atexit
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .api import LambdaCloud
from .engine import RUN_DIR

SSH_OPTS = [
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "ServerAliveInterval=30",
    "-o", "ExitOnForwardFailure=yes",
]


def _log(msg: str) -> None:
    print(f"[autoscale {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class AutoScaler:
    """Owns the box lifecycle + idle reaping. Thread-safe via a single lock."""

    def __init__(self, *, run_id: str, pipeline: str, upstream_port: int, idle_s: float,
                 budget: float, instance_type: str, filesystem: str | None, key_path: str):
        self.run_id = run_id
        self.pipeline = pipeline
        self.upstream = f"http://127.0.0.1:{upstream_port}"
        self.upstream_port = upstream_port
        self.idle_s = idle_s
        self.budget = budget
        self.instance_type = instance_type
        self.filesystem = filesystem
        self.key_path = key_path
        self.state = "down"            # down | warming | up | stopping
        self.last_activity = time.time()
        self.tunnel: subprocess.Popen | None = None
        self.lock = threading.Lock()
        self.api = LambdaCloud()

    # ---- helpers ----
    def _env(self) -> dict:
        e = dict(os.environ)
        e.setdefault("LAMBDA_LAB_UA", "curl/8.7.1")
        return e

    def _cli(self, *args: str) -> int:
        return subprocess.run([sys.executable, "-m", "lambda_lab.run", *args], env=self._env()).returncode

    def _box_ip(self) -> str | None:
        inst = self.api.find_instance_by_name(self.run_id)
        if inst and inst.get("ip"):
            return inst["ip"]
        p = RUN_DIR / f"{self.run_id}.json"
        if p.exists():
            return json.loads(p.read_text()).get("ctx", {}).get("ip")
        return None

    # ---- lifecycle ----
    def warm(self) -> None:
        with self.lock:
            if self.state in ("up", "warming"):
                return
            self.state = "warming"
        try:
            _log(f"cold start: launching {self.pipeline} (this takes ~2-5 min)…")
            args = ["start", self.pipeline, "--run-id", self.run_id,
                    "--budget", str(self.budget), "--instance-type", self.instance_type]
            if self.filesystem:
                args += ["--filesystem", self.filesystem]
            if self._cli(*args) != 0:
                _log("start failed; staying down")
                self.state = "down"
                return
            ip = self._box_ip()
            if not ip:
                _log("no instance IP after start; staying down")
                self.state = "down"
                return
            self._open_tunnel(ip)
            if self._wait_upstream_ready(240):
                self.last_activity = time.time()
                self.state = "up"
                _log(f"GPU ready at {ip} — proxying; will idle-stop after {int(self.idle_s)}s")
            else:
                _log("upstream never became ready; tearing down")
                self.teardown()
        except Exception as e:  # noqa: BLE001
            _log(f"warm error: {e}")
            self.state = "down"

    def _open_tunnel(self, ip: str) -> None:
        self._close_tunnel()
        cmd = ["ssh", *SSH_OPTS, "-i", self.key_path, "-N",
               "-L", f"{self.upstream_port}:localhost:8000", f"ubuntu@{ip}"]
        self.tunnel = subprocess.Popen(cmd)

    def _close_tunnel(self) -> None:
        if self.tunnel:
            try:
                self.tunnel.terminate()
            except Exception:
                pass
            self.tunnel = None

    def _wait_upstream_ready(self, timeout: float) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(self.upstream + "/health", timeout=8) as resp:
                    if json.loads(resp.read().decode()).get("ready"):
                        return True
            except Exception:
                pass
            time.sleep(5)
        return False

    def teardown(self) -> None:
        with self.lock:
            if self.state == "down":
                return
            self.state = "stopping"
        _log("scaling to zero: terminating box (meter stops)")
        self._close_tunnel()
        self._cli("teardown", self.run_id)
        self.state = "down"

    def reaper(self) -> None:
        while True:
            time.sleep(15)
            if self.state == "up":
                idle = time.time() - self.last_activity
                if idle > self.idle_s:
                    _log(f"idle {int(idle)}s > {int(self.idle_s)}s")
                    self.teardown()

    def proxy_post(self, path: str, body: bytes) -> tuple[int, bytes]:
        req = urllib.request.Request(self.upstream + path, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()


def _handler(scaler: AutoScaler):
    class Handler(BaseHTTPRequestHandler):
        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Private-Network", "true")

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _raw(self, code, body: bytes):
            self.send_response(code)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self):
            if not self.path.startswith("/health"):
                self._json(404, {"error": "not found"})
                return
            if scaler.state == "up":
                self._json(200, {
                    "ready": True, "status": "up",
                    "idle_s": int(time.time() - scaler.last_activity),
                    "model": "stabilityai/stable-diffusion-xl-base-1.0 (autoscaled)",
                })
            else:
                self._json(200, {
                    "ready": False, "status": scaler.state,
                    "message": "GPU is off — it starts automatically on your first generate (~2-5 min cold start).",
                })

        def do_POST(self):
            if not self.path.startswith("/generate"):
                self._json(404, {"error": "not found"})
                return
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n)  # always drain
            scaler.last_activity = time.time()
            if scaler.state == "up":
                try:
                    code, data = scaler.proxy_post("/generate", body)
                    self._raw(code, data)
                    return
                except Exception as e:  # upstream dropped (e.g. just reaped) — restart
                    _log(f"proxy failed ({e}); re-warming")
                    scaler.state = "down"
            if scaler.state == "down":
                threading.Thread(target=scaler.warm, daemon=True).start()
            self._json(503, {
                "status": "warming",
                "retry_in_s": 15,
                "message": "GPU is starting (~2-5 min). Keep your prompt — it will run as soon as the box is up.",
            })

        def log_message(self, *a):
            return

    return Handler


def serve_autoscaler(*, idle_min: float, budget: float, local_port: int,
                     instance_type: str, filesystem: str | None, run_id: str,
                     pipeline: str = "serve-sdxl") -> None:
    key_path = os.path.expanduser("~/.ssh/id_ed25519")
    scaler = AutoScaler(
        run_id=run_id, pipeline=pipeline, upstream_port=local_port + 1,
        idle_s=idle_min * 60, budget=budget, instance_type=instance_type,
        filesystem=filesystem, key_path=key_path,
    )

    def _bye(*_a):
        _log("shutting down — tearing the box down so nothing keeps billing")
        scaler.teardown()
        os._exit(0)

    atexit.register(scaler.teardown)
    signal.signal(signal.SIGINT, _bye)
    signal.signal(signal.SIGTERM, _bye)

    threading.Thread(target=scaler.reaper, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", local_port), _handler(scaler))
    _log(f"listening on http://127.0.0.1:{local_port}  (point the Studio here)")
    _log(f"GPU OFF until first generate · idle-stop {int(idle_min)} min · budget ${budget} · {instance_type}"
         + (f" · fs={filesystem}" if filesystem else " · no persistent fs (cold starts re-download weights)"))
    srv.serve_forever()
