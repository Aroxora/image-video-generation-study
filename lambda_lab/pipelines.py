"""
Declarative pipelines: each is an ordered list of steps plus default params that
pick the cheapest instance from the account lineup that comfortably fits the task.
Override any default from the CLI with `--set key=value` or a `--config` JSON file.

The default instance choices below favor x86 boxes for training (no aarch64
friction) and the cheapest GPU that fits; switch `instance_type` to gpu_1x_gh200
when you want the 96 GB / $2.29-hr box (see bootstrap.sh — it handles ARM).
"""

from __future__ import annotations

from . import steps as S
from .engine import Step

REMOTE_KIT = S.REMOTE_KIT
JOBDIR = f"{REMOTE_KIT}/jobs"


def _train_steps() -> list[Step]:
    return [
        Step("ensure_ssh_key", S.ensure_ssh_key),
        Step("ensure_filesystem", S.ensure_filesystem),
        Step("launch", S.launch, teardown_on_fail=True),
        Step("wait_active", S.wait_active, teardown_on_fail=True),
        Step("bootstrap", S.bootstrap, teardown_on_fail=True),
        Step("sync_up", S.sync_up, teardown_on_fail=True),
        Step("run_job", S.run_job, teardown_on_fail=True),
        Step("sync_down", S.sync_down, teardown_on_fail=True),
        Step("teardown", S.teardown),
    ]


def _serve_steps() -> list[Step]:
    # No teardown: a server stays up until you run `run.py teardown <run_id>`.
    return [
        Step("ensure_ssh_key", S.ensure_ssh_key),
        Step("ensure_filesystem", S.ensure_filesystem),
        Step("launch", S.launch, teardown_on_fail=True),
        Step("wait_active", S.wait_active, teardown_on_fail=True),
        Step("bootstrap", S.bootstrap, teardown_on_fail=True),
        Step("serve", S.serve_job, teardown_on_fail=True),
    ]


# pipeline name -> (steps, default params). job_command runs on the instance.
PIPELINES: dict[str, dict] = {
    # ---- image LoRA ----
    "train-sdxl-lora": {
        "steps": _train_steps,
        "defaults": {
            "instance_type": "gpu_1x_a10",          # 24 GB is plenty for SDXL LoRA — $1.29/hr
            "stack": "ai-toolkit",
            "budget_usd": 5.0,
            "upload": [("./configs/sdxl_lora.example.yaml", f"{REMOTE_KIT}/run.yaml"),
                       ("./dataset", f"{REMOTE_KIT}/dataset")],
            "job_command": f"bash {JOBDIR}/sdxl_lora.sh {REMOTE_KIT}/run.yaml",
            "download": [f"{REMOTE_KIT}/output"],
        },
    },
    "train-flux-lora": {
        "steps": _train_steps,
        "defaults": {
            "instance_type": "gpu_1x_a100_sxm4",     # 40 GB + FP8 fits FLUX LoRA — $1.99/hr
            "stack": "ai-toolkit",
            "budget_usd": 8.0,
            "upload": [("./configs/flux_lora.example.yaml", f"{REMOTE_KIT}/run.yaml"),
                       ("./dataset", f"{REMOTE_KIT}/dataset")],
            "job_command": f"bash {JOBDIR}/flux_lora.sh {REMOTE_KIT}/run.yaml",
            "download": [f"{REMOTE_KIT}/output"],
        },
    },
    # ---- video LoRA ----
    "train-video-lora": {
        "steps": _train_steps,
        "defaults": {
            "instance_type": "gpu_1x_h100_pcie",     # 80 GB, x86 — $3.29/hr; or gpu_1x_gh200 for 96 GB
            "stack": "diffusion-pipe",
            "budget_usd": 20.0,
            "upload": [("./configs/video_lora.example.toml", f"{REMOTE_KIT}/run.toml"),
                       ("./dataset", f"{REMOTE_KIT}/dataset")],
            "job_command": f"bash {JOBDIR}/video_lora.sh {REMOTE_KIT}/run.toml",
            "download": [f"{REMOTE_KIT}/output"],
        },
    },
    # ---- batch inference (generate N, pull, terminate) ----
    "batch-infer": {
        "steps": _train_steps,                       # same lifecycle: gen -> download -> teardown
        "defaults": {
            "instance_type": "gpu_1x_a100_sxm4",
            "stack": "diffusers",
            "budget_usd": 3.0,
            "model": "black-forest-labs/FLUX.1-schnell",
            "prompt": "a tiny astronaut hatching from an egg on the moon, cinematic",
            "n": 16,
            "job_command": "",   # filled in build() from model/prompt/n
            "download": [f"{REMOTE_KIT}/output"],
        },
    },
    # ---- SDXL JSON API for the web Studio (stays up; tear down manually) ----
    "serve-sdxl": {
        "steps": _serve_steps,
        "defaults": {
            "instance_type": "gpu_1x_a10",          # SDXL fits 24 GB — $1.29/hr
            "stack": "diffusers",
            "budget_usd": 10.0,
            "keep_alive": True,
            "port": 8000,
            "model": "stabilityai/stable-diffusion-xl-base-1.0",
            "job_command": "",                       # built in build() from model/port
        },
    },
    # ---- interactive ComfyUI server (stays up; tear down manually) ----
    "serve-comfyui": {
        "steps": _serve_steps,
        "defaults": {
            "instance_type": "gpu_1x_a10",
            "stack": "comfyui",
            "budget_usd": 6.0,
            "keep_alive": True,
            "job_command": f"bash {JOBDIR}/serve_comfyui.sh",
        },
    },
}


def build(pipeline: str, overrides: dict) -> tuple[list[Step], dict]:
    if pipeline not in PIPELINES:
        raise KeyError(f"unknown pipeline {pipeline!r}; known: {sorted(PIPELINES)}")
    spec = PIPELINES[pipeline]
    params = dict(spec["defaults"])
    params.update(overrides or {})

    # bootstrap.sh installs into ~/venv, so direct-python jobs must use that
    # interpreter (the *.sh job scripts already `source` it themselves).
    py = "$HOME/venv/bin/python"

    # serve-sdxl: assemble the remote command from model/port (so --set takes effect)
    if pipeline == "serve-sdxl":
        model = params["model"]
        port = int(params.get("port", 8000))
        params["job_command"] = f'{py} {JOBDIR}/serve_api.py --model "{model}" --port {port}'

    # batch-infer: assemble the remote command from model/prompt/n if not given
    if pipeline == "batch-infer" and not params.get("job_command"):
        model = params["model"]
        prompt = params["prompt"].replace('"', '\\"')
        n = int(params.get("n", 16))
        params["job_command"] = (
            f'{py} {JOBDIR}/batch_infer.py --model "{model}" '
            f'--prompt "{prompt}" --n {n} --out {REMOTE_KIT}/output'
        )

    steps = spec["steps"]()
    return steps, params
