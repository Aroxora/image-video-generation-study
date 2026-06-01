# lambda_lab — rent a GPU, train/serve open models, get out cheap

An **agentic, resumable orchestrator** for renting [Lambda Cloud](https://lambda.ai)
GPUs by the minute to train and serve open-weight image/video generation models,
then tearing the instance down before it can run up a bill.

> "Unlocked" here means **open-weight** models you fully control and fine-tune —
> FLUX, SDXL, SD 3.5, Wan, HunyuanVideo, LTX-Video — as opposed to closed,
> pay-per-call APIs. You rent the metal, run the weights, and own the pipeline.

It is **standard-library only** on your side (the Lambda API client is built on
`urllib`; remote work goes over your system `ssh`/`rsync`), so there's nothing to
`pip install` to drive it.

---

## Why a state machine and not a shell script

Renting a GPU is a long-horizon, money-burning, interruption-prone process. So a
**run is a persisted state machine** (`.lambda_lab/runs/<id>.json`) of idempotent
steps:

```
ensure_ssh_key → ensure_filesystem → launch → wait_active → bootstrap
  → sync_up → run_job → sync_down → teardown
```

- **Idempotent** — `launch` reuses any instance *tagged with the run id*, so a
  resume never double-spends; `bootstrap` is apt/pip-idempotent with a sentinel;
  jobs run **detached** (`setsid` + a sentinel exit-code file) so they survive SSH
  drops and a resumed run re-attaches instead of restarting.
- **Checkpointed** — state is flushed after every transition (atomic write), so the
  instant after `launch` the instance id is on disk — no orphaned GPUs.
- **Resumable** — `resume <run_id>` skips completed steps and continues. Laptop
  slept through a 6-hour train? Resume picks it back up.
- **Budget-guarded** — a hard USD cap is checked between steps and on every job
  poll; hitting it forces teardown. **Teardown is also the default success path.**

A rented GPU you forgot to kill is the only way to lose real money here; the guard
and the auto-teardown exist to make that nearly impossible.

---

## The economics (this account's lineup, 2026-06)

| instance (API id)        | VRAM  | $/hr  | arch   | best for |
|--------------------------|-------|-------|--------|----------|
| `gpu_1x_a10`             | 24 GB | 1.29  | x86_64 | SDXL train + infer, FLUX infer (quantized) |
| `gpu_1x_a100_sxm4`       | 40 GB | 1.99  | x86_64 | FLUX LoRA (FP8), fast SDXL batches |
| **`gpu_1x_gh200`**       | 96 GB | 2.29  | arm64  | **best $/VRAM** — big video models, FLUX full |
| `gpu_1x_h100_pcie`       | 80 GB | 3.29  | x86_64 | video LoRA, x86 (no aarch64 friction) |
| `gpu_1x_h100_sxm5`       | 80 GB | 4.29  | x86_64 | fastest single-GPU |
| `gpu_8x_v100`            | 16×8  | 6.32  | x86_64 | legacy parallelism (16 GB/GPU is limiting) |
| `gpu_2x_h100_sxm5`       | 80×2  | 8.38  | x86_64 | multi-GPU training |

**The GH200 (96 GB at $2.29/hr) is the standout value** — more VRAM than an H100
and cheaper than the H100 PCIe — but it is **ARM64** (Grace-Hopper). `bootstrap.sh`
handles aarch64 (PyTorch ships CUDA `aarch64`/sbsa wheels), but a couple of extras
may build from source. Prefer the x86 **A100 40 GB ($1.99)** or **H100 PCIe ($3.29)**
if you want zero ARM friction; reach for the GH200 when you need the 96 GB.

Billing is **per-minute while the instance exists** and stops at termination; a
**persistent filesystem** is billed separately per GB and survives teardown — use
it to cache 20–50 GB of weights so you never re-download them. `python -m
lambda_lab.run costs` prints the live fit matrix.

---

## Quickstart

```bash
# 1. credentials (one time)
export LAMBDA_API_KEY=...            # from https://cloud.lambda.ai/api-keys
ssh-keygen -t ed25519                # if you don't already have ~/.ssh/id_ed25519

# 2. look before you leap — no API key needed for plan/costs
python -m lambda_lab.run costs
python -m lambda_lab.run plan train-flux-lora

# 3. put 15-40 images + .txt captions in ./dataset  (see configs/dataset.README.md)

# 4. go: launch, install, train, pull results, auto-teardown — with a $8 ceiling
python -m lambda_lab.run start train-flux-lora --budget 8 \
    --instance-type gpu_1x_a100_sxm4

# interrupted? continue exactly where it stopped
python -m lambda_lab.run resume flux-<id>

# always-safe manual kill
python -m lambda_lab.run teardown flux-<id>
```

Pipelines: `train-sdxl-lora` · `train-flux-lora` · `train-video-lora` ·
`batch-infer` · `serve-comfyui`. Override any default with `--set key=value` or
`--config file.json`.

---

## Letting a coding agent drive it

Every step is also a discrete, verifiable CLI command, which is what makes this
safe to hand to an agent (e.g. Claude Code):

1. You create the API key + SSH key once and export `LAMBDA_API_KEY`.
2. The agent runs `plan` (no spend) to show the steps + estimate, then `start`
   with an explicit `--budget`. The budget guard is the hard safety rail.
3. If anything stalls, the agent runs `status <id>` (full step ladder + spend) and
   `resume <id>`; on completion or any fatal error the engine tears down on its own.

Because state is on disk and steps are idempotent, the agent can stop and resume
across turns, or even across sessions, without losing the run.

---

## Files

```
lambda_lab/
  api.py         Lambda Cloud REST client (stdlib only)
  costs.py       price table + cost math + budget guard
  engine.py      the resumable step state-machine
  steps.py       idempotent steps (ssh, fs, launch, bootstrap, job, teardown)
  ssh.py         ssh/rsync + detached-job helpers (survive disconnects)
  pipelines.py   declarative pipelines (cheapest GPU that fits each task)
  run.py         CLI: start / resume / status / teardown / plan / costs / types
  bootstrap.sh   arch-aware remote installer (handles GH200 aarch64)
  jobs/          remote launchers: flux_lora, sdxl_lora, video_lora, serve, batch_infer
  configs/       example ai-toolkit / diffusion-pipe configs + dataset guide
```

## Responsible use

Respect each model's license (FLUX.1-dev is **non-commercial**; SDXL and Wan are
more permissive — read them). Only train on data you have rights to; don't model a
real person without consent; don't generate illegal content. Removing a hosted
API's content filter by self-hosting open weights puts the responsibility for
lawful, ethical use entirely on you.
