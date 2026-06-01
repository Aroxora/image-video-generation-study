"""
The concrete, idempotent steps a GPU run is built from. Each takes a Context and
returns a small dict that is persisted into the step record. Every step is safe
to re-run: that is what makes `resume` work.
"""

from __future__ import annotations

import os
from pathlib import Path

from .costs import Cost
from .ssh import rsync

HERE = Path(__file__).resolve().parent
REMOTE_HOME = "/home/ubuntu"
REMOTE_KIT = f"{REMOTE_HOME}/lambda_lab"   # where we stage bootstrap.sh + jobs/


# ---------------------------------------------------------------- ssh key ----
def ensure_ssh_key(ctx) -> dict:
    """Register the local public key with Lambda (idempotent) so launch can use it."""
    name = ctx.params.get("ssh_key_name") or "lambda_lab"
    pub_path = ctx.params.get("ssh_pubkey_path") or os.path.expanduser("~/.ssh/id_ed25519.pub")
    pub = Path(pub_path).read_text().strip()
    ctx.api.ensure_ssh_key(name, pub)
    ctx.set(ssh_key_name=name)
    ctx.log(f"ssh key ready: {name}")
    return {"ssh_key_name": name}


# -------------------------------------------------------------- filesystem ----
def ensure_filesystem(ctx) -> dict:
    """Verify the named persistent filesystem exists (it caches weights/datasets
    across runs so you never re-download 30 GB). Creation is region-bound and is
    done once in the dashboard; we only attach it here."""
    name = ctx.params.get("filesystem")
    if not name:
        ctx.log("no filesystem requested — using ephemeral instance SSD only")
        return {"filesystem": None}
    fs = ctx.api.find_filesystem(name)
    if not fs:
        raise RuntimeError(
            f"filesystem {name!r} not found. Create it once at "
            f"https://cloud.lambda.ai/file-systems (pick a region with GPU capacity), "
            f"then re-run. It will mount at /home/ubuntu/{name} or /lambda/<name>."
        )
    ctx.set(filesystem=name)
    ctx.log(f"filesystem ready: {name}")
    return {"filesystem": name}


# ------------------------------------------------------------------ launch ----
def launch(ctx) -> dict:
    """Launch the instance — but reuse one already tagged with this run id, so a
    resumed run never double-spends. Records the instance id immediately."""
    run_id = ctx.state.run_id
    existing = ctx.api.find_instance_by_name(run_id)
    if existing:
        ctx.set(instance_id=existing["id"], ip=existing.get("ip"))
        ctx.log(f"reusing instance {existing['id']} (status={existing.get('status')})")
        return {"instance_id": existing["id"], "reused": True}

    itype = ctx.params["instance_type"]
    region = ctx.params.get("region")
    if not region:
        regions = ctx.api.regions_with_capacity(itype)
        if not regions:
            raise RuntimeError(f"no region currently has capacity for {itype}; try another type or wait")
        region = regions[0]
        ctx.log(f"auto-selected region with capacity: {region}")

    key = ctx.get("ssh_key_name")
    fs = [ctx.get("filesystem")] if ctx.get("filesystem") else None
    ids = ctx.api.launch(itype, region, [key], name=run_id, file_system_names=fs)
    if not ids:
        raise RuntimeError("launch returned no instance id")
    ctx.set(instance_id=ids[0], region=region)
    ctx.log(f"launched {ids[0]} ({itype} @ {region})")
    return {"instance_id": ids[0], "region": region, "reused": False}


def wait_active(ctx) -> dict:
    """Block until active + SSH-reachable; start the cost clock the moment it's live."""
    iid = ctx.get("instance_id")
    inst = ctx.api.wait_until_active(iid, timeout_s=float(ctx.params.get("launch_timeout_s", 1200)))
    ctx.set(ip=inst["ip"])
    cost = ctx.cost
    cost.start()
    ctx.save_cost(cost)
    ctx.log(f"active at {inst['ip']} — meter started at ${cost.usd_hr}/hr, budget ${cost.budget_usd}")
    r = ctx.remote()
    if not r.wait_for_ssh(timeout_s=float(ctx.params.get("ssh_timeout_s", 600))):
        raise RuntimeError("instance active but SSH never came up")
    ctx.log("ssh reachable")
    return {"ip": inst["ip"]}


# --------------------------------------------------------------- bootstrap ----
def bootstrap(ctx) -> dict:
    """Stage the kit and run the arch-aware installer once (detached, budget-aware).
    A remote sentinel makes re-runs return in seconds."""
    r = ctx.remote()
    # quick idempotency check
    done = r.capture(f"test -f {REMOTE_KIT}/.bootstrap.ok && echo yes || echo no").strip()
    if done == "yes" and not ctx.params.get("force_bootstrap"):
        ctx.log("bootstrap already complete (sentinel present) — skipping")
        return {"skipped": True}

    r.run(f"mkdir -p {REMOTE_KIT}", stream=False)
    ip = ctx.get("ip")
    rsync(str(HERE) + "/", f"{REMOTE_KIT}/", up=True, ip=ip,
          user=ctx.params.get("ssh_user", "ubuntu"), key_path=ctx.params.get("ssh_key_path"),
          excludes=("__pycache__", "*.pyc"))

    stack = ctx.params.get("stack", "diffusers")    # diffusers | comfyui | ai-toolkit | diffusion-pipe
    cmd = f"bash {REMOTE_KIT}/bootstrap.sh {stack} && touch {REMOTE_KIT}/.bootstrap.ok"
    r.start_detached(cmd, tag="bootstrap")
    ctx.log("bootstrap running (installing CUDA PyTorch + stack) …")

    def on_poll(status, log_tail):
        cost = ctx.cost
        ctx.save_cost(cost)
        ctx.log(f"  bootstrap … spent ${cost.spent_usd()} · {log_tail.splitlines()[-1] if log_tail.strip() else ''}")
        ctx.check_budget()

    code = r.wait_for_job(tag="bootstrap", poll_s=float(ctx.params.get("poll_s", 30)), on_poll=on_poll)
    if code != 0:
        raise RuntimeError(f"bootstrap failed (exit {code}); see {REMOTE_KIT}/bootstrap.log on the box")
    ctx.log("bootstrap complete")
    return {"stack": stack}


# ----------------------------------------------------------------- sync up ----
def sync_up(ctx) -> dict:
    """Push configs + dataset to the instance (skipped if nothing to send)."""
    ip = ctx.get("ip")
    sent = []
    for local, remote in ctx.params.get("upload", []):
        lpath = os.path.expanduser(local)
        if not os.path.exists(lpath):
            ctx.log(f"  upload skip (missing): {lpath}")
            continue
        ctx.remote().run(f"mkdir -p {os.path.dirname(remote) or '.'}", stream=False)
        rsync(lpath, remote, up=True, ip=ip, user=ctx.params.get("ssh_user", "ubuntu"),
              key_path=ctx.params.get("ssh_key_path"))
        sent.append(remote)
        ctx.log(f"  uploaded {lpath} -> {remote}")
    return {"uploaded": sent}


# ------------------------------------------------------------------- job ------
def run_job(ctx) -> dict:
    """Run the main long-horizon job detached; poll with the budget guard active.
    On resume this re-attaches to a still-running job via its sentinel files."""
    r = ctx.remote()
    command = ctx.params["job_command"]
    st = r.job_status(tag="job")
    if st["state"] == "done":
        ctx.log(f"job already finished (exit {st['code']}) — not restarting")
        if st["code"] != 0:
            raise RuntimeError(f"prior job exited {st['code']}; inspect {ctx.get('ip')}:~/.lambda_lab_job/job.log")
        return {"reattached": True, "code": st["code"]}
    if st["state"] != "running":
        ctx.log("starting job (detached, survives disconnects) …")
        r.start_detached(command, tag="job")
    else:
        ctx.log("re-attaching to job already running on the instance …")

    def on_poll(status, log_tail):
        cost = ctx.cost
        ctx.save_cost(cost)
        last = log_tail.strip().splitlines()[-1] if log_tail.strip() else "(no output yet)"
        ctx.log(f"  job … ${cost.spent_usd()}/${cost.budget_usd} · {last}")
        ctx.check_budget()

    code = r.wait_for_job(tag="job", poll_s=float(ctx.params.get("poll_s", 30)), on_poll=on_poll)
    if code != 0:
        raise RuntimeError(f"job exited {code}; full log at {ctx.get('ip')}:~/.lambda_lab_job/job.log")
    ctx.log("job finished cleanly")
    return {"code": 0}


# ------------------------------------------------------------------ serve ----
def serve_job(ctx) -> dict:
    """Start a long-lived server (e.g. ComfyUI) detached and return immediately,
    printing the SSH tunnel command. The instance stays up (keep_alive) until you
    run `python -m lambda_lab.run teardown <run_id>` — so DON'T forget it."""
    r = ctx.remote()
    st = r.job_status(tag="serve")
    if st["state"] != "running":
        r.start_detached(ctx.params["job_command"], tag="serve")
        ctx.log("server starting (detached) …")
    ip = ctx.get("ip")
    port = int(ctx.params.get("port", 8188))
    key = ctx.params.get("ssh_key_path") or "~/.ssh/id_ed25519"
    ctx.log(f"server up. Tunnel it locally:\n"
            f"    ssh -i {key} -N -L {port}:localhost:{port} ubuntu@{ip}\n"
            f"  then open http://localhost:{port}\n"
            f"  TEAR DOWN when done:  python -m lambda_lab.run teardown {ctx.state.run_id}")
    cost = ctx.cost
    ctx.save_cost(cost)
    return {"serving": True, "ip": ip, "port": port}


# --------------------------------------------------------------- sync down ----
def sync_down(ctx) -> dict:
    """Pull outputs/checkpoints back to ./outputs/<run_id> before teardown."""
    ip = ctx.get("ip")
    local_root = Path(ctx.params.get("download_dir", f"outputs/{ctx.state.run_id}"))
    local_root.mkdir(parents=True, exist_ok=True)
    pulled = []
    for remote in ctx.params.get("download", []):
        rsync(remote, str(local_root) + "/", up=False, ip=ip,
              user=ctx.params.get("ssh_user", "ubuntu"), key_path=ctx.params.get("ssh_key_path"))
        pulled.append(remote)
        ctx.log(f"  downloaded {remote} -> {local_root}/")
    return {"downloaded": pulled, "local": str(local_root)}


# ------------------------------------------------------------------ teardown --
def teardown(ctx) -> dict:
    """Terminate the instance (the only step that stops the meter). Idempotent:
    only ever touches the instance this run owns; the filesystem is left intact.

    NOTE: this always terminates when called. `keep_alive` keeps a serve pipeline
    up by simply NOT putting a teardown step in its success path — it must never
    block an EXPLICIT teardown (CLI / autoscaler / failure-path), or a serving box
    could be left billing forever."""
    iid = ctx.get("instance_id")
    # prefer matching by run-id tag, fall back to recorded id
    inst = ctx.api.find_instance_by_name(ctx.state.run_id)
    target = inst["id"] if inst else iid
    if not target:
        ctx.log("nothing to terminate")
        return {"terminated": False}
    ctx.api.terminate([target])
    cost = ctx.cost
    ctx.state.status = "torn_down"
    ctx.save_cost(cost)
    ctx.log(f"terminated {target} — final spend ${cost.spent_usd()}. Meter stopped.")
    return {"terminated": True, "instance_id": target, "final_usd": cost.spent_usd()}
