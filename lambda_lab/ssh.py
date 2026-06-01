"""
Thin wrappers over the system `ssh` / `rsync` so remote work needs no Python deps
on either side. The important piece for long-horizon runs is `start_detached`:
it launches a job inside its own `setsid` session that survives the SSH
connection dropping, tees output to a logfile, and writes the exit code to a
sentinel file on completion. A resumed run then distinguishes "still running"
from "done(code)" purely by inspecting those files — no babysitting required.
"""

from __future__ import annotations

import shlex
import subprocess
import time

SSH_OPTS = [
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "ServerAliveInterval=20",
    "-o", "ServerAliveCountMax=3",
    "-o", "ConnectTimeout=15",
]


class Remote:
    def __init__(self, ip: str, user: str = "ubuntu", key_path: str | None = None):
        self.ip = ip
        self.user = user
        self.key_path = key_path

    def _base(self) -> list[str]:
        cmd = ["ssh", *SSH_OPTS]
        if self.key_path:
            cmd += ["-i", self.key_path]
        cmd.append(f"{self.user}@{self.ip}")
        return cmd

    # ---- connectivity -----------------------------------------------------
    def wait_for_ssh(self, *, timeout_s: float = 600, poll_s: float = 8) -> bool:
        """Poll until `ssh true` succeeds (the box boots a bit after it goes active)."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            r = subprocess.run(self._base() + ["true"], capture_output=True, timeout=30)
            if r.returncode == 0:
                return True
            time.sleep(poll_s)
        return False

    # ---- synchronous command ----------------------------------------------
    def run(self, command: str, *, check: bool = True, stream: bool = True) -> subprocess.CompletedProcess:
        """Run a command and (by default) stream its output to this terminal."""
        full = self._base() + [command]
        if stream:
            proc = subprocess.run(full)
        else:
            proc = subprocess.run(full, capture_output=True, text=True)
        if check and proc.returncode != 0:
            tail = "" if stream else (proc.stderr or "")[-2000:]
            raise RuntimeError(f"remote command failed ({proc.returncode}): {command}\n{tail}")
        return proc

    def capture(self, command: str) -> str:
        return self.run(command, check=False, stream=False).stdout or ""

    # ---- detached long-running job ----------------------------------------
    def start_detached(self, command: str, *, job_dir: str = "~/.lambda_lab_job", tag: str = "job") -> None:
        """Start `command` so it outlives the SSH session.

        Writes: <job_dir>/<tag>.log (stdout+stderr), <tag>.pid, and on exit
        <tag>.done containing the integer exit code. Idempotent-friendly: callers
        should check `job_status` first and skip if already running/finished.
        """
        d = job_dir
        script = (
            f"mkdir -p {d} && "
            f"rm -f {d}/{tag}.done && "
            f"setsid bash -lc {shlex.quote(command + f'; echo $? > {d}/{tag}.done')} "
            f"> {d}/{tag}.log 2>&1 < /dev/null & "
            f"echo $! > {d}/{tag}.pid"
        )
        self.run(script, stream=False)

    def job_status(self, *, job_dir: str = "~/.lambda_lab_job", tag: str = "job") -> dict:
        """Return {state: 'absent'|'running'|'done', code: int|None}."""
        d = job_dir
        out = self.capture(
            f"if [ -f {d}/{tag}.done ]; then echo done $(cat {d}/{tag}.done); "
            f"elif [ -f {d}/{tag}.pid ] && kill -0 $(cat {d}/{tag}.pid) 2>/dev/null; then echo running; "
            f"else echo absent; fi"
        ).strip()
        parts = out.split()
        if not parts:
            return {"state": "absent", "code": None}
        if parts[0] == "done":
            return {"state": "done", "code": int(parts[1]) if len(parts) > 1 and parts[1].lstrip('-').isdigit() else None}
        return {"state": parts[0], "code": None}

    def tail_log(self, *, job_dir: str = "~/.lambda_lab_job", tag: str = "job", lines: int = 40) -> str:
        return self.capture(f"tail -n {lines} {job_dir}/{tag}.log 2>/dev/null")

    def wait_for_job(
        self, *, job_dir: str = "~/.lambda_lab_job", tag: str = "job",
        poll_s: float = 30, on_poll=None,
    ) -> int:
        """Block until the detached job finishes; return its exit code.

        `on_poll(status, log_tail)` is called each cycle — the engine uses it to
        check the budget guard and stream progress. Survives SSH hiccups because
        it re-checks the sentinel files each time rather than holding a channel.
        """
        while True:
            st = self.job_status(job_dir=job_dir, tag=tag)
            if on_poll:
                on_poll(st, self.tail_log(job_dir=job_dir, tag=tag))
            if st["state"] == "done":
                return st["code"] if st["code"] is not None else -1
            if st["state"] == "absent":
                raise RuntimeError(f"detached job {tag!r} vanished without a sentinel — check {job_dir}/{tag}.log")
            time.sleep(poll_s)


def rsync(src: str, dst: str, *, key_path: str | None = None, up: bool, ip: str, user: str = "ubuntu", excludes: tuple[str, ...] = ()) -> None:
    """rsync a directory up to or down from the instance.

    up=True : local `src` -> remote `dst`.  up=False : remote `src` -> local `dst`.
    """
    ssh_cmd = "ssh " + " ".join(shlex.quote(o) for o in SSH_OPTS)
    if key_path:
        ssh_cmd += f" -i {shlex.quote(key_path)}"
    remote = f"{user}@{ip}:"
    a = src if up else f"{remote}{src}"
    b = f"{remote}{dst}" if up else dst
    cmd = ["rsync", "-az", "--info=progress2", "-e", ssh_cmd]
    for ex in excludes:
        cmd += ["--exclude", ex]
    cmd += [a, b]
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"rsync failed ({proc.returncode}): {' '.join(cmd)}")
