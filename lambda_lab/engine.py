"""
The long-horizon engine: a run is a persisted state machine of idempotent steps.

Why a state machine instead of a shell script? Renting a GPU is a multi-step,
hours-long, money-burning process that *will* get interrupted — SSH drops, your
laptop sleeps, a step flakes. Each Step here is:
  * idempotent  — re-running it is always safe (launch reuses an instance tagged
    with the run id; bootstrap is apt/pip-idempotent; jobs check a remote sentinel),
  * checkpointed — state is flushed to disk after every transition, so the very
    next thing after `launch` records the instance id (no orphaned GPUs), and
  * resumable    — `resume <run_id>` skips completed steps and re-enters the rest.

A Cost watchdog runs between steps and on every job poll; if spend reaches the
budget the engine forces teardown. Teardown is also the default success path.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .api import LambdaCloud
from .costs import Cost


RUN_DIR = Path(os.environ.get("LAMBDA_LAB_HOME", ".lambda_lab")) / "runs"


def _now() -> float:
    return time.time()


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


@dataclass
class StepRecord:
    name: str
    status: str = "pending"          # pending | running | done | failed | skipped
    attempts: int = 0
    started: float | None = None
    ended: float | None = None
    output: dict = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "StepRecord":
        return cls(**{k: d.get(k) for k in cls.__annotations__})


class RunState:
    """The persisted record of one orchestration run."""

    def __init__(self, run_id: str, pipeline: str, params: dict):
        self.run_id = run_id
        self.pipeline = pipeline
        self.params = params
        self.status = "pending"      # pending | running | done | failed | torn_down
        self.created = _now()
        self.updated = _now()
        self.steps: dict[str, StepRecord] = {}
        # mutable shared context, persisted so resume can pick it back up:
        self.ctx: dict = {
            "instance_id": None,
            "ip": None,
            "ssh_key_name": params.get("ssh_key_name"),
            "filesystem": params.get("filesystem"),
            "instance_type": params.get("instance_type"),
            "region": params.get("region"),
            "cost": None,
            "log": [],
        }

    # ---- persistence ------------------------------------------------------
    @property
    def path(self) -> Path:
        return RUN_DIR / f"{self.run_id}.json"

    def save(self) -> None:
        self.updated = _now()
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "run_id": self.run_id, "pipeline": self.pipeline, "params": self.params,
            "status": self.status, "created": self.created, "updated": self.updated,
            "steps": {k: v.to_dict() for k, v in self.steps.items()}, "ctx": self.ctx,
        }
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self.path)   # atomic — a crash mid-write never corrupts state

    @classmethod
    def load(cls, run_id: str) -> "RunState":
        p = RUN_DIR / f"{run_id}.json"
        if not p.exists():
            raise FileNotFoundError(f"no such run: {run_id} ({p})")
        d = json.loads(p.read_text())
        s = cls(d["run_id"], d["pipeline"], d.get("params", {}))
        s.status = d.get("status", "pending")
        s.created = d.get("created", _now())
        s.updated = d.get("updated", _now())
        s.steps = {k: StepRecord.from_dict(v) for k, v in d.get("steps", {}).items()}
        s.ctx = d.get("ctx", s.ctx)
        return s

    @staticmethod
    def list_runs() -> list[str]:
        if not RUN_DIR.exists():
            return []
        return sorted(p.stem for p in RUN_DIR.glob("*.json"))

    def log(self, msg: str) -> None:
        line = f"[{_ts()}] {msg}"
        print(line, flush=True)
        self.ctx.setdefault("log", []).append(line)


@dataclass
class Step:
    name: str
    fn: Callable[["Context"], dict]
    max_attempts: int = 3
    backoff_s: float = 10.0
    # If this step fails after all retries, should the engine force teardown?
    teardown_on_fail: bool = False


class Context:
    """Handed to every step. Bundles the API client, shared run context, cost
    guard, and lazy SSH access, plus helpers to persist as the step runs."""

    def __init__(self, engine: "Engine", state: RunState):
        self.engine = engine
        self.state = state
        self.api: LambdaCloud = engine.api
        self.params = state.params
        self._remote = None

    # shared, persisted scratchpad ----------------------------------------
    def get(self, key: str, default=None):
        return self.state.ctx.get(key, default)

    def set(self, **kw) -> None:
        self.state.ctx.update(kw)
        self.state.save()

    def log(self, msg: str) -> None:
        self.state.log(msg)

    # cost -----------------------------------------------------------------
    @property
    def cost(self) -> Cost:
        c = self.state.ctx.get("cost")
        if c is None:
            cost = Cost(
                instance_type=self.params["instance_type"],
                budget_usd=float(self.params.get("budget_usd", 10.0)),
            )
            self.state.ctx["cost"] = cost.to_dict()
            return cost
        return Cost.from_dict(c)

    def save_cost(self, cost: Cost) -> None:
        self.state.ctx["cost"] = cost.to_dict()
        self.state.save()

    def check_budget(self) -> None:
        cost = self.cost
        if cost.started and cost.over_budget():
            raise BudgetExceeded(
                f"budget ${cost.budget_usd} reached (spent ${cost.spent_usd()}); forcing teardown"
            )

    # ssh ------------------------------------------------------------------
    def remote(self):
        from .ssh import Remote
        if self._remote is None:
            ip = self.get("ip")
            if not ip:
                raise RuntimeError("no instance ip yet")
            self._remote = Remote(ip, user=self.params.get("ssh_user", "ubuntu"),
                                  key_path=self.params.get("ssh_key_path"))
        return self._remote


class BudgetExceeded(RuntimeError):
    pass


class Engine:
    def __init__(self, state: RunState, api: LambdaCloud | None = None):
        self.state = state
        self.api = api or LambdaCloud()

    # ---- run / resume -----------------------------------------------------
    def execute(self, steps: list[Step]) -> RunState:
        st = self.state
        st.status = "running"
        st.save()
        ctx = Context(self, st)
        teardown_step = next((s for s in steps if s.name == "teardown"), None)

        for step in steps:
            rec = st.steps.setdefault(step.name, StepRecord(step.name))
            if rec.status == "done":
                st.log(f"✓ {step.name} (cached)")
                continue

            ok = self._run_step(ctx, step, rec)
            if not ok:
                st.status = "failed"
                st.save()
                if step.teardown_on_fail and teardown_step and step.name != "teardown":
                    st.log("running teardown after fatal failure …")
                    self._run_step(ctx, teardown_step, st.steps.setdefault("teardown", StepRecord("teardown")))
                return st

        st.status = "done"
        st.save()
        st.log(f"run {st.run_id} complete — total spend ${ctx.cost.spent_usd()}")
        return st

    def _run_step(self, ctx: Context, step: Step, rec: StepRecord) -> bool:
        st = self.state
        for attempt in range(1, step.max_attempts + 1):
            rec.status = "running"
            rec.attempts = attempt
            rec.started = _now()
            rec.error = ""
            st.save()
            st.log(f"▶ {step.name} (attempt {attempt}/{step.max_attempts})")
            try:
                # budget gate before any step that could keep the meter running
                ctx.check_budget()
                out = step.fn(ctx) or {}
                rec.output = out if isinstance(out, dict) else {"result": out}
                rec.status = "done"
                rec.ended = _now()
                st.save()
                st.log(f"✓ {step.name}")
                return True
            except BudgetExceeded as e:
                rec.status = "failed"
                rec.error = str(e)
                rec.ended = _now()
                st.save()
                st.log(f"✗ {step.name}: {e}")
                # force teardown regardless of step config
                td = next((s for s in [] if s.name == "teardown"), None)
                from .steps import teardown as do_teardown
                try:
                    do_teardown(ctx)
                except Exception as te:  # noqa: BLE001
                    st.log(f"teardown error: {te}")
                return False
            except Exception as e:  # noqa: BLE001
                rec.error = f"{e}\n{traceback.format_exc()}"
                rec.ended = _now()
                st.save()
                st.log(f"✗ {step.name} attempt {attempt}: {e}")
                if attempt < step.max_attempts:
                    time.sleep(step.backoff_s * attempt)
                else:
                    rec.status = "failed"
                    st.save()
                    return False
        return False
