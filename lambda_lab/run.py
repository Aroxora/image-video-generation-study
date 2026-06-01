"""
CLI for the orchestrator.  `python -m lambda_lab.run <command>`

  start <pipeline>    launch + run a pipeline end to end (auto-teardown on finish)
  resume <run_id>     continue an interrupted run (skips completed steps)
  status [run_id]     show a run's step ladder + spend (or list all runs)
  list                list run ids
  teardown <run_id>   terminate the run's instance now (stops the meter)
  plan <pipeline>     dry run: print steps, params, and a cost estimate (no API key)
  costs               print the price table + a cheapest-that-fits matrix
  types               live `GET /instance-types` (needs LAMBDA_API_KEY)
  instances           live `GET /instances`

Examples
  export LAMBDA_API_KEY=...                       # from https://cloud.lambda.ai/api-keys
  python -m lambda_lab.run plan train-flux-lora
  python -m lambda_lab.run start train-flux-lora --budget 8 --instance-type gpu_1x_a100_sxm4
  python -m lambda_lab.run resume flux-lora-0c1f      # after your laptop slept
  python -m lambda_lab.run teardown flux-lora-0c1f    # belt-and-suspenders
"""

from __future__ import annotations

import argparse
import binascii
import json
import os
import sys
import time

from .api import LambdaCloud, LambdaCloudError
from .costs import PRICES, cheapest_that_fits, estimate, price_per_hr
from .engine import Engine, RunState, Context
from .pipelines import PIPELINES, build
from . import steps as S


def _coerce(v: str):
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    if v and v[0] in "[{":
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            pass
    return v


def _overrides(args) -> dict:
    o: dict = {}
    if args.config:
        o.update(json.loads(open(args.config).read()))
    for pair in args.set or []:
        if "=" not in pair:
            sys.exit(f"--set expects key=value, got {pair!r}")
        k, val = pair.split("=", 1)
        o[k] = _coerce(val)
    for k in ("budget", "instance_type", "region", "filesystem", "stack", "run_id"):
        v = getattr(args, k, None)
        if v is not None:
            o["budget_usd" if k == "budget" else k] = v
    return o


def _new_run_id(pipeline: str) -> str:
    short = pipeline.replace("train-", "").replace("-lora", "").replace("_", "-")
    suffix = binascii.hexlify(os.urandom(2)).decode()
    return f"{short}-{suffix}"


# ---- commands -------------------------------------------------------------
def cmd_start(args):
    overrides = _overrides(args)
    run_id = overrides.pop("run_id", None) or _new_run_id(args.pipeline)
    steps, params = build(args.pipeline, overrides)
    state = RunState(run_id, args.pipeline, params)
    state.save()
    print(f"▶ run {run_id}  ({args.pipeline} on {params['instance_type']}, budget ${params.get('budget_usd')})")
    Engine(state).execute(steps)
    cmd_status(argparse.Namespace(run_id=run_id))


def cmd_resume(args):
    state = RunState.load(args.run_id)
    steps, params = build(state.pipeline, state.params)
    print(f"↻ resuming {args.run_id} ({state.pipeline}) — completed: "
          f"{[k for k,v in state.steps.items() if v.status=='done']}")
    Engine(state).execute(steps)
    cmd_status(argparse.Namespace(run_id=args.run_id))


def cmd_status(args):
    if not args.run_id:
        runs = RunState.list_runs()
        if not runs:
            print("no runs yet.")
            return
        for rid in runs:
            s = RunState.load(rid)
            print(f"  {rid:22s} {s.pipeline:18s} {s.status}")
        return
    s = RunState.load(args.run_id)
    print(f"\nrun {s.run_id}  [{s.status}]  pipeline={s.pipeline}")
    print(f"  instance: {s.ctx.get('instance_type')}  id={s.ctx.get('instance_id')}  ip={s.ctx.get('ip')}")
    if s.ctx.get("cost"):
        c = s.ctx["cost"]
        print(f"  spend: ${c.get('spent_usd')} / ${c.get('budget_usd')}  (@ ${c.get('usd_hr')}/hr)")
    print("  steps:")
    for name, rec in s.steps.items():
        mark = {"done": "✓", "failed": "✗", "running": "▶", "pending": "·", "skipped": "—"}.get(rec.status, "?")
        extra = f"  ! {rec.error.splitlines()[0]}" if rec.error else ""
        print(f"    {mark} {name:16s} {rec.status:8s} (tries {rec.attempts}){extra}")
    print()


def cmd_list(args):
    cmd_status(argparse.Namespace(run_id=None))


def cmd_teardown(args):
    state = RunState.load(args.run_id)
    ctx = Context(Engine(state), state)
    S.teardown(ctx)


def cmd_plan(args):
    steps, params = build(args.pipeline, _overrides(args))
    it = params["instance_type"]
    print(f"\npipeline: {args.pipeline}")
    print(f"instance: {it}  ({PRICES[it]['label']}, ${price_per_hr(it)}/hr, {PRICES[it]['arch']})")
    print(f"budget:   ${params.get('budget_usd')}")
    print("steps:")
    for st in steps:
        print(f"    - {st.name}")
    print("job_command:")
    print(f"    {params.get('job_command')}")
    for hrs in (0.5, 1, 2):
        print(f"estimate @ {hrs:>3} hr: ${estimate(it, hrs)}")
    print()


def cmd_costs(args):
    print("\nLambda on-demand lineup (per box / per hr):")
    for t, v in sorted(PRICES.items(), key=lambda kv: kv[1]["usd_hr"]):
        print(f"  {t:18s} {v['label']:22s} {v['vram_gb']:>3} GB  {v['arch']:7s} ${v['usd_hr']}/hr")
    print("\ncheapest instance that fits a given VRAM need:")
    for need in (12, 16, 24, 40, 80, 96):
        fits = cheapest_that_fits(need)
        best = fits[0] if fits else "—"
        print(f"  >= {need:>2} GB : {best}")
    print()


def cmd_types(args):
    data = LambdaCloud().instance_types()
    for name, info in sorted(data.items()):
        regions = [r.get("name") for r in info.get("regions_with_capacity_available", [])]
        price = info.get("instance_type", {}).get("price_cents_per_hour")
        pstr = f"${price/100:.2f}/hr" if price else "?"
        print(f"  {name:20s} {pstr:10s} capacity in: {regions or 'none right now'}")


def cmd_instances(args):
    for inst in LambdaCloud().list_instances():
        print(f"  {inst.get('id')}  {inst.get('name')}  {inst.get('instance_type',{}).get('name')}  "
              f"{inst.get('status')}  {inst.get('ip')}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="lambda_lab.run", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--budget", type=float, help="hard USD cap; auto-teardown when reached")
        sp.add_argument("--instance-type", dest="instance_type", help=f"one of {sorted(PRICES)}")
        sp.add_argument("--region", help="region name (default: first with capacity)")
        sp.add_argument("--filesystem", help="persistent filesystem name to attach")
        sp.add_argument("--stack", help="diffusers | comfyui | ai-toolkit | diffusion-pipe")
        sp.add_argument("--run-id", dest="run_id", help="explicit run id")
        sp.add_argument("--set", action="append", help="extra param: key=value (repeatable)")
        sp.add_argument("--config", help="JSON file of param overrides")

    sp = sub.add_parser("start", help="launch + run a pipeline"); sp.add_argument("pipeline", choices=sorted(PIPELINES)); add_common(sp); sp.set_defaults(fn=cmd_start)
    sp = sub.add_parser("resume", help="continue an interrupted run"); sp.add_argument("run_id"); sp.set_defaults(fn=cmd_resume)
    sp = sub.add_parser("status", help="show a run"); sp.add_argument("run_id", nargs="?"); sp.set_defaults(fn=cmd_status)
    sp = sub.add_parser("list", help="list runs"); sp.set_defaults(fn=cmd_list)
    sp = sub.add_parser("teardown", help="terminate a run's instance"); sp.add_argument("run_id"); sp.set_defaults(fn=cmd_teardown)
    sp = sub.add_parser("plan", help="dry run (no API key)"); sp.add_argument("pipeline", choices=sorted(PIPELINES)); add_common(sp); sp.set_defaults(fn=cmd_plan)
    sp = sub.add_parser("costs", help="price table + fit matrix"); sp.set_defaults(fn=cmd_costs)
    sp = sub.add_parser("types", help="live instance types"); sp.set_defaults(fn=cmd_types)
    sp = sub.add_parser("instances", help="live running instances"); sp.set_defaults(fn=cmd_instances)

    args = p.parse_args(argv)
    try:
        args.fn(args)
    except LambdaCloudError as e:
        sys.exit(f"Lambda API error: {e}")
    except FileNotFoundError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
