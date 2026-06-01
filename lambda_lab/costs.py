"""
Cost model for Lambda Cloud GPU rentals.

PRICES is grounded in the exact on-demand lineup shown on the account this repo was
built for (2026-06). Lambda bills on-demand by the minute while an instance exists
and stops billing the moment it is terminated; a persistent filesystem is billed
separately per GB-stored and survives termination. Update PRICES from
`python -m lambda_lab.run types` (live `GET /instance-types`) if Lambda changes them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time


# instance_type_name -> facts. The key is the Lambda Cloud API identifier used in
# `POST /instance-operations/launch`. usd_hr / vram are the per-GPU-box figures.
PRICES: dict[str, dict] = {
    "gpu_1x_gh200":     {"label": "1x GH200 (96 GB)",      "usd_hr": 2.29, "vram_gb": 96,  "arch": "arm64",  "gpus": 1},
    "gpu_1x_a10":       {"label": "1x A10 (24 GB PCIe)",   "usd_hr": 1.29, "vram_gb": 24,  "arch": "x86_64", "gpus": 1},
    "gpu_1x_a100_sxm4": {"label": "1x A100 (40 GB SXM4)",  "usd_hr": 1.99, "vram_gb": 40,  "arch": "x86_64", "gpus": 1},
    "gpu_1x_h100_pcie": {"label": "1x H100 (80 GB PCIe)",  "usd_hr": 3.29, "vram_gb": 80,  "arch": "x86_64", "gpus": 1},
    "gpu_1x_h100_sxm5": {"label": "1x H100 (80 GB SXM5)",  "usd_hr": 4.29, "vram_gb": 80,  "arch": "x86_64", "gpus": 1},
    "gpu_8x_v100":      {"label": "8x Tesla V100 (16 GB)", "usd_hr": 6.32, "vram_gb": 16,  "arch": "x86_64", "gpus": 8},
    "gpu_2x_h100_sxm5": {"label": "2x H100 (80 GB SXM5)",  "usd_hr": 8.38, "vram_gb": 80,  "arch": "x86_64", "gpus": 2},
}


def price_per_hr(instance_type: str) -> float:
    info = PRICES.get(instance_type)
    if not info:
        raise KeyError(f"unknown instance type {instance_type!r}; known: {sorted(PRICES)}")
    return float(info["usd_hr"])


def cheapest_that_fits(min_vram_gb: float, *, arch: str | None = None, gpus: int = 1) -> list[str]:
    """Return instance types whose single-GPU VRAM >= min_vram_gb, cheapest first.

    `arch` filters to 'arm64' or 'x86_64' (the GH200 is arm64 — see bootstrap.sh).
    Most diffusion training/inference uses one GPU, so we compare per-GPU VRAM.
    """
    candidates = [
        t for t, v in PRICES.items()
        if v["vram_gb"] >= min_vram_gb
        and v["gpus"] >= gpus
        and (arch is None or v["arch"] == arch)
    ]
    return sorted(candidates, key=price_per_hr)


@dataclass
class Cost:
    """Tracks accrued spend for one rented instance and enforces a hard budget.

    The budget guard is the safety net for long-horizon runs: the engine calls
    `over_budget()` on every poll and forces teardown before the bill can exceed
    `budget_usd`. `started` is a wall-clock epoch set when the instance goes active.
    """

    instance_type: str
    budget_usd: float = 10.0
    started: float | None = None
    # extra fixed costs you want counted against the budget (e.g. filesystem GB-month)
    fixed_usd: float = 0.0
    _clock: callable = field(default=time.time, repr=False)

    @property
    def usd_hr(self) -> float:
        return price_per_hr(self.instance_type)

    def start(self) -> None:
        if self.started is None:
            self.started = self._clock()

    def elapsed_hr(self) -> float:
        if self.started is None:
            return 0.0
        return max(0.0, (self._clock() - self.started) / 3600.0)

    def spent_usd(self) -> float:
        return round(self.fixed_usd + self.usd_hr * self.elapsed_hr(), 4)

    def remaining_usd(self) -> float:
        return round(self.budget_usd - self.spent_usd(), 4)

    def over_budget(self) -> bool:
        return self.spent_usd() >= self.budget_usd

    def budget_runs_out_in_min(self) -> float:
        """Minutes of runway left at the current burn rate."""
        if self.usd_hr <= 0:
            return float("inf")
        return max(0.0, self.remaining_usd() / self.usd_hr * 60.0)

    def to_dict(self) -> dict:
        return {
            "instance_type": self.instance_type,
            "budget_usd": self.budget_usd,
            "started": self.started,
            "fixed_usd": self.fixed_usd,
            "usd_hr": self.usd_hr,
            "spent_usd": self.spent_usd(),
            "remaining_usd": self.remaining_usd(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Cost":
        c = cls(
            instance_type=d["instance_type"],
            budget_usd=d.get("budget_usd", 10.0),
            started=d.get("started"),
            fixed_usd=d.get("fixed_usd", 0.0),
        )
        return c


def estimate(instance_type: str, hours: float, *, fixed_usd: float = 0.0) -> float:
    """One-line forward estimate: $/hr * hours (+ fixed)."""
    return round(price_per_hr(instance_type) * hours + fixed_usd, 2)


def cost_per_output(instance_type: str, outputs_per_hour: float) -> float:
    """$ per single output (image, or second-of-video) given a throughput.

    e.g. cost_per_output('gpu_1x_a10', 1800) -> $/image at 1800 img/hr on the A10.
    """
    if outputs_per_hour <= 0:
        return float("inf")
    return round(price_per_hr(instance_type) / outputs_per_hour, 6)
