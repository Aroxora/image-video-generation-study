"""
lambda_lab — an agentic, resumable orchestrator for renting Lambda Cloud GPUs to
train and serve open-weight image/video generation models, economically.

Design goals (why this exists):
  * Long-horizon: a "run" is a persisted state machine. It can be interrupted at
    any point (laptop sleeps, SSH drops, a training job runs for six hours) and
    resumed with `python -m lambda_lab.run resume <run_id>` — every step is
    idempotent, so re-entering is always safe.
  * Cheap-by-default: a hard USD budget guard auto-terminates the instance before
    it can overrun, and teardown is the default exit path. A rented GPU you forgot
    to kill is the only way to actually lose money here.
  * Agent-drivable: each step is also a standalone CLI subcommand, so a coding
    agent (or you) can drive the loop one verifiable step at a time, or fire the
    whole pipeline and walk away.

Nothing here needs third-party Python packages — the Lambda Cloud API client is
built on the standard library, and remote work goes over your system `ssh`/`rsync`.

See lambda_lab/README.md for the full guide.
"""

from .costs import PRICES, Cost, cheapest_that_fits
from .api import LambdaCloud, LambdaCloudError

__all__ = ["PRICES", "Cost", "cheapest_that_fits", "LambdaCloud", "LambdaCloudError"]

__version__ = "0.1.0"
