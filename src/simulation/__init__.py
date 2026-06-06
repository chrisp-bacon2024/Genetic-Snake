"""Headless game evaluation for genetic-algorithm training."""

from .headless import EvalResult, HeadlessSimulator, Scenario
from .parallel import EvalJob, evaluate_genomes_parallel, resolve_worker_count

__all__ = [
    "EvalJob",
    "EvalResult",
    "HeadlessSimulator",
    "Scenario",
    "evaluate_genomes_parallel",
    "resolve_worker_count",
]
