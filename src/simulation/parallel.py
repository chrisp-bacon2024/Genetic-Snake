"""
Parallel genome evaluation for genetic-algorithm training.

Each worker runs an independent headless simulation. The worker entry point
must stay at module level so Windows spawn can pickle it.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

import numpy as np

from evolution.genome import Genome
from models.grid import Grid
from simulation.headless import EvalResult, HeadlessSimulator, Scenario


@dataclass(frozen=True, slots=True)
class EvalJob:
    """Picklable evaluation request for one genome."""

    genes: np.ndarray
    grid_cols: int
    grid_rows: int
    max_steps: int  # -1 means no step cap (config.MAX_EVAL_STEPS is None)
    food_seeds: tuple[int, ...]
    random_runs: int
    use_random_food: bool


def resolve_worker_count(workers: int) -> int:
    """Map CLI workers (0 = auto) to a process count."""
    if workers > 0:
        return workers
    cpu_count = os.cpu_count() or 2
    return max(1, cpu_count - 1)


def build_eval_job(
    genes: np.ndarray,
    *,
    grid_cols: int,
    grid_rows: int,
    max_steps: int | None,
    food_seeds: tuple[int, ...],
    random_runs: int,
    use_random_food: bool,
) -> EvalJob:
    return EvalJob(
        genes=np.asarray(genes, dtype=np.float64),
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        max_steps=-1 if max_steps is None else max_steps,
        food_seeds=food_seeds,
        random_runs=random_runs,
        use_random_food=use_random_food,
    )


def evaluate_genome_worker(job: EvalJob) -> EvalResult:
    """Run one genome in a worker process (module-level for Windows spawn)."""
    max_steps = None if job.max_steps < 0 else job.max_steps
    simulator = HeadlessSimulator(Grid(job.grid_cols, job.grid_rows), max_steps=max_steps)
    genome = Genome(job.genes)

    if job.food_seeds:
        simulator.set_scenarios([Scenario(food_seed=seed) for seed in job.food_seeds])
        return simulator.evaluate(genome)

    if job.use_random_food:
        return simulator.evaluate(genome, runs=job.random_runs)

    raise ValueError("EvalJob requires food_seeds or random food evaluation.")


ProgressCallback = Callable[[int, int], None]


def evaluate_genomes_parallel(
    jobs: list[EvalJob],
    *,
    workers: int,
    progress_callback: ProgressCallback | None = None,
) -> list[EvalResult]:
    """Evaluate genomes serially or across a process pool."""
    total = len(jobs)
    if total == 0:
        return []

    worker_count = resolve_worker_count(workers)
    if worker_count <= 1:
        results: list[EvalResult] = []
        for index, job in enumerate(jobs):
            results.append(evaluate_genome_worker(job))
            if progress_callback is not None and (
                index == 0 or (index + 1) % 50 == 0 or index + 1 == total
            ):
                progress_callback(index + 1, total)
        return results

    results: list[EvalResult | None] = [None] * total
    completed = 0
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_index = {
            executor.submit(evaluate_genome_worker, job): index for index, job in enumerate(jobs)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()
            completed += 1
            if progress_callback is not None and (
                completed == 1 or completed % 50 == 0 or completed == total
            ):
                progress_callback(completed, total)

    if any(result is None for result in results):
        raise RuntimeError("parallel evaluation did not return all results")
    return results  # type: list[EvalResult]
