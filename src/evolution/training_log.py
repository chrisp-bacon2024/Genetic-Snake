"""Append-only training metrics log for dashboard history on resume."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import config
from evolution.fitness import compute_fitness
from evolution.training_metrics import GenerationMetrics

TRAINING_LOG_NAME = "training_log.jsonl"


def _json_float(value: float) -> float | None:
    return None if value != value else value


def _parse_float(value: object) -> float:
    if value is None:
        return float("nan")
    return float(value)


def _fitness_missing(value: float) -> bool:
    return value != value


def estimate_fitness_from_score(score: int, grid_cols: int, grid_rows: int) -> tuple[float, float]:
    """
    Approximate best/avg fitness for dashboard history when only score was saved.

    Uses the same formula as training with a rough step count derived from score
    and board size so the curve shape matches real runs reasonably well.
    """
    board = max(grid_cols, grid_rows)
    # Modest step count so the exponential score term dominates (matches real high-score runs).
    steps = max(50, score * 15 + board)
    best_fit = compute_fitness(score, steps)
    avg_score = max(0, score // 2)
    avg_steps = max(40, steps // 3)
    avg_fit = compute_fitness(avg_score, avg_steps)
    return best_fit, avg_fit


def enrich_metrics_fitness(metrics: GenerationMetrics) -> GenerationMetrics:
    """Fill missing fitness fields from score so dashboard charts can plot history."""
    if not _fitness_missing(metrics.best_fitness) and not _fitness_missing(metrics.avg_fitness):
        return metrics
    best_fit, avg_fit = estimate_fitness_from_score(
        metrics.best_score, metrics.grid_cols, metrics.grid_rows
    )
    return GenerationMetrics(
        generation=metrics.generation,
        grid_cols=metrics.grid_cols,
        grid_rows=metrics.grid_rows,
        best_fitness=best_fit if _fitness_missing(metrics.best_fitness) else metrics.best_fitness,
        avg_fitness=avg_fit if _fitness_missing(metrics.avg_fitness) else metrics.avg_fitness,
        best_score=metrics.best_score,
        max_score=metrics.max_score,
        avg_max10=metrics.avg_max10,
        best_ever_score=metrics.best_ever_score,
        death_cause=metrics.death_cause,
    )


def metrics_to_dict(metrics: GenerationMetrics) -> dict:
    return {
        "generation": metrics.generation,
        "grid_cols": metrics.grid_cols,
        "grid_rows": metrics.grid_rows,
        "best_fitness": _json_float(metrics.best_fitness),
        "avg_fitness": _json_float(metrics.avg_fitness),
        "best_score": metrics.best_score,
        "max_score": metrics.max_score,
        "avg_max10": metrics.avg_max10,
        "best_ever_score": metrics.best_ever_score,
        "death_cause": metrics.death_cause,
    }


def metrics_from_dict(data: dict) -> GenerationMetrics:
    return GenerationMetrics(
        generation=int(data["generation"]),
        grid_cols=int(data["grid_cols"]),
        grid_rows=int(data["grid_rows"]),
        best_fitness=_parse_float(data.get("best_fitness")),
        avg_fitness=_parse_float(data.get("avg_fitness")),
        best_score=int(data["best_score"]),
        max_score=int(data["max_score"]),
        avg_max10=float(data["avg_max10"]),
        best_ever_score=int(data["best_ever_score"]),
        death_cause=str(data["death_cause"]),
    )


def append_training_log(path: Path, metrics: GenerationMetrics) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics_to_dict(metrics)) + "\n")


def load_training_log(path: Path) -> list[GenerationMetrics]:
    metrics: list[GenerationMetrics] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            metrics.append(metrics_from_dict(json.loads(line)))
    metrics.sort(key=lambda m: m.generation)
    return metrics


def _parse_generation(path: Path) -> int:
    return int(path.stem.split("_", 1)[1])


def backfill_metrics_from_gen_npz(replays_dir: Path) -> list[GenerationMetrics]:
    """
    Reconstruct partial metrics from saved gen_XXXX.npz files when JSONL is incomplete.

    Fitness fields are estimated from score when not stored in the checkpoint npz.
    """
    files = sorted(replays_dir.glob("gen_*.npz"), key=_parse_generation)
    if not files:
        return []

    metrics: list[GenerationMetrics] = []
    best_ever = 0
    recent_scores: list[int] = []

    for path in files:
        data = np.load(path)
        generation = int(data["generation"])
        score = int(data["score"])
        grid_cols = int(data["grid_cols"]) if "grid_cols" in data else config.GRID_COLS
        grid_rows = int(data["grid_rows"]) if "grid_rows" in data else config.GRID_ROWS
        death_cause = str(data["death_cause"]) if "death_cause" in data else "wall"

        best_ever = max(best_ever, score)
        recent_scores.append(score)
        if len(recent_scores) > 10:
            recent_scores.pop(0)
        avg_max10 = sum(recent_scores) / len(recent_scores)
        best_fit, avg_fit = estimate_fitness_from_score(score, grid_cols, grid_rows)

        metrics.append(
            GenerationMetrics(
                generation=generation,
                grid_cols=grid_cols,
                grid_rows=grid_rows,
                best_fitness=best_fit,
                avg_fitness=avg_fit,
                best_score=score,
                max_score=score,
                avg_max10=avg_max10,
                best_ever_score=best_ever,
                death_cause=death_cause,
            )
        )

    return metrics


def load_training_history(replays_dir: Path) -> list[GenerationMetrics]:
    """
    Load metrics for the dashboard: JSONL rows plus any missing gens from gen_*.npz.

    If no log exists yet, backfilled rows are written once so future resumes are fast.
    """
    log_path = replays_dir / TRAINING_LOG_NAME
    by_generation: dict[int, GenerationMetrics] = {}

    if log_path.exists():
        for metrics in load_training_log(log_path):
            by_generation[metrics.generation] = metrics

    for metrics in backfill_metrics_from_gen_npz(replays_dir):
        existing = by_generation.get(metrics.generation)
        if existing is None:
            by_generation[metrics.generation] = metrics
        elif _fitness_missing(existing.best_fitness):
            by_generation[metrics.generation] = metrics

    merged = sorted(by_generation.values(), key=lambda m: m.generation)
    merged = [enrich_metrics_fitness(m) for m in merged]

    if not log_path.exists() and merged:
        for metrics in merged:
            append_training_log(log_path, metrics)

    return merged
