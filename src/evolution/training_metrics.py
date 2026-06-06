"""Per-generation training metrics shared by console logging and the live dashboard."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrainingStartInfo:
    """Metadata printed once at the start of a run."""

    population: int
    start_generation: int
    end_generation: int
    resume_note: str
    arch_label: str
    genome_length: int
    breeding_note: str
    curriculum_note: str
    eval_note: str
    refine_note: str
    workers_note: str = "serial"


@dataclass(frozen=True, slots=True)
class GenerationMetrics:
    """One line of training progress (matches the CLI log columns)."""

    generation: int
    grid_cols: int
    grid_rows: int
    best_fitness: float
    avg_fitness: float
    best_score: int
    max_score: int
    avg_max10: float
    best_ever_score: int
    death_cause: str
    win_count: int = 0
    win_needed: int = 0

    @property
    def grid_label(self) -> str:
        return f"{self.grid_cols}x{self.grid_rows}"


def format_generation_line(metrics: GenerationMetrics) -> str:
    wins = ""
    if metrics.win_needed > 0:
        wins = f" | wins {metrics.win_count}/{metrics.win_needed}"
    return (
        f"Gen {metrics.generation:4d} | grid {metrics.grid_label:>5s} | "
        f"best_fit {metrics.best_fitness:10.2f} | "
        f"avg_fit {metrics.avg_fitness:9.2f} | "
        f"best_score {metrics.best_score:3d} | max_score {metrics.max_score:3d} | "
        f"avg_max10 {metrics.avg_max10:4.1f} | "
        f"best_ever {metrics.best_ever_score:3d}{wins} | died {metrics.death_cause}"
    )
