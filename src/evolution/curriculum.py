"""Grid-size curriculum: advance when most of the population wins the current board."""

from __future__ import annotations

from dataclasses import dataclass

import config


@dataclass(frozen=True, slots=True)
class CurriculumStage:
    """One training phase on a fixed grid size."""

    cols: int
    rows: int
    max_generations: int = 0  # 0 = no cap; advance on win fraction only


class Curriculum:
    """
    Tracks the active curriculum stage and advances when enough snakes win.

    Stage progression is based on population win rate, not a fixed generation quota.
    """

    def __init__(
        self,
        stages: tuple[CurriculumStage, ...],
        *,
        stage_index: int = 0,
        local_generations: int = 0,
    ) -> None:
        if not stages:
            raise ValueError("Curriculum requires at least one stage.")
        if not 0 <= stage_index < len(stages):
            raise ValueError(f"stage_index {stage_index} out of range for {len(stages)} stages.")
        self._stages = stages
        self._stage_index = stage_index
        self._local_generations = local_generations

    @property
    def stages(self) -> tuple[CurriculumStage, ...]:
        return self._stages

    @property
    def stage_index(self) -> int:
        return self._stage_index

    @property
    def local_generations(self) -> int:
        """Completed generations on the current stage (before the in-flight generation)."""
        return self._local_generations

    def current(self) -> tuple[CurriculumStage, int, int]:
        """Return (stage, local_generation_index, stage_index)."""
        return self._stages[self._stage_index], self._local_generations, self._stage_index

    def is_final_stage(self) -> bool:
        return self._stage_index >= len(self._stages) - 1

    def increment_generation(self) -> None:
        self._local_generations += 1

    def win_count(self, individuals) -> int:
        """Count individuals whose latest evaluation ended in a board fill (win)."""
        return sum(1 for ind in individuals if ind.death_cause == "win")

    def should_advance(self, win_count: int, population_size: int) -> bool:
        """True when the population has mastered the current grid and a next stage exists."""
        if self.is_final_stage() or population_size <= 0:
            return False
        if self._local_generations < config.CURRICULUM_MIN_GENS_PER_STAGE:
            return False

        stage = self._stages[self._stage_index]
        if stage.max_generations > 0 and self._local_generations >= stage.max_generations:
            return True

        fraction = win_count / float(population_size)
        return fraction >= config.CURRICULUM_ADVANCE_WIN_FRACTION

    def should_stop(self, win_count: int, population_size: int) -> bool:
        """True when the final stage is mastered and training can end early."""
        if not config.TRAINING_STOP_ON_WIN or population_size <= 0:
            return False
        if not self.is_final_stage():
            return False
        if self._local_generations < config.TRAINING_STOP_MIN_GENS:
            return False
        fraction = win_count / float(population_size)
        return fraction >= config.TRAINING_STOP_WIN_FRACTION

    def advance(self) -> CurriculumStage:
        """Move to the next stage and reset the per-stage generation counter."""
        if self.is_final_stage():
            return self._stages[self._stage_index]
        self._stage_index += 1
        self._local_generations = 0
        return self._stages[self._stage_index]

    def summary_label(self) -> str:
        """Short description for training logs."""
        grids = " -> ".join(f"{stage.cols}x{stage.rows}" for stage in self._stages)
        threshold = config.CURRICULUM_ADVANCE_WIN_FRACTION
        return f"{grids} (advance >= {threshold:.0%} wins)"


def build_curriculum(
    stage_config: tuple[tuple[int, ...], ...] | tuple[tuple[int, int, int], ...],
) -> Curriculum:
    """
    Build stages from config tuples.

    Each entry is (cols, rows) or (cols, rows, max_generations).
    max_generations=0 means no generation cap — only win-rate advancement.
    """
    stages: list[CurriculumStage] = []
    for item in stage_config:
        if len(item) == 2:
            cols, rows = item
            max_generations = 0
        elif len(item) == 3:
            cols, rows, max_generations = item
        else:
            raise ValueError(f"Invalid curriculum stage config: {item!r}")
        stages.append(
            CurriculumStage(
                int(cols),
                int(rows),
                max_generations=max(0, int(max_generations)),
            )
        )
    return Curriculum(tuple(stages))


def stages_to_array(stages: tuple[CurriculumStage, ...]) -> "np.ndarray":
    import numpy as np

    return np.asarray(
        [(stage.cols, stage.rows, stage.max_generations) for stage in stages],
        dtype=np.int64,
    )


def stages_from_array(data: "np.ndarray") -> tuple[CurriculumStage, ...]:
    return tuple(
        CurriculumStage(int(cols), int(rows), max(0, int(max_gens)))
        for cols, rows, max_gens in data
    )
