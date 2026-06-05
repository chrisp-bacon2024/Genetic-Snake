"""Grid-size curriculum: train on small boards before the full game."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CurriculumStage:
    """One training phase on a fixed grid size."""

    cols: int
    rows: int
    generations: int


class Curriculum:
    """Maps global generation indices to curriculum stages."""

    def __init__(self, stages: tuple[CurriculumStage, ...]) -> None:
        if not stages:
            raise ValueError("Curriculum requires at least one stage.")
        self._stages = stages
        self._ends: list[int] = []
        total = 0
        for stage in stages:
            total += stage.generations
            self._ends.append(total)

    @property
    def stages(self) -> tuple[CurriculumStage, ...]:
        return self._stages

    def total_generations(self) -> int:
        return self._ends[-1]

    def stage_for_generation(self, generation: int) -> tuple[CurriculumStage, int, int]:
        """
        Return (stage, local_generation_index, stage_index) for a global generation.

        Generations at or beyond total map to the last stage.
        """
        if generation < 0:
            raise ValueError("generation must be non-negative")

        for index, end in enumerate(self._ends):
            if generation < end:
                start = 0 if index == 0 else self._ends[index - 1]
                return self._stages[index], generation - start, index

        last_index = len(self._stages) - 1
        last_start = 0 if last_index == 0 else self._ends[last_index - 1]
        return self._stages[last_index], generation - last_start, last_index


def build_curriculum(
    stage_config: tuple[tuple[int, int, int], ...],
    total_generations: int,
) -> Curriculum:
    """
    Scale configured stage lengths to ``total_generations`` while keeping proportions.

    The last stage absorbs rounding remainder so the sum matches exactly.
    """
    if total_generations <= 0:
        raise ValueError("total_generations must be positive")

    base_total = sum(gens for _, _, gens in stage_config)
    if base_total <= 0:
        raise ValueError("stage_config must include positive generation counts")

    stages: list[CurriculumStage] = []
    allocated = 0
    for index, (cols, rows, gens) in enumerate(stage_config):
        if index == len(stage_config) - 1:
            stage_gens = total_generations - allocated
        else:
            stage_gens = max(1, round(total_generations * gens / base_total))
            allocated += stage_gens
        stages.append(CurriculumStage(cols, rows, stage_gens))

    return Curriculum(tuple(stages))


def stages_to_array(stages: tuple[CurriculumStage, ...]) -> "np.ndarray":
    import numpy as np

    return np.asarray([(s.cols, s.rows, s.generations) for s in stages], dtype=np.int64)


def stages_from_array(data: "np.ndarray") -> tuple[CurriculumStage, ...]:
    return tuple(
        CurriculumStage(int(cols), int(rows), int(gens))
        for cols, rows, gens in data
    )
