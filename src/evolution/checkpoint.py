"""Save/load training checkpoints and per-generation best genomes."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import config
from evolution.curriculum import (
    Curriculum,
    CurriculumStage,
    build_curriculum,
    stages_from_array,
    stages_to_array,
)
from evolution.genome import Genome
from evolution.population import Individual, Population
from neural.network import NeuralNetwork

CHECKPOINT_NAME = "checkpoint.npz"


def architecture_array() -> np.ndarray:
    return np.asarray(NeuralNetwork.architecture())


def save_best_genome(
    path: Path,
    generation: int,
    genes: np.ndarray,
    score: int,
    food_seed: int,
    grid_cols: int,
    grid_rows: int,
    *,
    death_cause: str = "",
) -> None:
    """Persist a generation's best genome plus the seed needed to replay it."""
    np.savez(
        path,
        genes=genes,
        generation=generation,
        score=score,
        food_seed=food_seed,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        architecture=architecture_array(),
        nn_arch=config.NN_ARCH,
        death_cause=death_cause,
    )


def save_checkpoint(
    path: Path,
    *,
    next_generation: int,
    population: Population,
    best_ever_score: int,
    best_overall_fitness: float,
    hall_of_fame: Individual | None,
    curriculum_enabled: bool,
    curriculum: Curriculum | None,
    crossover_rate: float,
) -> None:
    """Save full population state so training can resume later."""
    curriculum_stage_index = -1
    curriculum_local_generations = 0
    if curriculum is not None:
        curriculum_stage_index = curriculum.stage_index
        curriculum_local_generations = curriculum.local_generations

    np.savez(
        path,
        next_generation=next_generation,
        population_genes=np.stack([ind.genome.genes for ind in population.individuals]),
        population_size=len(population.individuals),
        best_ever_score=best_ever_score,
        best_overall_fitness=best_overall_fitness,
        has_hall_of_fame=hall_of_fame is not None,
        hall_of_fame_genes=hall_of_fame.genome.genes if hall_of_fame is not None else np.array([]),
        architecture=architecture_array(),
        curriculum_enabled=curriculum_enabled,
        curriculum_stages=stages_to_array(curriculum.stages) if curriculum is not None else np.array([]),
        curriculum_stage_index=curriculum_stage_index,
        curriculum_local_generations=curriculum_local_generations,
        crossover_rate=crossover_rate,
    )


def load_checkpoint(
    path: Path,
    population_size: int,
) -> tuple[
    int,
    Population,
    int,
    float,
    Individual | None,
    bool,
    Curriculum | None,
    float,
    int | None,
    int | None,
]:
    """Restore population and training metadata from a checkpoint file."""
    if not path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {path.resolve()}. Train without --resume first."
        )

    data = np.load(path)
    expected_arch = architecture_array()
    if not np.array_equal(data["architecture"], expected_arch):
        raise ValueError(
            "Checkpoint architecture "
            f"{tuple(int(x) for x in data['architecture'])} "
            f"does not match current {tuple(expected_arch)}."
        )

    saved_pop_size = int(data["population_size"])
    if saved_pop_size != population_size:
        raise ValueError(
            f"Checkpoint population size ({saved_pop_size}) "
            f"does not match --population ({population_size})."
        )

    genes = np.asarray(data["population_genes"], dtype=np.float64)
    if genes.shape != (population_size, NeuralNetwork.genome_length()):
        raise ValueError(
            f"Checkpoint population shape {genes.shape} is invalid for "
            f"population={population_size}, genome_length={NeuralNetwork.genome_length()}."
        )

    population = Population([Individual(genome=Genome(genes[i])) for i in range(population_size)])
    next_generation = int(data["next_generation"])
    best_ever_score = int(data["best_ever_score"])
    best_overall_fitness = float(data["best_overall_fitness"])

    hall_of_fame: Individual | None = None
    if bool(data["has_hall_of_fame"]):
        hall_of_fame = Individual(genome=Genome(np.asarray(data["hall_of_fame_genes"], dtype=np.float64)))

    curriculum_enabled = bool(data["curriculum_enabled"]) if "curriculum_enabled" in data else False
    curriculum: Curriculum | None = None
    curriculum_stage_index: int | None = None
    curriculum_local_generations: int | None = None
    if curriculum_enabled:
        if "curriculum_stage_index" in data:
            curriculum_stage_index = int(data["curriculum_stage_index"])
        if "curriculum_local_generations" in data:
            curriculum_local_generations = int(data["curriculum_local_generations"])

        if "curriculum_stages" in data and len(data["curriculum_stages"]) > 0:
            raw_stages = stages_from_array(data["curriculum_stages"])
            stages = tuple(
                CurriculumStage(stage.cols, stage.rows, max_generations=0) for stage in raw_stages
            )
        else:
            stages = build_curriculum(config.CURRICULUM_STAGES).stages

        stage_index = curriculum_stage_index if curriculum_stage_index is not None else 0
        stage_index = max(0, min(stage_index, len(stages) - 1))
        local_generations = curriculum_local_generations if curriculum_local_generations is not None else 0
        curriculum = Curriculum(
            stages,
            stage_index=stage_index,
            local_generations=max(0, local_generations),
        )

    crossover_rate = float(data["crossover_rate"]) if "crossover_rate" in data else config.CROSSOVER_RATE

    return (
        next_generation,
        population,
        best_ever_score,
        best_overall_fitness,
        hall_of_fame,
        curriculum_enabled,
        curriculum,
        crossover_rate,
        curriculum_stage_index,
        curriculum_local_generations,
    )
