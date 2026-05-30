"""
Genetic-algorithm training CLI for the Snake AI.

Run from the ``src/`` directory:

    python train.py                      # train with config defaults
    python train.py --generations 50     # shorter run
    python train.py --watch              # train, then watch the saved best snakes
    python train.py --generations 200 --resume   # continue from checkpoint

Each generation every individual (elites included) is re-evaluated with a fresh
random food board, ranked by fitness, and bred into the next generation. The best
genome of each generation is saved to ``replays/`` so it can be watched later.

A full-population checkpoint is written to ``replays/checkpoint.npz`` after every
generation so training can be resumed with ``--resume`` (runs ``--generations``
more generations from where the last run stopped).
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

import config
from evolution.genome import Genome
from evolution.population import Individual, Population
from models.grid import Grid
from neural.network import NeuralNetwork
from simulation.headless import HeadlessSimulator

CHECKPOINT_NAME = "checkpoint.npz"


def _architecture_array() -> np.ndarray:
    return np.asarray(NeuralNetwork.architecture())


def replays_dir() -> Path:
    path = Path(config.REPLAYS_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def clear_replays_dir() -> Path:
    """Wipe and recreate the replays directory so each run starts clean."""
    replays = Path(config.REPLAYS_DIR)
    if replays.exists():
        shutil.rmtree(replays)
    replays.mkdir(parents=True, exist_ok=True)
    return replays


def save_best_genome(path: Path, generation: int, genes: np.ndarray, score: int, food_seed: int) -> None:
    """Persist a generation's best genome plus the seed needed to replay it."""
    np.savez(
        path,
        genes=genes,
        generation=generation,
        score=score,
        food_seed=food_seed,
        architecture=_architecture_array(),
    )


def save_checkpoint(
    path: Path,
    *,
    next_generation: int,
    population: Population,
    best_ever_score: int,
    best_overall_fitness: float,
    hall_of_fame: Individual | None,
) -> None:
    """Save full population state so training can resume later."""
    np.savez(
        path,
        next_generation=next_generation,
        population_genes=np.stack([ind.genome.genes for ind in population.individuals]),
        population_size=len(population.individuals),
        best_ever_score=best_ever_score,
        best_overall_fitness=best_overall_fitness,
        has_hall_of_fame=hall_of_fame is not None,
        hall_of_fame_genes=hall_of_fame.genome.genes if hall_of_fame is not None else np.array([]),
        architecture=_architecture_array(),
    )


def load_checkpoint(path: Path, population_size: int) -> tuple[int, Population, int, float, Individual | None]:
    """Restore population and training metadata from a checkpoint file."""
    if not path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {path.resolve()}. Train without --resume first."
        )

    data = np.load(path)
    expected_arch = _architecture_array()
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

    return next_generation, population, best_ever_score, best_overall_fitness, hall_of_fame


def _apply_result(individual: Individual, result) -> None:
    individual.fitness = result.fitness
    individual.score = result.score
    individual.steps = result.steps
    individual.best_food_seed = result.best_food_seed
    individual.death_cause = result.death_cause


def run_training(args: argparse.Namespace) -> None:
    checkpoint_path = replays_dir() / CHECKPOINT_NAME

    if args.resume:
        replays = replays_dir()
        start_generation, population, best_ever_score, best_overall_fitness, hall_of_fame = load_checkpoint(
            checkpoint_path, args.population
        )
        resume_note = f"resuming from gen {start_generation}"
    else:
        replays = clear_replays_dir()
        start_generation = 0
        population = Population.random(args.population)
        best_overall_fitness = -1.0
        best_ever_score = 0
        hall_of_fame = None
        resume_note = "fresh start"

    grid = Grid(config.GRID_COLS, config.GRID_ROWS)
    simulator = HeadlessSimulator(grid)

    top_fraction = config.SELECT_TOP_FRACTION
    refine_runs = config.SELECT_EVAL_RUNS
    end_generation = start_generation + args.generations

    print(
        f"Training: pop={args.population} gens={start_generation}-{end_generation - 1} ({resume_note}) "
        f"arch={tuple(int(x) for x in _architecture_array())} "
        f"random_food={config.RANDOM_FOOD_EVAL} "
        f"refine_top={top_fraction:.0%}x{refine_runs}runs",
        flush=True,
    )

    for generation in range(start_generation, end_generation):
        # Phase 1: one quick random game per snake (screening).
        for individual in population.individuals:
            result = simulator.evaluate(individual.genome, runs=1)
            _apply_result(individual, result)

        # Phase 2: re-evaluate top fraction with multiple boards for stable ranking.
        ranked_prelim = population.sorted_by_fitness()
        refine_count = max(1, int(len(ranked_prelim) * top_fraction))
        for individual in ranked_prelim[:refine_count]:
            result = simulator.evaluate(individual.genome, runs=refine_runs)
            _apply_result(individual, result)

        max_score = max(ind.score for ind in population.individuals)
        ranked = population.sorted_by_fitness()
        best = ranked[0]
        best_seed = best.best_food_seed

        if best.score > best_ever_score:
            best_ever_score = best.score
            hall_of_fame = Individual(
                genome=best.genome.copy(),
                fitness=best.fitness,
                score=best.score,
                steps=best.steps,
                best_food_seed=best.best_food_seed,
                death_cause=best.death_cause,
            )

        save_best_genome(
            replays / f"gen_{generation:04d}.npz",
            generation,
            best.genome.genes,
            best.score,
            best_seed,
        )

        if best.fitness > best_overall_fitness:
            best_overall_fitness = best.fitness
            save_best_genome(
                replays / "best.npz", generation, best.genome.genes, best.score, best_seed
            )

        if hall_of_fame is not None and best_ever_score > 0:
            save_best_genome(
                replays / "best_score.npz",
                generation,
                hall_of_fame.genome.genes,
                hall_of_fame.score,
                hall_of_fame.best_food_seed,
            )

        print(
            f"Gen {generation:4d} | best_fit {best.fitness:10.2f} | "
            f"avg_fit {population.average_fitness():9.2f} | "
            f"best_score {best.score:3d} | max_score {max_score:3d} | "
            f"best_ever {best_ever_score:3d} | died {best.death_cause}",
            flush=True,
        )

        population = population.evolve_next_generation()
        if hall_of_fame is not None:
            # Keep the best-ever genome in the pool so breakthroughs are not lost to luck.
            population.individuals[-1] = Individual(genome=hall_of_fame.genome.copy())

        save_checkpoint(
            checkpoint_path,
            next_generation=generation + 1,
            population=population,
            best_ever_score=best_ever_score,
            best_overall_fitness=best_overall_fitness,
            hall_of_fame=hall_of_fame,
        )

    print(f"Done. Best genomes saved in {replays.resolve()}", flush=True)


def watch_best() -> None:
    """Replay the saved per-generation best snakes (requires pygame)."""
    from ui.replay_viewer import ReplayViewer

    ReplayViewer(Path(config.REPLAYS_DIR)).run()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Snake AI with a genetic algorithm.")
    parser.add_argument("--generations", type=int, default=config.GENERATIONS)
    parser.add_argument("--population", type=int, default=config.POPULATION_SIZE)
    parser.add_argument("--eval-runs", type=int, default=config.EVAL_RUNS_PER_GENOME)
    parser.add_argument("--watch", action="store_true", help="Watch saved best snakes after training.")
    parser.add_argument("--watch-only", action="store_true", help="Skip training; just watch existing replays.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from replays/checkpoint.npz (runs --generations more generations).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    if not args.watch_only:
        run_training(args)
    if args.watch or args.watch_only:
        watch_best()


if __name__ == "__main__":
    main()
