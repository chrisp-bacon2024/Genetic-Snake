"""
Genetic-algorithm training CLI for the Snake AI.

Run from the ``src/`` directory:

    python train.py                      # train with config defaults
    python train.py --generations 50     # shorter run
    python train.py --watch              # train, then watch the saved best snakes

Each generation every individual (elites included) is re-evaluated on the current
seeded scenario set, ranked by fitness, and bred into the next generation. The best
genome of each generation is saved to ``replays/`` so it can be watched later.
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

import config
from evolution.population import Population
from models.grid import Grid
from simulation.headless import HeadlessSimulator


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
        architecture=np.asarray((config.NN_INPUT_SIZE, *config.NN_HIDDEN_SIZES, config.NN_OUTPUT_SIZE)),
    )


def run_training(args: argparse.Namespace) -> None:
    replays = clear_replays_dir()
    grid = Grid(config.GRID_COLS, config.GRID_ROWS)
    simulator = HeadlessSimulator(grid)
    population = Population.random(args.population)

    reseed_every = max(1, config.SCENARIO_RESEED_FREQUENCY)
    best_overall = -1.0

    print(
        f"Training: pop={args.population} gens={args.generations} "
        f"arch={(config.NN_INPUT_SIZE, *config.NN_HIDDEN_SIZES, config.NN_OUTPUT_SIZE)} "
        f"eval_runs={args.eval_runs}",
        flush=True,
    )

    for generation in range(args.generations):
        if generation % reseed_every == 0:
            simulator.set_scenarios(
                simulator.build_scenarios(args.eval_runs, seed=generation)
            )

        max_score = 0
        for individual in population.individuals:
            result = simulator.evaluate(individual.genome)
            individual.fitness = result.fitness
            individual.score = result.score
            individual.steps = result.steps
            individual.best_food_seed = result.best_food_seed
            max_score = max(max_score, result.score)

        ranked = population.sorted_by_fitness()
        best = ranked[0]
        best_seed = best.best_food_seed

        save_best_genome(
            replays / f"gen_{generation:04d}.npz",
            generation,
            best.genome.genes,
            best.score,
            best_seed,
        )

        if best.fitness > best_overall:
            best_overall = best.fitness
            save_best_genome(
                replays / "best.npz", generation, best.genome.genes, best.score, best_seed
            )

        print(
            f"Gen {generation:4d} | best_fit {best.fitness:10.2f} | "
            f"avg_fit {population.average_fitness():9.2f} | "
            f"best_score {best.score:3d} | max_score {max_score:3d}",
            flush=True,
        )

        if generation < args.generations - 1:
            population = population.evolve_next_generation()

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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    if not args.watch_only:
        run_training(args)
    if args.watch or args.watch_only:
        watch_best()


if __name__ == "__main__":
    main()
