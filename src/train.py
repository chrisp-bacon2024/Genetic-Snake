"""
Genetic-algorithm training CLI for the Snake AI.

Run from the ``src/`` directory:

    python train.py                      # train with config defaults (grid->MLP->4)
    python train.py --generations 50     # shorter run
    python train.py --curriculum         # 5x5 -> 10x10 -> 20x20 drill schedule
    python train.py --asexual            # clone+mutate only (no SBX crossover)
    python train.py --watch              # train, then watch the saved best snakes
    python train.py --watch-live         # train while replaying each gen as it finishes
    python train.py --dashboard          # train with live score/fitness charts
    python train.py --generations 200 --resume   # continue from checkpoint

Architecture or encoder changes require a fresh training run (old checkpoints
are incompatible). Default network: grid -> MLP -> 4 (see config.NN_HIDDEN_SIZES).

Each generation every individual (elites included) is re-evaluated, ranked by
fitness, and bred into the next generation. The best genome of each generation
is saved to ``replays/`` so it can be watched later.

A full-population checkpoint is written to ``replays/checkpoint.npz`` after every
generation so training can be resumed with ``--resume`` (runs ``--generations``
more generations from where the last run stopped).
"""

import argparse
import shutil
import sys
import threading
from pathlib import Path
from typing import Protocol

import numpy as np

import config
from evolution.curriculum import (
    Curriculum,
    build_curriculum,
    stages_from_array,
    stages_to_array,
)
from evolution.genome import Genome
from evolution.population import Individual, Population
from evolution.training_log import TRAINING_LOG_NAME, append_training_log, load_training_history
from evolution.training_metrics import (
    GenerationMetrics,
    TrainingStartInfo,
    format_generation_line,
)
from models.grid import Grid
from neural.network import NeuralNetwork
from simulation.headless import HeadlessSimulator

CHECKPOINT_NAME = "checkpoint.npz"


class TrainingObserver(Protocol):
    """Optional hooks for live dashboard or external loggers."""

    def on_start(self, info: TrainingStartInfo) -> None: ...

    def on_curriculum(self, message: str) -> None: ...

    def on_progress(self, message: str) -> None: ...

    def on_generation(self, metrics: GenerationMetrics) -> None: ...

    def on_done(self, replays_path: Path) -> None: ...


def _print_start_info(info: TrainingStartInfo) -> None:
    print(
        f"Training: pop={info.population} gens={info.start_generation}-{info.end_generation - 1} "
        f"({info.resume_note}) "
        f"arch={info.arch_label} genes={info.genome_length} "
        f"breeding={info.breeding_note} "
        f"curriculum=[{info.curriculum_note}] "
        f"eval={info.eval_note} "
        f"refine_top={info.refine_note}",
        flush=True,
    )


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
        architecture=_architecture_array(),
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
        curriculum_enabled=curriculum_enabled,
        curriculum_stages=stages_to_array(curriculum.stages) if curriculum is not None else np.array([]),
        crossover_rate=crossover_rate,
    )


def load_checkpoint(
    path: Path,
    population_size: int,
) -> tuple[int, Population, int, float, Individual | None, bool, Curriculum | None, float]:
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

    curriculum_enabled = bool(data["curriculum_enabled"]) if "curriculum_enabled" in data else False
    curriculum: Curriculum | None = None
    if curriculum_enabled and "curriculum_stages" in data and len(data["curriculum_stages"]) > 0:
        curriculum = Curriculum(stages_from_array(data["curriculum_stages"]))

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
    )


def _apply_result(individual: Individual, result) -> None:
    individual.fitness = result.fitness
    individual.score = result.score
    individual.steps = result.steps
    individual.best_food_seed = result.best_food_seed
    individual.death_cause = result.death_cause


def _crossover_rate_for_args(args: argparse.Namespace) -> float:
    return 0.0 if args.asexual else config.CROSSOVER_RATE


def _resolve_curriculum(args: argparse.Namespace, checkpoint_curriculum: Curriculum | None) -> Curriculum | None:
    if not args.curriculum:
        return None
    if checkpoint_curriculum is not None:
        return checkpoint_curriculum
    return build_curriculum(config.CURRICULUM_STAGES, args.generations)


def _apply_stage_grid(simulator: HeadlessSimulator, stage) -> None:
    simulator.set_grid(stage.cols, stage.rows)


def _set_generation_scenarios(simulator: HeadlessSimulator, generation: int, run_count: int) -> None:
    """Pin shared food seeds for the generation so every snake plays the same boards."""
    scenarios = simulator.build_scenarios(run_count, seed=generation)
    simulator.set_scenarios(scenarios)


def resolve_generation_span(args: argparse.Namespace) -> tuple[int, int]:
    """
    Return (start_generation, end_generation) for this run.

    end_generation is exclusive (same as the training loop range).
    """
    start_generation = 0
    if args.resume:
        checkpoint_path = replays_dir() / CHECKPOINT_NAME
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"No checkpoint at {checkpoint_path.resolve()}. Train without --resume first."
            )
        data = np.load(checkpoint_path)
        start_generation = int(data["next_generation"])
    return start_generation, start_generation + args.generations


def _report_progress(
    observer: TrainingObserver | None,
    message: str,
) -> None:
    if observer is not None:
        observer.on_progress(message)
    else:
        print(message, flush=True)


def run_training(
    args: argparse.Namespace,
    *,
    observer: TrainingObserver | None = None,
) -> None:
    checkpoint_path = replays_dir() / CHECKPOINT_NAME
    checkpoint_curriculum: Curriculum | None = None
    crossover_rate = _crossover_rate_for_args(args)

    if args.resume:
        replays = replays_dir()
        (
            start_generation,
            population,
            best_ever_score,
            best_overall_fitness,
            hall_of_fame,
            checkpoint_curriculum_enabled,
            checkpoint_curriculum,
            checkpoint_crossover_rate,
        ) = load_checkpoint(checkpoint_path, args.population)
        if checkpoint_curriculum_enabled != args.curriculum:
            raise ValueError(
                "Checkpoint curriculum setting "
                f"({checkpoint_curriculum_enabled}) does not match this run "
                f"({'curriculum' if args.curriculum else 'no-curriculum'})."
            )
        if checkpoint_crossover_rate != crossover_rate:
            raise ValueError(
                f"Checkpoint crossover rate ({checkpoint_crossover_rate}) does not match "
                f"this run ({crossover_rate}). Use the same --asexual setting when resuming."
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

    curriculum = _resolve_curriculum(args, checkpoint_curriculum)
    end_generation = start_generation + args.generations

    if args.curriculum:
        simulator = HeadlessSimulator()
        stage, _, _ = curriculum.stage_for_generation(start_generation)
        _apply_stage_grid(simulator, stage)
    else:
        simulator = HeadlessSimulator(Grid(config.GRID_COLS, config.GRID_ROWS))

    top_fraction = config.SELECT_TOP_FRACTION
    refine_runs = config.SELECT_EVAL_RUNS
    current_stage_index: int | None = None

    curriculum_note = (
        " -> ".join(f"{s.cols}x{s.rows}x{s.generations}" for s in curriculum.stages)
        if curriculum is not None
        else f"{config.GRID_COLS}x{config.GRID_ROWS} only"
    )

    breeding_note = "asexual" if crossover_rate <= 0.0 else f"crossover top {config.PARENT_POOL_FRACTION:.0%}"
    eval_note = (
        f"shared_seeds x{config.EVAL_RUNS_PER_GENOME}"
        if config.SHARED_EVAL_SEEDS
        else f"random x{config.EVAL_RUNS_PER_GENOME}"
    )
    genome_len = NeuralNetwork.genome_length()

    arch_label = NeuralNetwork.architecture_label()
    start_info = TrainingStartInfo(
        population=args.population,
        start_generation=start_generation,
        end_generation=end_generation,
        resume_note=resume_note,
        arch_label=arch_label,
        genome_length=genome_len,
        breeding_note=breeding_note,
        curriculum_note=curriculum_note,
        eval_note=eval_note,
        refine_note=f"{top_fraction:.0%}x{refine_runs}runs",
    )
    if observer is not None:
        observer.on_start(start_info)
    else:
        _print_start_info(start_info)

    recent_max_scores: list[int] = []
    training_log_path = replays / TRAINING_LOG_NAME

    for generation in range(start_generation, end_generation):
        if curriculum is not None:
            stage, _, stage_index = curriculum.stage_for_generation(generation)
            if stage_index != current_stage_index:
                _apply_stage_grid(simulator, stage)
                if current_stage_index is not None:
                    best_ever_score = 0
                    curriculum_msg = (
                        f"--- Curriculum: now training on {stage.cols}x{stage.rows} "
                        f"(stage {stage_index + 1}/{len(curriculum.stages)}) ---"
                    )
                    if observer is not None:
                        observer.on_curriculum(curriculum_msg)
                    else:
                        print(curriculum_msg, flush=True)
                current_stage_index = stage_index
            grid_cols, grid_rows = stage.cols, stage.rows
        else:
            grid_cols, grid_rows = config.GRID_COLS, config.GRID_ROWS

        screening_runs = config.EVAL_RUNS_PER_GENOME
        if config.SHARED_EVAL_SEEDS:
            _set_generation_scenarios(simulator, generation, screening_runs)

        # Phase 1: screening evaluation (averaged over screening_runs boards).
        pop_size = len(population.individuals)
        for index, individual in enumerate(population.individuals):
            if index == 0 or (index + 1) % 50 == 0 or index + 1 == pop_size:
                _report_progress(
                    observer,
                    f"  gen {generation} screening {index + 1}/{pop_size}...",
                )
            result = (
                simulator.evaluate(individual.genome)
                if config.SHARED_EVAL_SEEDS
                else simulator.evaluate(individual.genome, runs=screening_runs)
            )
            _apply_result(individual, result)

        # Phase 2: re-evaluate top fraction with multiple boards for stable ranking.
        ranked_prelim = population.sorted_by_fitness()
        refine_count = max(1, int(len(ranked_prelim) * top_fraction))
        if config.SHARED_EVAL_SEEDS:
            _set_generation_scenarios(simulator, generation + 1_000_000, refine_runs)
        for index, individual in enumerate(ranked_prelim[:refine_count]):
            if index == 0 or (index + 1) % 25 == 0 or index + 1 == refine_count:
                _report_progress(
                    observer,
                    f"  gen {generation} refining top {index + 1}/{refine_count}...",
                )
            result = (
                simulator.evaluate(individual.genome)
                if config.SHARED_EVAL_SEEDS
                else simulator.evaluate(individual.genome, runs=refine_runs)
            )
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
            grid_cols,
            grid_rows,
            death_cause=best.death_cause,
        )

        if best.fitness > best_overall_fitness:
            best_overall_fitness = best.fitness
            save_best_genome(
                replays / "best.npz",
                generation,
                best.genome.genes,
                best.score,
                best_seed,
                grid_cols,
                grid_rows,
                death_cause=best.death_cause,
            )

        if hall_of_fame is not None and best_ever_score > 0:
            save_best_genome(
                replays / "best_score.npz",
                generation,
                hall_of_fame.genome.genes,
                hall_of_fame.score,
                hall_of_fame.best_food_seed,
                grid_cols,
                grid_rows,
                death_cause=hall_of_fame.death_cause,
            )

        grid_label = f"{grid_cols}x{grid_rows}"
        recent_max_scores.append(max_score)
        if len(recent_max_scores) > 10:
            recent_max_scores.pop(0)
        avg_max = sum(recent_max_scores) / len(recent_max_scores)
        metrics = GenerationMetrics(
            generation=generation,
            grid_cols=grid_cols,
            grid_rows=grid_rows,
            best_fitness=best.fitness,
            avg_fitness=population.average_fitness(),
            best_score=best.score,
            max_score=max_score,
            avg_max10=avg_max,
            best_ever_score=best_ever_score,
            death_cause=best.death_cause,
        )
        append_training_log(training_log_path, metrics)
        if observer is not None:
            observer.on_generation(metrics)
        else:
            print(format_generation_line(metrics), flush=True)

        population = population.evolve_next_generation(crossover_rate=crossover_rate)
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
            curriculum_enabled=args.curriculum,
            curriculum=curriculum,
            crossover_rate=crossover_rate,
        )

    done_msg = f"Done. Best genomes saved in {replays.resolve()}"
    if observer is not None:
        observer.on_done(replays)
    print(done_msg, flush=True)


def watch_best() -> None:
    """Replay the saved per-generation best snakes (requires pygame)."""
    from ui.replay_viewer import ReplayViewer

    ReplayViewer(Path(config.REPLAYS_DIR)).run()


class _DashboardObserver:
    """Adapts TrainingDashboard to TrainingObserver."""

    def __init__(self, dashboard: "TrainingDashboard") -> None:
        self._dashboard = dashboard

    def on_start(self, info: TrainingStartInfo) -> None:
        self._dashboard.set_start_info(info)
        _print_start_info(info)

    def on_curriculum(self, message: str) -> None:
        self._dashboard.add_curriculum_note(message)
        print(message, flush=True)

    def on_progress(self, message: str) -> None:
        self._dashboard.set_progress(message)

    def on_generation(self, metrics: GenerationMetrics) -> None:
        self._dashboard.log_generation(metrics)

    def on_done(self, replays_path: Path) -> None:
        self._dashboard.set_done(replays_path)


def run_training_with_dashboard(args: argparse.Namespace) -> None:
    """Train in a background thread while the main thread shows live charts."""
    from ui.training_dashboard import TrainingDashboard

    training_done = threading.Event()
    training_error: list[BaseException] = []
    start_generation, end_generation = resolve_generation_span(args)
    dashboard = TrainingDashboard(training_done)
    dashboard.set_generation_span(start_generation, end_generation)
    history = load_training_history(replays_dir())
    if history:
        dashboard.load_metrics(history)
    observer = _DashboardObserver(dashboard)

    def training_worker() -> None:
        try:
            run_training(args, observer=observer)
        except BaseException as exc:
            training_error.append(exc)
        finally:
            training_done.set()

    thread = threading.Thread(target=training_worker, name="training", daemon=False)
    thread.start()
    dashboard.run()
    thread.join()
    if training_error:
        raise training_error[0]


def run_training_with_live_view(args: argparse.Namespace) -> None:
    """Train in a background thread while the main thread replays each generation live."""
    from ui.replay_viewer import LiveReplayViewer

    training_done = threading.Event()
    training_error: list[BaseException] = []

    def training_worker() -> None:
        try:
            run_training(args)
        except BaseException as exc:
            training_error.append(exc)
        finally:
            training_done.set()

    thread = threading.Thread(target=training_worker, name="training", daemon=True)
    thread.start()
    LiveReplayViewer(replays_dir(), training_done=training_done).run()
    thread.join(timeout=0.1)
    if training_error:
        raise training_error[0]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Snake AI with a genetic algorithm.")
    parser.add_argument("--generations", type=int, default=config.GENERATIONS)
    parser.add_argument("--population", type=int, default=config.POPULATION_SIZE)
    parser.add_argument("--eval-runs", type=int, default=config.EVAL_RUNS_PER_GENOME)
    parser.add_argument("--watch", action="store_true", help="Watch saved best snakes after training.")
    parser.add_argument(
        "--watch-live",
        action="store_true",
        help="Replay each generation live while training (Esc closes the viewer; training continues).",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Show live charts of scores and fitness while training (close window to exit UI).",
    )
    parser.add_argument("--watch-only", action="store_true", help="Skip training; just watch existing replays.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from replays/checkpoint.npz (runs --generations more generations).",
    )
    curriculum_group = parser.add_mutually_exclusive_group()
    curriculum_group.add_argument(
        "--no-curriculum",
        dest="curriculum",
        action="store_false",
        help=f"Train on {config.GRID_COLS}x{config.GRID_ROWS} only (skip curriculum stages).",
    )
    curriculum_group.add_argument(
        "--curriculum",
        dest="curriculum",
        action="store_true",
        help="Use curriculum stages from config (default when CURRICULUM_ENABLED).",
    )
    parser.set_defaults(curriculum=config.CURRICULUM_ENABLED)
    breeding_group = parser.add_mutually_exclusive_group()
    breeding_group.add_argument(
        "--crossover",
        dest="asexual",
        action="store_false",
        help="SBX crossover between top snakes (default).",
    )
    breeding_group.add_argument(
        "--asexual",
        dest="asexual",
        action="store_true",
        help="Clone+mutate only; no crossover between parents.",
    )
    parser.set_defaults(asexual=False)
    parser.add_argument(
        "--replays-dir",
        type=str,
        default=None,
        help="Override config.REPLAYS_DIR for saves and checkpoints.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    if args.replays_dir:
        config.REPLAYS_DIR = args.replays_dir
    if args.watch_live and args.watch_only:
        raise SystemExit("--watch-live cannot be used with --watch-only.")
    if args.watch_live and args.watch:
        raise SystemExit("Use either --watch-live or --watch, not both.")
    if args.dashboard and args.watch_live:
        raise SystemExit("Use either --dashboard or --watch-live, not both.")
    if args.dashboard:
        run_training_with_dashboard(args)
    elif args.watch_live:
        run_training_with_live_view(args)
    else:
        if not args.watch_only:
            run_training(args)
        if args.watch or args.watch_only:
            watch_best()


if __name__ == "__main__":
    main()
