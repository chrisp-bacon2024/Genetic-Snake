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
    python train.py --generations 200 --resume   # continue from checkpoint (same grid stage)
    python train.py --workers 0                  # parallel eval (0 = auto, use all cores-1)
    python train.py --max-steps 3200             # cap long evaluation games

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

import config
from evolution.checkpoint import (
    CHECKPOINT_NAME,
    load_checkpoint,
    save_best_genome,
    save_checkpoint,
)
from evolution.curriculum import (
    Curriculum,
    build_curriculum,
    parent_pool_wins,
    wins_required,
)
from evolution.population import Individual, Population
from evolution.training_progress import InlineProgress
from evolution.training_log import TRAINING_LOG_NAME, append_training_log, load_training_history
from evolution.training_metrics import (
    GenerationMetrics,
    TrainingStartInfo,
    format_generation_line,
)
from models.grid import Grid
from neural.network import NeuralNetwork
from simulation.headless import HeadlessSimulator
from simulation.parallel import (
    build_eval_job,
    evaluate_genomes_parallel,
    resolve_worker_count,
)


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
        f"refine_top={info.refine_note} "
        f"max_steps={info.max_steps_note} "
        f"workers={info.workers_note}",
        flush=True,
    )


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


def _apply_result(individual: Individual, result) -> None:
    individual.fitness = result.fitness
    individual.score = result.score
    individual.steps = result.steps
    individual.best_food_seed = result.best_food_seed
    individual.death_cause = result.death_cause


def _scenario_food_seeds(simulator: HeadlessSimulator) -> tuple[int, ...]:
    return tuple(scenario.food_seed for scenario in simulator.scenarios)


def _evaluate_individuals(
    individuals: list[Individual],
    *,
    workers: int,
    simulator: HeadlessSimulator,
    grid_cols: int,
    grid_rows: int,
    food_seeds: tuple[int, ...],
    random_runs: int,
    observer: TrainingObserver | None,
    progress_label: str,
) -> None:
    """Evaluate a batch of genomes in parallel (or serially when workers <= 1)."""
    use_random_food = not food_seeds and config.RANDOM_FOOD_EVAL
    jobs = [
        build_eval_job(
            individual.genome.genes,
            grid_cols=grid_cols,
            grid_rows=grid_rows,
            max_steps=simulator.max_steps,
            food_seeds=food_seeds,
            random_runs=random_runs,
            use_random_food=use_random_food,
        )
        for individual in individuals
    ]

    def on_progress(done: int, total: int) -> None:
        _report_progress(observer, f"{progress_label} {done}/{total}...")

    results = evaluate_genomes_parallel(
        jobs,
        workers=workers,
        progress_callback=on_progress,
    )
    for individual, result in zip(individuals, results):
        _apply_result(individual, result)


def _crossover_rate_for_args(args: argparse.Namespace) -> float:
    return 0.0 if args.asexual else config.CROSSOVER_RATE


def _apply_cli_config(args: argparse.Namespace) -> None:
    if args.replays_dir:
        config.REPLAYS_DIR = args.replays_dir

    fraction = args.curriculum_win_pct / 100.0
    if not 0.0 < fraction <= 1.0:
        raise SystemExit("--curriculum-win-pct must be greater than 0 and at most 100.")
    config.CURRICULUM_ADVANCE_WIN_FRACTION = fraction
    config.TRAINING_STOP_WIN_FRACTION = fraction


def _resolve_curriculum(args: argparse.Namespace, checkpoint_curriculum: Curriculum | None) -> Curriculum | None:
    if not args.curriculum:
        return None
    if checkpoint_curriculum is not None:
        return checkpoint_curriculum
    return build_curriculum(config.CURRICULUM_STAGES)


def _apply_stage_grid(simulator: HeadlessSimulator, stage) -> None:
    simulator.set_grid(stage.cols, stage.rows)


def _set_generation_scenarios(simulator: HeadlessSimulator, generation: int, run_count: int) -> None:
    """Pin shared food seeds for the generation so every snake plays the same boards."""
    scenarios = simulator.build_scenarios(run_count, seed=generation)
    simulator.set_scenarios(scenarios)


def _report_progress(
    observer: TrainingObserver | None,
    message: str,
) -> None:
    if observer is not None:
        observer.on_progress(message)
    else:
        InlineProgress.update(message)


def _print_generation_line(line: str, *, observer: TrainingObserver | None) -> None:
    InlineProgress.finish(line)


def run_training(
    args: argparse.Namespace,
    *,
    observer: TrainingObserver | None = None,
) -> None:
    checkpoint_path = replays_dir() / CHECKPOINT_NAME
    checkpoint_curriculum: Curriculum | None = None
    crossover_rate = _crossover_rate_for_args(args)
    stop_on_win = args.stop_on_win

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
            _checkpoint_stage_index,
            _checkpoint_local_generations,
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

    max_steps = args.max_steps
    if args.curriculum:
        simulator = HeadlessSimulator(max_steps=max_steps)
        stage, local_generation, stage_index = curriculum.current()
        _apply_stage_grid(simulator, stage)
        if args.resume:
            resume_curriculum_msg = (
                f"--- Resuming curriculum on {stage.cols}x{stage.rows} "
                f"(stage {stage_index + 1}/{len(curriculum.stages)}, "
                f"local gen {local_generation}) ---"
            )
            if observer is not None:
                observer.on_curriculum(resume_curriculum_msg)
            else:
                print(resume_curriculum_msg, flush=True)
    else:
        simulator = HeadlessSimulator(
            Grid(config.GRID_COLS, config.GRID_ROWS),
            max_steps=max_steps,
        )

    single_shot = args.single_shot or config.TRAINING_SINGLE_SHOT_EVAL
    top_fraction = config.SELECT_TOP_FRACTION
    refine_runs = config.SELECT_EVAL_RUNS
    screening_runs = 1 if single_shot else args.eval_runs

    curriculum_note = (
        curriculum.summary_label()
        if curriculum is not None
        else f"{config.GRID_COLS}x{config.GRID_ROWS} only"
    )

    if crossover_rate <= 0.0:
        breeding_note = "asexual"
    elif config.CHAMPION_ASEXUAL_FRACTION > 0.0:
        breeding_note = (
            f"crossover top {config.PARENT_POOL_FRACTION:.0%} "
            f"+ champion {config.CHAMPION_ASEXUAL_FRACTION:.0%}"
        )
    else:
        breeding_note = f"crossover top {config.PARENT_POOL_FRACTION:.0%}"
    if single_shot:
        eval_note = "single-shot (1 board per snake)"
        refine_note = "off"
    else:
        eval_note = (
            f"shared_seeds x{screening_runs}"
            if config.SHARED_EVAL_SEEDS
            else f"random x{screening_runs}"
        )
        refine_note = f"{top_fraction:.0%}x{refine_runs}runs"
    genome_len = NeuralNetwork.genome_length()
    worker_count = resolve_worker_count(args.workers)
    workers_note = "serial" if worker_count <= 1 else f"{worker_count} processes"
    max_steps_note = "none" if simulator.max_steps is None else str(simulator.max_steps)

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
        refine_note=refine_note,
        max_steps_note=max_steps_note,
        workers_note=workers_note,
    )
    if observer is not None:
        observer.on_start(start_info)
    else:
        _print_start_info(start_info)

    training_log_path = replays / TRAINING_LOG_NAME

    for generation in range(start_generation, end_generation):
        if curriculum is not None:
            stage, _, stage_index = curriculum.current()
            grid_cols, grid_rows = stage.cols, stage.rows
        else:
            grid_cols, grid_rows = config.GRID_COLS, config.GRID_ROWS
        # Metrics and replays must use the grid snakes were evaluated on, not a post-advance grid.
        eval_grid_cols, eval_grid_rows = grid_cols, grid_rows

        if config.SHARED_EVAL_SEEDS:
            _set_generation_scenarios(simulator, generation, screening_runs)
        screening_seeds = _scenario_food_seeds(simulator) if config.SHARED_EVAL_SEEDS else ()

        progress_label = (
            f"Gen {generation} eval"
            if single_shot
            else f"Gen {generation} screening"
        )
        _evaluate_individuals(
            population.individuals,
            workers=args.workers,
            simulator=simulator,
            grid_cols=grid_cols,
            grid_rows=grid_rows,
            food_seeds=screening_seeds,
            random_runs=screening_runs,
            observer=observer,
            progress_label=progress_label,
        )

        if not single_shot:
            # Phase 2: re-evaluate top fraction with multiple boards for stable ranking.
            ranked_prelim = population.sorted_by_fitness()
            refine_count = max(1, int(len(ranked_prelim) * top_fraction))
            refine_targets = ranked_prelim[:refine_count]
            if config.SHARED_EVAL_SEEDS:
                _set_generation_scenarios(simulator, generation + 1_000_000, refine_runs)
            refine_seeds = _scenario_food_seeds(simulator) if config.SHARED_EVAL_SEEDS else ()

            _evaluate_individuals(
                refine_targets,
                workers=args.workers,
                simulator=simulator,
                grid_cols=grid_cols,
                grid_rows=grid_rows,
                food_seeds=refine_seeds,
                random_runs=refine_runs,
                observer=observer,
                progress_label=f"Gen {generation} refining",
            )

        pop_size = len(population.individuals)
        gen_win_count = 0
        gen_win_needed = 0
        stop_training = False
        if curriculum is not None:
            curriculum.increment_generation()
            gen_win_count, parent_pool_size = parent_pool_wins(population.individuals)
            if curriculum.is_final_stage():
                gen_win_needed = wins_required(pop_size, config.TRAINING_STOP_WIN_FRACTION)
                if curriculum.should_stop(gen_win_count, pop_size, stop_on_win=stop_on_win):
                    stop_training = True
            else:
                gen_win_needed = wins_required(pop_size, config.CURRICULUM_ADVANCE_WIN_FRACTION)
                if curriculum.should_advance(gen_win_count, pop_size):
                    new_stage = curriculum.advance()
                    _apply_stage_grid(simulator, new_stage)
                    best_ever_score = 0
                    win_fraction = gen_win_count / float(pop_size)
                    curriculum_msg = (
                        f"--- Curriculum: {gen_win_count}/{parent_pool_size} wins in top "
                        f"{config.SELECT_TOP_FRACTION:.0%} ({win_fraction:.0%} of pop) — advancing to "
                        f"{new_stage.cols}x{new_stage.rows} "
                        f"(stage {curriculum.stage_index + 1}/{len(curriculum.stages)}) ---"
                    )
                    if observer is not None:
                        observer.on_curriculum(curriculum_msg)
                    else:
                        print(curriculum_msg, flush=True)
                    grid_cols, grid_rows = new_stage.cols, new_stage.rows
        elif stop_on_win:
            gen_win_count, _ = parent_pool_wins(population.individuals)
            gen_win_needed = wins_required(pop_size, config.TRAINING_STOP_WIN_FRACTION)
            local_gens = generation - start_generation + 1
            if (
                local_gens >= config.TRAINING_STOP_MIN_GENS
                and gen_win_count >= gen_win_needed
            ):
                stop_training = True

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
            eval_grid_cols,
            eval_grid_rows,
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
                eval_grid_cols,
                eval_grid_rows,
                death_cause=best.death_cause,
            )

        if hall_of_fame is not None and best_ever_score > 0:
            save_best_genome(
                replays / "best_score.npz",
                generation,
                hall_of_fame.genome.genes,
                hall_of_fame.score,
                hall_of_fame.best_food_seed,
                eval_grid_cols,
                eval_grid_rows,
                death_cause=hall_of_fame.death_cause,
            )

        grid_label = f"{eval_grid_cols}x{eval_grid_rows}"
        pop_size = len(population.individuals)
        avg_score = (
            sum(ind.score for ind in population.individuals) / pop_size if pop_size else 0.0
        )
        metrics = GenerationMetrics(
            generation=generation,
            grid_cols=eval_grid_cols,
            grid_rows=eval_grid_rows,
            best_fitness=best.fitness,
            avg_fitness=population.average_fitness(),
            best_score=best.score,
            max_score=max_score,
            avg_score=avg_score,
            best_ever_score=best_ever_score,
            death_cause=best.death_cause,
            win_count=gen_win_count,
            win_needed=gen_win_needed,
            population_scores=tuple(ind.score for ind in population.individuals),
            population_death_causes=tuple(ind.death_cause for ind in population.individuals),
        )
        append_training_log(training_log_path, metrics)
        if observer is not None:
            observer.on_generation(metrics)
        else:
            _print_generation_line(format_generation_line(metrics), observer=observer)

        if not stop_training:
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

        if stop_training:
            win_fraction = gen_win_count / float(pop_size)
            stop_msg = (
                f"--- Training complete: {gen_win_count}/{pop_size} wins "
                f"({win_fraction:.0%}) on {grid_cols}x{grid_rows} at gen {generation} ---"
            )
            if observer is not None:
                observer.on_curriculum(stop_msg)
            else:
                print(stop_msg, flush=True)
            break

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
        InlineProgress.update(message)

    def on_generation(self, metrics: GenerationMetrics) -> None:
        self._dashboard.add_generation(metrics)
        InlineProgress.finish(format_generation_line(metrics))

    def on_done(self, replays_path: Path) -> None:
        self._dashboard.set_done(replays_path)


def run_training_with_dashboard(args: argparse.Namespace) -> None:
    """Train in a background thread while the main thread shows live charts."""
    from ui.training_dashboard import TrainingDashboard

    training_done = threading.Event()
    training_error: list[BaseException] = []
    dashboard = TrainingDashboard(training_done)
    if args.resume:
        try:
            history = load_training_history(replays_dir())
        except (KeyError, TypeError, ValueError) as exc:
            print(f"Warning: could not load prior training history: {exc}", flush=True)
            history = []
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
    parser.add_argument(
        "--eval-runs",
        type=int,
        default=config.EVAL_RUNS_PER_GENOME,
        help="Screening boards averaged per snake (ignored with --single-shot).",
    )
    parser.add_argument(
        "--single-shot",
        action="store_true",
        help="One board per snake per generation; skip the top-fraction refine pass.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        metavar="N",
        help="Cap simulation steps per evaluation game (default: no cap).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel evaluation processes (0=auto: CPU count minus 1, 1=serial).",
    )
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
        help="Continue from replays/checkpoint.npz (runs --generations more generations; restores curriculum stage).",
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
    parser.add_argument(
        "--curriculum-win-pct",
        type=float,
        default=config.CURRICULUM_ADVANCE_WIN_FRACTION * 100.0,
        metavar="PCT",
        help=(
            "Percent of the population that must win to advance a curriculum stage "
            "or stop early on the final board (default: %(default)g)."
        ),
    )
    stop_group = parser.add_mutually_exclusive_group()
    stop_group.add_argument(
        "--stop-on-win",
        dest="stop_on_win",
        action="store_true",
        help="End early when enough snakes win the target board (default).",
    )
    stop_group.add_argument(
        "--no-stop-on-win",
        dest="stop_on_win",
        action="store_false",
        help="Run the full --generations count even after the population wins.",
    )
    parser.set_defaults(stop_on_win=config.TRAINING_STOP_ON_WIN)
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
    _apply_cli_config(args)
    if args.dashboard:
        import matplotlib

        matplotlib.use("TkAgg")
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
