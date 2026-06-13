"""
Scan saved per-generation replays and plot training progress.

Run from the ``src/`` directory:

    python analyze_training.py
    python analyze_training.py --output replays/training_analysis.png
    python analyze_training.py --skip-resim          # scores only (no death-cause pass)
    python analyze_training.py --window 25 --show

Reads ``replays/gen_XXXX.npz`` (score + genome + food seed). By default each file
is re-simulated once headlessly to recover death cause for the saved best run.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config
from evolution.genome import Genome
from evolution.training_log import load_training_history
from evolution.training_metrics import GenerationMetrics
from game.game_state import DeathCause
from models.grid import Grid
from neural.network import NeuralNetwork
from simulation.headless import HeadlessSimulator, Scenario

DEATH_CAUSES: tuple[DeathCause, ...] = ("body", "wall", "starved", "win")


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    generation: int
    score: int
    grid_cols: int
    grid_rows: int
    food_seed: int
    death_cause: DeathCause | None = None
    resim_score: int | None = None


def _parse_generation(path: Path) -> int:
    return int(path.stem.split("_", 1)[1])


def load_records(replays_dir: Path, *, resim: bool) -> list[GenerationRecord]:
    files = sorted(replays_dir.glob("gen_*.npz"), key=_parse_generation)
    if not files:
        raise FileNotFoundError(f"No gen_*.npz files in {replays_dir.resolve()}")

    records: list[GenerationRecord] = []
    sim_cache: dict[tuple[int, int], HeadlessSimulator] = {}

    for index, path in enumerate(files):
        data = np.load(path)
        generation = int(data["generation"])
        score = int(data["score"])
        food_seed = int(data["food_seed"])
        grid_cols = int(data["grid_cols"]) if "grid_cols" in data else config.GRID_COLS
        grid_rows = int(data["grid_rows"]) if "grid_rows" in data else config.GRID_ROWS

        death_cause: DeathCause | None = None
        if "death_cause" in data:
            raw = str(data["death_cause"])
            death_cause = _normalize_death_cause(raw)

        resim_score: int | None = None
        if resim and death_cause is None:
            grid_key = (grid_cols, grid_rows)
            if grid_key not in sim_cache:
                sim_cache[grid_key] = HeadlessSimulator(Grid(grid_cols, grid_rows))
            simulator = sim_cache[grid_key]
            simulator.set_scenarios([Scenario(food_seed=food_seed)])
            genome = Genome(np.asarray(data["genes"], dtype=np.float64))
            result = simulator.evaluate(genome)
            death_cause = result.death_cause
            resim_score = result.score

        records.append(
            GenerationRecord(
                generation=generation,
                score=score,
                grid_cols=grid_cols,
                grid_rows=grid_rows,
                food_seed=food_seed,
                death_cause=death_cause,
                resim_score=resim_score,
            )
        )

        if resim and (index + 1) % 250 == 0:
            print(f"  re-simulated {index + 1}/{len(files)} generations…", flush=True)

    return records


def write_csv(path: Path, records: list[GenerationRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["generation", "score", "best_ever", "grid", "death_cause", "resim_score"]
        )
        best_ever = 0
        for record in records:
            best_ever = max(best_ever, record.score)
            writer.writerow(
                [
                    record.generation,
                    record.score,
                    best_ever,
                    f"{record.grid_cols}x{record.grid_rows}",
                    record.death_cause or "",
                    record.resim_score if record.resim_score is not None else "",
                ]
            )


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="same")


def _bar_width(generations: np.ndarray) -> float:
    if len(generations) <= 1:
        return 0.8
    gaps = np.diff(generations.astype(np.float64))
    positive = gaps[gaps > 0]
    if len(positive) == 0:
        return 0.8
    return max(0.35, float(np.min(positive)) * 0.85)


def _death_cause_colors() -> dict[str, str]:
    return {
        "body": "#e45756",
        "wall": "#f58518",
        "starved": "#72b7b2",
        "win": "#54a24b",
    }


def _normalize_death_cause(raw: str) -> DeathCause | None:
    if raw == "timeout":
        raw = "starved"
    if raw in DEATH_CAUSES:
        return raw  # type: ignore[return-value]
    return None


def _population_death_fractions(metric: GenerationMetrics) -> dict[str, float]:
    """Per-cause share of the population for one generation (sums to 1)."""
    if metric.population_death_causes:
        counts: Counter[str] = Counter()
        for raw in metric.population_death_causes:
            cause = _normalize_death_cause(str(raw))
            if cause is not None:
                counts[cause] += 1
        total = sum(counts.values())
    else:
        cause = _normalize_death_cause(metric.death_cause)
        counts = Counter([cause] if cause is not None else [])
        total = sum(counts.values())
    if total <= 0:
        return dict.fromkeys(DEATH_CAUSES, 0.0)
    return {cause: counts.get(cause, 0) / total for cause in DEATH_CAUSES}


def _metrics_for_records(
    records: list[GenerationRecord],
    metrics: list[GenerationMetrics],
) -> list[GenerationMetrics]:
    by_generation = {metric.generation: metric for metric in metrics}
    aligned: list[GenerationMetrics] = []
    for record in records:
        metric = by_generation.get(record.generation)
        if metric is not None:
            aligned.append(metric)
    return aligned


def _plot_death_cause_bars(
    ax,
    metrics: list[GenerationMetrics],
    generations: np.ndarray,
) -> None:
    from matplotlib.patches import Patch

    if not metrics:
        return

    colors = _death_cause_colors()
    width = _bar_width(generations)
    gens = [metric.generation for metric in metrics]
    bottom = [0.0] * len(metrics)
    present_causes: set[str] = set()

    for cause in DEATH_CAUSES:
        fracs = [_population_death_fractions(metric)[cause] for metric in metrics]
        if not any(frac > 0 for frac in fracs):
            continue
        present_causes.add(cause)
        ax.bar(
            gens,
            fracs,
            width=width,
            bottom=bottom,
            color=colors[cause],
            align="center",
            edgecolor="none",
        )
        bottom = [base + frac for base, frac in zip(bottom, fracs)]

    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Population outcome share")
    ax.grid(True, alpha=0.25, axis="y")

    handles = [
        Patch(facecolor=colors[cause], label=cause)
        for cause in DEATH_CAUSES
        if cause in present_causes
    ]
    if handles:
        ax.legend(handles=handles, loc="upper right", ncol=min(4, len(handles)), fontsize=9)

def plot_analysis(
    records: list[GenerationRecord],
    metrics: list[GenerationMetrics],
    *,
    output: Path | None,
    show: bool,
    window: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plots. Install with: pip install matplotlib"
        ) from exc

    generations = np.array([record.generation for record in records], dtype=np.int64)
    scores = np.array([record.score for record in records], dtype=np.int64)
    best_ever = np.maximum.accumulate(scores)
    rolling = _rolling_mean(scores.astype(np.float64), window)

    bar_metrics = _metrics_for_records(records, metrics)
    has_causes = bool(bar_metrics)
    figure_height = 10 if has_causes else 7
    fig, axes = plt.subplots(3 if has_causes else 2, 1, figsize=(12, figure_height), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    ax_score, ax_best = axes[0], axes[1]
    ax_score.plot(generations, scores, color="#5a9fd4", alpha=0.35, linewidth=1, label="Best per gen")
    ax_score.plot(generations, rolling, color="#1f6fb2", linewidth=2, label=f"{window}-gen rolling avg")
    ax_score.set_ylabel("Score (apples)")
    ax_score.set_title("Training progress from saved replays")
    ax_score.grid(True, alpha=0.25)
    ax_score.legend(loc="upper left")

    ax_best.plot(generations, best_ever, color="#3ecf8e", linewidth=2, label="Best ever")
    ax_best.scatter(
        generations[np.where(np.diff(best_ever, prepend=0) > 0)[0]],
        best_ever[np.where(np.diff(best_ever, prepend=0) > 0)[0]],
        color="#ffd166",
        s=18,
        zorder=3,
        label="New record",
    )
    ax_best.set_ylabel("Best ever")
    ax_best.grid(True, alpha=0.25)
    ax_best.legend(loc="upper left")

    summary_lines = [
        f"Generations: {records[0].generation}–{records[-1].generation}",
        f"Peak score: {int(best_ever[-1])} (gen {int(generations[int(np.argmax(best_ever))])})",
        f"Final rolling avg: {rolling[-1]:.1f}",
    ]
    if has_causes:
        population_totals: Counter[str] = Counter()
        for metric in bar_metrics:
            if metric.population_death_causes:
                for raw in metric.population_death_causes:
                    cause = _normalize_death_cause(str(raw))
                    if cause is not None:
                        population_totals[cause] += 1
            else:
                cause = _normalize_death_cause(metric.death_cause)
                if cause is not None:
                    population_totals[cause] += 1
        dominant = max(population_totals, key=population_totals.get)
        summary_lines.append(
            "Population outcomes: "
            + ", ".join(f"{name} {population_totals[name]}" for name in DEATH_CAUSES if population_totals[name])
            + f" (dominant: {dominant})"
        )
    fig.text(0.12, 0.02, "  ·  ".join(summary_lines), fontsize=10, color="#444")

    if has_causes:
        ax_cause = axes[2]
        _plot_death_cause_bars(ax_cause, bar_metrics, generations)
        ax_cause.set_xlabel("Generation")
    else:
        ax_best.set_xlabel("Generation")

    fig.tight_layout(rect=(0, 0.04, 1, 1))

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output.resolve()}", flush=True)

    if show:
        plt.show()
    elif output is None:
        plt.show()
    else:
        plt.close(fig)


def print_summary(
    records: list[GenerationRecord],
    metrics: list[GenerationMetrics],
    *,
    window: int,
) -> None:
    scores = [record.score for record in records]
    best_ever = 0
    milestones = {5: None, 10: None, 15: None, 20: None, 23: None, 24: None}
    last_increase = 0
    peak = 0

    for record in records:
        if record.score > peak:
            peak = record.score
            last_increase = record.generation
        best_ever = max(best_ever, record.score)
        for threshold in milestones:
            if milestones[threshold] is None and record.score >= threshold:
                milestones[threshold] = record.generation

    recent = scores[-window:]
    print(f"Generations scanned: {len(records)} ({records[0].generation}-{records[-1].generation})")
    print(f"Peak score: {peak} (last improved at gen {last_increase})")
    print("First gen to reach score >= N:")
    for threshold in sorted(milestones):
        hit = milestones[threshold]
        label = str(hit) if hit is not None else "never"
        print(f"  >={threshold:2d}: {label}")
    print(f"Last {window}-gen avg best score: {sum(recent) / len(recent):.1f}")
    print(f"Last {window}-gen max best score: {max(recent)}")

    bar_metrics = _metrics_for_records(records, metrics)
    if bar_metrics:
        tail = bar_metrics[-window:]
        counts: Counter[str] = Counter()
        for metric in tail:
            if metric.population_death_causes:
                for raw in metric.population_death_causes:
                    cause = _normalize_death_cause(str(raw))
                    if cause is not None:
                        counts[cause] += 1
            else:
                cause = _normalize_death_cause(metric.death_cause)
                if cause is not None:
                    counts[cause] += 1
        print(f"Last {window}-gen population outcomes: {dict(counts)}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze saved GA training replays.")
    parser.add_argument(
        "--replays",
        type=Path,
        default=Path(config.REPLAYS_DIR),
        help=f"Directory with gen_*.npz files (default: {config.REPLAYS_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(config.REPLAYS_DIR) / "training_analysis.png",
        help="PNG path for the chart (default: replays/training_analysis.png)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(config.REPLAYS_DIR) / "training_summary.csv",
        help="Optional CSV export path (default: replays/training_summary.csv)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="Rolling window size for smoothed lines (default: 10)",
    )
    parser.add_argument(
        "--skip-resim",
        action="store_true",
        help="Skip headless re-simulation (scores only, no death-cause chart)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive matplotlib window",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Print summary + CSV only; do not write PNG",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    replays_dir = args.replays
    resim = not args.skip_resim

    print(f"Loading {replays_dir.resolve()}…", flush=True)
    if resim:
        print("Re-simulating saved best runs for death causes (may take a minute)…", flush=True)
    records = load_records(replays_dir, resim=resim)
    metrics = load_training_history(replays_dir)

    write_csv(args.csv, records)
    print(f"Wrote CSV to {args.csv.resolve()}", flush=True)
    print_summary(records, metrics, window=args.window)

    if not args.no_plot:
        plot_analysis(
            records,
            metrics,
            output=None if args.show else args.output,
            show=args.show,
            window=args.window,
        )


if __name__ == "__main__":
    main()
