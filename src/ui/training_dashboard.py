"""
Live matplotlib dashboard for genetic-algorithm training.

Training runs on a background thread; the main thread owns the Tk/matplotlib
window and polls shared metrics on a timer (more reliable on Windows than
FuncAnimation with threaded writers).
"""

from __future__ import annotations

import math
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import config
from evolution.training_metrics import GenerationMetrics, TrainingStartInfo

_DEATH_CAUSES = ("body", "wall", "starved", "timeout", "win")
_DEATH_COLORS = {
    "body": "#e45756",
    "wall": "#f58518",
    "starved": "#72b7b2",
    "timeout": "#4c78a8",
    "win": "#54a24b",
}
_WIN_LINE_COLOR = "#54a24b"
_GRID_MARKER_COLOR = "#9aa0a8"
_MAX_BOX_PLOTS = 220


def _finite(value: float, default: float = 0.0) -> float:
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _generation_xlim(
    start_info: TrainingStartInfo | None,
    metrics: list[GenerationMetrics],
    *,
    min_span: int = 10,
    right_pad: int = 2,
) -> tuple[float, float] | None:
    """X-axis from the first logged generation through the latest (+ small padding)."""
    if metrics:
        start = metrics[0].generation
        end = metrics[-1].generation
    elif start_info is not None:
        start = start_info.start_generation
        end = start
    else:
        return None

    if start_info is not None:
        start = min(start, start_info.start_generation)

    if end - start < min_span - 1:
        end = start + min_span - 1

    return (float(start) - 0.5, float(end + right_pad) + 0.5)


def _current_stage_win_score(
    metrics: list[GenerationMetrics],
    start_info: TrainingStartInfo | None,
) -> int:
    """Win threshold for the active grid (line chart Y-axis tracks the current stage only)."""
    if metrics:
        latest = metrics[-1]
        return config.max_win_score(latest.grid_cols, latest.grid_rows)
    if start_info is not None and "->" in start_info.curriculum_note:
        cols, rows = config.CURRICULUM_STAGES[0]
        return config.max_win_score(cols, rows)
    return config.max_win_score(config.GRID_COLS, config.GRID_ROWS)


def _line_score_ylim(
    metrics: list[GenerationMetrics],
    start_info: TrainingStartInfo | None,
) -> tuple[float, float]:
    peak = _current_stage_win_score(metrics, start_info)
    return (-0.5, peak + max(2, peak * 0.05))


def _grid_transitions(
    metrics: list[GenerationMetrics],
) -> list[tuple[int, str]]:
    """Return (generation, grid_label) for each curriculum stage change."""
    transitions: list[tuple[int, str]] = []
    for index in range(1, len(metrics)):
        prev = metrics[index - 1]
        cur = metrics[index]
        if (prev.grid_cols, prev.grid_rows) != (cur.grid_cols, cur.grid_rows):
            transitions.append((prev.generation, cur.grid_label))
    return transitions


def _win_line_segments(
    metrics: list[GenerationMetrics],
) -> list[tuple[int, int, int]]:
    """Horizontal win thresholds: (start_gen, end_gen, apples_to_win) per grid stage."""
    if not metrics:
        return []
    segments: list[tuple[int, int, int]] = []
    start_index = 0
    for index in range(1, len(metrics)):
        prev = metrics[index - 1]
        cur = metrics[index]
        if (prev.grid_cols, prev.grid_rows) != (cur.grid_cols, cur.grid_rows):
            segments.append(
                (
                    metrics[start_index].generation,
                    prev.generation,
                    config.max_win_score(prev.grid_cols, prev.grid_rows),
                )
            )
            start_index = index
    last = metrics[-1]
    segments.append(
        (
            metrics[start_index].generation,
            last.generation,
            config.max_win_score(last.grid_cols, last.grid_rows),
        )
    )
    return segments


def _normalized_population_scores(metric: GenerationMetrics) -> list[float]:
    """Score as a fraction of apples needed to win on that generation's grid."""
    win = config.max_win_score(metric.grid_cols, metric.grid_rows)
    if win <= 0 or not metric.population_scores:
        return []
    return [min(1.0, score / win) for score in metric.population_scores]


def _death_cause_fractions(metric: GenerationMetrics) -> dict[str, float]:
    """Per-cause share of the population for one generation (sums to 1)."""
    if metric.population_death_causes:
        counts = Counter(metric.population_death_causes)
        total = sum(counts.values())
    else:
        counts = Counter([metric.death_cause])
        total = 1
    if total <= 0:
        return dict.fromkeys(_DEATH_CAUSES, 0.0)
    return {cause: counts.get(cause, 0) / total for cause in _DEATH_CAUSES}


def _bar_width(generation_count: int, limits: tuple[float, float] | None) -> float:
    span = max(1.0, (limits[1] - limits[0]) if limits else float(generation_count))
    return min(0.85, max(0.15, span / max(generation_count, 1) * 0.7))


def _box_plot_metrics(metrics: list[GenerationMetrics]) -> list[GenerationMetrics]:
    """Thin long histories so box plots stay responsive."""
    with_scores = [m for m in metrics if m.population_scores]
    if len(with_scores) <= _MAX_BOX_PLOTS:
        return with_scores
    step = max(1, len(with_scores) // _MAX_BOX_PLOTS)
    sampled = with_scores[::step]
    if sampled[-1] is not with_scores[-1]:
        sampled.append(with_scores[-1])
    return sampled


@dataclass
class _Snapshot:
    start_info: TrainingStartInfo | None
    metrics: list[GenerationMetrics]
    curriculum_notes: list[str]
    progress: str | None
    done_path: str | None
    error: str | None


class TrainingDashboard:
    """Thread-safe metrics store with a live matplotlib window."""

    def __init__(
        self,
        training_done: threading.Event,
        *,
        refresh_ms: int = 400,
    ) -> None:
        self._training_done = training_done
        self._refresh_ms = refresh_ms
        self._lock = threading.Lock()
        self._start_info: TrainingStartInfo | None = None
        self._metrics: list[GenerationMetrics] = []
        self._curriculum_notes: list[str] = []
        self._progress: str | None = None
        self._done_path: str | None = None
        self._error: str | None = None
        self._closed = False

    def set_start_info(self, info: TrainingStartInfo) -> None:
        with self._lock:
            self._start_info = info

    def load_metrics(self, metrics: list[GenerationMetrics]) -> None:
        with self._lock:
            self._metrics = sorted(metrics, key=lambda m: m.generation)

    def add_generation(self, metrics: GenerationMetrics) -> None:
        with self._lock:
            self._metrics.append(metrics)

    def add_curriculum_note(self, note: str) -> None:
        with self._lock:
            self._curriculum_notes.append(note)

    def set_progress(self, message: str) -> None:
        with self._lock:
            self._progress = message

    def set_done(self, replays_path: Path) -> None:
        with self._lock:
            self._done_path = str(replays_path.resolve())

    def _snapshot(self) -> _Snapshot:
        with self._lock:
            return _Snapshot(
                start_info=self._start_info,
                metrics=list(self._metrics),
                curriculum_notes=list(self._curriculum_notes),
                progress=self._progress,
                done_path=self._done_path,
                error=self._error,
            )

    def run(self) -> None:
        import matplotlib.pyplot as plt

        plt.style.use("ggplot")
        fig = plt.figure(figsize=(12, 8.5), constrained_layout=True)
        try:
            fig.canvas.manager.set_window_title("Genetic Snake - Training")
        except AttributeError:
            pass

        grid = fig.add_gridspec(
            3,
            2,
            height_ratios=[1.0, 0.95, 0.4],
            width_ratios=[1.15, 1.0],
        )
        ax_scores = fig.add_subplot(grid[0, 0])
        ax_boxes = fig.add_subplot(grid[1, 0], sharex=ax_scores)
        ax_fitness = fig.add_subplot(grid[0, 1])
        ax_death = fig.add_subplot(grid[1, 1], sharex=ax_fitness)
        ax_status = fig.add_subplot(grid[2, :])
        ax_status.axis("off")

        score_artists = {
            "best_score": ax_scores.plot([], [], color="#5a9fd4", label="best", linewidth=1.5)[0],
            "max_score": ax_scores.plot([], [], color="#9ecae1", alpha=0.85, label="max", linewidth=1)[0],
            "best_ever": ax_scores.plot([], [], color="#3ecf8e", label="best_ever", linewidth=2)[0],
            "avg_score": ax_scores.plot(
                [], [], color="#ffd166", linestyle="--", label="avg", linewidth=1.2
            )[0],
        }
        ax_scores.set_title("Scores (all curriculum stages)")
        ax_scores.set_ylabel("Apples")
        ax_scores.grid(True, alpha=0.35)
        ax_scores.legend(loc="upper left", fontsize=7, ncol=2)
        plt.setp(ax_scores.get_xticklabels(), visible=False)

        ax_boxes.set_title("Population score distribution (normalized)")
        ax_boxes.set_xlabel("Generation")
        ax_boxes.set_ylabel("Score / win")
        ax_boxes.set_ylim(-0.02, 1.08)
        ax_boxes.grid(True, alpha=0.35, axis="y")

        fitness_artists = {
            "best_fit": ax_fitness.plot([], [], color="#b07aa1", label="best_fit", linewidth=1.5)[0],
            "avg_fit": ax_fitness.plot([], [], color="#d4a6c8", linestyle="--", label="avg_fit", linewidth=1)[0],
        }
        ax_fitness.set_title("Fitness (all stages)")
        ax_fitness.set_ylabel("Fitness")
        ax_fitness.set_yscale("symlog", linthresh=100.0)
        ax_fitness.grid(True, alpha=0.35)
        ax_fitness.legend(loc="upper left", fontsize=8)
        plt.setp(ax_fitness.get_xticklabels(), visible=False)

        ax_death.set_title("Death cause (population, normalized)")
        ax_death.set_xlabel("Generation")
        ax_death.set_ylabel("Fraction")
        ax_death.set_ylim(0.0, 1.0)
        ax_death.grid(True, alpha=0.35, axis="y")

        chart_decorations: list = []

        def _clear_chart_decorations() -> None:
            for artist in chart_decorations:
                artist.remove()
            chart_decorations.clear()

        def _draw_grid_markers(
            axes: list,
            metrics: list[GenerationMetrics],
            *,
            show_labels: bool = False,
        ) -> None:
            for last_old_gen, grid_label in _grid_transitions(metrics):
                marker_x = last_old_gen + 0.5
                for axis in axes:
                    chart_decorations.append(
                        axis.axvline(
                            marker_x,
                            color=_GRID_MARKER_COLOR,
                            linewidth=1.0,
                            alpha=0.65,
                            zorder=0,
                        )
                    )
                if show_labels:
                    chart_decorations.append(
                        ax_scores.text(
                            marker_x,
                            0.02,
                            grid_label,
                            transform=ax_scores.get_xaxis_transform(),
                            fontsize=7,
                            color=_GRID_MARKER_COLOR,
                            ha="left",
                            va="bottom",
                            rotation=90,
                            alpha=0.9,
                        )
                    )

        def _draw_win_lines(axes: list, metrics: list[GenerationMetrics]) -> None:
            for start_gen, end_gen, win_level in _win_line_segments(metrics):
                for axis in axes:
                    (line,) = axis.plot(
                        [start_gen, end_gen],
                        [win_level, win_level],
                        color=_WIN_LINE_COLOR,
                        linestyle=":",
                        linewidth=1.3,
                        alpha=0.8,
                        zorder=1,
                    )
                    chart_decorations.append(line)

        def _draw_normalized_win_line(
            axis,
            limits: tuple[float, float] | None,
        ) -> None:
            if limits is None:
                return
            (line,) = axis.plot(
                [limits[0], limits[1]],
                [1.0, 1.0],
                color=_WIN_LINE_COLOR,
                linestyle=":",
                linewidth=1.3,
                alpha=0.8,
                zorder=1,
            )
            chart_decorations.append(line)

        def _update_score_panels(
            metrics: list[GenerationMetrics],
            snap: _Snapshot,
        ) -> None:
            ylim = _line_score_ylim(metrics, snap.start_info)
            box_ylim = (-0.02, 1.08)
            ax_scores.set_ylim(ylim)
            ax_boxes.set_ylim(box_ylim)
            limits = _generation_xlim(snap.start_info, metrics)
            if limits is not None:
                ax_scores.set_xlim(limits)
                ax_boxes.set_xlim(limits)

            if not metrics:
                for artist in score_artists.values():
                    artist.set_data([], [])
                ax_boxes.cla()
                ax_boxes.set_title("Population score distribution (normalized)")
                ax_boxes.set_xlabel("Generation")
                ax_boxes.set_ylabel("Score / win")
                ax_boxes.grid(True, alpha=0.35, axis="y")
                ax_boxes.set_ylim(box_ylim)
                if limits is not None:
                    ax_boxes.set_xlim(limits)
                return

            gens = [m.generation for m in metrics]
            score_artists["best_score"].set_data(gens, [m.best_score for m in metrics])
            score_artists["max_score"].set_data(gens, [m.max_score for m in metrics])
            score_artists["best_ever"].set_data(gens, [m.best_ever_score for m in metrics])
            score_artists["avg_score"].set_data(gens, [m.avg_score for m in metrics])

            ax_boxes.cla()
            ax_boxes.set_title("Population score distribution (normalized)")
            ax_boxes.set_xlabel("Generation")
            ax_boxes.set_ylabel("Score / win")
            ax_boxes.grid(True, alpha=0.35, axis="y")
            ax_boxes.set_ylim(box_ylim)
            if limits is not None:
                ax_boxes.set_xlim(limits)

            box_metrics = _box_plot_metrics(metrics)
            if box_metrics:
                positions = [m.generation for m in box_metrics]
                data = [_normalized_population_scores(m) for m in box_metrics]
                width = _bar_width(len(box_metrics), limits)
                ax_boxes.boxplot(
                    data,
                    positions=positions,
                    widths=width,
                    showfliers=False,
                    patch_artist=True,
                    boxprops={"facecolor": "#5a9fd4", "alpha": 0.35, "linewidth": 0.8},
                    medianprops={"color": "#1f4e79", "linewidth": 1.2},
                    whiskerprops={"color": "#5a9fd4", "linewidth": 0.8},
                    capprops={"color": "#5a9fd4", "linewidth": 0.8},
                )

            _draw_grid_markers([ax_scores, ax_boxes], metrics, show_labels=True)
            _draw_win_lines([ax_scores], metrics)
            _draw_normalized_win_line(ax_boxes, limits)

        def _update_death_chart(metrics: list[GenerationMetrics], snap: _Snapshot) -> None:
            ax_death.clear()
            ax_death.set_title("Death cause (population, normalized)")
            ax_death.set_xlabel("Generation")
            ax_death.set_ylabel("Fraction")
            ax_death.set_ylim(0.0, 1.0)
            ax_death.grid(True, alpha=0.35, axis="y")
            limits = _generation_xlim(snap.start_info, metrics)
            if limits is not None:
                ax_death.set_xlim(limits)
            if not metrics:
                return

            bar_metrics = _box_plot_metrics(metrics)
            gens = [m.generation for m in bar_metrics]
            width = _bar_width(len(bar_metrics), limits)
            bottom = [0.0] * len(bar_metrics)
            for cause in _DEATH_CAUSES:
                fracs = [_death_cause_fractions(m)[cause] for m in bar_metrics]
                ax_death.bar(
                    gens,
                    fracs,
                    width=width,
                    bottom=bottom,
                    color=_DEATH_COLORS[cause],
                    label=cause,
                    align="center",
                    edgecolor="none",
                )
                bottom = [base + frac for base, frac in zip(bottom, fracs)]
            ax_death.legend(loc="upper left", ncol=5, fontsize=7)
            _draw_grid_markers([ax_death], metrics)

        def _update_status(ax, snap: _Snapshot, metrics: list[GenerationMetrics]) -> None:
            ax.clear()
            ax.axis("off")
            if snap.start_info is not None:
                header = _status_header(snap.start_info)
            else:
                header = "Training…"

            if not metrics:
                body = header
                if snap.progress:
                    body += f"\n{snap.progress}"
                if snap.error:
                    body += f"\nError: {snap.error}"
                ax.text(
                    0.02,
                    0.92,
                    body,
                    transform=ax.transAxes,
                    fontsize=9,
                    family="monospace",
                    verticalalignment="top",
                )
                return

            latest = metrics[-1]
            recent = metrics[-min(20, len(metrics)) :]
            cause_counts = Counter(item.death_cause for item in recent)
            wins = ""
            if latest.win_needed > 0:
                wins = f"  wins {latest.win_count}/{latest.win_needed}"

            body = (
                f"{header}\n"
                f"Latest  gen {latest.generation}  grid {latest.grid_label}  died {latest.death_cause}{wins}\n"
                f"         best_score {latest.best_score}  max_score {latest.max_score}  "
                f"best_ever {latest.best_ever_score}  avg_score {latest.avg_score:.1f}\n"
                f"         best_fit {latest.best_fitness:.2f}  avg_fit {latest.avg_fitness:.2f}\n"
                f"Recent deaths (20 gens): {dict(cause_counts)}"
            )
            if snap.curriculum_notes:
                body += f"\nCurriculum: {snap.curriculum_notes[-1]}"
            if snap.progress:
                body += f"\n{snap.progress}"
            if snap.done_path:
                body += f"\nDone — saved to {snap.done_path}"
            elif self._training_done.is_set():
                body += "\nTraining finished (close window to exit)."
            if snap.error:
                body += f"\nError: {snap.error}"
            ax.text(
                0.02,
                0.95,
                body,
                transform=ax.transAxes,
                fontsize=9,
                family="monospace",
                verticalalignment="top",
            )

        def _refresh() -> None:
            if self._closed:
                return
            try:
                snap = self._snapshot()
                metrics = snap.metrics

                _clear_chart_decorations()
                _update_score_panels(metrics, snap)

                if metrics:
                    gens = [m.generation for m in metrics]
                    limits = _generation_xlim(snap.start_info, metrics)
                    if limits is not None:
                        ax_fitness.set_xlim(limits)

                    best_fit = [_finite(m.best_fitness) for m in metrics]
                    avg_fit = [_finite(m.avg_fitness) for m in metrics]
                    fitness_artists["best_fit"].set_data(gens, best_fit)
                    fitness_artists["avg_fit"].set_data(gens, avg_fit)
                    ax_fitness.relim()
                    ax_fitness.autoscale(axis="y")
                    positive = [v for v in best_fit + avg_fit if v > 0]
                    if positive:
                        ax_fitness.set_ylim(min(positive) * 0.5, max(positive) * 2.0)

                    _draw_grid_markers([ax_fitness], metrics)
                    _update_death_chart(metrics, snap)
                else:
                    ax_death.clear()
                    ax_death.set_title("Death cause (population, normalized)")
                    ax_death.set_xlabel("Generation")
                    ax_death.set_ylabel("Fraction")
                    ax_death.set_ylim(0.0, 1.0)
                    ax_death.grid(True, alpha=0.35, axis="y")

                _update_status(ax_status, snap, metrics)
            except Exception as exc:
                with self._lock:
                    self._error = str(exc)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

        def _schedule_refresh() -> None:
            if self._closed:
                return
            _refresh()
            try:
                fig.canvas.get_tk_widget().after(self._refresh_ms, _schedule_refresh)
            except AttributeError:
                timer = fig.canvas.new_timer(interval=self._refresh_ms, single=True)
                timer.add_callback(_schedule_refresh)
                timer.start()

        def _on_close(_event) -> None:
            self._closed = True

        fig.canvas.mpl_connect("close_event", _on_close)
        _refresh()
        _schedule_refresh()
        plt.show()


def _status_header(info: TrainingStartInfo) -> str:
    return (
        f"pop={info.population}  gens={info.start_generation}-{info.end_generation - 1}  "
        f"({info.resume_note})\n"
        f"arch={info.arch_label}  genes={info.genome_length}  {info.breeding_note}\n"
        f"curriculum=[{info.curriculum_note}]  eval={info.eval_note}  {info.refine_note}\n"
        f"max_steps={info.max_steps_note}  workers={info.workers_note}"
    )
