"""
Live matplotlib dashboard for genetic-algorithm training.

Shows the same metrics as the CLI log as updating charts while training runs
in a background thread.
"""

from __future__ import annotations

import threading
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from evolution.training_metrics import GenerationMetrics, TrainingStartInfo, format_generation_line

_DEATH_CAUSES = ("body", "wall", "starved", "timeout")
_DEATH_COLORS = {
    "body": "#e45756",
    "wall": "#f58518",
    "starved": "#72b7b2",
    "timeout": "#4c78a8",
}


def _generation_xlim(
    start_info: TrainingStartInfo | None,
    *,
    span: tuple[int, int] | None = None,
    metrics: list[GenerationMetrics] | None = None,
) -> tuple[float, float] | None:
    """Full training span for the x-axis (inclusive generation indices)."""
    if span is not None:
        start, end_exclusive = span
    elif start_info is not None:
        start = start_info.start_generation
        end_exclusive = start_info.end_generation
    else:
        return None

    end_inclusive = end_exclusive - 1
    if metrics:
        start = min(start, metrics[0].generation)
        end_inclusive = max(end_inclusive, metrics[-1].generation)
    return (float(start) - 0.5, float(end_inclusive) + 0.5)


def _apply_generation_xlim(
    axes,
    start_info: TrainingStartInfo | None,
    *,
    span: tuple[int, int] | None = None,
    metrics: list[GenerationMetrics] | None = None,
) -> None:
    limits = _generation_xlim(start_info, span=span, metrics=metrics)
    if limits is None:
        return
    for ax in axes:
        ax.set_xlim(limits)


class TrainingDashboard:
    """Thread-safe collector + live charts for training progress."""

    def __init__(
        self,
        training_done: threading.Event,
        *,
        refresh_ms: int = 500,
        death_window: int = 80,
    ) -> None:
        self._training_done = training_done
        self._refresh_ms = refresh_ms
        self._death_window = death_window
        self._lock = threading.Lock()
        self._start_info: TrainingStartInfo | None = None
        self._metrics: list[GenerationMetrics] = []
        self._curriculum_notes: list[str] = []
        self._done_path: str | None = None
        self._closed = False
        self._generation_span: tuple[int, int] | None = None

    def set_generation_span(self, start_generation: int, end_generation: int) -> None:
        """Preset x-axis range before the first metrics arrive (end_generation is exclusive)."""
        with self._lock:
            self._generation_span = (start_generation, end_generation)

    def set_start_info(self, info: TrainingStartInfo) -> None:
        with self._lock:
            self._start_info = info
            if self._metrics:
                x_start = min(m.generation for m in self._metrics)
                x_start = min(x_start, info.start_generation)
            else:
                x_start = info.start_generation
            self._generation_span = (x_start, info.end_generation)

    def load_metrics(self, metrics: list[GenerationMetrics]) -> None:
        """Pre-load history (e.g. from a prior run) before live updates arrive."""
        with self._lock:
            self._metrics = sorted(metrics, key=lambda m: m.generation)
            if self._metrics:
                end = self._generation_span[1] if self._generation_span else self._metrics[-1].generation + 1
                self._generation_span = (self._metrics[0].generation, end)

    def add_generation(self, metrics: GenerationMetrics) -> None:
        with self._lock:
            self._metrics.append(metrics)

    def add_curriculum_note(self, note: str) -> None:
        with self._lock:
            self._curriculum_notes.append(note)

    def set_done(self, replays_path: Path) -> None:
        with self._lock:
            self._done_path = str(replays_path.resolve())

    def log_generation(self, metrics: GenerationMetrics) -> None:
        """Push to charts and mirror the CLI log line."""
        self.add_generation(metrics)
        print(format_generation_line(metrics), flush=True)

    def run(self) -> None:
        """Block on the dashboard window until the user closes it."""
        plt.style.use("ggplot")
        fig = plt.figure(figsize=(12, 8))
        try:
            fig.canvas.manager.set_window_title("Genetic Snake - Training")
        except AttributeError:
            pass
        grid = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.0, 0.55], hspace=0.38, wspace=0.28)

        ax_scores = fig.add_subplot(grid[0, 0])
        ax_fitness = fig.add_subplot(grid[0, 1])
        ax_death = fig.add_subplot(grid[1, :])
        ax_status = fig.add_subplot(grid[2, :])
        ax_status.axis("off")

        lines: dict[str, object] = {}

        def _init_axes() -> None:
            ax_scores.set_title("Scores (apples)")
            ax_scores.set_xlabel("Generation")
            ax_scores.grid(True, alpha=0.35)
            lines["best_score"], = ax_scores.plot([], [], color="#5a9fd4", label="best_score", linewidth=1.5)
            lines["max_score"], = ax_scores.plot([], [], color="#9ecae1", alpha=0.8, label="max_score", linewidth=1)
            lines["best_ever"], = ax_scores.plot([], [], color="#3ecf8e", label="best_ever", linewidth=2)
            lines["avg_max10"], = ax_scores.plot([], [], color="#ffd166", linestyle="--", label="avg_max10", linewidth=1.5)
            ax_scores.legend(loc="upper left", fontsize=8)

            ax_fitness.set_title("Fitness")
            ax_fitness.set_xlabel("Generation")
            ax_fitness.set_yscale("symlog", linthresh=100.0)
            ax_fitness.grid(True, alpha=0.35)
            lines["best_fit"], = ax_fitness.plot([], [], color="#b07aa1", label="best_fit", linewidth=1.5)
            lines["avg_fit"], = ax_fitness.plot([], [], color="#d4a6c8", linestyle="--", label="avg_fit", linewidth=1)
            ax_fitness.legend(loc="upper left", fontsize=8)

            ax_death.set_title(f"Death cause (best snake, last {self._death_window} gens)")
            ax_death.set_xlabel("Generation")
            ax_death.set_ylabel("Count in window")
            ax_death.grid(True, alpha=0.35, axis="y")

        _init_axes()

        def _snapshot() -> tuple[
            TrainingStartInfo | None,
            tuple[int, int] | None,
            list[GenerationMetrics],
            list[str],
            str | None,
            bool,
        ]:
            with self._lock:
                return (
                    self._start_info,
                    self._generation_span,
                    list(self._metrics),
                    list(self._curriculum_notes),
                    self._done_path,
                    self._closed,
                )

        plot_axes = (ax_scores, ax_fitness, ax_death)

        def _update_death_bars(
            ax,
            metrics: list[GenerationMetrics],
            start_info: TrainingStartInfo | None,
            generation_span: tuple[int, int] | None,
        ) -> None:
            ax.clear()
            ax.set_title(f"Death cause counts (rolling {self._death_window}-gen window)")
            ax.set_xlabel("Generation")
            ax.set_ylabel("Count in window")
            ax.grid(True, alpha=0.35, axis="y")
            _apply_generation_xlim((ax,), start_info, span=generation_span, metrics=metrics)
            if len(metrics) < 2:
                return
            gens = [m.generation for m in metrics]
            window = min(self._death_window, len(metrics))
            for cause in _DEATH_CAUSES:
                counts = []
                for end_idx in range(len(metrics)):
                    start_idx = max(0, end_idx - window + 1)
                    slice_ = metrics[start_idx : end_idx + 1]
                    counts.append(sum(1 for m in slice_ if m.death_cause == cause))
                ax.plot(
                    gens,
                    counts,
                    label=cause,
                    color=_DEATH_COLORS[cause],
                    linewidth=1.5,
                    alpha=0.9,
                )
            ax.legend(loc="upper right", ncol=4, fontsize=8)

        def _update(_frame: int) -> None:
            start_info, generation_span, metrics, curriculum_notes, done_path, closed = _snapshot()
            _apply_generation_xlim(plot_axes, start_info, span=generation_span, metrics=metrics)

            if not metrics:
                if start_info is not None:
                    ax_status.clear()
                    ax_status.axis("off")
                    ax_status.text(
                        0.02,
                        0.85,
                        _status_header(start_info),
                        transform=ax_status.transAxes,
                        fontsize=10,
                        family="monospace",
                        verticalalignment="top",
                    )
                return

            gens = [m.generation for m in metrics]
            lines["best_score"].set_data(gens, [m.best_score for m in metrics])
            lines["max_score"].set_data(gens, [m.max_score for m in metrics])
            lines["best_ever"].set_data(gens, [m.best_ever_score for m in metrics])
            lines["avg_max10"].set_data(gens, [m.avg_max10 for m in metrics])
            ax_scores.relim()
            ax_scores.autoscale(axis="y")

            best_fit = [m.best_fitness for m in metrics]
            avg_fit = [m.avg_fitness for m in metrics]
            lines["best_fit"].set_data(gens, best_fit)
            lines["avg_fit"].set_data(gens, avg_fit)
            ax_fitness.relim()
            ax_fitness.autoscale(axis="y")
            if best_fit:
                positive = [v for v in best_fit + avg_fit if v > 0]
                if positive:
                    ymin = min(positive) * 0.5
                    ymax = max(positive) * 2.0
                    ax_fitness.set_ylim(ymin, ymax)

            _update_death_bars(ax_death, metrics, start_info, generation_span)

            latest = metrics[-1]
            cause_counts = Counter(m.death_cause for m in metrics[-min(20, len(metrics)) :])

            ax_status.clear()
            ax_status.axis("off")
            header = _status_header(start_info) if start_info else "Training…"
            status = (
                f"{header}\n"
                f"Latest  gen {latest.generation}  grid {latest.grid_label}  "
                f"died {latest.death_cause}\n"
                f"         best_score {latest.best_score}  max_score {latest.max_score}  "
                f"best_ever {latest.best_ever_score}  avg_max10 {latest.avg_max10:.1f}\n"
                f"         best_fit {latest.best_fitness:.2f}  avg_fit {latest.avg_fitness:.2f}\n"
                f"Recent deaths (20 gens): {dict(cause_counts)}"
            )
            if curriculum_notes:
                status += f"\nCurriculum: {curriculum_notes[-1]}"
            if done_path:
                status += f"\nDone — saved to {done_path}"
            elif self._training_done.is_set():
                status += "\nTraining finished (close window to exit)."
            ax_status.text(
                0.02,
                0.95,
                status,
                transform=ax_status.transAxes,
                fontsize=9,
                family="monospace",
                verticalalignment="top",
            )

            if closed:
                anim.event_source.stop()

        def _on_close(_event) -> None:
            with self._lock:
                self._closed = True

        fig.canvas.mpl_connect("close_event", _on_close)
        anim = FuncAnimation(fig, _update, interval=self._refresh_ms, cache_frame_data=False)
        _ = anim
        plt.show()


def _status_header(info: TrainingStartInfo) -> str:
    return (
        f"pop={info.population}  gens={info.start_generation}-{info.end_generation - 1}  "
        f"({info.resume_note})\n"
        f"arch={info.arch_label}  genes={info.genome_length}  {info.breeding_note}\n"
        f"curriculum=[{info.curriculum_note}]  eval={info.eval_note}  {info.refine_note}"
    )
