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
from dataclasses import dataclass, field
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


def _finite(value: float, default: float = 0.0) -> float:
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _score_stages(start_info: TrainingStartInfo | None) -> tuple[tuple[int, int], ...]:
    """Board sizes to show as separate score panels (curriculum vs fixed grid)."""
    if start_info is not None and "->" not in start_info.curriculum_note:
        return ((config.GRID_COLS, config.GRID_ROWS),)
    return config.CURRICULUM_STAGES


def _metrics_for_stage(
    metrics: list[GenerationMetrics],
    cols: int,
    rows: int,
) -> list[GenerationMetrics]:
    return [m for m in metrics if m.grid_cols == cols and m.grid_rows == rows]


def _stage_xlim(
    stage_metrics: list[GenerationMetrics],
    *,
    min_span: int = 10,
    right_pad: int = 2,
) -> tuple[float, float] | None:
    if not stage_metrics:
        return None
    start = stage_metrics[0].generation
    end = stage_metrics[-1].generation
    if end - start < min_span - 1:
        end = start + min_span - 1
    return (float(start) - 0.5, float(end + right_pad) + 0.5)


def _rolling_avg_max10(stage_metrics: list[GenerationMetrics], window: int = 10) -> list[float]:
    avgs: list[float] = []
    for end_idx in range(len(stage_metrics)):
        start_idx = max(0, end_idx - window + 1)
        slice_ = stage_metrics[start_idx : end_idx + 1]
        avgs.append(sum(m.max_score for m in slice_) / len(slice_))
    return avgs


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


@dataclass
class _StageScorePanel:
    cols: int
    rows: int
    ax: object
    artists: dict[str, object] = field(default_factory=dict)
    win_line: object | None = None

    @property
    def grid_label(self) -> str:
        return f"{self.cols}x{self.rows}"

    @property
    def max_win(self) -> int:
        return config.max_win_score(self.cols, self.rows)


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
        death_window: int = 40,
    ) -> None:
        self._training_done = training_done
        self._refresh_ms = refresh_ms
        self._death_window = death_window
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
        snap = self._snapshot()
        stages = _score_stages(snap.start_info)
        n_stages = len(stages)
        fig_height = 4.5 + n_stages * 1.6
        fig = plt.figure(figsize=(12, fig_height), constrained_layout=True)
        try:
            fig.canvas.manager.set_window_title("Genetic Snake - Training")
        except AttributeError:
            pass

        height_ratios = [0.9] * n_stages + [1.0, 0.45]
        grid = fig.add_gridspec(n_stages + 2, 2, height_ratios=height_ratios, width_ratios=[1.15, 1.0])

        score_panels: list[_StageScorePanel] = []
        for row, (cols, rows) in enumerate(stages):
            ax = fig.add_subplot(grid[row, 0])
            max_win = config.max_win_score(cols, rows)
            panel = _StageScorePanel(cols=cols, rows=rows, ax=ax)
            panel.artists = {
                "best_score": ax.plot([], [], color="#5a9fd4", label="best", linewidth=1.5)[0],
                "max_score": ax.plot([], [], color="#9ecae1", alpha=0.85, label="max", linewidth=1)[0],
                "best_ever": ax.plot([], [], color="#3ecf8e", label="best_ever", linewidth=2)[0],
                "avg_max10": ax.plot(
                    [], [], color="#ffd166", linestyle="--", label="avg_max10", linewidth=1.2
                )[0],
            }
            panel.win_line = ax.axhline(
                max_win,
                color="#54a24b",
                linestyle=":",
                linewidth=1.2,
                alpha=0.75,
                label="win",
            )
            ax.set_title(f"Scores — {panel.grid_label} (max {max_win})")
            ax.set_xlabel("Generation")
            ax.set_ylabel("Apples")
            ax.set_ylim(-0.5, max_win + max(2, max_win * 0.05))
            ax.grid(True, alpha=0.35)
            ax.legend(loc="upper left", fontsize=7, ncol=2)
            score_panels.append(panel)

        ax_fitness = fig.add_subplot(grid[0:n_stages, 1])
        ax_death = fig.add_subplot(grid[n_stages, :])
        ax_status = fig.add_subplot(grid[n_stages + 1, :])
        ax_status.axis("off")

        fitness_artists = {
            "best_fit": ax_fitness.plot([], [], color="#b07aa1", label="best_fit", linewidth=1.5)[0],
            "avg_fit": ax_fitness.plot([], [], color="#d4a6c8", linestyle="--", label="avg_fit", linewidth=1)[0],
        }
        ax_fitness.set_title("Fitness (all stages)")
        ax_fitness.set_xlabel("Generation")
        ax_fitness.set_yscale("symlog", linthresh=100.0)
        ax_fitness.grid(True, alpha=0.35)
        ax_fitness.legend(loc="upper left", fontsize=8)

        def _update_stage_scores(
            panels: list[_StageScorePanel],
            metrics: list[GenerationMetrics],
        ) -> None:
            for panel in panels:
                stage_metrics = _metrics_for_stage(metrics, panel.cols, panel.rows)
                ax = panel.ax
                if not stage_metrics:
                    for artist in panel.artists.values():
                        artist.set_data([], [])
                    ax.set_title(f"Scores — {panel.grid_label} (max {panel.max_win}) — not started")
                    continue

                gens = [m.generation for m in stage_metrics]
                panel.artists["best_score"].set_data(gens, [m.best_score for m in stage_metrics])
                panel.artists["max_score"].set_data(gens, [m.max_score for m in stage_metrics])
                panel.artists["best_ever"].set_data(gens, [m.best_ever_score for m in stage_metrics])
                panel.artists["avg_max10"].set_data(gens, _rolling_avg_max10(stage_metrics))

                limits = _stage_xlim(stage_metrics)
                if limits is not None:
                    ax.set_xlim(limits)
                ax.set_ylim(-0.5, panel.max_win + max(2, panel.max_win * 0.05))

                if metrics and metrics[-1].grid_cols == panel.cols and metrics[-1].grid_rows == panel.rows:
                    status = "active"
                else:
                    status = "complete"
                ax.set_title(f"Scores — {panel.grid_label} (max {panel.max_win}) — {status}")

        def _update_death_chart(metrics: list[GenerationMetrics], snap: _Snapshot) -> None:
            ax_death.clear()
            ax_death.set_title(f"Death cause (best snake, rolling {self._death_window}-gen window)")
            ax_death.set_xlabel("Generation")
            ax_death.set_ylabel("Count in window")
            ax_death.grid(True, alpha=0.35, axis="y")
            limits = _generation_xlim(snap.start_info, metrics)
            if limits is not None:
                ax_death.set_xlim(limits)
            if len(metrics) < 2:
                return

            gens = [m.generation for m in metrics]
            window = min(self._death_window, len(metrics))
            for cause in _DEATH_CAUSES:
                counts = []
                for end_idx in range(len(metrics)):
                    start_idx = max(0, end_idx - window + 1)
                    slice_ = metrics[start_idx : end_idx + 1]
                    counts.append(sum(1 for item in slice_ if item.death_cause == cause))
                ax_death.plot(
                    gens,
                    counts,
                    label=cause,
                    color=_DEATH_COLORS[cause],
                    linewidth=1.5,
                    alpha=0.9,
                )
            ax_death.legend(loc="upper right", ncol=5, fontsize=8)

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
                f"best_ever {latest.best_ever_score}  avg_max10 {latest.avg_max10:.1f}\n"
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

                _update_stage_scores(score_panels, metrics)

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

                    _update_death_chart(metrics, snap)

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
