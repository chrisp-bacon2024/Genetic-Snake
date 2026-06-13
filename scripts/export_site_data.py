"""
Export training replays and metrics for the recruiter demo site.

Run from the repository root:

    python scripts/export_site_data.py --replays-dir src/replays
    python scripts/export_site_data.py --replays-dir src/replays --generations 0,215 --full
    python scripts/export_site_data.py --metrics-only --replays-dir src/replays
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
DEFAULT_OUTPUT = REPO_ROOT / "site" / "public" / "data"
DEFAULT_GRID_GENERATIONS = [0, 10, 50, 100, 135, 150, 160, 176, 199, 215]
DEFAULT_FEATURED_GENERATIONS = [0, 215]
DEFAULT_LITE_MAX_TICKS = 0
DEFAULT_GRID_MAX_TICKS: dict[int, int] = {}
DEFAULT_GRID_TAIL_FRAMES: dict[int, int] = {}
DEFAULT_FEATURED_MAX_TICKS: dict[int, int] = {}


def _ensure_src_path() -> None:
    src = str(SRC_DIR)
    if src not in sys.path:
        sys.path.insert(0, src)


def _load_npz(path: Path) -> dict:
    data = np.load(path)
    return {
        "generation": int(data["generation"]),
        "score": int(data["score"]),
        "food_seed": int(data["food_seed"]),
        "grid_cols": int(data["grid_cols"]) if "grid_cols" in data else 20,
        "grid_rows": int(data["grid_rows"]) if "grid_rows" in data else 20,
        "death_cause": str(data["death_cause"]) if "death_cause" in data else "unknown",
        "genes": np.asarray(data["genes"], dtype=np.float64),
    }


def _record_replay(
    npz_path: Path,
    *,
    lite: bool,
    max_ticks: int | None,
    tail_frames: int | None = None,
) -> dict:
    _ensure_src_path()
    from evolution.genome import Genome
    from models.grid import Grid
    from replay.recorder import GameRecorder
    from simulation.headless import HeadlessSimulator, Scenario

    meta = _load_npz(npz_path)
    genome = Genome(meta["genes"])
    grid = Grid(meta["grid_cols"], meta["grid_rows"])
    scenario = Scenario(food_seed=meta["food_seed"])
    simulator = HeadlessSimulator(grid)
    recorder = simulator._record_run(genome, scenario)

    document = recorder.to_dict()
    if max_ticks is not None and max_ticks > 0:
        document["frames"] = document["frames"][:max_ticks]
    if tail_frames is not None and tail_frames > 0 and len(document["frames"]) > tail_frames:
        document["frames"] = document["frames"][-tail_frames:]
    document["frame_count"] = len(document["frames"])

    if lite:
        document["frames"] = [_lite_frame(frame) for frame in document["frames"]]

    document["generation"] = meta["generation"]
    document["saved_score"] = meta["score"]
    document["death_cause"] = meta["death_cause"]
    document["lite"] = lite
    return document


def _lite_frame(frame: dict) -> dict:
    return {
        "tick": frame["tick"],
        "direction": frame["direction"],
        "snake": frame["snake"],
        "food": frame["food"],
        "score": frame["score"],
        "alive": frame["alive"],
        "ate_food": frame["ate_food"],
        "died": frame["died"],
        "starved": frame.get("starved", False),
    }


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))


def _export_replays(
    replays_dir: Path,
    output_dir: Path,
    generations: list[int],
    *,
    lite: bool,
    max_ticks: int | None,
    per_generation_max_ticks: dict[int, int],
    per_generation_tail_frames: dict[int, int],
    suffix: str,
) -> list[dict]:
    entries: list[dict] = []
    for generation in generations:
        npz_path = replays_dir / f"gen_{generation:04d}.npz"
        if not npz_path.exists():
            print(f"Skipping missing {npz_path}", flush=True)
            continue

        tick_cap = per_generation_max_ticks.get(generation, max_ticks)
        tail_cap = per_generation_tail_frames.get(generation)
        document = _record_replay(
            npz_path,
            lite=lite,
            max_ticks=tick_cap,
            tail_frames=tail_cap,
        )
        filename = f"gen_{generation:04d}{suffix}.json"
        output_path = output_dir / filename
        _write_json(output_path, document)

        entries.append(
            {
                "generation": generation,
                "score": document["saved_score"],
                "grid_cols": document["grid"]["cols"],
                "grid_rows": document["grid"]["rows"],
                "death_cause": document["death_cause"],
                "frame_count": document["frame_count"],
                "ticks_per_second": document["ticks_per_second"],
                "lite": lite,
                "path": f"data/{filename}",
                "narrative": _narrative_for_generation(generation),
            }
        )
        print(
            f"Exported gen {generation:4d} -> {output_path.name} "
            f"({document['frame_count']} frames, lite={lite})",
            flush=True,
        )
    return entries


def _narrative_for_generation(generation: int) -> str:
    narratives = {
        0: "Untrained population baseline",
        10: "Early learning on 5×5",
        50: "Plateau on the small board",
        100: "5×5 mastered — consistent wins",
        135: "Curriculum advance — first steps on 10×10",
        150: "Still adapting to the larger grid",
        160: "10×10 progress — longer survival",
        176: "10×10 mastered — board cleared",
        199: "Learning 20×20 — first big-board run",
        215: "Full-board win on 20×20",
    }
    return narratives.get(generation, f"Generation {generation}")


def _export_metrics(replays_dir: Path, output_dir: Path) -> None:
    _ensure_src_path()
    from evolution.training_log import load_training_history, metrics_to_dict

    metrics = load_training_history(replays_dir)
    payload = []
    for entry in metrics:
        row = metrics_to_dict(entry)
        row.pop("population_scores", None)
        row.pop("population_death_causes", None)
        payload.append(row)
    _write_json(output_dir / "metrics.json", payload)
    print(f"Exported {len(payload)} metric rows to metrics.json", flush=True)


def _export_chart(replays_dir: Path, output_dir: Path) -> None:
    chart_path = output_dir / "training_chart.png"
    analyze_script = SRC_DIR / "analyze_training.py"
    command = [
        sys.executable,
        str(analyze_script),
        "--replays",
        str(replays_dir.resolve()),
        "--output",
        str(chart_path.resolve()),
        "--skip-resim",
    ]
    subprocess.run(command, cwd=str(SRC_DIR), check=True)
    print(f"Exported training chart to {chart_path}", flush=True)


def _write_manifest(
    output_dir: Path,
    grid_entries: list[dict],
    featured_entries: list[dict],
) -> None:
    payload = {
        "grid_generations": grid_entries,
        "featured_generations": featured_entries,
        "default_featured_generation": 215,
    }
    _write_json(output_dir / "manifest.json", payload)
    print("Wrote manifest.json", flush=True)


def _parse_generations(raw: str | None, default: list[int]) -> list[int]:
    if not raw:
        return default
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Genetic Snake site data.")
    parser.add_argument(
        "--replays-dir",
        type=Path,
        default=SRC_DIR / "replays",
        help="Directory containing gen_*.npz and training_log.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Site public data directory",
    )
    parser.add_argument(
        "--generations",
        type=str,
        default=None,
        help="Comma-separated generations for grid lite exports",
    )
    parser.add_argument(
        "--featured-generations",
        type=str,
        default=None,
        help="Comma-separated generations for full featured exports",
    )
    parser.add_argument(
        "--lite",
        action="store_true",
        help="Export lite JSON (board state only) for --generations",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Export full JSON (with neural activations) for --featured-generations",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=DEFAULT_LITE_MAX_TICKS,
        help="Cap frames for lite exports (0 = no cap)",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Export metrics.json and training chart only",
    )
    parser.add_argument(
        "--skip-chart",
        action="store_true",
        help="Skip matplotlib chart export",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    replays_dir = args.replays_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not replays_dir.exists():
        raise FileNotFoundError(f"Replays directory not found: {replays_dir}")

    if args.metrics_only:
        _export_metrics(replays_dir, output_dir)
        if not args.skip_chart:
            _export_chart(replays_dir, output_dir)
        return

    grid_generations = _parse_generations(args.generations, DEFAULT_GRID_GENERATIONS)
    featured_generations = _parse_generations(
        args.featured_generations, DEFAULT_FEATURED_GENERATIONS
    )
    max_ticks = None if args.max_ticks <= 0 else args.max_ticks

    grid_entries = _export_replays(
        replays_dir,
        output_dir,
        grid_generations,
        lite=True,
        max_ticks=max_ticks,
        per_generation_max_ticks=DEFAULT_GRID_MAX_TICKS,
        per_generation_tail_frames=DEFAULT_GRID_TAIL_FRAMES,
        suffix="_lite",
    )
    featured_entries = _export_replays(
        replays_dir,
        output_dir,
        featured_generations,
        lite=False,
        max_ticks=None,
        per_generation_max_ticks=DEFAULT_FEATURED_MAX_TICKS,
        per_generation_tail_frames={},
        suffix="_full",
    )

    _export_metrics(replays_dir, output_dir)
    if not args.skip_chart:
        _export_chart(replays_dir, output_dir)
    _write_manifest(output_dir, grid_entries, featured_entries)


if __name__ == "__main__":
    main()
