"""
Accumulates ReplayFrame objects and writes them to JSON.

Typical lifecycle:
    recorder.start(genome, game)
    loop: recorder.record_frame(game, snapshot, tick_result)
    recorder.save("replays/best.json")   # e.g. best snake per GA epoch
"""

import json
from pathlib import Path

import numpy as np

import config
from controllers.ai_controller import NetworkSnapshot
from evolution.genome import Genome
from game.game import Game
from game.game_state import TickResult
from models.position import Position

from .frame import ReplayFrame

REPLAY_FORMAT_VERSION = 1


class GameRecorder:
    """
    In-memory recording of a full game for later web playback.

    Does not save to disk automatically — call save() explicitly when keeping
    a run (planned: best individual each genetic-algorithm epoch).
    """

    def __init__(self) -> None:
        self._frames: list[ReplayFrame] = []
        self._genome: Genome | None = None
        self._grid_cols = config.GRID_COLS
        self._grid_rows = config.GRID_ROWS
        self._ticks_per_second = config.TICKS_PER_SECOND

    @property
    def frames(self) -> tuple[ReplayFrame, ...]:
        return tuple(self._frames)

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def is_recording(self) -> bool:
        return self._genome is not None

    def start(self, genome: Genome, game: Game) -> None:
        """Begin a new recording session (clears any previous frames)."""
        self._frames.clear()
        self._genome = genome.copy()
        self._grid_cols = game.grid.width
        self._grid_rows = game.grid.height

    def clear(self) -> None:
        """Discard all frames and stop recording."""
        self._frames.clear()
        self._genome = None

    def record_frame(
        self,
        game: Game,
        snapshot: NetworkSnapshot,
        tick_result: TickResult,
    ) -> None:
        """Append one tick after get_direction() and Game.tick() have run."""
        if self._genome is None:
            raise RuntimeError("Call start() before recording frames.")

        frame = ReplayFrame(
            tick=len(self._frames),
            direction=snapshot.chosen_direction,
            inputs=snapshot.inputs.copy(),
            hidden_layers=tuple(layer.copy() for layer in snapshot.hidden_layers),
            rnn_hidden=snapshot.rnn_hidden.copy(),
            outputs=snapshot.outputs.copy(),
            snake=self._serialize_snake(game.snake.body),
            food=(game.food.position.x, game.food.position.y),
            score=game.score,
            alive=game.alive,
            ate_food=tick_result.ate_food,
            died=tick_result.died,
            starved=tick_result.starved,
        )
        self._frames.append(frame)

    def to_dict(self) -> dict:
        """Full replay document for JSON export."""
        if self._genome is None:
            raise RuntimeError("No recording in progress.")

        return {
            "version": REPLAY_FORMAT_VERSION,
            "grid": {"cols": self._grid_cols, "rows": self._grid_rows},
            "ticks_per_second": self._ticks_per_second,
            "genome": self._genome.genes.tolist(),
            "frame_count": len(self._frames),
            "frames": [frame.to_dict() for frame in self._frames],
        }

    def save(self, path: str | Path) -> Path:
        """Write the recording to a JSON file. Returns the resolved path."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=2)
        return output.resolve()

    @classmethod
    def load(cls, path: str | Path) -> "GameRecorder":
        """Load a replay JSON file into a recorder instance."""
        with Path(path).open(encoding="utf-8") as file:
            data = json.load(file)

        recorder = cls()
        recorder._grid_cols = int(data["grid"]["cols"])
        recorder._grid_rows = int(data["grid"]["rows"])
        recorder._ticks_per_second = int(data.get("ticks_per_second", config.TICKS_PER_SECOND))
        recorder._genome = Genome(np.asarray(data["genome"], dtype=np.float64))
        recorder._frames = [ReplayFrame.from_dict(frame) for frame in data["frames"]]
        return recorder

    @staticmethod
    def _serialize_snake(body: tuple[Position, ...]) -> tuple[tuple[int, int], ...]:
        return tuple((segment.x, segment.y) for segment in body)
