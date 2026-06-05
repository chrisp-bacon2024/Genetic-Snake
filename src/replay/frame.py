"""
One recorded simulation tick for web replay.

Each frame captures everything needed to redraw the board and neural-network
panel without re-running the network in the browser.
"""

from dataclasses import dataclass

import numpy as np

from models.direction import Direction


@dataclass(frozen=True, slots=True)
class ReplayFrame:
    """
    Snapshot after one Game.tick().

    The neural fields match NetworkSnapshot; snake/food/score describe the
    visible board state at that moment.
    """

    tick: int
    direction: Direction
    inputs: np.ndarray
    hidden_layers: tuple[np.ndarray, ...]
    rnn_hidden: np.ndarray | None = None
    outputs: np.ndarray
    snake: tuple[tuple[int, int], ...]
    food: tuple[int, int]
    score: int
    alive: bool
    ate_food: bool
    died: bool
    starved: bool = False

    def to_dict(self) -> dict:
        """Serialize to JSON-friendly primitives."""
        return {
            "tick": self.tick,
            "direction": self.direction.name,
            "inputs": self.inputs.tolist(),
            "hidden_layers": [layer.tolist() for layer in self.hidden_layers],
            "rnn_hidden": self.rnn_hidden.tolist() if self.rnn_hidden is not None else [],
            "outputs": self.outputs.tolist(),
            "snake": [list(segment) for segment in self.snake],
            "food": list(self.food),
            "score": self.score,
            "alive": self.alive,
            "ate_food": self.ate_food,
            "died": self.died,
            "starved": self.starved,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReplayFrame":
        """Reconstruct a frame from loaded JSON (handles legacy single-hidden format)."""
        if "hidden_layers" in data:
            hidden_layers = tuple(
                np.asarray(layer, dtype=np.float64) for layer in data["hidden_layers"]
            )
        else:
            hidden_layers = (np.asarray(data.get("hidden", []), dtype=np.float64),)
        rnn_hidden = None
        if "rnn_hidden" in data and len(data["rnn_hidden"]) > 0:
            rnn_hidden = np.asarray(data["rnn_hidden"], dtype=np.float64)

        return cls(
            tick=int(data["tick"]),
            direction=Direction[data["direction"]],
            inputs=np.asarray(data["inputs"], dtype=np.float64),
            hidden_layers=hidden_layers,
            rnn_hidden=rnn_hidden,
            outputs=np.asarray(data["outputs"], dtype=np.float64),
            snake=tuple(tuple(segment) for segment in data["snake"]),
            food=tuple(data["food"]),
            score=int(data["score"]),
            alive=bool(data["alive"]),
            ate_food=bool(data["ate_food"]),
            died=bool(data["died"]),
            starved=bool(data.get("starved", False)),
        )
