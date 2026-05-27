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
    hidden: np.ndarray
    outputs: np.ndarray
    snake: tuple[tuple[int, int], ...]
    food: tuple[int, int]
    score: int
    alive: bool
    ate_food: bool
    died: bool

    def to_dict(self) -> dict:
        """Serialize to JSON-friendly primitives."""
        return {
            "tick": self.tick,
            "direction": self.direction.name,
            "inputs": self.inputs.tolist(),
            "hidden": self.hidden.tolist(),
            "outputs": self.outputs.tolist(),
            "snake": [list(segment) for segment in self.snake],
            "food": list(self.food),
            "score": self.score,
            "alive": self.alive,
            "ate_food": self.ate_food,
            "died": self.died,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReplayFrame":
        """Reconstruct a frame from loaded JSON."""
        return cls(
            tick=int(data["tick"]),
            direction=Direction[data["direction"]],
            inputs=np.asarray(data["inputs"], dtype=np.float64),
            hidden=np.asarray(data["hidden"], dtype=np.float64),
            outputs=np.asarray(data["outputs"], dtype=np.float64),
            snake=tuple(tuple(segment) for segment in data["snake"]),
            food=tuple(data["food"]),
            score=int(data["score"]),
            alive=bool(data["alive"]),
            ate_food=bool(data["ate_food"]),
            died=bool(data["died"]),
        )
