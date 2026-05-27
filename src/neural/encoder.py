"""
Convert a live Game into 24 normalized vision features for the neural network.

Eight rays are cast relative to the snake heading. Each ray contributes three
values: Manhattan distance (in steps) to wall, body, and food, normalized to [0, 1].
"""

import numpy as np

import config
from game.game import Game
from models.direction import relative_ray_deltas
from models.position import Position


class GameStateEncoder:
    """
    Encodes game state as a 24-element float vector.

    Layout: for each of 8 rays → [wall_dist, body_dist, food_dist].
    """

    def __init__(self, max_steps: int | None = None) -> None:
        self._max_steps = max_steps or max(config.GRID_COLS, config.GRID_ROWS)

    def encode(self, game: Game) -> np.ndarray:
        """Build the input vector from the current board state."""
        head = game.snake.head()
        body = set(game.snake.body[1:])
        food_pos = game.food.position
        grid = game.grid
        features: list[float] = []

        for dx, dy in relative_ray_deltas(game.snake.direction):
            wall_steps, body_steps, food_steps = self._cast_ray(
                head, dx, dy, grid, body, food_pos
            )
            features.extend(
                [
                    self._normalize(wall_steps),
                    self._normalize(body_steps),
                    self._normalize(food_steps),
                ]
            )

        return np.asarray(features, dtype=np.float64)

    def _cast_ray(
        self,
        head: Position,
        dx: int,
        dy: int,
        grid,
        body: set[Position],
        food_pos: Position,
    ) -> tuple[int, int, int]:
        """
        Walk cell-by-cell along (dx, dy) until leaving the grid.

        Records step count to the first wall, body segment, and food encountered.
        If a target is never hit, returns max_steps for that target.
        """
        wall_steps = self._max_steps
        body_steps = self._max_steps
        food_steps = self._max_steps

        steps = 0
        x, y = head.x, head.y
        while True:
            x += dx
            y += dy
            steps += 1
            pos = Position(x, y)

            if not grid.in_bounds(pos):
                wall_steps = steps
                break

            if pos in body and body_steps == self._max_steps:
                body_steps = steps

            if pos == food_pos and food_steps == self._max_steps:
                food_steps = steps

        return wall_steps, body_steps, food_steps

    def _normalize(self, steps: int) -> float:
        return min(steps / self._max_steps, 1.0)
