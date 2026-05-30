"""
Convert a live Game into 32 vision features for the neural network.

Layout (32 features):
  - 8 rays cast relative to the snake heading, each contributing 3 inverse-distance
    values [wall, food, body] where close = high signal (1.0 adjacent, ~0 far, 0 absent).
  - 4 one-hot head-direction features (UP, DOWN, LEFT, RIGHT).
  - 4 one-hot tail-direction features (UP, DOWN, LEFT, RIGHT).

Inverse distance (rather than raw normalized distance) puts the strongest signal on
the nearest obstacle/food, and the dense direction one-hots give an always-present
sense of heading and body layout.
"""

import numpy as np

import config
from game.game import Game
from models.direction import Direction, relative_ray_deltas
from models.position import Position

# Absolute direction order for the one-hot features.
_DIRECTION_ORDER = (Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT)


class GameStateEncoder:
    """Encodes game state as a 32-element float vector in [0, 1]."""

    def encode(self, game: Game) -> np.ndarray:
        """Build the input vector from the current board state."""
        head = game.snake.head()
        body = set(game.snake.body[1:])
        food_pos = game.food.position
        grid = game.grid
        features: list[float] = []

        for dx, dy in relative_ray_deltas(game.snake.direction):
            wall_steps, food_steps, body_steps = self._cast_ray(
                head, dx, dy, grid, body, food_pos
            )
            features.append(self._inverse(wall_steps))
            features.append(self._inverse(food_steps))
            features.append(self._inverse(body_steps))

        features.extend(self._one_hot(game.snake.direction))
        features.extend(self._one_hot(game.snake.tail_direction))

        return np.asarray(features, dtype=np.float64)

    def _cast_ray(
        self,
        head: Position,
        dx: int,
        dy: int,
        grid,
        body: set[Position],
        food_pos: Position,
    ) -> tuple[int | None, int | None, int | None]:
        """
        Walk cell-by-cell along (dx, dy) until leaving the grid.

        Returns step counts to the first wall, food, and body segment encountered;
        None when that target is never hit along the ray (food/body only).
        """
        food_steps: int | None = None
        body_steps: int | None = None

        steps = 0
        x, y = head.x, head.y
        while True:
            x += dx
            y += dy
            steps += 1
            pos = Position(x, y)

            if not grid.in_bounds(pos):
                return steps, food_steps, body_steps

            if food_steps is None and pos == food_pos:
                food_steps = steps

            if body_steps is None and pos in body:
                body_steps = steps

    def _one_hot(self, direction: Direction) -> list[float]:
        """One-hot encode a cardinal direction in _DIRECTION_ORDER."""
        return [1.0 if direction is d else 0.0 for d in _DIRECTION_ORDER]

    def _inverse(self, steps: int | None) -> float:
        """Inverse distance: adjacent -> 1.0, far -> ~0, absent (None) -> 0.0."""
        if steps is None or steps <= 0:
            return 0.0
        return 1.0 / float(steps)
