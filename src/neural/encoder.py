"""
Convert a live Game into 37 vision features for the neural network.

Layout (37 features):
  - 8 rays cast relative to the snake heading, each contributing [wall, food, body]:
    wall/body use inverse distance along the ray; food uses angular alignment toward
    food (always-on baseline + boost on rays pointing at food).
  - 1 inverse Manhattan distance to food (always-on when food exists).
  - 4 heading-relative food offsets [fwd, right, back, left] normalized by grid size.
  - 4 one-hot head-direction features (UP, DOWN, LEFT, RIGHT).
  - 4 one-hot tail-direction features (UP, DOWN, LEFT, RIGHT).
"""

import numpy as np

import config
from game.game import Game
from models.direction import Direction, heading_frame, relative_ray_deltas
from models.position import Position

# Absolute direction order for the one-hot features.
_DIRECTION_ORDER = (Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT)


class GameStateEncoder:
    """Encodes game state as a 37-element float vector in [0, 1]."""

    def __init__(self, max_steps: int | None = None) -> None:
        self._max_manhattan = max_steps or (config.GRID_COLS - 1 + config.GRID_ROWS - 1)

    def encode(self, game: Game) -> np.ndarray:
        """Build the input vector from the current board state."""
        head = game.snake.head()
        body = set(game.snake.body[1:])
        food_pos = game.food.position
        grid = game.grid
        features: list[float] = []

        for dx, dy in relative_ray_deltas(game.snake.direction):
            wall_steps, body_steps = self._cast_ray(head, dx, dy, grid, body)
            features.append(self._inverse(wall_steps))
            features.append(self._ray_food_alignment(head, food_pos, dx, dy))
            features.append(self._inverse(body_steps))

        features.extend(self._food_features(game.snake.direction, head, food_pos))
        features.extend(self._one_hot(game.snake.direction))
        features.extend(self._one_hot(game.snake.tail_direction))

        return np.asarray(features, dtype=np.float64)

    def _food_features(
        self,
        facing: Direction,
        head: Position,
        food_pos: Position,
    ) -> list[float]:
        """
        Always-on food location: [inverse_manhattan, fwd, right, back, left].

        Manhattan distance tells the snake how far away food is; the four offsets
        (normalized cell counts in the heading frame) tell it which way to turn.
        """
        dx = food_pos.x - head.x
        dy = food_pos.y - head.y
        manhattan = abs(dx) + abs(dy)

        (fx, fy), (rx, ry) = heading_frame(facing)
        forward = dx * fx + dy * fy
        right = dx * rx + dy * ry

        norm = float(self._max_manhattan)
        return [
            self._inverse_manhattan(manhattan),
            max(0.0, forward) / norm,
            max(0.0, right) / norm,
            max(0.0, -forward) / norm,
            max(0.0, -right) / norm,
        ]

    def _ray_food_alignment(
        self,
        head: Position,
        food_pos: Position,
        ray_dx: int,
        ray_dy: int,
    ) -> float:
        """
        Food signal for one ray: baseline on all rays plus boost toward food.

        Uses cosine alignment between the ray and the vector to food, scaled by
        inverse Manhattan distance so closer food produces a stronger signal.
        """
        fx = food_pos.x - head.x
        fy = food_pos.y - head.y
        manhattan = abs(fx) + abs(fy)
        if manhattan == 0:
            return 1.0

        ray_len = (ray_dx**2 + ray_dy**2) ** 0.5
        food_len = (fx**2 + fy**2) ** 0.5
        dot = (ray_dx * fx + ray_dy * fy) / (ray_len * food_len)
        alignment = 0.25 + 0.75 * max(0.0, dot)
        return alignment * self._inverse_manhattan(manhattan)

    def _cast_ray(
        self,
        head: Position,
        dx: int,
        dy: int,
        grid,
        body: set[Position],
    ) -> tuple[int | None, int | None]:
        """
        Walk cell-by-cell along (dx, dy) until leaving the grid.

        Returns step counts to the first wall and body segment encountered;
        None for body when no segment lies on the ray.
        """
        body_steps: int | None = None

        steps = 0
        x, y = head.x, head.y
        while True:
            x += dx
            y += dy
            steps += 1
            pos = Position(x, y)

            if not grid.in_bounds(pos):
                return steps, body_steps

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

    @staticmethod
    def _inverse_manhattan(manhattan: int) -> float:
        """Inverse Manhattan distance: adjacent -> 0.5, same cell -> 1.0, far -> ~0."""
        return 1.0 / (float(manhattan) + 1.0)
