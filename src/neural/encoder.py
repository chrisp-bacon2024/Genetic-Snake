"""
Convert a live Game into a compact ray-based vector for the neural network.

Layout (44 features):
  - 8 rays relative to heading, each [wall, food, body]:
    wall/body use inverse distance; food uses angular alignment toward food.
  - 5 food cues: inverse Manhattan distance + heading-relative offsets.
  - 4 one-hot head direction (UP, DOWN, LEFT, RIGHT).
  - 4 one-hot tail direction (UP, DOWN, LEFT, RIGHT).
  - 4 one-step lookahead flags (safe moves, matches output order).
  - 1 reachable-empty ratio (BFS from head / total empty).
  - 1 empty-cell fraction (remaining board space).
"""

from __future__ import annotations

from collections import deque

import numpy as np

import config
from game.game import Game
from models.direction import Direction, heading_frame, relative_ray_deltas
from models.position import Position
from neural.vision_rays import cast_ray, proximity_activation

_DIRECTION_ORDER = (Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT)


class GameStateEncoder:
    """Encodes game state as a fixed-length float vector in [0, 1]."""

    def __init__(self) -> None:
        self.last_reachable_empty_ratio = 0.0

    @staticmethod
    def input_size() -> int:
        return config.nn_input_size()

    @staticmethod
    def ray_feature_count() -> int:
        return config.ENCODER_RAY_COUNT * 3

    @staticmethod
    def food_feature_count() -> int:
        return 5

    @staticmethod
    def meta_feature_count() -> int:
        return 10

    @staticmethod
    def feature_offsets() -> dict[str, tuple[int, int]]:
        """Start index and count for each encoder section (used by the UI)."""
        start = 0
        rays = config.ENCODER_RAY_COUNT * 3
        food = 5
        head = 4
        tail = 4
        lookahead = 4
        space = 3
        return {
            "rays": (start, rays),
            "food": (start := start + rays, food),
            "head": (start := start + food, head),
            "tail": (start := start + head, tail),
            "lookahead": (start := start + tail, lookahead),
            "space": (start := start + lookahead, space),
        }

    def encode(self, game: Game) -> np.ndarray:
        """Build the input vector from the current board state."""
        head = game.snake.head()
        body = set(game.snake.body[1:])
        food_pos = game.food.position
        grid = game.grid
        features: list[float] = []

        for dx, dy in relative_ray_deltas(game.snake.direction):
            wall_steps, body_steps = self._cast_ray(head, dx, dy, grid, body)
            features.append(proximity_activation(wall_steps))
            features.append(self._ray_food_alignment(head, food_pos, dx, dy))
            features.append(proximity_activation(body_steps))

        features.extend(self._food_features(game.snake.direction, head, food_pos, grid))
        features.extend(self._one_hot(game.snake.direction))
        features.extend(self._one_hot(game.snake.tail_direction))
        features.extend(self._lookahead(game))
        reachable = self._reachable_empty_ratio(game)
        self.last_reachable_empty_ratio = reachable
        features.append(reachable)
        features.append(self._empty_fraction(game))
        features.append(self._body_length_fraction(game))

        return np.asarray(features, dtype=np.float64)

    def safe_move_mask(self, game: Game) -> np.ndarray:
        """Boolean mask over output directions (True = one-step safe)."""
        return np.asarray(self._lookahead(game), dtype=bool)

    def reachable_empty_ratio(self, game: Game) -> float:
        """Share of empty cells reachable from the head without crossing body."""
        ratio = self._reachable_empty_ratio(game)
        self.last_reachable_empty_ratio = ratio
        return ratio

    def _lookahead(self, game: Game) -> list[float]:
        snake = game.snake
        grid = game.grid
        return [
            1.0 if snake.is_step_safe(direction, grid) else 0.0
            for direction in _DIRECTION_ORDER
        ]

    def _food_features(
        self,
        facing: Direction,
        head: Position,
        food_pos: Position,
        grid,
    ) -> list[float]:
        dx = food_pos.x - head.x
        dy = food_pos.y - head.y
        manhattan = abs(dx) + abs(dy)

        (fx, fy), (rx, ry) = heading_frame(facing)
        forward = dx * fx + dy * fy
        right = dx * rx + dy * ry
        # Use board span (not full-grid Manhattan max) so nearby food direction cues are brighter.
        norm = float(max(1, max(grid.width, grid.height)))
        direction_scale = config.ENCODER_FOOD_DIRECTION_SCALE / norm

        def _scaled(component: float) -> float:
            return min(1.0, max(0.0, component) * direction_scale)

        return [
            self._inverse_manhattan(manhattan),
            _scaled(forward),
            _scaled(right),
            _scaled(-forward),
            _scaled(-right),
        ]

    def _ray_food_alignment(
        self,
        head: Position,
        food_pos: Position,
        ray_dx: int,
        ray_dy: int,
    ) -> float:
        fx = food_pos.x - head.x
        fy = food_pos.y - head.y
        manhattan = abs(fx) + abs(fy)
        if manhattan == 0:
            return 1.0

        ray_len = (ray_dx**2 + ray_dy**2) ** 0.5
        food_len = (fx**2 + fy**2) ** 0.5
        dot = (ray_dx * fx + ray_dy * fy) / (ray_len * food_len)
        floor = config.ENCODER_FOOD_RAY_ALIGN_FLOOR
        alignment = floor + (1.0 - floor) * max(0.0, dot)
        # Match wall-style falloff (1/distance) so food at similar range is similarly bright.
        distance = proximity_activation(manhattan)
        return min(1.0, alignment * distance * config.ENCODER_FOOD_RAY_BOOST)

    def _cast_ray(
        self,
        head: Position,
        dx: int,
        dy: int,
        grid,
        body: set[Position],
    ) -> tuple[int | None, int | None]:
        return cast_ray(head, dx, dy, grid, body)

    def _empty_fraction(self, game: Game) -> float:
        total = game.grid.width * game.grid.height
        occupied = len(game.snake.body)
        return max(0.0, (total - occupied) / float(total))

    @staticmethod
    def _body_length_fraction(game: Game) -> float:
        total = game.grid.width * game.grid.height
        return len(game.snake.body) / float(max(1, total))

    def _reachable_empty_ratio(self, game: Game) -> float:
        grid = game.grid
        width = grid.width
        height = grid.height
        cell_count = width * height
        body_len = len(game.snake.body)
        all_empty = cell_count - body_len
        if all_empty <= 0:
            return 0.0

        blocked = bytearray(cell_count)
        for segment in game.snake.body:
            blocked[segment.y * width + segment.x] = 1

        head = game.snake.head()
        start = head.y * width + head.x
        visited = bytearray(cell_count)
        queue: deque[int] = deque([start])
        visited[start] = 1
        reached_empty = 0

        while queue:
            index = queue.popleft()
            x = index % width
            y = index // width
            if index != start and not blocked[index]:
                reached_empty += 1
            for nx, ny in ((x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)):
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                next_index = ny * width + nx
                if visited[next_index] or blocked[next_index]:
                    continue
                visited[next_index] = 1
                queue.append(next_index)

        return min(1.0, reached_empty / float(all_empty))

    def _one_hot(self, direction: Direction) -> list[float]:
        return [1.0 if direction is d else 0.0 for d in _DIRECTION_ORDER]

    @staticmethod
    def _inverse_manhattan(manhattan: int) -> float:
        return 1.0 / (float(manhattan) + 1.0)
