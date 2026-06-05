"""
Convert a live Game into a fixed-size vector for the neural network.

Layout (MAX_GRID_COLS * MAX_GRID_ROWS + 10 features):
  - Full board raster (row-major, padded to max grid size for curriculum):
      0.0 empty, 0.4 body, 0.7 head, 1.0 food (off-board padding cells = 0.0).
  - 4 one-hot head direction (UP, DOWN, LEFT, RIGHT).
  - 4 one-step lookahead flags (safe moves, matches output order).
  - 1 reachable-empty ratio (BFS from head through non-body cells / total empty).
  - 1 empty-cell fraction (remaining space on the board).
"""

from __future__ import annotations

from collections import deque

import numpy as np

import config
from game.game import Game
from models.direction import Direction
from models.position import Position

_DIRECTION_ORDER = (Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT)

_CELL_EMPTY = 0.0
_CELL_BODY = 0.4
_CELL_HEAD = 0.7
_CELL_FOOD = 1.0


class GameStateEncoder:
    """Encodes game state as a fixed-length float vector in [0, 1]."""

    @staticmethod
    def input_size() -> int:
        return config.nn_input_size()

    @staticmethod
    def grid_feature_count() -> int:
        return config.MAX_GRID_COLS * config.MAX_GRID_ROWS

    @staticmethod
    def meta_feature_count() -> int:
        return 10

    def encode(self, game: Game) -> np.ndarray:
        """Build the input vector from the current board state."""
        features: list[float] = []
        features.extend(self._grid_cells(game))
        features.extend(self._one_hot(game.snake.direction))
        features.extend(self._lookahead(game))
        features.append(self._reachable_empty_ratio(game))
        features.append(self._empty_fraction(game))
        return np.asarray(features, dtype=np.float64)

    def safe_move_mask(self, game: Game) -> np.ndarray:
        """Boolean mask over output directions (True = one-step safe)."""
        return np.asarray(self._lookahead(game), dtype=bool)

    def reachable_empty_ratio(self, game: Game) -> float:
        """Share of empty cells reachable from the head without crossing body."""
        return self._reachable_empty_ratio(game)

    def _grid_cells(self, game: Game) -> list[float]:
        """Rasterize the live grid, padded into the max training board size."""
        grid = game.grid
        head = game.snake.head()
        food = game.food.position
        body = set(game.snake.body[1:])
        cells: list[float] = []

        for row in range(config.MAX_GRID_ROWS):
            for col in range(config.MAX_GRID_COLS):
                if col >= grid.width or row >= grid.height:
                    cells.append(_CELL_EMPTY)
                    continue
                pos = Position(col, row)
                if pos == food:
                    cells.append(_CELL_FOOD)
                elif pos == head:
                    cells.append(_CELL_HEAD)
                elif pos in body:
                    cells.append(_CELL_BODY)
                else:
                    cells.append(_CELL_EMPTY)
        return cells

    def _lookahead(self, game: Game) -> list[float]:
        snake = game.snake
        grid = game.grid
        return [
            1.0 if snake.is_step_safe(direction, grid) else 0.0
            for direction in _DIRECTION_ORDER
        ]

    def _empty_fraction(self, game: Game) -> float:
        total = game.grid.width * game.grid.height
        occupied = len(game.snake.body)
        return max(0.0, (total - occupied) / float(total))

    def _reachable_empty_ratio(self, game: Game) -> float:
        """
        Fraction of empty cells reachable from the head without crossing body.

        Low values warn the policy that the snake is boxing itself in.
        """
        grid = game.grid
        head = game.snake.head()
        body = set(game.snake.body)
        all_empty = grid.width * grid.height - len(body)
        if all_empty <= 0:
            return 0.0

        visited: set[Position] = {head}
        queue: deque[Position] = deque([head])
        while queue:
            pos = queue.popleft()
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nxt = Position(pos.x + dx, pos.y + dy)
                if not grid.in_bounds(nxt) or nxt in body or nxt in visited:
                    continue
                visited.add(nxt)
                queue.append(nxt)

        reached_empty = sum(1 for pos in visited if pos not in body)
        return min(1.0, reached_empty / float(all_empty))

    def _one_hot(self, direction: Direction) -> list[float]:
        return [1.0 if direction is d else 0.0 for d in _DIRECTION_ORDER]
