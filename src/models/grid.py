"""Rectangular playfield with bounds checking and random spawn helpers."""

import random

from .position import Position


class Grid:
    """Fixed-size grid. Coordinates are valid when 0 <= x < width and 0 <= y < height."""

    def __init__(self, width: int, height: int) -> None:
        self._width = width
        self._height = height

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def in_bounds(self, position: Position) -> bool:
        """Return True if position lies inside the grid."""
        return 0 <= position.x < self._width and 0 <= position.y < self._height

    def random_empty_cell(
        self, occupied: set[Position], rng: random.Random | None = None
    ) -> Position:
        """
        Pick a random cell not in occupied. Raises if the grid is full.

        Pass a seeded ``random.Random`` for deterministic food placement during
        genetic-algorithm evaluation; otherwise the module-global RNG is used.
        """
        empty_cells = [
            Position(x, y)
            for x in range(self._width)
            for y in range(self._height)
            if Position(x, y) not in occupied
        ]
        if not empty_cells:
            raise RuntimeError("No empty cells available on the grid.")
        chooser = rng.choice if rng is not None else random.choice
        return chooser(empty_cells)

    def center(self) -> Position:
        """Return the approximate center cell (where the snake spawns)."""
        return Position(self._width // 2, self._height // 2)
