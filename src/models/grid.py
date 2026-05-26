import random

from .position import Position


class Grid:
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
        return 0 <= position.x < self._width and 0 <= position.y < self._height

    def random_empty_cell(self, occupied: set[Position]) -> Position:
        empty_cells = [
            Position(x, y)
            for x in range(self._width)
            for y in range(self._height)
            if Position(x, y) not in occupied
        ]
        if not empty_cells:
            raise RuntimeError("No empty cells available on the grid.")
        return random.choice(empty_cells)

    def center(self) -> Position:
        return Position(self._width // 2, self._height // 2)
