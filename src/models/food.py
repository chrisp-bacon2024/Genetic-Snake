from .grid import Grid
from .position import Position


class Food:
    def __init__(self, position: Position) -> None:
        self._position = position

    @property
    def position(self) -> Position:
        return self._position

    def respawn(self, grid: Grid, occupied: set[Position]) -> None:
        self._position = grid.random_empty_cell(occupied)
