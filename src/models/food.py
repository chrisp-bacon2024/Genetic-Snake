"""Food placement on the grid."""

from .grid import Grid
from .position import Position


class Food:
    """Single food pellet at a grid position. Respawns on an empty cell when eaten."""

    def __init__(self, position: Position) -> None:
        self._position = position

    @property
    def position(self) -> Position:
        return self._position

    def respawn(self, grid: Grid, occupied: set[Position]) -> None:
        """Move food to a random unoccupied cell."""
        self._position = grid.random_empty_cell(occupied)
