"""Immutable integer grid coordinate."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Position:
    """A cell on the game grid, identified by (x, y) with origin at top-left."""

    x: int
    y: int

    def offset(self, dx: int, dy: int) -> "Position":
        """Return a new position shifted by (dx, dy)."""
        return Position(self.x + dx, self.y + dy)
