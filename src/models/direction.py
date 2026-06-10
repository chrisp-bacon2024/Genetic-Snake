"""
Cardinal movement directions and relative vision-ray helpers.

Directions map to (dx, dy) grid deltas. relative_ray_deltas() returns eight step
vectors (forward, diagonals, sideways, backward) based on where the snake is facing.
These rays feed the GameStateEncoder.
"""

from enum import Enum


class Direction(Enum):
    """One of four grid movement directions."""

    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

    def opposite(self) -> "Direction":
        """Return the 180° reverse of this direction."""
        opposites = {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT,
        }
        return opposites[self]

    def to_delta(self) -> tuple[int, int]:
        """Return (dx, dy) grid offset for one step in this direction."""
        return self.value

    @classmethod
    def from_name(cls, name: str) -> "Direction":
        """Parse a direction from its enum name (e.g. ``"UP"``)."""
        return cls[name]

    @classmethod
    def from_delta(cls, dx: int, dy: int) -> "Direction":
        """
        Map a one-step (dx, dy) offset onto its cardinal Direction.

        Non-unit deltas are reduced to their sign first so segment-to-segment
        vectors (always axis-aligned and length one in this game) resolve cleanly.
        """
        sx = (dx > 0) - (dx < 0)
        sy = (dy > 0) - (dy < 0)
        for direction in cls:
            if direction.value == (sx, sy):
                return direction
        raise ValueError(f"No direction matches delta ({dx}, {dy}).")


def heading_frame(facing: Direction) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Return (forward_vec, right_vec) unit deltas for the snake's heading.

    Used to project world offsets into heading-relative forward/right components
    for dense food-direction encoding.
    """
    forward = facing.to_delta()
    right = (-forward[1], forward[0])
    return forward, right


def relative_ray_deltas(facing: Direction) -> list[tuple[int, int]]:
    """
    Return eight (dx, dy) unit steps relative to the snake's current heading.

    Order: forward, forward-right, right, back-right, back, back-left, left,
    forward-left. Used by GameStateEncoder to cast vision rays.
    """
    forward_x, forward_y = facing.to_delta()
    right_x, right_y = (-forward_y, forward_x)
    back_x, back_y = (-forward_x, -forward_y)
    left_x, left_y = (forward_y, -forward_x)

    def combine(ax: int, ay: int, bx: int, by: int) -> tuple[int, int]:
        dx = ax + bx
        dy = ay + by
        if dx != 0:
            dx = 1 if dx > 0 else -1
        if dy != 0:
            dy = 1 if dy > 0 else -1
        return (dx, dy)

    return [
        (forward_x, forward_y),
        combine(forward_x, forward_y, right_x, right_y),
        (right_x, right_y),
        combine(back_x, back_y, right_x, right_y),
        (back_x, back_y),
        combine(back_x, back_y, left_x, left_y),
        (left_x, left_y),
        combine(forward_x, forward_y, left_x, left_y),
    ]
