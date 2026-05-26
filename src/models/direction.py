from enum import Enum


class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

    def opposite(self) -> "Direction":
        opposites = {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT,
        }
        return opposites[self]

    def to_delta(self) -> tuple[int, int]:
        return self.value

    @classmethod
    def from_name(cls, name: str) -> "Direction":
        return cls[name]


def relative_ray_deltas(facing: Direction) -> list[tuple[int, int]]:
    """Return 8 (dx, dy) ray directions relative to the snake heading."""
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
