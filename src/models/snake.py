"""Snake body storage, movement, growth, and self-collision detection."""

from collections import deque

from .direction import Direction
from .position import Position


class Snake:
    """
    Linked body segments stored in a deque (head at index 0).

    Movement adds a new head each tick and removes the tail unless grow() was
    called on the previous tick (after eating food).
    """

    def __init__(self, start: Position, direction: Direction = Direction.RIGHT) -> None:
        self._body: deque[Position] = deque([start])
        self._direction = direction
        self._grow_pending = False

    @property
    def direction(self) -> Direction:
        return self._direction

    @property
    def body(self) -> tuple[Position, ...]:
        """All segments from head to tail."""
        return tuple(self._body)

    @property
    def tail_direction(self) -> Direction:
        """
        Direction the tail trails (from the segment before the tail toward the tail).

        Falls back to the head heading for a length-1 snake where no tail exists.
        """
        if len(self._body) < 2:
            return self._direction
        tail = self._body[-1]
        before_tail = self._body[-2]
        return Direction.from_delta(tail.x - before_tail.x, tail.y - before_tail.y)

    def head(self) -> Position:
        return self._body[0]

    def occupies(self, position: Position) -> bool:
        return position in self._body

    def set_direction(self, direction: Direction) -> None:
        """Change heading. Ignored if direction is the 180° opposite (prevents instant death)."""
        if direction != self._direction.opposite():
            self._direction = direction

    def move(self) -> Position:
        """Advance one cell in the current direction. Returns the new head position."""
        dx, dy = self._direction.to_delta()
        new_head = self.head().offset(dx, dy)
        self._body.appendleft(new_head)
        if self._grow_pending:
            self._grow_pending = False
        else:
            self._body.pop()
        return new_head

    def grow(self) -> None:
        """Keep the tail on the next move (called when food is eaten)."""
        self._grow_pending = True

    def collides_with_self(self, head: Position) -> bool:
        """Return True if head overlaps any body segment other than itself."""
        return head in list(self._body)[1:]
