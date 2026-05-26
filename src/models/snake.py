from collections import deque

from .direction import Direction
from .position import Position


class Snake:
    def __init__(self, start: Position, direction: Direction = Direction.RIGHT) -> None:
        self._body: deque[Position] = deque([start])
        self._direction = direction
        self._grow_pending = False

    @property
    def direction(self) -> Direction:
        return self._direction

    @property
    def body(self) -> tuple[Position, ...]:
        return tuple(self._body)

    def head(self) -> Position:
        return self._body[0]

    def occupies(self, position: Position) -> bool:
        return position in self._body

    def set_direction(self, direction: Direction) -> None:
        if direction != self._direction.opposite():
            self._direction = direction

    def move(self) -> Position:
        dx, dy = self._direction.to_delta()
        new_head = self.head().offset(dx, dy)
        self._body.appendleft(new_head)
        if self._grow_pending:
            self._grow_pending = False
        else:
            self._body.pop()
        return new_head

    def grow(self) -> None:
        self._grow_pending = True

    def collides_with_self(self, head: Position) -> bool:
        return head in list(self._body)[1:]
