from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Position:
    x: int
    y: int

    def offset(self, dx: int, dy: int) -> "Position":
        return Position(self.x + dx, self.y + dy)
