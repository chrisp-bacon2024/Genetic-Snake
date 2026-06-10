"""Heading-relative vision rays for encoding and on-board visualization."""

from __future__ import annotations

from dataclasses import dataclass

from game.game import Game
from models.direction import relative_ray_deltas
from models.position import Position

_RAY_COMPASS_LABELS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


def proximity_activation(steps: int | None) -> float:
    """Inverse step count in (0, 1]; used by the encoder and playfield rays."""
    if steps is None or steps <= 0:
        return 0.0
    return 1.0 / float(steps)


@dataclass(frozen=True, slots=True)
class VisionRay:
    """One cast ray from the snake head (matches encoder ray order)."""

    dx: int
    dy: int
    compass: str
    wall_steps: int
    body_steps: int | None

    def hits_body_first(self) -> bool:
        return self.body_steps is not None and self.body_steps < self.wall_steps

    def obstacle_proximity(self) -> float:
        """Activation for the nearest obstacle along this ray (wall or body)."""
        if self.hits_body_first():
            return proximity_activation(self.body_steps)
        return proximity_activation(self.wall_steps)

    def end_cell(self, head: Position) -> Position:
        """Last in-bounds cell along the ray (body or wall stop)."""
        if self.hits_body_first():
            steps = self.body_steps
        else:
            steps = max(0, self.wall_steps - 1)
        assert steps is not None
        return Position(head.x + self.dx * steps, head.y + self.dy * steps)


def cast_ray(
    head: Position,
    dx: int,
    dy: int,
    grid,
    body: set[Position],
) -> tuple[int, int | None]:
    """
    Steps along (dx, dy) until the grid edge.

    Returns (wall_steps, body_steps). body_steps is the first step onto a body cell.
    """
    body_steps: int | None = None
    steps = 0
    x, y = head.x, head.y
    while True:
        x += dx
        y += dy
        steps += 1
        pos = Position(x, y)
        if not grid.in_bounds(pos):
            return steps, body_steps
        if body_steps is None and pos in body:
            body_steps = steps


def vision_rays_for_game(game: Game) -> list[VisionRay]:
    """Eight rays relative to the snake heading (same geometry as the encoder)."""
    head = game.snake.head()
    body = set(game.snake.body[1:])
    grid = game.grid
    rays: list[VisionRay] = []

    for (dx, dy), compass in zip(
        relative_ray_deltas(game.snake.direction),
        _RAY_COMPASS_LABELS,
    ):
        wall_steps, body_steps = cast_ray(head, dx, dy, grid, body)
        rays.append(
            VisionRay(
                dx=dx,
                dy=dy,
                compass=compass,
                wall_steps=wall_steps,
                body_steps=body_steps,
            )
        )
    return rays
