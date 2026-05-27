"""
Core Snake simulation: movement, collisions, scoring, and reset.

This module has no pygame imports so the same rules can run headless during
genetic-algorithm training.
"""

import config
from models.direction import Direction
from models.food import Food
from models.grid import Grid
from models.snake import Snake

from .game_state import GameState, TickResult


class Game:
    """
    Owns the snake, food, grid, score, and alive flag.

    Call tick(direction) once per simulation step. The controller (human or AI)
    supplies the direction; Game applies rules and returns what happened.
    """

    def __init__(self, grid: Grid) -> None:
        self._grid = grid
        self._state = GameState()
        self._snake: Snake
        self._food: Food
        self._reset_entities()

    @property
    def grid(self) -> Grid:
        return self._grid

    @property
    def snake(self) -> Snake:
        return self._snake

    @property
    def food(self) -> Food:
        return self._food

    @property
    def score(self) -> int:
        return self._state.score

    @property
    def alive(self) -> bool:
        return self._state.alive

    @property
    def starved(self) -> bool:
        """True if the snake died from going too long without eating."""
        return self._state.starved

    def reset(self) -> None:
        """Start a new game on the same grid (score 0, snake at center)."""
        self._state = GameState()
        self._reset_entities()

    def tick(self, direction: Direction | None = None) -> TickResult:
        """
        Advance one simulation step.

        1. Apply direction (optional)
        2. Move snake head
        3. Die on wall or self collision
        4. Eat food → grow, increment score, respawn food, reset starvation counter
        5. Die if too many ticks pass without eating (kills infinite spin loops)
        """
        if not self._state.alive:
            return TickResult()

        if direction is not None:
            self._snake.set_direction(direction)

        new_head = self._snake.move()

        if not self._grid.in_bounds(new_head):
            self._state.alive = False
            return TickResult(died=True)

        if self._snake.collides_with_self(new_head):
            self._state.alive = False
            return TickResult(died=True)

        if new_head == self._food.position:
            self._snake.grow()
            self._state.score += 1
            self._state.steps_since_food = 0
            occupied = set(self._snake.body)
            self._food.respawn(self._grid, occupied)
            return TickResult(ate_food=True)

        self._state.steps_since_food += 1
        limit = config.starvation_limit(len(self._snake.body), self._grid.width, self._grid.height)
        if self._state.steps_since_food >= limit:
            self._state.alive = False
            self._state.starved = True
            return TickResult(died=True, starved=True)

        return TickResult()

    def _reset_entities(self) -> None:
        start = self._grid.center()
        self._snake = Snake(start)
        occupied = set(self._snake.body)
        food_position = self._grid.random_empty_cell(occupied)
        self._food = Food(food_position)
