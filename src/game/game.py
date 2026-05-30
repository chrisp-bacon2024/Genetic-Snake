"""
Core Snake simulation: movement, collisions, scoring, and reset.

This module has no pygame imports so the same rules can run headless during
genetic-algorithm training.
"""

import random

import config
from models.direction import Direction
from models.food import Food
from models.grid import Grid
from models.position import Position
from models.snake import Snake

from .game_state import DeathCause, GameState, TickResult


class Game:
    """
    Owns the snake, food, grid, score, and alive flag.

    Call tick(direction) once per simulation step. The controller (human or AI)
    supplies the direction; Game applies rules and returns what happened.

    For deterministic genetic-algorithm evaluation, pass ``food_seed`` (and
    optionally a fixed ``start_position`` / ``start_direction``) so every snake
    faces an identical apple sequence.
    """

    def __init__(
        self,
        grid: Grid,
        food_seed: int | None = None,
        start_position: Position | None = None,
        start_direction: Direction | None = None,
    ) -> None:
        self._grid = grid
        self._food_seed = food_seed
        self._start_position = start_position
        self._start_direction = start_direction
        self._food_rng = random.Random(food_seed) if food_seed is not None else None
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

    @property
    def death_cause(self) -> DeathCause | None:
        """How the snake died: wall, body, starved, or None if still alive."""
        return self._state.death_cause

    def reset(
        self,
        food_seed: int | None = None,
        start_position: Position | None = None,
        start_direction: Direction | None = None,
    ) -> None:
        """
        Start a new game on the same grid (score 0).

        Optional arguments override the seed/start configured at construction so a
        single Game instance can be reused across evaluation scenarios.
        """
        if food_seed is not None:
            self._food_seed = food_seed
        if start_position is not None:
            self._start_position = start_position
        if start_direction is not None:
            self._start_direction = start_direction
        self._food_rng = (
            random.Random(self._food_seed) if self._food_seed is not None else None
        )
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
            self._state.death_cause = "wall"
            return TickResult(died=True, death_cause="wall")

        if self._snake.collides_with_self(new_head):
            self._state.alive = False
            self._state.death_cause = "body"
            return TickResult(died=True, death_cause="body")

        if new_head == self._food.position:
            self._snake.grow()
            self._state.score += 1
            self._state.steps_since_food = 0
            occupied = set(self._snake.body)
            self._food.respawn(self._grid, occupied, self._food_rng)
            return TickResult(ate_food=True)

        self._state.steps_since_food += 1
        limit = config.starvation_limit(len(self._snake.body), self._grid.width, self._grid.height)
        if self._state.steps_since_food >= limit:
            self._state.alive = False
            self._state.starved = True
            self._state.death_cause = "starved"
            return TickResult(died=True, starved=True, death_cause="starved")

        return TickResult()

    def _reset_entities(self) -> None:
        start = self._start_position if self._start_position is not None else self._grid.center()
        if self._start_direction is not None:
            self._snake = Snake(start, self._start_direction)
        else:
            self._snake = Snake(start)
        occupied = set(self._snake.body)
        food_position = self._grid.random_empty_cell(occupied, self._food_rng)
        self._food = Food(food_position)
