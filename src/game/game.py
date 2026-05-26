from models.direction import Direction
from models.food import Food
from models.grid import Grid
from models.snake import Snake

from .game_state import GameState, TickResult


class Game:
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

    def reset(self) -> None:
        self._state = GameState()
        self._reset_entities()

    def tick(self, direction: Direction | None = None) -> TickResult:
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
            occupied = set(self._snake.body)
            self._food.respawn(self._grid, occupied)
            return TickResult(ate_food=True)

        return TickResult()

    def _reset_entities(self) -> None:
        start = self._grid.center()
        self._snake = Snake(start)
        occupied = set(self._snake.body)
        food_position = self._grid.random_empty_cell(occupied)
        self._food = Food(food_position)
