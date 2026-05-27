"""
Keyboard-driven snake control (legacy / debugging).

Not wired in the main app — the snake is AI-controlled. Kept so human play
can be re-enabled without rewriting the Controller abstraction.
"""

import pygame

from models.direction import Direction

from .controller import Controller


class KeyboardController(Controller):
    """Maps arrow keys to directions with buffering and highlight-on-press."""

    _KEY_TO_DIRECTION = {
        pygame.K_UP: Direction.UP,
        pygame.K_DOWN: Direction.DOWN,
        pygame.K_LEFT: Direction.LEFT,
        pygame.K_RIGHT: Direction.RIGHT,
    }

    def __init__(self, initial_direction: Direction = Direction.RIGHT) -> None:
        self._current_direction = initial_direction
        self._pending_direction: Direction | None = None
        self._pressed_keys: set[int] = set()

    def update(self, events: list) -> None:
        for event in events:
            if event.type == pygame.KEYDOWN and event.key in self._KEY_TO_DIRECTION:
                self._pressed_keys.add(event.key)
                direction = self._KEY_TO_DIRECTION[event.key]
                if direction != self._current_direction.opposite():
                    self._pending_direction = direction
            elif event.type == pygame.KEYUP and event.key in self._KEY_TO_DIRECTION:
                self._pressed_keys.discard(event.key)

    def get_direction(self) -> Direction | None:
        if self._pending_direction is not None:
            self._current_direction = self._pending_direction
            self._pending_direction = None
        return self._current_direction

    def get_active_direction(self) -> Direction | None:
        if not self._pressed_keys:
            return self._current_direction
        for key in (pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT):
            if key in self._pressed_keys:
                return self._KEY_TO_DIRECTION[key]
        return self._current_direction

    def reset(self) -> None:
        self._current_direction = Direction.RIGHT
        self._pending_direction = None
        self._pressed_keys.clear()
