"""
Abstract interface for anything that chooses snake direction each tick.

Implementations: KeyboardController (human), AIController (neural net).
"""

from abc import ABC, abstractmethod

from models.direction import Direction


class Controller(ABC):
    """Strategy object that maps input or inference to a Direction."""

    @abstractmethod
    def update(self, events: list) -> None:
        """Process pygame events for the current frame (may be no-op for AI)."""

    @abstractmethod
    def get_direction(self) -> Direction | None:
        """Direction to apply on the next simulation tick."""

    @abstractmethod
    def get_active_direction(self) -> Direction | None:
        """Direction currently shown in the UI (held key or last AI choice)."""

    def reset(self) -> None:
        """Clear controller state after a game restart."""
