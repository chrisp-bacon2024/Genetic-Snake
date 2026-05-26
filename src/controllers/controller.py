from abc import ABC, abstractmethod

from models.direction import Direction


class Controller(ABC):
    @abstractmethod
    def update(self, events: list) -> None:
        """Process input events for the current frame."""

    @abstractmethod
    def get_direction(self) -> Direction | None:
        """Direction to apply on the next simulation tick."""

    @abstractmethod
    def get_active_direction(self) -> Direction | None:
        """Direction currently highlighted (key held or AI choice)."""

    def reset(self) -> None:
        """Reset controller state after a game restart."""
