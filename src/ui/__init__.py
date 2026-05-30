"""Pygame window, main loop, and left-sidebar + game-grid rendering."""

__all__ = ["SnakeApp"]


def __getattr__(name: str):
    if name == "SnakeApp":
        from .app import SnakeApp

        return SnakeApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
