"""Outcome flags and score/alive tracking for a single game session."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TickResult:
    """Events that occurred during one Game.tick() call."""

    ate_food: bool = False
    died: bool = False


@dataclass
class GameState:
    """Mutable session state: score and whether the snake is still alive."""

    score: int = 0
    alive: bool = True
