"""Outcome flags and score/alive tracking for a single game session."""

from dataclasses import dataclass
from typing import Literal

DeathCause = Literal["wall", "body", "starved", "win"]


@dataclass(frozen=True, slots=True)
class TickResult:
    """Events that occurred during one Game.tick() call."""

    ate_food: bool = False
    died: bool = False
    won: bool = False
    starved: bool = False
    death_cause: DeathCause | None = None


@dataclass
class GameState:
    """Mutable session state: score and whether the snake is still alive."""

    score: int = 0
    alive: bool = True
    won: bool = False
    steps_since_food: int = 0
    starved: bool = False
    death_cause: DeathCause | None = None
