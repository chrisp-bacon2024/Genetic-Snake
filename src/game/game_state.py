from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TickResult:
    ate_food: bool = False
    died: bool = False


@dataclass
class GameState:
    score: int = 0
    alive: bool = True
