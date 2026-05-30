"""
Headless evaluation of a genome over one or more deterministic scenarios.

A Scenario pins the snake's start position/direction and the food RNG seed so every
genome in a generation faces an identical apple sequence. Evaluation runs the network
directly (no pygame, no per-tick snapshot) for speed; the optional record=True path
reuses AIController + GameRecorder to capture a replay of the best run.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

import config
from evolution.fitness import compute_fitness
from evolution.genome import Genome
from game.game import Game
from models.direction import Direction
from models.grid import Grid
from models.position import Position
from neural.encoder import GameStateEncoder
from neural.network import NeuralNetwork

if TYPE_CHECKING:
    from replay.recorder import GameRecorder

_OUTPUT_TO_DIRECTION = {
    0: Direction.UP,
    1: Direction.DOWN,
    2: Direction.LEFT,
    3: Direction.RIGHT,
}


@dataclass(frozen=True, slots=True)
class Scenario:
    """A reproducible evaluation board."""

    food_seed: int
    start_position: Position | None = None
    start_direction: Direction | None = None


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Averaged outcome of evaluating a genome across its scenarios."""

    fitness: float
    score: int  # best score across runs
    steps: int  # steps of the best-scoring run
    avg_score: float
    best_food_seed: int  # food seed of the best-scoring run (for replay)


class HeadlessSimulator:
    """Runs games without rendering to score genomes during training."""

    def __init__(self, grid: Grid | None = None, max_steps: int | None = None) -> None:
        self._grid = grid or Grid(config.GRID_COLS, config.GRID_ROWS)
        self._max_steps = max_steps or config.MAX_EVAL_STEPS
        self._encoder = GameStateEncoder()
        self._scenarios: list[Scenario] = [Scenario(food_seed=0)]

    def build_scenarios(self, count: int, seed: int | None = None) -> list[Scenario]:
        """Generate `count` centered scenarios with distinct food seeds."""
        rng = random.Random(seed)
        return [Scenario(food_seed=rng.randrange(2**31)) for _ in range(max(1, count))]

    def set_scenarios(self, scenarios: list[Scenario]) -> None:
        """Pin the scenario set used by evaluate() (all genomes share it)."""
        if not scenarios:
            raise ValueError("At least one scenario is required.")
        self._scenarios = list(scenarios)

    @property
    def scenarios(self) -> list[Scenario]:
        return list(self._scenarios)

    def evaluate(
        self, genome: Genome, *, record: bool = False
    ) -> "EvalResult | tuple[EvalResult, GameRecorder]":
        """
        Run the genome over every active scenario.

        Returns averaged fitness plus the best score/steps. With record=True, also
        returns a GameRecorder holding the best-scoring run for replay.
        """
        network = NeuralNetwork.from_genome(genome)

        fitnesses: list[float] = []
        scores: list[int] = []
        best_index = 0
        best_key = (-1, 0)  # (score, steps) — prefer higher score, then more steps
        for i, scenario in enumerate(self._scenarios):
            score, steps = self._run(network, scenario)
            fitnesses.append(compute_fitness(score, steps))
            scores.append(score)
            if (score, steps) > best_key:
                best_key = (score, steps)
                best_index = i

        result = EvalResult(
            fitness=float(np.mean(fitnesses)),
            score=max(scores),
            steps=best_key[1],
            avg_score=float(np.mean(scores)),
            best_food_seed=self._scenarios[best_index].food_seed,
        )

        if not record:
            return result

        recorder = self._record_run(genome, self._scenarios[best_index])
        return result, recorder

    def _run(self, network: NeuralNetwork, scenario: Scenario) -> tuple[int, int]:
        """Simulate one game; return (score, steps)."""
        game = Game(
            self._grid,
            food_seed=scenario.food_seed,
            start_position=scenario.start_position,
            start_direction=scenario.start_direction,
        )
        steps = 0
        current = game.snake.direction
        while game.alive and steps < self._max_steps:
            inputs = self._encoder.encode(game)
            outputs = network.forward(inputs).outputs
            current = self._pick_direction(outputs, current)
            game.tick(current)
            steps += 1
        return game.score, steps

    def _record_run(self, genome: Genome, scenario: Scenario) -> "GameRecorder":
        """Re-run a scenario with AIController so the full replay is captured."""
        from controllers.ai_controller import AIController
        from replay.recorder import GameRecorder

        game = Game(
            self._grid,
            food_seed=scenario.food_seed,
            start_position=scenario.start_position,
            start_direction=scenario.start_direction,
        )
        network = NeuralNetwork.from_genome(genome)
        controller = AIController(game, network)
        recorder = GameRecorder()
        recorder.start(genome, game)

        steps = 0
        while game.alive and steps < self._max_steps:
            direction = controller.get_direction()
            tick_result = game.tick(direction)
            recorder.record_frame(game, controller.last_snapshot, tick_result)
            steps += 1
        return recorder

    @staticmethod
    def _pick_direction(outputs: np.ndarray, current: Direction) -> Direction:
        """Argmax over outputs, skipping any 180-degree reversal."""
        for index in np.argsort(outputs)[::-1]:
            direction = _OUTPUT_TO_DIRECTION[int(index)]
            if direction != current.opposite():
                return direction
        return current
