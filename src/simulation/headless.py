"""
Headless evaluation of a genome for genetic-algorithm training.

When RANDOM_FOOD_EVAL is True (default), each evaluate() call runs one game with a
fresh random food seed (centered start, default heading). Otherwise falls back to
running over a pinned scenario list (legacy deterministic mode).
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
from game.game_state import DeathCause
from models.direction import Direction
from models.grid import Grid
from models.position import Position
from neural.encoder import GameStateEncoder
from neural.network import NeuralNetwork
from neural.policy import decide_step, new_rnn_hidden

if TYPE_CHECKING:
    from replay.recorder import GameRecorder


@dataclass(frozen=True, slots=True)
class Scenario:
    """A reproducible evaluation board."""

    food_seed: int
    start_position: Position | None = None
    start_direction: Direction | None = None


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Outcome of evaluating a genome."""

    fitness: float
    score: int
    steps: int
    avg_score: float
    best_food_seed: int
    death_cause: DeathCause


class HeadlessSimulator:
    """Runs games without rendering to score genomes during training."""

    def __init__(self, grid: Grid | None = None, max_steps: int | None = None) -> None:
        self._grid = grid or Grid(config.GRID_COLS, config.GRID_ROWS)
        self._max_steps = (
            max_steps if max_steps is not None else config.MAX_EVAL_STEPS
        )
        self._encoder = GameStateEncoder()
        self._scenarios: list[Scenario] = [Scenario(food_seed=0)]

    def build_scenarios(self, count: int, seed: int | None = None) -> list[Scenario]:
        """Generate centered scenarios with distinct food seeds (deterministic mode)."""
        rng = random.Random(seed)
        return [Scenario(food_seed=rng.randrange(2**31)) for _ in range(max(1, count))]

    def set_scenarios(self, scenarios: list[Scenario]) -> None:
        """Pin the scenario set used by evaluate() in deterministic mode."""
        if not scenarios:
            raise ValueError("At least one scenario is required.")
        self._scenarios = list(scenarios)

    def set_grid(self, cols: int, rows: int) -> None:
        """Switch evaluation to a different board size (curriculum training)."""
        self._grid = Grid(cols, rows)
        if config.MAX_EVAL_STEPS is not None:
            self._max_steps = cols * rows * 4

    @property
    def grid(self) -> Grid:
        return self._grid

    @property
    def scenarios(self) -> list[Scenario]:
        return list(self._scenarios)

    def evaluate(
        self, genome: Genome, *, runs: int | None = None, record: bool = False
    ) -> "EvalResult | tuple[EvalResult, GameRecorder]":
        """Run the genome and return fitness plus metrics for replay."""
        if config.SHARED_EVAL_SEEDS:
            return self._evaluate_scenarios(genome, record=record)
        run_count = runs if runs is not None else config.EVAL_RUNS_PER_GENOME
        if config.RANDOM_FOOD_EVAL:
            return self._evaluate_random(genome, runs=run_count, record=record)
        return self._evaluate_scenarios(genome, record=record)

    def _evaluate_random(
        self, genome: Genome, *, runs: int = 1, record: bool = False
    ) -> "EvalResult | tuple[EvalResult, GameRecorder]":
        """Run one or more games with fresh random food seeds each."""
        network = NeuralNetwork.from_genome(genome)
        run_count = max(1, runs)

        fitnesses: list[float] = []
        scores: list[int] = []
        best_index = 0
        best_key = (-1, 0)
        scenarios: list[Scenario] = []
        death_causes: list[DeathCause] = []

        for _ in range(run_count):
            food_seed = random.randrange(2**31)
            scenario = Scenario(food_seed=food_seed)
            score, steps, cause, shaping, won, space_ratio = self._run(network, scenario)
            fitnesses.append(
                compute_fitness(
                    score,
                    steps,
                    shaping,
                    won=won,
                    space_ratio=space_ratio,
                    grid_cols=self._grid.width,
                    grid_rows=self._grid.height,
                )
            )
            scores.append(score)
            scenarios.append(scenario)
            death_causes.append(cause)
            if (score, steps) > best_key:
                best_key = (score, steps)
                best_index = len(scenarios) - 1

        result = EvalResult(
            fitness=float(np.mean(fitnesses)),
            score=max(scores),
            steps=best_key[1],
            avg_score=float(np.mean(scores)),
            best_food_seed=scenarios[best_index].food_seed,
            death_cause=death_causes[best_index],
        )
        if not record:
            return result
        recorder = self._record_run(genome, scenarios[best_index])
        return result, recorder

    def _evaluate_scenarios(
        self, genome: Genome, *, record: bool = False
    ) -> "EvalResult | tuple[EvalResult, GameRecorder]":
        """Run over pinned scenarios; fitness is the mean across runs."""
        network = NeuralNetwork.from_genome(genome)

        fitnesses: list[float] = []
        scores: list[int] = []
        best_index = 0
        best_key = (-1, 0)
        death_causes: list[DeathCause] = []
        for i, scenario in enumerate(self._scenarios):
            score, steps, cause, shaping, won, space_ratio = self._run(network, scenario)
            fitnesses.append(
                compute_fitness(
                    score,
                    steps,
                    shaping,
                    won=won,
                    space_ratio=space_ratio,
                    grid_cols=self._grid.width,
                    grid_rows=self._grid.height,
                )
            )
            scores.append(score)
            death_causes.append(cause)
            if (score, steps) > best_key:
                best_key = (score, steps)
                best_index = i

        result = EvalResult(
            fitness=float(np.mean(fitnesses)),
            score=max(scores),
            steps=best_key[1],
            avg_score=float(np.mean(scores)),
            best_food_seed=self._scenarios[best_index].food_seed,
            death_cause=death_causes[best_index],
        )

        if not record:
            return result

        recorder = self._record_run(genome, self._scenarios[best_index])
        return result, recorder

    def _run(
        self, network: NeuralNetwork, scenario: Scenario
    ) -> tuple[int, int, DeathCause, float, bool, float]:
        """Simulate one game; return score, steps, cause, shaping, won, space_ratio."""
        game = Game(
            self._grid,
            food_seed=scenario.food_seed,
            start_position=scenario.start_position,
            start_direction=scenario.start_direction,
        )
        steps = 0
        shaping_bonus = 0.0
        space_ratio_sum = 0.0
        hidden = new_rnn_hidden()
        current = game.snake.direction
        while game.alive:
            if self._max_steps is not None and steps >= self._max_steps:
                return game.score, steps, "timeout", shaping_bonus, False, 0.0
            prev_dist = _manhattan_distance(game.snake.head(), game.food.position)
            current, hidden, _ = decide_step(
                network, game, self._encoder, hidden, current
            )
            game.tick(current)
            steps += 1
            curr_dist = _manhattan_distance(game.snake.head(), game.food.position)
            shaping_bonus += config.FITNESS_DISTANCE_SHAPING * float(prev_dist - curr_dist)
            space_ratio_sum += self._encoder.reachable_empty_ratio(game)
        space_ratio = space_ratio_sum / max(1, steps)
        cause = game.death_cause or "wall"
        return game.score, steps, cause, shaping_bonus, game.won, space_ratio

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
        while game.alive:
            if self._max_steps is not None and steps >= self._max_steps:
                break
            direction = controller.get_direction()
            tick_result = game.tick(direction)
            recorder.record_frame(game, controller.last_snapshot, tick_result)
            steps += 1
        return recorder


def _manhattan_distance(a: Position, b: Position) -> int:
    return abs(a.x - b.x) + abs(a.y - b.y)
