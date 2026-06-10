"""Fitness rewards eating and fast first-apple runs over idle survival."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import config  # noqa: E402
from evolution.fitness import compute_fitness  # noqa: E402


class FitnessFoodSeekingTests(unittest.TestCase):
    def test_scoreless_wandering_beats_nothing_but_loses_to_first_eat(self) -> None:
        wander = compute_fitness(0, steps=200, shaping_bonus=0.0)
        eater = compute_fitness(
            1,
            steps=25,
            shaping_bonus=40.0,
            steps_to_first_eat=25,
        )
        self.assertLess(wander, eater)

    def test_faster_first_eat_scores_higher(self) -> None:
        slow = compute_fitness(1, steps=60, steps_to_first_eat=60)
        fast = compute_fitness(1, steps=20, steps_to_first_eat=20)
        self.assertGreater(fast, slow)

    def test_shaping_cap_higher_before_first_apple(self) -> None:
        hungry_gain = compute_fitness(0, steps=0, shaping_bonus=200.0) - compute_fitness(
            0, steps=0, shaping_bonus=100.0
        )
        fed_gain = compute_fitness(3, steps=0, shaping_bonus=200.0) - compute_fitness(
            3, steps=0, shaping_bonus=100.0
        )
        self.assertGreater(hungry_gain, fed_gain)

    def test_space_bonus_suppressed_at_score_zero(self) -> None:
        self.assertAlmostEqual(
            compute_fitness(0, steps=5, space_ratio=1.0),
            compute_fitness(0, steps=5, space_ratio=0.0),
        )
        fed_gain = compute_fitness(3, steps=5, space_ratio=1.0) - compute_fitness(
            3, steps=5, space_ratio=0.0
        )
        self.assertAlmostEqual(fed_gain, config.FITNESS_SPACE_WEIGHT, places=3)


if __name__ == "__main__":
    unittest.main()
