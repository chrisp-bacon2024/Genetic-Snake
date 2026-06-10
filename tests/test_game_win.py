"""Tests for board-fill win detection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import config  # noqa: E402
from game.game import Game  # noqa: E402
from models.direction import Direction  # noqa: E402
from models.grid import Grid  # noqa: E402


class WinDetectionTests(unittest.TestCase):
    def test_max_win_score_5x5(self) -> None:
        self.assertEqual(config.max_win_score(5, 5), 24)

    def test_eating_last_apple_on_2x2_is_win(self) -> None:
        game = Game(Grid(2, 2), food_seed=0)
        directions = (
            Direction.UP,
            Direction.LEFT,
            Direction.DOWN,
            Direction.RIGHT,
            Direction.UP,
            Direction.LEFT,
            Direction.DOWN,
        )
        for direction in directions:
            if not game.alive:
                break
            game.tick(direction)
        self.assertTrue(game.won)
        self.assertEqual(game.death_cause, "win")
        self.assertGreaterEqual(game.score, config.max_win_score(2, 2))
        self.assertEqual(len(game.snake.body), 4)


if __name__ == "__main__":
    unittest.main()
