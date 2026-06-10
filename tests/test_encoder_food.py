"""Food inputs are comparable in strength to nearby wall proximity."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from game.game import Game  # noqa: E402
from models.direction import Direction  # noqa: E402
from models.grid import Grid  # noqa: E402
from models.position import Position  # noqa: E402
from neural.encoder import GameStateEncoder  # noqa: E402
from neural.vision_rays import proximity_activation, vision_rays_for_game  # noqa: E402


class EncoderFoodSignalTests(unittest.TestCase):
    def test_near_diagonal_food_ray_matches_near_wall_brightness(self) -> None:
        game = Game(
            Grid(15, 15),
            food_seed=1,
            start_position=Position(12, 2),
            start_direction=Direction.DOWN,
        )
        # Place food one diagonal step from head for a deterministic check.
        game.food._position = Position(11, 3)  # type: ignore[attr-defined]

        inputs = GameStateEncoder().encode(game)
        rays = vision_rays_for_game(game)
        best_food_ray = max(float(inputs[i * 3 + 1]) for i in range(8))

        nearest_wall = max(
            proximity_activation(ray.wall_steps)
            for ray in rays
            if not ray.hits_body_first()
        )
        self.assertGreater(best_food_ray, 0.35)
        self.assertGreaterEqual(best_food_ray, nearest_wall * 0.55)

    def test_food_direction_cues_are_brighter_than_before(self) -> None:
        game = Game(
            Grid(15, 15),
            food_seed=1,
            start_position=Position(7, 7),
            start_direction=Direction.UP,
        )
        game.food._position = Position(8, 8)  # type: ignore[attr-defined]
        inputs = GameStateEncoder().encode(game)
        # Food section starts after 24 ray features; indices 25-28 are direction cues.
        direction_cues = [float(inputs[i]) for i in range(25, 29)]
        self.assertGreater(max(direction_cues), 0.15)


if __name__ == "__main__":
    unittest.main()
