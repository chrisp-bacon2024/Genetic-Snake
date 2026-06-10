"""Vision ray casting matches encoder geometry."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from game.game import Game  # noqa: E402
from models.grid import Grid  # noqa: E402
from models.position import Position  # noqa: E402
from models.direction import Direction  # noqa: E402
from neural.encoder import GameStateEncoder  # noqa: E402
from neural.vision_rays import cast_ray, proximity_activation, vision_rays_for_game  # noqa: E402


class VisionRayTests(unittest.TestCase):
    def test_cast_ray_hits_wall(self) -> None:
        head = Position(2, 2)
        wall_steps, body = cast_ray(head, 1, 0, Grid(5, 5), set())
        self.assertEqual(wall_steps, 3)
        self.assertIsNone(body)

    def test_eight_rays_on_game(self) -> None:
        game = Game(Grid(10, 10), food_seed=1)
        rays = vision_rays_for_game(game)
        self.assertEqual(len(rays), 8)
        self.assertEqual(rays[0].compass, "N")

    def test_proximity_falls_off_with_distance(self) -> None:
        near = proximity_activation(2)
        far = proximity_activation(8)
        self.assertGreater(near, far)

    def test_ray_obstacle_proximity_matches_encoder(self) -> None:
        game = Game(
            Grid(15, 15),
            food_seed=1,
            start_position=Position(12, 2),
            start_direction=Direction.DOWN,
        )
        inputs = GameStateEncoder().encode(game)
        rays = vision_rays_for_game(game)
        for index, ray in enumerate(rays):
            expected = float(inputs[index * 3 + (2 if ray.hits_body_first() else 0)])
            self.assertAlmostEqual(ray.obstacle_proximity(), expected)


if __name__ == "__main__":
    unittest.main()
