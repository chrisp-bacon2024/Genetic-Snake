"""
Watch the saved best snakes from a training run.

Each generation's best genome is stored in ``replays/gen_XXXX.npz`` (genes + the food
seed of its best run). The viewer re-simulates each one live, reusing the same game
renderer and neural-network panel as the interactive app, and cycles through them in
generation order. The current generation and score are shown in the window caption.

Controls: Right/Space = next, Left = previous, Esc = quit.
"""

from pathlib import Path

import numpy as np
import pygame

import config
from controllers.ai_controller import AIController
from evolution.genome import Genome
from game.game import Game
from models.grid import Grid
from neural.network import NeuralNetwork

from .control_panel import ControlPanel
from .game_renderer import GameRenderer


class ReplayViewer:
    """Cycles through saved per-generation best genomes, re-simulated live."""

    def __init__(self, replays_dir: Path, ticks_per_second: int | None = None) -> None:
        self._files = sorted(replays_dir.glob("gen_*.npz"))
        if not self._files:
            raise FileNotFoundError(
                f"No gen_*.npz replays found in {replays_dir.resolve()}. Train first."
            )
        self._tick_interval = 1.0 / (ticks_per_second or config.TICKS_PER_SECOND)
        self._index = 0

    def run(self) -> None:
        pygame.init()
        pygame.display.set_caption("Genetic Snake - Replay")
        screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
        clock = pygame.time.Clock()

        panel_surface = pygame.Surface((config.PANEL_WIDTH, config.WINDOW_HEIGHT))
        game_surface = pygame.Surface(
            (config.WINDOW_WIDTH - config.PANEL_WIDTH, config.WINDOW_HEIGHT)
        )
        grid = Grid(config.GRID_COLS, config.GRID_ROWS)

        running = True
        while running and self._files:
            generation, score, genome, food_seed = self._load(self._index)
            game = Game(grid, food_seed=food_seed)
            controller = AIController(game, NeuralNetwork.from_genome(genome))
            control_panel = ControlPanel(panel_surface)
            renderer = GameRenderer(game_surface, game)
            pygame.display.set_caption(
                f"Genetic Snake - Gen {generation} (saved score {score})"
            )

            controller.get_direction()
            tick_accumulator = 0.0
            advance = 0  # -1 prev, +1 next
            steps = 0
            max_steps = config.MAX_EVAL_STEPS
            dead_linger = 0.0

            while running and advance == 0:
                delta = clock.tick(config.RENDER_FPS) / 1000.0
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key in (pygame.K_RIGHT, pygame.K_SPACE):
                            advance = 1
                        elif event.key == pygame.K_LEFT:
                            advance = -1

                if game.alive and steps < max_steps:
                    tick_accumulator += delta
                    while tick_accumulator >= self._tick_interval:
                        tick_accumulator -= self._tick_interval
                        direction = controller.get_direction()
                        game.tick(direction)
                        steps += 1
                        if not game.alive or steps >= max_steps:
                            break
                else:
                    dead_linger += delta
                    if dead_linger >= 1.5:
                        advance = 1

                control_panel.draw(controller.last_snapshot)
                renderer.draw()
                screen.blit(panel_surface, (0, 0))
                screen.blit(game_surface, (config.PANEL_WIDTH, 0))
                pygame.display.flip()

            if running:
                self._index = (self._index + advance) % len(self._files)

        pygame.quit()

    def _load(self, index: int) -> tuple[int, int, Genome, int]:
        data = np.load(self._files[index])
        genome = Genome(np.asarray(data["genes"], dtype=np.float64))
        generation = int(data["generation"])
        score = int(data["score"])
        food_seed = int(data["food_seed"])
        return generation, score, genome, food_seed
