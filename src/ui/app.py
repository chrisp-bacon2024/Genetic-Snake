import pygame

import config
from controllers.keyboard_controller import KeyboardController
from game.game import Game
from models.grid import Grid

from .control_panel import ControlPanel
from .game_renderer import GameRenderer


class SnakeApp:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Genetic Snake")
        self._screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
        self._clock = pygame.time.Clock()

        panel_surface = pygame.Surface((config.PANEL_WIDTH, config.WINDOW_HEIGHT))
        game_surface = pygame.Surface(
            (config.WINDOW_WIDTH - config.PANEL_WIDTH, config.WINDOW_HEIGHT)
        )

        grid = Grid(config.GRID_COLS, config.GRID_ROWS)
        self._game = Game(grid)
        self._controller = KeyboardController(self._game.snake.direction)
        self._control_panel = ControlPanel(panel_surface)
        self._renderer = GameRenderer(game_surface, self._game)

        self._panel_surface = panel_surface
        self._game_surface = game_surface
        self._tick_accumulator = 0.0
        self._tick_interval = 1.0 / config.TICKS_PER_SECOND
        self._running = True

    def run(self) -> None:
        while self._running:
            delta = self._clock.tick(config.RENDER_FPS) / 1000.0
            events = pygame.event.get()
            self._handle_events(events)
            self._controller.update(events)
            self._update_simulation(delta)
            self._render()
            pygame.display.flip()

        pygame.quit()

    def _handle_events(self, events: list) -> None:
        for event in events:
            if event.type == pygame.QUIT:
                self._running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._running = False
                elif event.key == pygame.K_r:
                    self._restart()

    def _restart(self) -> None:
        self._game.reset()
        self._controller.reset()

    def _update_simulation(self, delta: float) -> None:
        if not self._game.alive:
            return

        self._tick_accumulator += delta
        while self._tick_accumulator >= self._tick_interval:
            self._tick_accumulator -= self._tick_interval
            direction = self._controller.get_direction()
            self._game.tick(direction)

    def _render(self) -> None:
        active_direction = self._controller.get_active_direction()
        self._control_panel.draw(active_direction)
        self._renderer.draw()

        self._screen.blit(self._panel_surface, (0, 0))
        self._screen.blit(self._game_surface, (config.PANEL_WIDTH, 0))
