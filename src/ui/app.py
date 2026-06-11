"""
Main pygame application: wires game logic, AI, recording, and rendering.

The simulation runs at TICKS_PER_SECOND; the display refreshes at RENDER_FPS.
Each sim tick: decide direction → tick game → record frame → draw UI.
"""

import pygame

import config
from controllers.ai_controller import AIController
from game.game import Game
from models.grid import Grid
from neural.network import NeuralNetwork
from replay.recorder import GameRecorder

from .control_panel import ControlPanel
from .game_renderer import GameRenderer
from .score_jump_dialog import (
    ScoreJumpDialog,
    _DoubleClickDetector,
    try_open_score_dialog_on_double_click,
)


class SnakeApp:
    """
    Top-level application object.

    Creates a random genome/network on startup, runs until quit, and records
    every tick in a GameRecorder (save to disk is manual / future GA hook).
    """

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
        self._genome = NeuralNetwork.random_genome()
        self._network = NeuralNetwork.from_genome(self._genome)
        self._controller = AIController(self._game, self._network)
        self._recorder = GameRecorder()
        self._control_panel = ControlPanel(panel_surface)
        self._renderer = GameRenderer(game_surface, self._game)

        self._panel_surface = panel_surface
        self._game_surface = game_surface
        self._tick_accumulator = 0.0
        self._tick_interval = 1.0 / config.TICKS_PER_SECOND
        self._running = True
        self._score_dialog = ScoreJumpDialog(game_surface.get_width(), game_surface.get_height())
        self._score_dialog.set_panel_origin(config.PANEL_WIDTH)
        self._score_double_click = _DoubleClickDetector()

        self._begin_recording()
        self._controller.get_direction()

    @property
    def recorder(self) -> GameRecorder:
        """Access the in-memory replay (call save() when persisting a run)."""
        return self._recorder

    def _begin_recording(self) -> None:
        self._recorder.start(self._genome, self._game)

    def run(self) -> None:
        """Main loop: events → simulation → render until window closes."""
        while self._running:
            delta = self._clock.tick(config.RENDER_FPS) / 1000.0
            events = pygame.event.get()
            self._handle_events(events)
            if not self._score_dialog.active:
                self._controller.update(events)
            self._update_simulation(delta)
            self._render()
            pygame.display.flip()

        pygame.quit()

    def _max_win_score(self) -> int:
        return config.max_win_score(self._game.grid.width, self._game.grid.height)

    def _apply_score_jump(self, target_score: int) -> None:
        self._controller.seek_to_score(target_score)
        self._begin_recording()
        if self._game.alive:
            self._controller.get_direction()

    def _handle_events(self, events: list) -> None:
        now = pygame.time.get_ticks() / 1000.0
        max_score = self._max_win_score()

        for event in events:
            if event.type == pygame.QUIT:
                self._running = False
                return

            if self._score_dialog.active:
                result = self._score_dialog.handle_event(event, max_score=max_score)
                if result is not False:
                    if isinstance(result, int):
                        self._apply_score_jump(result)
                    continue

            if try_open_score_dialog_on_double_click(
                event,
                score_rect=self._renderer.score_rect,
                panel_width=config.PANEL_WIDTH,
                dialog=self._score_dialog,
                detector=self._score_double_click,
                now=now,
            ):
                continue

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._running = False
                    return
                elif event.key == pygame.K_r:
                    self._restart()

    def _restart(self) -> None:
        self._game.reset()
        if config.RESTART_NEW_GENOME:
            self._genome = NeuralNetwork.random_genome()
            self._network = NeuralNetwork.from_genome(self._genome)
            self._controller = AIController(self._game, self._network)
        else:
            self._controller.reset()
        self._begin_recording()
        self._controller.get_direction()

    def _update_simulation(self, delta: float) -> None:
        """Fixed-rate game ticks decoupled from render frame rate."""
        if not self._game.alive:
            return

        self._tick_accumulator += delta
        while self._tick_accumulator >= self._tick_interval:
            self._tick_accumulator -= self._tick_interval
            direction = self._controller.get_direction()
            tick_result = self._game.tick(direction)
            self._recorder.record_frame(
                self._game,
                self._controller.last_snapshot,
                tick_result,
            )

    def _render(self) -> None:
        snapshot = self._controller.visual_snapshot()
        self._control_panel.draw(snapshot)
        self._renderer.draw(inputs=snapshot.inputs)
        self._score_dialog.draw(self._game_surface, max_score=self._max_win_score())

        self._screen.blit(self._panel_surface, (0, 0))
        self._screen.blit(self._game_surface, (config.PANEL_WIDTH, 0))
