"""Left sidebar: neural-network diagram and keyboard hints."""

import pygame

import config
from controllers.ai_controller import NetworkSnapshot

from .network_visualizer import NetworkVisualizer


class ControlPanel:
    """Draws the panel surface passed in at construction time."""

    def __init__(self, surface: pygame.Surface) -> None:
        self._surface = surface
        self._font = pygame.font.SysFont("consolas", 18)
        self._network_visualizer = NetworkVisualizer(surface)

    def draw(self, snapshot: NetworkSnapshot) -> None:
        """Fill the panel and render the network diagram plus footer hints."""
        self._surface.fill(config.COLOR_PANEL)
        self._network_visualizer.draw(snapshot)
        self._draw_hints(self._network_visualizer.bottom_y)

    def _draw_hints(self, start_y: int) -> None:
        hint_lines = [
            "AI-controlled",
            "R to restart",
            "Esc to quit",
        ]
        y = max(start_y, config.WINDOW_HEIGHT - 24 - len(hint_lines) * config.HINT_LINE_HEIGHT)
        for line in hint_lines:
            text = self._font.render(line, True, config.COLOR_TEXT_DIM)
            text_rect = text.get_rect(centerx=config.PANEL_WIDTH // 2, top=y)
            self._surface.blit(text, text_rect)
            y += config.HINT_LINE_HEIGHT
