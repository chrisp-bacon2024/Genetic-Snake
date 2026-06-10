"""Left sidebar: neural-network diagram and keyboard hints."""

import pygame

import config
from controllers.ai_controller import NetworkSnapshot

from .network_visualizer import NetworkVisualizer

# Space reserved below the network diagram for gen-jump UI + nav bar (replay viewer).
REPLAY_GEN_INPUT_GAP = 8
REPLAY_FOOTER_HEIGHT = 4 + 16 + 6 + 22 + 12 + 6 + REPLAY_GEN_INPUT_GAP


class ControlPanel:
    """Draws the panel surface passed in at construction time."""

    def __init__(self, surface: pygame.Surface) -> None:
        self._surface = surface
        self._font = pygame.font.SysFont("consolas", 14)
        self._network_visualizer = NetworkVisualizer(surface)

    @property
    def network_bottom_y(self) -> int:
        """Y coordinate just below the output arrow cluster."""
        return self._network_visualizer.bottom_y

    def draw(self, snapshot: NetworkSnapshot, *, replay_mode: bool = False) -> None:
        """Fill the panel and render the network diagram plus footer hints."""
        self._surface.fill(config.COLOR_PANEL)
        if replay_mode:
            bottom_limit = config.WINDOW_HEIGHT - REPLAY_FOOTER_HEIGHT
        else:
            hint_height = 3 * config.HINT_LINE_HEIGHT + 16
            bottom_limit = config.WINDOW_HEIGHT - hint_height - 8
        self._network_visualizer.draw(snapshot, bottom_limit=bottom_limit)
        if not replay_mode:
            self._draw_hints(self._network_visualizer.bottom_y)

    def _draw_hints(self, start_y: int) -> None:
        hint_lines = [
            "AI-controlled",
            "R to restart",
            "Esc to quit",
        ]
        y = max(
            start_y + 8,
            config.WINDOW_HEIGHT - len(hint_lines) * config.HINT_LINE_HEIGHT - 10,
        )
        for line in hint_lines:
            text = self._font.render(line, True, config.COLOR_TEXT_DIM)
            text_rect = text.get_rect(centerx=config.PANEL_WIDTH // 2, top=y)
            self._surface.blit(text, text_rect)
            y += config.HINT_LINE_HEIGHT
