import pygame

import config
from models.direction import Direction


class ControlPanel:
    def __init__(self, surface: pygame.Surface) -> None:
        self._surface = surface
        self._font = pygame.font.SysFont("consolas", 18)
        self._title_font = pygame.font.SysFont("consolas", 22, bold=True)

    def draw(self, active_direction: Direction | None) -> None:
        self._surface.fill(config.COLOR_PANEL)

        title = self._title_font.render("Controls", True, config.COLOR_TEXT)
        title_rect = title.get_rect(centerx=config.PANEL_WIDTH // 2, top=24)
        self._surface.blit(title, title_rect)

        layout = self._button_layout()
        for direction, rect in layout.items():
            self._draw_button(direction, rect, active_direction == direction)

        hint_lines = [
            "Arrow keys to move",
            "R to restart",
            "Esc to quit",
        ]
        y = config.WINDOW_HEIGHT - 24 - len(hint_lines) * 22
        for line in hint_lines:
            text = self._font.render(line, True, config.COLOR_TEXT_DIM)
            text_rect = text.get_rect(centerx=config.PANEL_WIDTH // 2, top=y)
            self._surface.blit(text, text_rect)
            y += 22

    def _button_layout(self) -> dict[Direction, pygame.Rect]:
        size = config.CONTROL_BUTTON_SIZE
        gap = config.CONTROL_BUTTON_GAP
        center_x = config.PANEL_WIDTH // 2
        cluster_top = 100

        up_rect = pygame.Rect(0, 0, size, size)
        up_rect.center = (center_x, cluster_top + size // 2)

        down_rect = pygame.Rect(0, 0, size, size)
        down_rect.center = (center_x, cluster_top + size + gap + size // 2)

        left_rect = pygame.Rect(0, 0, size, size)
        left_rect.center = (center_x - size - gap, down_rect.centery)

        right_rect = pygame.Rect(0, 0, size, size)
        right_rect.center = (center_x + size + gap, down_rect.centery)

        return {
            Direction.UP: up_rect,
            Direction.DOWN: down_rect,
            Direction.LEFT: left_rect,
            Direction.RIGHT: right_rect,
        }

    def _draw_button(
        self,
        direction: Direction,
        rect: pygame.Rect,
        active: bool,
    ) -> None:
        if active:
            glow_rect = rect.inflate(8, 8)
            pygame.draw.rect(self._surface, config.COLOR_CONTROL_ACTIVE_GLOW, glow_rect, border_radius=10)
            pygame.draw.rect(self._surface, config.COLOR_CONTROL_ACTIVE, rect, border_radius=8)
            label_color = config.COLOR_BACKGROUND
        else:
            pygame.draw.rect(self._surface, config.COLOR_CONTROL_INACTIVE, rect, border_radius=8)
            pygame.draw.rect(self._surface, config.COLOR_CONTROL_BORDER, rect, width=2, border_radius=8)
            label_color = config.COLOR_TEXT_DIM

        label = self._direction_label(direction)
        text = self._font.render(label, True, label_color)
        text_rect = text.get_rect(center=rect.center)
        self._surface.blit(text, text_rect)

    @staticmethod
    def _direction_label(direction: Direction) -> str:
        labels = {
            Direction.UP: "^",
            Direction.DOWN: "v",
            Direction.LEFT: "<",
            Direction.RIGHT: ">",
        }
        return labels[direction]
