"""Draws the Snake playfield: grid lines, snake, food, score, and game-over overlay."""

import numpy as np
import pygame

import config
from game.game import Game
from neural.encoder import GameStateEncoder
from neural.vision_rays import vision_rays_for_game
from ui.input_feature_color import input_feature_color


class GameRenderer:
    """Renders game state onto a dedicated surface (right side of the window)."""

    def __init__(
        self,
        surface: pygame.Surface,
        game: Game,
        *,
        cell_size: int | None = None,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> None:
        self._surface = surface
        self._game = game
        self._cell_size = cell_size or config.CELL_SIZE
        self._offset_x = offset_x
        self._offset_y = offset_y
        self._score_font = pygame.font.SysFont("consolas", 22, bold=True)
        self._overlay_font = pygame.font.SysFont("consolas", 28, bold=True)

    @property
    def cols(self) -> int:
        return self._game.grid.width

    @property
    def rows(self) -> int:
        return self._game.grid.height

    @property
    def board_width(self) -> int:
        return self.cols * self._cell_size

    @property
    def board_height(self) -> int:
        return self.rows * self._cell_size

    def draw(self, *, inputs: np.ndarray | None = None) -> None:
        """Redraw the full game area for the current frame."""
        self._surface.fill(config.COLOR_BACKGROUND)
        self._draw_grid()
        if config.VISION_RAYS_ENABLED:
            self._draw_vision_rays(inputs)
        self._draw_food()
        self._draw_snake()
        self._draw_hud()

        if not self._game.alive:
            self._draw_game_over()

    def _death_message(self) -> tuple[str, str]:
        if self._game.starved:
            return "Starved", "Press R to restart"
        return "Game Over", "Press R to restart"

    def _draw_grid(self) -> None:
        for col in range(self.cols + 1):
            x = self._offset_x + col * self._cell_size
            pygame.draw.line(
                self._surface,
                config.COLOR_GRID_LINE,
                (x, self._offset_y),
                (x, self._offset_y + self.board_height),
            )
        for row in range(self.rows + 1):
            y = self._offset_y + row * self._cell_size
            pygame.draw.line(
                self._surface,
                config.COLOR_GRID_LINE,
                (self._offset_x, y),
                (self._offset_x + self.board_width, y),
            )

    def _cell_center(self, col: int, row: int) -> tuple[int, int]:
        return (
            self._offset_x + col * self._cell_size + self._cell_size // 2,
            self._offset_y + row * self._cell_size + self._cell_size // 2,
        )

    def _draw_vision_rays(self, inputs: np.ndarray | None) -> None:
        """Draw heading-relative rays from the head toward walls, body, and food."""
        head = self._game.snake.head()
        head_xy = self._cell_center(head.x, head.y)
        food = self._game.food.position
        line_width = max(1, self._cell_size // 22)
        end_radius = max(1, self._cell_size // 16)

        if inputs is None:
            inputs = GameStateEncoder().encode(self._game)
        ray_count = config.ENCODER_RAY_COUNT

        for ray_index, ray in enumerate(vision_rays_for_game(self._game)):
            end = ray.end_cell(head)
            end_xy = self._cell_center(end.x, end.y)
            value = ray.obstacle_proximity()
            feature_row = 2 if ray.hits_body_first() else 0
            color = input_feature_color(feature_row, value)
            if end_xy != head_xy:
                pygame.draw.line(self._surface, color, head_xy, end_xy, line_width)
                pygame.draw.circle(self._surface, color, end_xy, end_radius)

        if food != head:
            food_xy = self._cell_center(food.x, food.y)
            # Match the brightest food-alignment neuron in the ray grid (middle row).
            best_food_align = max(
                float(inputs[i * 3 + 1]) for i in range(ray_count)
            )
            food_color = input_feature_color(1, best_food_align)
            pygame.draw.line(self._surface, food_color, head_xy, food_xy, line_width)
            pygame.draw.circle(self._surface, food_color, food_xy, end_radius)

    def _draw_snake(self) -> None:
        body = self._game.snake.body
        for index, segment in enumerate(body):
            color = config.COLOR_SNAKE_HEAD if index == 0 else config.COLOR_SNAKE_BODY
            self._draw_cell(segment.x, segment.y, color, inset=2)

    def _draw_food(self) -> None:
        food = self._game.food.position
        self._draw_cell(food.x, food.y, config.COLOR_FOOD, inset=4)

    def _draw_cell(self, col: int, row: int, color: tuple[int, int, int], inset: int) -> None:
        rect = pygame.Rect(
            self._offset_x + col * self._cell_size + inset,
            self._offset_y + row * self._cell_size + inset,
            self._cell_size - inset * 2,
            self._cell_size - inset * 2,
        )
        pygame.draw.rect(self._surface, color, rect, border_radius=4)

    def _draw_hud(self) -> None:
        score_text = self._score_font.render(f"Score: {self._game.score}", True, config.COLOR_TEXT)
        text_rect = score_text.get_rect()
        if self._offset_y >= text_rect.height + 10:
            text_rect.topleft = (self._offset_x + 8, self._offset_y - text_rect.height - 6)
        else:
            text_rect.topright = (
                self._offset_x + self.board_width - 8,
                self._offset_y + 8,
            )
        self._surface.blit(score_text, text_rect)

    def _draw_game_over(self) -> None:
        overlay = pygame.Surface((self.board_width, self.board_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 140))
        self._surface.blit(overlay, (self._offset_x, self._offset_y))

        title, subtitle = self._death_message()
        center_x = self._offset_x + self.board_width // 2
        center_y = self._offset_y + self.board_height // 2
        game_over = self._overlay_font.render(title, True, config.COLOR_GAME_OVER)
        restart = self._score_font.render(subtitle, True, config.COLOR_TEXT)
        game_over_rect = game_over.get_rect(center=(center_x, center_y - 16))
        restart_rect = restart.get_rect(center=(center_x, center_y + 20))
        self._surface.blit(game_over, game_over_rect)
        self._surface.blit(restart, restart_rect)


def board_layout(
    cols: int,
    rows: int,
    surface_width: int,
    surface_height: int,
) -> tuple[int, int, int]:
    """Return (cell_size, offset_x, offset_y) to center the board on a surface."""
    cell_size = min(surface_width // cols, surface_height // rows)
    offset_x = (surface_width - cols * cell_size) // 2
    offset_y = (surface_height - rows * cell_size) // 2
    return cell_size, offset_x, offset_y
