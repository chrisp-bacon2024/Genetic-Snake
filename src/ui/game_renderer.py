import pygame

import config
from game.game import Game


class GameRenderer:
    def __init__(self, surface: pygame.Surface, game: Game) -> None:
        self._surface = surface
        self._game = game
        self._score_font = pygame.font.SysFont("consolas", 22, bold=True)
        self._overlay_font = pygame.font.SysFont("consolas", 28, bold=True)

    def draw(self) -> None:
        self._surface.fill(config.COLOR_BACKGROUND)
        self._draw_grid()
        self._draw_food()
        self._draw_snake()
        self._draw_hud()

        if not self._game.alive:
            self._draw_game_over()

    def _draw_grid(self) -> None:
        grid_width = config.GRID_COLS * config.CELL_SIZE
        for col in range(config.GRID_COLS + 1):
            x = col * config.CELL_SIZE
            pygame.draw.line(
                self._surface,
                config.COLOR_GRID_LINE,
                (x, 0),
                (x, config.WINDOW_HEIGHT),
            )
        for row in range(config.GRID_ROWS + 1):
            y = row * config.CELL_SIZE
            pygame.draw.line(
                self._surface,
                config.COLOR_GRID_LINE,
                (0, y),
                (grid_width, y),
            )

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
            col * config.CELL_SIZE + inset,
            row * config.CELL_SIZE + inset,
            config.CELL_SIZE - inset * 2,
            config.CELL_SIZE - inset * 2,
        )
        pygame.draw.rect(self._surface, color, rect, border_radius=4)

    def _draw_hud(self) -> None:
        score_text = self._score_font.render(f"Score: {self._game.score}", True, config.COLOR_TEXT)
        self._surface.blit(score_text, (12, 10))

    def _draw_game_over(self) -> None:
        grid_width = config.GRID_COLS * config.CELL_SIZE
        overlay = pygame.Surface((grid_width, config.WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 140))
        self._surface.blit(overlay, (0, 0))

        game_over = self._overlay_font.render("Game Over", True, config.COLOR_GAME_OVER)
        restart = self._score_font.render("Press R to restart", True, config.COLOR_TEXT)
        game_over_rect = game_over.get_rect(
            center=(grid_width // 2, config.WINDOW_HEIGHT // 2 - 16)
        )
        restart_rect = restart.get_rect(
            center=(grid_width // 2, config.WINDOW_HEIGHT // 2 + 20)
        )
        self._surface.blit(game_over, game_over_rect)
        self._surface.blit(restart, restart_rect)
