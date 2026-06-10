"""Draws the Snake playfield: grid lines, snake, food, score, and game-over overlay."""

from __future__ import annotations

import math

import numpy as np
import pygame

import config
from game.game import Game
from models.position import Position
from neural.encoder import GameStateEncoder
from neural.vision_rays import vision_rays_for_game
from ui.input_feature_color import input_feature_color, lerp_color


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
        if not self._game.won:
            self._draw_food()
        self._draw_snake()
        if self._game.won:
            self._draw_win_fill()
        self._draw_hud()

        if not self._game.alive:
            self._draw_game_over()

    def _death_message(self) -> tuple[str, str, tuple[int, int, int]]:
        if self._game.won:
            return "You Win!", "Press R to restart", config.COLOR_WIN
        if self._game.starved:
            return "Starved", "Press R to restart", config.COLOR_GAME_OVER
        return "Game Over", "Press R to restart", config.COLOR_GAME_OVER

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
        snake = self._game.snake
        body = snake.body
        length = len(body)
        move_dx, move_dy = snake.direction.to_delta()

        for index in range(length - 1):
            self._draw_segment_joint(body, index, length)

        for index, segment in enumerate(body):
            color = self._snake_segment_color(index, length)
            axis = self._segment_axis(body, index, move_dx, move_dy)
            if axis is None:
                continue
            ax, ay = axis
            self._draw_segment_pad(
                segment.x,
                segment.y,
                color,
                index,
                length,
                body,
                move_dx,
                move_dy,
            )
            if index == 0:
                dead = not self._game.alive and not self._game.won
                self._draw_head_eyes(
                    segment.x,
                    segment.y,
                    move_dx,
                    move_dy,
                    dead=dead,
                )
            elif length > 1:
                self._draw_scale_chevron(segment.x, segment.y, ax, ay, color)

    def _snake_segment_color(self, index: int, length: int) -> tuple[int, int, int]:
        """Dark head → lighter tail along the full body."""
        if length <= 1:
            return config.COLOR_SNAKE_HEAD
        t = index / float(length - 1)
        return lerp_color(config.COLOR_SNAKE_HEAD, config.COLOR_SNAKE_TAIL, t)

    @staticmethod
    def _segment_axis(
        body: tuple,
        index: int,
        move_dx: int,
        move_dy: int,
    ) -> tuple[int, int] | None:
        """Unit step along the body toward the tail (head uses movement heading)."""
        if len(body) < 1:
            return None
        if index == 0:
            return move_dx, move_dy
        if index < len(body) - 1:
            cur, nxt = body[index], body[index + 1]
            return nxt.x - cur.x, nxt.y - cur.y
        prev, cur = body[index - 1], body[index]
        return cur.x - prev.x, cur.y - prev.y

    def _segment_width_frac(self, index: int, length: int) -> float:
        """Head uses max width; tail uses min; body interpolates between."""
        if length <= 1:
            return config.SNAKE_SEGMENT_WIDTH_HEAD_FRAC
        t = index / float(length - 1)
        head = config.SNAKE_SEGMENT_WIDTH_HEAD_FRAC
        tail = config.SNAKE_SEGMENT_WIDTH_TAIL_FRAC
        return head + (tail - head) * t

    @staticmethod
    def _segment_neighbor_dirs(
        body: tuple[Position, ...],
        index: int,
    ) -> list[tuple[int, int]]:
        """Grid steps toward body segments this cell touches (prev and/or next)."""
        dirs: list[tuple[int, int]] = []
        cur = body[index]
        if index > 0:
            prev = body[index - 1]
            dirs.append((prev.x - cur.x, prev.y - cur.y))
        if index < len(body) - 1:
            nxt = body[index + 1]
            dirs.append((nxt.x - cur.x, nxt.y - cur.y))
        return dirs

    def _draw_segment_joint(
        self,
        body: tuple[Position, ...],
        index: int,
        length: int,
    ) -> None:
        """Rounded knuckle bridging two adjacent segments at the cell border."""
        a = body[index]
        b = body[index + 1]
        ax, ay = self._cell_center(a.x, a.y)
        bx, by = self._cell_center(b.x, b.y)
        mx = (ax + bx) * 0.5
        my = (ay + by) * 0.5
        width_a = self._segment_width_frac(index, length)
        width_b = self._segment_width_frac(index + 1, length)
        radius = (
            self._cell_size
            * (width_a + width_b)
            * 0.5
            * config.SNAKE_JOINT_RADIUS_SCALE
        )
        color = lerp_color(
            self._snake_segment_color(index, length),
            self._snake_segment_color(index + 1, length),
            0.5,
        )
        center = (int(mx), int(my))
        pygame.draw.circle(self._surface, color, center, max(2, int(radius)))
        ring = tuple(max(0, c - 42) for c in color)
        pygame.draw.circle(
            self._surface,
            ring,
            center,
            max(2, int(radius)),
            width=max(1, self._cell_size // 18),
        )
        highlight = tuple(min(255, c + 28) for c in color)
        pygame.draw.circle(
            self._surface,
            highlight,
            (int(mx - radius * 0.22), int(my - radius * 0.22)),
            max(1, int(radius * 0.22)),
        )

    def _draw_segment_pad(
        self,
        col: int,
        row: int,
        color: tuple[int, int, int],
        index: int,
        length: int,
        body: tuple[Position, ...],
        move_dx: int,
        move_dy: int,
    ) -> None:
        """Segment body with stubs only toward linked neighbors (plus head nose)."""
        cx, cy = self._cell_center(col, row)
        half_w = self._cell_size * self._segment_width_frac(index, length)
        stub_len = self._cell_size * config.SNAKE_STUB_FRAC
        core_r = max(2, int(half_w * 0.78))

        for direction in self._segment_neighbor_dirs(body, index):
            self._draw_segment_stub(cx, cy, direction, half_w, stub_len, color)

        if index == 0 and (move_dx, move_dy) != (0, 0):
            nose_len = self._cell_size * config.SNAKE_HEAD_NOSE_FRAC
            self._draw_segment_stub(cx, cy, (move_dx, move_dy), half_w, nose_len, color)

        pygame.draw.circle(self._surface, color, (int(cx), int(cy)), core_r)
        shadow = tuple(max(0, c - 34) for c in color)
        pygame.draw.circle(
            self._surface,
            shadow,
            (int(cx), int(cy)),
            core_r,
            width=max(1, self._cell_size // 16),
        )

    def _draw_segment_stub(
        self,
        cx: float,
        cy: float,
        direction: tuple[int, int],
        half_w: float,
        length: float,
        color: tuple[int, int, int],
    ) -> None:
        """Short connector from the segment core toward one neighbor."""
        dx, dy = direction
        span = math.hypot(dx, dy)
        if span == 0:
            return
        ux, uy = dx / span, dy / span
        px, py = -uy, ux
        tip_x = cx + ux * length
        tip_y = cy + uy * length
        points = [
            (int(cx + px * half_w), int(cy + py * half_w)),
            (int(tip_x + px * half_w), int(tip_y + py * half_w)),
            (int(tip_x), int(tip_y)),
            (int(tip_x - px * half_w), int(tip_y - py * half_w)),
            (int(cx - px * half_w), int(cy - py * half_w)),
        ]
        pygame.draw.polygon(self._surface, color, points)
        shadow = tuple(max(0, c - 34) for c in color)
        pygame.draw.polygon(
            self._surface,
            shadow,
            points,
            width=max(1, self._cell_size // 18),
        )
        pygame.draw.circle(
            self._surface,
            color,
            (int(tip_x), int(tip_y)),
            max(2, int(half_w * 0.55)),
        )

    def _draw_head_eyes(
        self,
        col: int,
        row: int,
        dx: int,
        dy: int,
        *,
        dead: bool = False,
    ) -> None:
        """Eyes on the leading face, looking along the movement direction."""
        if dx == 0 and dy == 0:
            return
        cx, cy = self._cell_center(col, row)
        span = math.hypot(dx, dy)
        ux, uy = dx / span, dy / span
        px, py = -uy, ux
        size = float(self._cell_size)
        forward = size * 0.16
        side = size * 0.13
        eye_r = max(2, self._cell_size // 8)
        pupil_r = max(1, eye_r // 2)
        pupil_fwd = max(1, eye_r // 3)
        for sign in (-1, 1):
            ex = cx + ux * forward + px * side * sign
            ey = cy + uy * forward + py * side * sign
            eye_pos = (int(ex), int(ey))
            pygame.draw.circle(self._surface, config.COLOR_SNAKE_EYE, eye_pos, eye_r)
            if dead:
                self._draw_eye_x(eye_pos, eye_r)
            else:
                pygame.draw.circle(
                    self._surface,
                    config.COLOR_SNAKE_PUPIL,
                    (int(ex + ux * pupil_fwd), int(ey + uy * pupil_fwd)),
                    pupil_r,
                )

    def _draw_eye_x(
        self,
        center: tuple[int, int],
        radius: int,
    ) -> None:
        """Dead-eye X mark over the white eye."""
        x, y = center
        half = max(2, radius - 1)
        width = max(2, self._cell_size // 14)
        pygame.draw.line(
            self._surface,
            config.COLOR_SNAKE_PUPIL,
            (x - half, y - half),
            (x + half, y + half),
            width,
        )
        pygame.draw.line(
            self._surface,
            config.COLOR_SNAKE_PUPIL,
            (x - half, y + half),
            (x + half, y - half),
            width,
        )

    def _draw_scale_chevron(
        self,
        col: int,
        row: int,
        dx: int,
        dy: int,
        base_color: tuple[int, int, int],
    ) -> None:
        """Directional scale chevron pointing along the body (head → tail)."""
        center = self._cell_center(col, row)
        points = self._chevron_points(center[0], center[1], dx, dy)
        if points is None:
            return
        fill = tuple(min(255, int(c + 42)) for c in base_color)
        edge = tuple(max(0, int(c - 50)) for c in base_color)
        pygame.draw.polygon(self._surface, fill, points)
        pygame.draw.polygon(
            self._surface,
            edge,
            points,
            width=max(1, self._cell_size // 16),
        )

    def _chevron_points(
        self,
        cx: float,
        cy: float,
        dx: int,
        dy: int,
    ) -> list[tuple[int, int]] | None:
        if dx == 0 and dy == 0:
            return None
        length = math.hypot(dx, dy)
        ux, uy = dx / length, dy / length
        px, py = -uy, ux
        size = float(self._cell_size)
        tip = (cx + ux * size * 0.30, cy + uy * size * 0.30)
        base_x = cx - ux * size * 0.10
        base_y = cy - uy * size * 0.10
        half = size * 0.20
        left = (base_x + px * half, base_y + py * half)
        right = (base_x - px * half, base_y - py * half)
        notch = (cx - ux * size * 0.22, cy - uy * size * 0.22)
        return [
            (int(tip[0]), int(tip[1])),
            (int(left[0]), int(left[1])),
            (int(notch[0]), int(notch[1])),
            (int(right[0]), int(right[1])),
        ]

    def _draw_win_fill(self) -> None:
        """Fill any remaining empty cells with tail styling (fallback after a win)."""
        body = self._game.snake.body
        occupied = set(body)
        length = len(body)
        if length == 0:
            return
        tail_index = length - 1
        tail_color = self._snake_segment_color(tail_index, length)
        for row in range(self.rows):
            for col in range(self.cols):
                pos = Position(col, row)
                if pos in occupied:
                    continue
                neighbor_dirs = self._win_fill_neighbor_dirs(pos, occupied)
                if not neighbor_dirs:
                    continue
                self._draw_win_fill_pad(
                    col,
                    row,
                    tail_color,
                    tail_index,
                    length,
                    neighbor_dirs,
                )

    @staticmethod
    def _win_fill_neighbor_dirs(
        pos: Position,
        occupied: set[Position],
    ) -> list[tuple[int, int]]:
        dirs: list[tuple[int, int]] = []
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            if Position(pos.x + dx, pos.y + dy) in occupied:
                dirs.append((dx, dy))
        return dirs

    def _draw_win_fill_pad(
        self,
        col: int,
        row: int,
        color: tuple[int, int, int],
        index: int,
        length: int,
        neighbor_dirs: list[tuple[int, int]],
    ) -> None:
        cx, cy = self._cell_center(col, row)
        half_w = self._cell_size * self._segment_width_frac(index, length)
        stub_len = self._cell_size * config.SNAKE_STUB_FRAC
        core_r = max(2, int(half_w * 0.78))
        for direction in neighbor_dirs:
            self._draw_segment_stub(cx, cy, direction, half_w, stub_len, color)
        pygame.draw.circle(self._surface, color, (int(cx), int(cy)), core_r)

    def _draw_food(self) -> None:
        food = self._game.food.position
        self._draw_apple(food.x, food.y)

    def _draw_apple(self, col: int, row: int) -> None:
        """Red apple with shading, stem, and leaf."""
        cx, cy = self._cell_center(col, row)
        size = float(self._cell_size)
        body_w = size * 0.58
        body_h = size * 0.64
        body_rect = pygame.Rect(0, 0, int(body_w), int(body_h))
        body_rect.center = (int(cx), int(cy + size * 0.05))

        shadow_rect = body_rect.copy()
        shadow_rect.move_ip(int(size * 0.03), int(size * 0.04))
        pygame.draw.ellipse(self._surface, config.COLOR_APPLE_SHADOW, shadow_rect)

        pygame.draw.ellipse(self._surface, config.COLOR_FOOD, body_rect)
        pygame.draw.ellipse(
            self._surface,
            config.COLOR_APPLE_OUTLINE,
            body_rect,
            width=max(1, int(size // 28)),
        )

        highlight_rect = pygame.Rect(0, 0, int(body_w * 0.36), int(body_h * 0.30))
        highlight_rect.center = (
            int(cx - body_w * 0.16),
            int(cy - body_h * 0.04 + size * 0.05),
        )
        pygame.draw.ellipse(self._surface, config.COLOR_APPLE_HIGHLIGHT, highlight_rect)

        dimple_r = max(2, int(size * 0.06))
        dimple_y = int(body_rect.top + body_h * 0.10)
        pygame.draw.circle(
            self._surface,
            config.COLOR_APPLE_SHADOW,
            (int(cx), dimple_y),
            dimple_r,
        )

        stem_w = max(2, int(size * 0.07))
        stem_top = (int(cx), int(body_rect.top - size * 0.10))
        stem_base = (int(cx), int(body_rect.top + size * 0.06))
        pygame.draw.line(
            self._surface,
            config.COLOR_APPLE_STEM,
            stem_base,
            stem_top,
            width=stem_w,
        )

        leaf_tip = (
            int(cx + size * 0.18),
            int(body_rect.top - size * 0.02),
        )
        leaf_mid = (
            int(cx + size * 0.06),
            int(body_rect.top - size * 0.10),
        )
        leaf_base = (
            int(cx - size * 0.02),
            int(body_rect.top + size * 0.02),
        )
        pygame.draw.polygon(
            self._surface,
            config.COLOR_APPLE_LEAF,
            [leaf_base, leaf_mid, leaf_tip],
        )
        leaf_vein = tuple(max(0, c - 30) for c in config.COLOR_APPLE_LEAF)
        pygame.draw.line(
            self._surface,
            leaf_vein,
            leaf_base,
            leaf_tip,
            max(1, int(size // 24)),
        )

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

        title, subtitle, title_color = self._death_message()
        center_x = self._offset_x + self.board_width // 2
        center_y = self._offset_y + self.board_height // 2
        game_over = self._overlay_font.render(title, True, title_color)
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
