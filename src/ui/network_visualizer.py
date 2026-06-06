"""
Live diagram of the neural network: input rays, hidden layers, output arrows.

Drawn on the left sidebar. Input dot brightness reflects encoder features;
hidden/output brightness reflects activation strength.
"""

from __future__ import annotations

import pygame
import numpy as np

import config
from controllers.ai_controller import NetworkSnapshot
from models.direction import Direction
from neural.encoder import GameStateEncoder


class NetworkVisualizer:
    """Renders the Input -> Hidden -> Output panel from a NetworkSnapshot."""

    _INPUT_LEGEND_HEADER = "Proximity to"
    _INPUT_LABELS = ("Wall", "Food", "Body")
    _INPUT_LEGEND_GAP = 6
    _OUTPUT_DIRECTIONS = (
        Direction.UP,
        Direction.DOWN,
        Direction.LEFT,
        Direction.RIGHT,
    )

    def __init__(self, surface: pygame.Surface) -> None:
        self._surface = surface
        self._input_legend_font = pygame.font.SysFont("consolas", config.NN_VIZ_LEGEND_SIZE)
        self._title_font = pygame.font.SysFont("consolas", config.NN_VIZ_TITLE_SIZE, bold=True)
        self._layer_font = pygame.font.SysFont("consolas", config.NN_VIZ_LAYER_LABEL_SIZE)
        self._arrow_font = pygame.font.SysFont("consolas", config.NN_VIZ_ARROW_GLYPH_SIZE, bold=True)
        self._bottom_y = config.NETWORK_VIZ_TOP

        self._layer_spacing = config.NN_VIZ_LAYER_SPACING
        self._label_gap = config.NN_VIZ_LABEL_TO_NODES_GAP
        self._panel_margin = 12
        self._input_radius = config.NN_INPUT_NODE_RADIUS
        self._input_row_height = config.NN_INPUT_ROW_HEIGHT
        self._input_col_gap = config.NN_INPUT_COL_GAP
        self._offsets = GameStateEncoder.feature_offsets()

    @property
    def bottom_y(self) -> int:
        return self._bottom_y

    def draw(self, snapshot: NetworkSnapshot, *, bottom_limit: int | None = None) -> None:
        """Lay out layers top-to-bottom, staying above ``bottom_limit`` when set."""
        y = config.NETWORK_VIZ_TOP
        limit = bottom_limit if bottom_limit is not None else config.WINDOW_HEIGHT - 8
        usable_width = config.PANEL_WIDTH - 2 * self._panel_margin

        title = self._title_font.render("Neural Net", True, config.COLOR_TEXT)
        title_rect = title.get_rect(centerx=config.PANEL_WIDTH // 2, top=y)
        self._surface.blit(title, title_rect)
        y = title_rect.bottom + self._layer_spacing

        label_bottom = self._draw_layer_label(
            f"Rays ({config.ENCODER_RAY_COUNT})", y, suffix="wall / food / body"
        )
        input_top = label_bottom + self._label_gap
        input_center_y = input_top + self._input_radius
        self._draw_input_legend(input_center_y)
        self._draw_input_rays(snapshot.inputs, input_center_y)
        y = input_top + self._input_row_height * 2 + self._input_radius * 2 + self._layer_spacing

        for section, label, color in (
            ("food", "Food", config.COLOR_NEURON_INPUT_FOOD),
            ("head", "Head dir", config.COLOR_CONTROL_ACTIVE),
            ("tail", "Tail dir", config.COLOR_CONTROL_ACTIVE),
            ("lookahead", "Lookahead", config.COLOR_NEURON_INPUT_FOOD),
            ("space", "Space", config.COLOR_NEURON_INPUT_BODY),
        ):
            start, count = self._offsets[section]
            label_bottom = self._draw_layer_label(f"{label} ({count})", y)
            row_top = label_bottom + self._label_gap
            self._draw_feature_row(
                snapshot.inputs,
                start_index=start,
                count=count,
                top=row_top,
                width=usable_width,
                radius=self._input_radius,
                base_color=color,
            )
            y = row_top + self._input_radius * 2 + self._layer_spacing

        for layer_index, hidden in enumerate(snapshot.hidden_layers):
            if config.NN_ARCH == "gru" and len(snapshot.hidden_layers) == 1:
                name = f"Memory ({hidden.shape[0]})"
            else:
                name = f"Hidden {layer_index + 1} ({hidden.shape[0]})"
            label_bottom = self._draw_layer_label(name, y)
            hidden_top = label_bottom + self._label_gap
            hidden_height = self._draw_memory_neurons(
                hidden,
                top=hidden_top,
                width=usable_width,
            )
            y = hidden_top + hidden_height + self._layer_spacing

        arrow_size = 28
        arrow_gap = 6
        cluster_height = arrow_size * 2 + arrow_gap
        if y + cluster_height + 8 > limit:
            y = max(config.NETWORK_VIZ_TOP, limit - cluster_height - 8)

        label_bottom = self._draw_layer_label("Output", y)
        output_top = label_bottom + self._label_gap
        self._draw_output_arrows(
            snapshot.outputs,
            snapshot.chosen_direction,
            output_top,
            arrow_size=arrow_size,
            arrow_gap=arrow_gap,
        )
        self._bottom_y = output_top + cluster_height + 6

    def _draw_layer_label(self, text: str, top: int, *, suffix: str | None = None) -> int:
        label = self._layer_font.render(text, True, config.COLOR_TEXT_DIM)
        label_rect = label.get_rect(centerx=config.PANEL_WIDTH // 2, top=top)
        self._surface.blit(label, label_rect)
        if suffix:
            hint_font = pygame.font.SysFont("consolas", 9)
            hint = hint_font.render(suffix, True, config.COLOR_TEXT_DIM)
            hint_rect = hint.get_rect(centerx=config.PANEL_WIDTH // 2, top=label_rect.bottom + 1)
            self._surface.blit(hint, hint_rect)
            return hint_rect.bottom
        return label_rect.bottom

    def _input_grid_start_x(self) -> int:
        cols = config.ENCODER_RAY_COUNT
        total_width = cols * (self._input_radius * 2 + self._input_col_gap) - self._input_col_gap
        return (config.PANEL_WIDTH - total_width) // 2 + self._input_radius

    def _draw_input_rays(self, inputs: np.ndarray, first_row_center_y: int) -> None:
        start_x = self._input_grid_start_x()
        for ray in range(config.ENCODER_RAY_COUNT):
            x = start_x + ray * (self._input_radius * 2 + self._input_col_gap)
            for row in range(3):
                index = ray * 3 + row
                value = float(inputs[index]) if index < len(inputs) else 0.0
                color = self._input_color(row, value)
                node_y = first_row_center_y + row * self._input_row_height
                pygame.draw.circle(self._surface, color, (x, node_y), self._input_radius)

    def _draw_input_legend(self, first_row_center_y: int) -> None:
        first_column_left = self._input_grid_start_x() - self._input_radius
        label_right = first_column_left - self._INPUT_LEGEND_GAP

        header = self._input_legend_font.render(
            self._INPUT_LEGEND_HEADER, True, config.COLOR_TEXT_DIM
        )
        header_y = first_row_center_y - self._input_row_height // 2 - 4
        header_rect = header.get_rect(right=label_right, bottom=header_y)
        self._surface.blit(header, header_rect)

        for i, label in enumerate(self._INPUT_LABELS):
            text = self._input_legend_font.render(label, True, config.COLOR_TEXT_DIM)
            text_rect = text.get_rect(
                right=label_right, centery=first_row_center_y + i * self._input_row_height
            )
            self._surface.blit(text, text_rect)

    def _draw_feature_row(
        self,
        inputs: np.ndarray,
        *,
        start_index: int,
        count: int,
        top: int,
        width: int,
        radius: int,
        base_color: tuple[int, int, int],
    ) -> None:
        gap = self._input_col_gap
        pitch = radius * 2 + gap
        row_width = count * pitch - gap
        start_x = self._panel_margin + (width - row_width) // 2 + radius
        center_y = top + radius

        for i in range(count):
            index = start_index + i
            value = float(inputs[index]) if index < len(inputs) else 0.0
            color = self._lerp_color(config.COLOR_NEURON_INACTIVE, base_color, value)
            x = start_x + i * pitch
            pygame.draw.circle(self._surface, color, (x, center_y), radius)

    def _draw_memory_neurons(
        self,
        hidden: np.ndarray,
        *,
        top: int,
        width: int,
    ) -> int:
        """Wrap GRU hidden units across multiple rows so they fit the panel."""
        radius = 4
        gap = 5
        pitch = radius * 2 + gap
        per_row = max(1, (width + gap) // pitch)
        rows = (len(hidden) + per_row - 1) // per_row
        row_height = pitch

        for i, value in enumerate(hidden):
            row = i // per_row
            col = i % per_row
            count_in_row = min(per_row, len(hidden) - row * per_row)
            row_width = count_in_row * pitch - gap
            start_x = self._panel_margin + (width - row_width) // 2 + radius
            cx = start_x + col * pitch
            cy = top + row * row_height + radius
            color = self._activation_color(float(value))
            pygame.draw.circle(self._surface, color, (cx, cy), radius)

        return rows * row_height

    def _draw_output_arrows(
        self,
        outputs: np.ndarray,
        chosen: Direction,
        nodes_top: int,
        *,
        arrow_size: int,
        arrow_gap: int = 6,
    ) -> None:
        layout = self._output_arrow_layout(nodes_top, arrow_size, arrow_gap)
        max_output = float(np.max(outputs)) if len(outputs) else 1.0
        if max_output <= 0:
            max_output = 1.0

        for i, direction in enumerate(self._OUTPUT_DIRECTIONS):
            value = float(outputs[i]) if i < len(outputs) else 0.0
            normalized = max(0.0, value / max_output)
            is_chosen = direction == chosen
            self._draw_arrow_button(
                layout[direction], direction, normalized, is_chosen, arrow_size=arrow_size
            )

    def _output_arrow_layout(
        self, cluster_top: int, size: int, gap: int
    ) -> dict[Direction, pygame.Rect]:
        center_x = config.PANEL_WIDTH // 2

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

    def _draw_arrow_button(
        self,
        rect: pygame.Rect,
        direction: Direction,
        activation: float,
        is_chosen: bool,
        *,
        arrow_size: int = 28,
    ) -> None:
        glyph = self._direction_glyph(direction)
        corner_radius = max(4, rect.width // 7)
        glyph_size = max(12, arrow_size // 2)
        glyph_font = pygame.font.SysFont("consolas", glyph_size, bold=True)

        if is_chosen:
            glow_rect = rect.inflate(4, 4)
            pygame.draw.rect(
                self._surface,
                config.COLOR_CONTROL_ACTIVE_GLOW,
                glow_rect,
                border_radius=corner_radius + 2,
            )
            fill_color = config.COLOR_CONTROL_ACTIVE
            label_color = config.COLOR_BACKGROUND
        else:
            fill_color = self._activation_color(activation)
            border_color = self._lerp_color(
                config.COLOR_CONTROL_BORDER,
                config.COLOR_CONTROL_ACTIVE,
                activation,
            )
            pygame.draw.rect(self._surface, fill_color, rect, border_radius=corner_radius)
            pygame.draw.rect(self._surface, border_color, rect, width=2, border_radius=corner_radius)
            label_color = self._lerp_color(config.COLOR_TEXT_DIM, config.COLOR_TEXT, activation)

        if is_chosen:
            pygame.draw.rect(self._surface, fill_color, rect, border_radius=corner_radius)

        label = glyph_font.render(glyph, True, label_color)
        self._surface.blit(label, label.get_rect(center=rect.center))

    @staticmethod
    def _direction_glyph(direction: Direction) -> str:
        return {
            Direction.UP: "^",
            Direction.DOWN: "v",
            Direction.LEFT: "<",
            Direction.RIGHT: ">",
        }[direction]

    @staticmethod
    def _input_color(feature_row: int, value: float) -> tuple[int, int, int]:
        base_colors = (
            config.COLOR_NEURON_INPUT_WALL,
            config.COLOR_NEURON_INPUT_FOOD,
            config.COLOR_NEURON_INPUT_BODY,
        )
        low = config.COLOR_NEURON_INACTIVE
        high = base_colors[feature_row]
        t = min(max(value, 0.0), 1.0)
        return tuple(int(low[i] + (high[i] - low[i]) * t) for i in range(3))

    def _activation_color(self, value: float) -> tuple[int, int, int]:
        normalized = min(max(value, 0.0), 1.0)
        return self._lerp_color(config.COLOR_NEURON_INACTIVE, config.COLOR_NEURON_ACTIVE, normalized)

    @staticmethod
    def _lerp_color(
        low: tuple[int, int, int],
        high: tuple[int, int, int],
        t: float,
    ) -> tuple[int, int, int]:
        t = min(max(t, 0.0), 1.0)
        return tuple(int(low[i] + (high[i] - low[i]) * t) for i in range(3))
