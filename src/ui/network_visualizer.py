"""
Live diagram of the neural network: input rays, hidden layer, output arrows.

Drawn on the left sidebar. Input dot brightness reflects normalized vision
distances; hidden/output brightness reflects activation strength. Output arrows
replace separate direction buttons — the chosen move glows bright blue.
"""

import pygame
import numpy as np

import config
from controllers.ai_controller import NetworkSnapshot
from models.direction import Direction


class NetworkVisualizer:
    """Renders the Input → Hidden → Output panel from a NetworkSnapshot."""

    _INPUT_LEGEND_HEADER = "Distance to"
    _INPUT_LABELS = ("Wall", "Body", "Food")
    _INPUT_LEGEND_GAP = 6
    _OUTPUT_DIRECTIONS = (
        Direction.UP,
        Direction.DOWN,
        Direction.LEFT,
        Direction.RIGHT,
    )

    def __init__(self, surface: pygame.Surface) -> None:
        self._surface = surface
        self._label_font = pygame.font.SysFont("consolas", 11)
        self._input_legend_font = pygame.font.SysFont("consolas", config.NN_VIZ_LEGEND_SIZE)
        self._title_font = pygame.font.SysFont("consolas", config.NN_VIZ_TITLE_SIZE, bold=True)
        self._layer_font = pygame.font.SysFont("consolas", config.NN_VIZ_LAYER_LABEL_SIZE)
        self._arrow_font = pygame.font.SysFont("consolas", config.NN_VIZ_ARROW_GLYPH_SIZE, bold=True)
        self._bottom_y = config.NETWORK_VIZ_TOP

        self._layer_spacing = config.NN_VIZ_LAYER_SPACING
        self._label_to_nodes_gap = config.NN_VIZ_LABEL_TO_NODES_GAP
        self._input_radius = config.NN_INPUT_NODE_RADIUS
        self._input_row_height = config.NN_INPUT_ROW_HEIGHT
        self._input_col_gap = config.NN_INPUT_COL_GAP
        self._input_layer_height = self._input_row_height * 2 + self._input_radius * 2
        self._hidden_radius = config.NN_HIDDEN_NODE_RADIUS
        self._hidden_gap = config.NN_HIDDEN_NODE_GAP

    @property
    def bottom_y(self) -> int:
        return self._bottom_y

    def draw(self, snapshot: NetworkSnapshot) -> None:
        y = config.NETWORK_VIZ_TOP

        title = self._title_font.render("Neural Net", True, config.COLOR_TEXT)
        title_rect = title.get_rect(centerx=config.PANEL_WIDTH // 2, top=y)
        self._surface.blit(title, title_rect)
        y += title_rect.height + self._layer_spacing

        label_bottom = self._draw_layer_label("Input (8 rays)", y)
        input_top = self._nodes_top(label_bottom)
        input_center_y = input_top + self._input_radius
        self._draw_input_legend(input_center_y)
        self._draw_input_layer(snapshot.inputs, input_center_y)
        y = input_top + self._input_layer_height + self._layer_spacing

        label_bottom = self._draw_layer_label("Hidden", y)
        hidden_top = self._nodes_top(label_bottom)
        self._draw_hidden_layer(snapshot.hidden, hidden_top)
        y = hidden_top + self._hidden_radius * 2 + self._layer_spacing

        label_bottom = self._draw_layer_label("Output", y)
        output_top = self._nodes_top(label_bottom)
        self._draw_output_arrows(snapshot.outputs, snapshot.chosen_direction, output_top)

        size = config.NN_OUTPUT_ARROW_SIZE
        gap = config.NN_OUTPUT_ARROW_GAP
        cluster_height = size + gap + size
        self._bottom_y = output_top + cluster_height + 10

    def _nodes_top(self, label_bottom: int) -> int:
        return label_bottom + self._label_to_nodes_gap

    def _draw_layer_label(self, text: str, top: int) -> int:
        label = self._layer_font.render(text, True, config.COLOR_TEXT_DIM)
        label_rect = label.get_rect(centerx=config.PANEL_WIDTH // 2, top=top)
        self._surface.blit(label, label_rect)
        return label_rect.bottom

    def _output_arrow_layout(self, cluster_top: int) -> dict[Direction, pygame.Rect]:
        size = config.NN_OUTPUT_ARROW_SIZE
        gap = config.NN_OUTPUT_ARROW_GAP
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

    def _draw_output_arrows(
        self,
        outputs: np.ndarray,
        chosen: Direction,
        nodes_top: int,
    ) -> None:
        layout = self._output_arrow_layout(nodes_top)
        max_output = float(np.max(outputs)) if len(outputs) else 1.0
        if max_output <= 0:
            max_output = 1.0

        for i, direction in enumerate(self._OUTPUT_DIRECTIONS):
            value = float(outputs[i]) if i < len(outputs) else 0.0
            normalized = max(0.0, value / max_output)
            is_chosen = direction == chosen
            self._draw_arrow_button(layout[direction], direction, normalized, is_chosen)

    def _draw_arrow_button(
        self,
        rect: pygame.Rect,
        direction: Direction,
        activation: float,
        is_chosen: bool,
    ) -> None:
        glyph = self._direction_glyph(direction)
        corner_radius = max(6, rect.width // 7)

        if is_chosen:
            glow_rect = rect.inflate(8, 8)
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

        label = self._arrow_font.render(glyph, True, label_color)
        self._surface.blit(label, label.get_rect(center=rect.center))

    @staticmethod
    def _direction_glyph(direction: Direction) -> str:
        return {
            Direction.UP: "^",
            Direction.DOWN: "v",
            Direction.LEFT: "<",
            Direction.RIGHT: ">",
        }[direction]

    def _input_grid_start_x(self) -> int:
        cols = 8
        radius = self._input_radius
        total_width = cols * (radius * 2 + self._input_col_gap) - self._input_col_gap
        return (config.PANEL_WIDTH - total_width) // 2 + radius

    def _draw_input_layer(self, inputs: np.ndarray, first_row_center_y: int) -> None:
        cols = 8
        rows = 3
        radius = self._input_radius
        start_x = self._input_grid_start_x()

        for ray in range(cols):
            x = start_x + ray * (radius * 2 + self._input_col_gap)
            for row in range(rows):
                index = ray * 3 + row
                value = float(inputs[index]) if index < len(inputs) else 0.0
                color = self._input_color(row, value)
                node_y = first_row_center_y + row * self._input_row_height
                pygame.draw.circle(self._surface, color, (x, node_y), radius)

    def _draw_input_legend(self, first_row_center_y: int) -> None:
        first_column_left = self._input_grid_start_x() - self._input_radius
        label_right = first_column_left - self._INPUT_LEGEND_GAP

        header = self._input_legend_font.render(self._INPUT_LEGEND_HEADER, True, config.COLOR_TEXT_DIM)
        header_y = first_row_center_y - self._input_row_height // 2 - 4
        header_rect = header.get_rect(right=label_right, bottom=header_y)
        self._surface.blit(header, header_rect)

        for i, label in enumerate(self._INPUT_LABELS):
            text = self._input_legend_font.render(label, True, config.COLOR_TEXT_DIM)
            text_rect = text.get_rect(right=label_right, centery=first_row_center_y + i * self._input_row_height)
            self._surface.blit(text, text_rect)

    def _draw_hidden_layer(self, hidden: np.ndarray, nodes_top: int) -> None:
        count = len(hidden)
        radius = self._hidden_radius
        gap = self._hidden_gap
        center_y = nodes_top + radius
        total_width = count * (radius * 2 + gap) - gap
        start_x = (config.PANEL_WIDTH - total_width) // 2 + radius

        for i, value in enumerate(hidden):
            x = start_x + i * (radius * 2 + gap)
            color = self._activation_color(float(value))
            pygame.draw.circle(self._surface, color, (x, center_y), radius)

    def _input_color(self, feature_row: int, value: float) -> tuple[int, int, int]:
        base_colors = (
            config.COLOR_NEURON_INPUT_WALL,
            config.COLOR_NEURON_INPUT_BODY,
            config.COLOR_NEURON_INPUT_FOOD,
        )
        return self._lerp_color(config.COLOR_NEURON_INACTIVE, base_colors[feature_row], value)

    def _activation_color(self, value: float, bright: bool = False) -> tuple[int, int, int]:
        target = config.COLOR_NEURON_ACTIVE if bright else config.COLOR_CONTROL_ACTIVE
        return self._lerp_color(config.COLOR_NEURON_INACTIVE, target, min(max(value, 0.0), 1.0))

    @staticmethod
    def _lerp_color(
        low: tuple[int, int, int],
        high: tuple[int, int, int],
        t: float,
    ) -> tuple[int, int, int]:
        t = min(max(t, 0.0), 1.0)
        return tuple(int(low[i] + (high[i] - low[i]) * t) for i in range(3))
