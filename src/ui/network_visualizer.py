import pygame
import numpy as np

import config
from controllers.ai_controller import NetworkSnapshot
from models.direction import Direction


class NetworkVisualizer:
    _INPUT_LABELS = ("W", "B", "F")
    _OUTPUT_LABELS = ("U", "D", "L", "R")
    _LABEL_TO_NODES_GAP = 6
    _LAYER_SPACING = 8
    _INPUT_RADIUS = 4
    _INPUT_ROW_HEIGHT = 12
    _INPUT_LAYER_HEIGHT = 28
    _HIDDEN_RADIUS = 5
    _OUTPUT_NODE_RADIUS = 10

    def __init__(self, surface: pygame.Surface) -> None:
        self._surface = surface
        self._label_font = pygame.font.SysFont("consolas", 11)
        self._title_font = pygame.font.SysFont("consolas", 16, bold=True)
        self._layer_font = pygame.font.SysFont("consolas", 10)
        self._bottom_y = config.NETWORK_VIZ_TOP

    @property
    def bottom_y(self) -> int:
        return self._bottom_y

    def draw(self, snapshot: NetworkSnapshot) -> None:
        y = config.NETWORK_VIZ_TOP

        title = self._title_font.render("Neural Net", True, config.COLOR_TEXT)
        title_rect = title.get_rect(centerx=config.PANEL_WIDTH // 2, top=y)
        self._surface.blit(title, title_rect)
        y += title_rect.height + self._LAYER_SPACING

        label_bottom = self._draw_layer_label("Input (8 rays)", y)
        input_top = self._nodes_top(label_bottom)
        input_center_y = input_top + self._INPUT_RADIUS
        self._draw_input_legend(input_center_y)
        self._draw_input_layer(snapshot.inputs, input_center_y)
        y = input_top + self._INPUT_LAYER_HEIGHT + self._LAYER_SPACING

        label_bottom = self._draw_layer_label("Hidden", y)
        hidden_top = self._nodes_top(label_bottom)
        self._draw_hidden_layer(snapshot.hidden, hidden_top)
        y = hidden_top + self._HIDDEN_RADIUS * 2 + self._LAYER_SPACING

        label_bottom = self._draw_layer_label("Output", y)
        output_top = self._nodes_top(label_bottom)
        output_center_y = output_top + self._OUTPUT_NODE_RADIUS
        self._draw_output_layer(snapshot.outputs, snapshot.chosen_direction, output_center_y)

        self._bottom_y = output_top + self._OUTPUT_NODE_RADIUS * 2 + 18

    def _nodes_top(self, label_bottom: int) -> int:
        return label_bottom + self._LABEL_TO_NODES_GAP

    def _draw_layer_label(self, text: str, top: int) -> int:
        label = self._layer_font.render(text, True, config.COLOR_TEXT_DIM)
        label_rect = label.get_rect(centerx=config.PANEL_WIDTH // 2, top=top)
        self._surface.blit(label, label_rect)
        return label_rect.bottom

    def _draw_input_layer(self, inputs: np.ndarray, first_row_center_y: int) -> None:
        cols = 8
        rows = 3
        col_gap = 6
        radius = self._INPUT_RADIUS
        total_width = cols * (radius * 2 + col_gap) - col_gap
        start_x = (config.PANEL_WIDTH - total_width) // 2 + radius

        for ray in range(cols):
            x = start_x + ray * (radius * 2 + col_gap)
            for row in range(rows):
                index = ray * 3 + row
                value = float(inputs[index]) if index < len(inputs) else 0.0
                color = self._input_color(row, value)
                node_y = first_row_center_y + row * self._INPUT_ROW_HEIGHT
                pygame.draw.circle(self._surface, color, (x, node_y), radius)

    def _draw_input_legend(self, first_row_center_y: int) -> None:
        for i, label in enumerate(self._INPUT_LABELS):
            text = self._label_font.render(label, True, config.COLOR_TEXT_DIM)
            text_rect = text.get_rect(left=10, centery=first_row_center_y + i * self._INPUT_ROW_HEIGHT)
            self._surface.blit(text, text_rect)

    def _draw_hidden_layer(self, hidden: np.ndarray, nodes_top: int) -> None:
        count = len(hidden)
        radius = self._HIDDEN_RADIUS
        gap = 4
        center_y = nodes_top + radius
        total_width = count * (radius * 2 + gap) - gap
        start_x = (config.PANEL_WIDTH - total_width) // 2 + radius

        for i, value in enumerate(hidden):
            x = start_x + i * (radius * 2 + gap)
            color = self._activation_color(float(value))
            pygame.draw.circle(self._surface, color, (x, center_y), radius)

    def _draw_output_layer(
        self,
        outputs: np.ndarray,
        chosen: Direction,
        center_y: int,
    ) -> None:
        directions = [
            Direction.UP,
            Direction.DOWN,
            Direction.LEFT,
            Direction.RIGHT,
        ]
        radius = self._OUTPUT_NODE_RADIUS
        gap = 16
        total_width = len(directions) * (radius * 2 + gap) - gap
        start_x = (config.PANEL_WIDTH - total_width) // 2 + radius

        max_output = float(np.max(outputs)) if len(outputs) else 1.0
        if max_output <= 0:
            max_output = 1.0

        for i, direction in enumerate(directions):
            x = start_x + i * (radius * 2 + gap)
            value = float(outputs[i]) if i < len(outputs) else 0.0
            normalized = max(0.0, value / max_output)
            is_chosen = direction == chosen

            if is_chosen:
                glow_rect = pygame.Rect(0, 0, radius * 2 + 8, radius * 2 + 8)
                glow_rect.center = (x, center_y)
                pygame.draw.rect(
                    self._surface,
                    config.COLOR_CONTROL_ACTIVE_GLOW,
                    glow_rect,
                    border_radius=radius + 4,
                )

            color = self._activation_color(normalized, bright=is_chosen)
            pygame.draw.circle(self._surface, color, (x, center_y), radius)

            label = self._label_font.render(self._OUTPUT_LABELS[i], True, config.COLOR_TEXT_DIM)
            label_rect = label.get_rect(centerx=x, top=center_y + radius + 4)
            self._surface.blit(label, label_rect)

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
