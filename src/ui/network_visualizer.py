"""
Live diagram of the neural network: input rays, hidden layers, output arrows.

Drawn on the left sidebar. Input dot brightness reflects encoder features;
hidden/output brightness reflects activation strength.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pygame

import config
from controllers.ai_controller import NetworkSnapshot
from models.direction import Direction
from neural.encoder import GameStateEncoder
from ui.input_feature_color import feature_row_color, input_feature_color, lerp_color


@dataclass(frozen=True)
class _FeatureSection:
    key: str
    label: str
    color: tuple[int, int, int]


@dataclass(frozen=True)
class _LayoutMetrics:
    layer_spacing: int
    label_gap: int
    input_radius: int
    input_row_height: int
    input_col_gap: int
    ray_suffix: bool
    feature_sections: tuple[_FeatureSection, ...]
    hidden_radius: int
    hidden_gap: int
    arrow_size: int
    arrow_gap: int


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

    _FULL_FEATURES = (
        _FeatureSection("food", "Food", config.COLOR_NEURON_INPUT_FOOD),
        _FeatureSection("head", "Head dir", config.COLOR_CONTROL_ACTIVE),
        _FeatureSection("tail", "Tail dir", config.COLOR_CONTROL_ACTIVE),
        _FeatureSection("lookahead", "Lookahead", config.COLOR_NEURON_INPUT_FOOD),
        _FeatureSection("space", "Space", config.COLOR_NEURON_INPUT_BODY),
    )
    _COMPACT_FEATURES = (
        _FeatureSection("food", "Food", config.COLOR_NEURON_INPUT_FOOD),
        _FeatureSection("dirs", "Dirs", config.COLOR_CONTROL_ACTIVE),
    )

    def __init__(self, surface: pygame.Surface) -> None:
        self._surface = surface
        self._input_legend_font = pygame.font.SysFont("consolas", config.NN_VIZ_LEGEND_SIZE)
        self._title_font = pygame.font.SysFont("consolas", config.NN_VIZ_TITLE_SIZE, bold=True)
        self._layer_font = pygame.font.SysFont("consolas", config.NN_VIZ_LAYER_LABEL_SIZE)
        self._arrow_font = pygame.font.SysFont("consolas", config.NN_VIZ_ARROW_GLYPH_SIZE, bold=True)
        self._bottom_y = config.NETWORK_VIZ_TOP

        self._panel_margin = 12
        self._offsets = GameStateEncoder.feature_offsets()

    @property
    def bottom_y(self) -> int:
        return self._bottom_y

    def draw(self, snapshot: NetworkSnapshot, *, bottom_limit: int | None = None) -> None:
        """Lay out layers top-to-bottom, staying above ``bottom_limit`` when set."""
        limit = bottom_limit if bottom_limit is not None else config.WINDOW_HEIGHT - 8
        usable_width = config.PANEL_WIDTH - 2 * self._panel_margin
        metrics = self._metrics_for_limit(snapshot, limit, usable_width)
        self._bottom_y = self._draw_with_metrics(snapshot, metrics, limit, usable_width)

    def _layout_templates(self) -> tuple[_LayoutMetrics, ...]:
        return (
            _LayoutMetrics(
                layer_spacing=config.NN_VIZ_LAYER_SPACING,
                label_gap=config.NN_VIZ_LABEL_TO_NODES_GAP,
                input_radius=config.NN_INPUT_NODE_RADIUS,
                input_row_height=config.NN_INPUT_ROW_HEIGHT,
                input_col_gap=config.NN_INPUT_COL_GAP,
                ray_suffix=True,
                feature_sections=self._FULL_FEATURES,
                hidden_radius=4,
                hidden_gap=5,
                arrow_size=28,
                arrow_gap=6,
            ),
            _LayoutMetrics(
                layer_spacing=10,
                label_gap=6,
                input_radius=4,
                input_row_height=15,
                input_col_gap=6,
                ray_suffix=True,
                feature_sections=self._FULL_FEATURES,
                hidden_radius=4,
                hidden_gap=4,
                arrow_size=26,
                arrow_gap=5,
            ),
            _LayoutMetrics(
                layer_spacing=8,
                label_gap=5,
                input_radius=4,
                input_row_height=14,
                input_col_gap=5,
                ray_suffix=False,
                feature_sections=self._FULL_FEATURES,
                hidden_radius=3,
                hidden_gap=3,
                arrow_size=24,
                arrow_gap=4,
            ),
            _LayoutMetrics(
                layer_spacing=7,
                label_gap=4,
                input_radius=4,
                input_row_height=13,
                input_col_gap=5,
                ray_suffix=False,
                feature_sections=self._COMPACT_FEATURES,
                hidden_radius=3,
                hidden_gap=3,
                arrow_size=22,
                arrow_gap=4,
            ),
            _LayoutMetrics(
                layer_spacing=6,
                label_gap=4,
                input_radius=3,
                input_row_height=12,
                input_col_gap=4,
                ray_suffix=False,
                feature_sections=self._COMPACT_FEATURES,
                hidden_radius=3,
                hidden_gap=2,
                arrow_size=20,
                arrow_gap=3,
            ),
        )

    def _metrics_for_limit(
        self,
        snapshot: NetworkSnapshot,
        limit: int,
        usable_width: int,
    ) -> _LayoutMetrics:
        """Pick the richest layout that fits, scaled to fill ``limit`` vertically."""
        target = limit - 6
        templates = self._layout_templates()

        for template in templates:
            base_height = self._estimate_height(snapshot, template, usable_width)
            if base_height <= 0:
                continue
            scale = target / base_height
            if scale < 0.62:
                continue
            metrics = self._scale_metrics(template, scale, usable_width)
            return self._fit_metrics_to_target(snapshot, template, metrics, target, usable_width)

        fallback = templates[-1]
        return self._fit_metrics_to_target(
            snapshot,
            fallback,
            self._scale_metrics(fallback, 1.0, usable_width),
            target,
            usable_width,
        )

    def _fit_metrics_to_target(
        self,
        snapshot: NetworkSnapshot,
        template: _LayoutMetrics,
        metrics: _LayoutMetrics,
        target: int,
        usable_width: int,
    ) -> _LayoutMetrics:
        height = self._estimate_height(snapshot, metrics, usable_width)
        if height == target:
            return metrics

        base_height = self._estimate_height(snapshot, template, usable_width)
        if base_height <= 0:
            return metrics

        scale = target / base_height
        for _ in range(24):
            metrics = self._scale_metrics(template, scale, usable_width)
            height = self._estimate_height(snapshot, metrics, usable_width)
            if height == target:
                break
            if height > target:
                scale -= 0.008
            else:
                scale += 0.008
        return metrics

    def _scale_metrics(
        self,
        metrics: _LayoutMetrics,
        factor: float,
        usable_width: int,
    ) -> _LayoutMetrics:
        def scaled(value: float, minimum: int) -> int:
            return max(minimum, round(value * factor))

        arrow_gap = scaled(metrics.arrow_gap, 3)
        arrow_size = min(
            scaled(metrics.arrow_size, 16),
            self._max_arrow_size(arrow_gap, usable_width),
        )
        return replace(
            metrics,
            layer_spacing=scaled(metrics.layer_spacing, 4),
            label_gap=scaled(metrics.label_gap, 3),
            input_radius=scaled(metrics.input_radius, 3),
            input_row_height=scaled(metrics.input_row_height, 10),
            input_col_gap=scaled(metrics.input_col_gap, 3),
            hidden_radius=scaled(metrics.hidden_radius, 2),
            hidden_gap=scaled(metrics.hidden_gap, 2),
            arrow_size=arrow_size,
            arrow_gap=arrow_gap,
        )

    @staticmethod
    def _max_arrow_size(gap: int, usable_width: int) -> int:
        # Cross layout width: left arm + gap + center + gap + right arm.
        return max(16, (usable_width - 2 * gap) // 3)

    def _estimate_height(
        self,
        snapshot: NetworkSnapshot,
        metrics: _LayoutMetrics,
        usable_width: int,
    ) -> int:
        y = config.NETWORK_VIZ_TOP
        y += self._title_font.get_height() + metrics.layer_spacing

        ray_suffix_rows = 10 if metrics.ray_suffix else 0
        y += self._layer_font.get_height() + ray_suffix_rows + metrics.label_gap
        y += metrics.input_row_height * 2 + metrics.input_radius * 2 + metrics.layer_spacing

        for section in metrics.feature_sections:
            _, count = self._section_slice(section.key)
            y += self._layer_font.get_height() + metrics.label_gap
            y += metrics.input_radius * 2 + metrics.layer_spacing

        for hidden in snapshot.hidden_layers:
            y += self._layer_font.get_height() + metrics.label_gap
            y += self._hidden_block_height(len(hidden), usable_width, metrics)
            y += metrics.layer_spacing

        cluster_height = metrics.arrow_size * 2 + metrics.arrow_gap
        y += self._layer_font.get_height() + metrics.label_gap + cluster_height + 6
        return y

    def _draw_with_metrics(
        self,
        snapshot: NetworkSnapshot,
        metrics: _LayoutMetrics,
        limit: int,
        usable_width: int,
    ) -> int:
        y = config.NETWORK_VIZ_TOP

        title = self._title_font.render("Neural Net", True, config.COLOR_TEXT)
        title_rect = title.get_rect(centerx=config.PANEL_WIDTH // 2, top=y)
        self._surface.blit(title, title_rect)
        y = title_rect.bottom + metrics.layer_spacing

        suffix = "wall / food / body" if metrics.ray_suffix else None
        label_bottom = self._draw_layer_label(
            f"Rays ({config.ENCODER_RAY_COUNT})", y, suffix=suffix
        )
        input_top = label_bottom + metrics.label_gap
        input_center_y = input_top + metrics.input_radius
        self._draw_input_legend(input_center_y, metrics.input_row_height)
        self._draw_input_rays(snapshot.inputs, input_center_y, metrics)
        y = input_top + metrics.input_row_height * 2 + metrics.input_radius * 2 + metrics.layer_spacing

        for section in metrics.feature_sections:
            start, count = self._section_slice(section.key)
            label_bottom = self._draw_layer_label(f"{section.label} ({count})", y)
            row_top = label_bottom + metrics.label_gap
            self._draw_feature_row(
                snapshot.inputs,
                start_index=start,
                count=count,
                top=row_top,
                width=usable_width,
                radius=metrics.input_radius,
                col_gap=metrics.input_col_gap,
                base_color=section.color,
            )
            y = row_top + metrics.input_radius * 2 + metrics.layer_spacing

        for layer_index, hidden in enumerate(snapshot.hidden_layers):
            if config.NN_ARCH == "gru" and len(snapshot.hidden_layers) == 1:
                name = f"Memory ({hidden.shape[0]})"
            else:
                name = f"Hidden {layer_index + 1} ({hidden.shape[0]})"
            label_bottom = self._draw_layer_label(name, y)
            hidden_top = label_bottom + metrics.label_gap
            hidden_height = self._draw_memory_neurons(
                hidden,
                top=hidden_top,
                width=usable_width,
                radius=metrics.hidden_radius,
                gap=metrics.hidden_gap,
            )
            y = hidden_top + hidden_height + metrics.layer_spacing

        cluster_height = metrics.arrow_size * 2 + metrics.arrow_gap
        label_bottom = self._draw_layer_label("Output", y)
        output_top = label_bottom + metrics.label_gap
        self._draw_output_arrows(
            snapshot.outputs,
            snapshot.chosen_direction,
            output_top,
            arrow_size=metrics.arrow_size,
            arrow_gap=metrics.arrow_gap,
        )
        return output_top + cluster_height + 6

    def _section_slice(self, key: str) -> tuple[int, int]:
        if key == "dirs":
            head_start, head_count = self._offsets["head"]
            _, tail_count = self._offsets["tail"]
            return head_start, head_count + tail_count
        return self._offsets[key]

    @staticmethod
    def _hidden_block_height(count: int, width: int, metrics: _LayoutMetrics) -> int:
        pitch = metrics.hidden_radius * 2 + metrics.hidden_gap
        per_row = max(1, (width + metrics.hidden_gap) // pitch)
        rows = (count + per_row - 1) // per_row
        return rows * pitch

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

    def _input_grid_start_x(self, col_gap: int, radius: int) -> int:
        cols = config.ENCODER_RAY_COUNT
        total_width = cols * (radius * 2 + col_gap) - col_gap
        return (config.PANEL_WIDTH - total_width) // 2 + radius

    def _draw_input_rays(self, inputs: np.ndarray, first_row_center_y: int, metrics: _LayoutMetrics) -> None:
        start_x = self._input_grid_start_x(metrics.input_col_gap, metrics.input_radius)
        for ray in range(config.ENCODER_RAY_COUNT):
            x = start_x + ray * (metrics.input_radius * 2 + metrics.input_col_gap)
            for row in range(3):
                index = ray * 3 + row
                value = float(inputs[index]) if index < len(inputs) else 0.0
                color = input_feature_color(row, value)
                node_y = first_row_center_y + row * metrics.input_row_height
                pygame.draw.circle(self._surface, color, (x, node_y), metrics.input_radius)

    def _draw_input_legend(self, first_row_center_y: int, row_height: int) -> None:
        first_column_left = self._input_grid_start_x(config.NN_INPUT_COL_GAP, config.NN_INPUT_NODE_RADIUS) - config.NN_INPUT_NODE_RADIUS
        label_right = first_column_left - self._INPUT_LEGEND_GAP

        header = self._input_legend_font.render(
            self._INPUT_LEGEND_HEADER, True, config.COLOR_TEXT_DIM
        )
        header_y = first_row_center_y - row_height // 2 - 4
        header_rect = header.get_rect(right=label_right, bottom=header_y)
        self._surface.blit(header, header_rect)

        for i, label in enumerate(self._INPUT_LABELS):
            text = self._input_legend_font.render(label, True, config.COLOR_TEXT_DIM)
            text_rect = text.get_rect(
                right=label_right, centery=first_row_center_y + i * row_height
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
        col_gap: int,
        base_color: tuple[int, int, int],
    ) -> None:
        pitch = radius * 2 + col_gap
        row_width = count * pitch - col_gap
        start_x = self._panel_margin + (width - row_width) // 2 + radius
        center_y = top + radius

        for i in range(count):
            index = start_index + i
            value = float(inputs[index]) if index < len(inputs) else 0.0
            color = feature_row_color(base_color, value)
            x = start_x + i * pitch
            pygame.draw.circle(self._surface, color, (x, center_y), radius)

    def _draw_memory_neurons(
        self,
        hidden: np.ndarray,
        *,
        top: int,
        width: int,
        radius: int,
        gap: int,
    ) -> int:
        """Wrap hidden units across multiple rows so they fit the panel."""
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
            border_color = lerp_color(
                config.COLOR_CONTROL_BORDER,
                config.COLOR_CONTROL_ACTIVE,
                activation,
            )
            pygame.draw.rect(self._surface, fill_color, rect, border_radius=corner_radius)
            pygame.draw.rect(self._surface, border_color, rect, width=2, border_radius=corner_radius)
            label_color = lerp_color(config.COLOR_TEXT_DIM, config.COLOR_TEXT, activation)

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

    def _activation_color(self, value: float) -> tuple[int, int, int]:
        normalized = min(max(value, 0.0), 1.0)
        return lerp_color(config.COLOR_NEURON_INACTIVE, config.COLOR_NEURON_ACTIVE, normalized)
