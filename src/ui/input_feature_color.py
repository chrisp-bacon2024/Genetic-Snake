"""Shared input-neuron coloring (network panel and playfield vision rays)."""

from __future__ import annotations

import config

_FEATURE_BASE_COLORS = (
    config.COLOR_NEURON_INPUT_WALL,
    config.COLOR_NEURON_INPUT_FOOD,
    config.COLOR_NEURON_INPUT_BODY,
)


def lerp_color(
    low: tuple[int, int, int],
    high: tuple[int, int, int],
    t: float,
) -> tuple[int, int, int]:
    t = min(max(t, 0.0), 1.0)
    return tuple(int(low[i] + (high[i] - low[i]) * t) for i in range(3))


def input_feature_color(feature_row: int, value: float) -> tuple[int, int, int]:
    """Match NetworkVisualizer ray rows: 0=wall, 1=food, 2=body."""
    high = _FEATURE_BASE_COLORS[feature_row]
    t = min(max(float(value), 0.0), 1.0)
    return lerp_color(config.COLOR_NEURON_INACTIVE, high, t)


def feature_row_color(base_color: tuple[int, int, int], value: float) -> tuple[int, int, int]:
    """Match NetworkVisualizer non-ray input rows (food cues, dirs, etc.)."""
    return lerp_color(config.COLOR_NEURON_INACTIVE, base_color, value)
