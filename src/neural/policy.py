"""
Shared decision step for headless training and the pygame controller.

Keeps GRU state handling and direction selection in one place.
"""

from __future__ import annotations

import numpy as np

from game.game import Game
from models.direction import Direction
from neural.encoder import GameStateEncoder
from neural.network import ForwardResult, NeuralNetwork

_OUTPUT_TO_DIRECTION = {
    0: Direction.UP,
    1: Direction.DOWN,
    2: Direction.LEFT,
    3: Direction.RIGHT,
}


def new_rnn_hidden() -> np.ndarray:
    return NeuralNetwork.new_hidden_state()


def direction_from_outputs(outputs: np.ndarray, current: Direction) -> Direction:
    """Argmax over logits, skipping any 180-degree reversal."""
    ranked = np.argsort(outputs)[::-1]
    for index in ranked:
        direction = _OUTPUT_TO_DIRECTION[int(index)]
        if direction != current.opposite():
            return direction
    return current


def decide_step(
    network: NeuralNetwork,
    game: Game,
    encoder: GameStateEncoder,
    hidden: np.ndarray,
    current_direction: Direction,
) -> tuple[Direction, np.ndarray, ForwardResult]:
    """Encode, forward one GRU step, pick a valid direction."""
    inputs = encoder.encode(game)
    result, hidden_new = network.forward(inputs, hidden)
    direction = direction_from_outputs(result.outputs, current_direction)
    return direction, hidden_new, result
