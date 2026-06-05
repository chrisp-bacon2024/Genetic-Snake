"""
Neural-network controller: encodes game state, runs forward pass, picks direction.

Also produces NetworkSnapshot each tick for the UI panel and GameRecorder.
"""

from dataclasses import dataclass

import numpy as np

import config
from controllers.controller import Controller
from game.game import Game
from models.direction import Direction
from neural.encoder import GameStateEncoder
from neural.network import ForwardResult, NeuralNetwork
from neural.policy import decide_step, new_rnn_hidden


@dataclass(frozen=True, slots=True)
class NetworkSnapshot:
    """
    All neural activations and the chosen move for one decision step.

    Stored in replays and drawn by NetworkVisualizer.
    """

    inputs: np.ndarray
    hidden_layers: tuple[np.ndarray, ...]
    outputs: np.ndarray
    chosen_direction: Direction
    rnn_hidden: np.ndarray


class AIController(Controller):
    """
    Drives the snake from a NeuralNetwork (GRU).

    Output index mapping (must match config.OUTPUT_DIRECTIONS):
        0 → UP, 1 → DOWN, 2 → LEFT, 3 → RIGHT
    """

    def __init__(self, game: Game, network: NeuralNetwork) -> None:
        self._game = game
        self._network = network
        self._encoder = GameStateEncoder()
        self._rnn_hidden = new_rnn_hidden()
        self._last_direction = game.snake.direction
        self._last_snapshot = self._empty_snapshot()

    @property
    def last_snapshot(self) -> NetworkSnapshot:
        """Most recent forward pass (for rendering and replay frames)."""
        return self._last_snapshot

    @property
    def network(self) -> NeuralNetwork:
        return self._network

    def update(self, events: list) -> None:
        pass

    def get_direction(self) -> Direction | None:
        """
        Encode current game → GRU step → pick highest valid output.

        Updates last_snapshot. Returns last direction if game is already over.
        """
        if not self._game.alive:
            return self._last_direction

        direction, self._rnn_hidden, result = decide_step(
            self._network,
            self._game,
            self._encoder,
            self._rnn_hidden,
            self._game.snake.direction,
        )
        self._last_direction = direction
        self._last_snapshot = NetworkSnapshot(
            inputs=result.inputs,
            hidden_layers=result.hidden_layers,
            outputs=result.outputs,
            chosen_direction=direction,
            rnn_hidden=result.rnn_hidden,
        )
        return direction

    def get_active_direction(self) -> Direction | None:
        return self._last_direction

    def reset(self) -> None:
        self._rnn_hidden = new_rnn_hidden()
        self._last_direction = self._game.snake.direction
        self._last_snapshot = self._empty_snapshot()

    def _empty_snapshot(self) -> NetworkSnapshot:
        hidden_size = NeuralNetwork.rnn_hidden_size()
        return NetworkSnapshot(
            inputs=np.zeros(config.NN_INPUT_SIZE),
            hidden_layers=(np.zeros(hidden_size),),
            outputs=np.zeros(config.NN_OUTPUT_SIZE),
            chosen_direction=self._game.snake.direction,
            rnn_hidden=np.zeros(hidden_size),
        )
