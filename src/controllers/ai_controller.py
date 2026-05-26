from dataclasses import dataclass

import numpy as np

import config
from controllers.controller import Controller
from game.game import Game
from models.direction import Direction
from neural.encoder import GameStateEncoder
from neural.network import ForwardResult, NeuralNetwork


@dataclass(frozen=True, slots=True)
class NetworkSnapshot:
    inputs: np.ndarray
    hidden: np.ndarray
    outputs: np.ndarray
    chosen_direction: Direction


class AIController(Controller):
    _OUTPUT_TO_DIRECTION = {
        0: Direction.UP,
        1: Direction.DOWN,
        2: Direction.LEFT,
        3: Direction.RIGHT,
    }

    def __init__(self, game: Game, network: NeuralNetwork) -> None:
        self._game = game
        self._network = network
        self._encoder = GameStateEncoder()
        self._last_direction = game.snake.direction
        self._last_snapshot = self._empty_snapshot()

    @property
    def last_snapshot(self) -> NetworkSnapshot:
        return self._last_snapshot

    @property
    def network(self) -> NeuralNetwork:
        return self._network

    def update(self, events: list) -> None:
        pass

    def get_direction(self) -> Direction | None:
        if not self._game.alive:
            return self._last_direction

        inputs = self._encoder.encode(self._game)
        result = self._network.forward(inputs)
        direction = self._direction_from_outputs(result, self._game.snake.direction)
        self._last_direction = direction
        self._last_snapshot = NetworkSnapshot(
            inputs=result.inputs,
            hidden=result.hidden,
            outputs=result.outputs,
            chosen_direction=direction,
        )
        return direction

    def get_active_direction(self) -> Direction | None:
        return self._last_direction

    def reset(self) -> None:
        self._last_direction = self._game.snake.direction
        self._last_snapshot = self._empty_snapshot()

    def _direction_from_outputs(
        self,
        result: ForwardResult,
        current_direction: Direction,
    ) -> Direction:
        ranked = np.argsort(result.outputs)[::-1]
        for index in ranked:
            direction = self._OUTPUT_TO_DIRECTION[int(index)]
            if direction != current_direction.opposite():
                return direction
        return current_direction

    def _empty_snapshot(self) -> NetworkSnapshot:
        return NetworkSnapshot(
            inputs=np.zeros(config.NN_INPUT_SIZE),
            hidden=np.zeros(config.NN_HIDDEN_SIZE),
            outputs=np.zeros(config.NN_OUTPUT_SIZE),
            chosen_direction=self._game.snake.direction,
        )
