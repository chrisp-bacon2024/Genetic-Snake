"""
Neural network mapped to/from a Genome.

Default architecture (NN_ARCH=\"gru\"): grid inputs -> GRU(48) -> 4 linear outputs.
Recurrent hidden state must be reset at the start of each game and carried across
ticks within a game.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config
from evolution.genome import Genome


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


@dataclass(frozen=True, slots=True)
class ForwardResult:
    """Activations produced by one forward pass (used for UI and replay recording)."""

    inputs: np.ndarray
    hidden_layers: tuple[np.ndarray, ...]
    outputs: np.ndarray
    rnn_hidden: np.ndarray


@dataclass(frozen=True, slots=True)
class _GRUWeights:
    """Gate weights for one GRU cell plus the output projection."""

    Wz: np.ndarray
    Uz: np.ndarray
    bz: np.ndarray
    Wr: np.ndarray
    Ur: np.ndarray
    br: np.ndarray
    Wh: np.ndarray
    Uh: np.ndarray
    bh: np.ndarray
    Wo: np.ndarray
    bo: np.ndarray


class NeuralNetwork:
    """
    GRU policy network whose weights are stored in a flat Genome.

  Genome layout: Wz, Uz, bz, Wr, Ur, br, Wh, Uh, bh, Wo, bo (row-major flatten).
    """

    def __init__(self, gru: _GRUWeights) -> None:
        self._gru = gru

    @classmethod
    def architecture(cls) -> tuple[int, ...]:
        """(input_size, rnn_hidden, output_size) for checkpoint compatibility."""
        return (config.NN_INPUT_SIZE, config.NN_RNN_HIDDEN, config.NN_OUTPUT_SIZE)

    @classmethod
    def rnn_hidden_size(cls) -> int:
        return config.NN_RNN_HIDDEN

    @classmethod
    def new_hidden_state(cls) -> np.ndarray:
        """Zero recurrent state for a new game."""
        return np.zeros(cls.rnn_hidden_size(), dtype=np.float64)

    @classmethod
    def genome_length(cls) -> int:
        input_size = config.NN_INPUT_SIZE
        hidden = config.NN_RNN_HIDDEN
        output_size = config.NN_OUTPUT_SIZE
        gate = input_size * hidden + hidden * hidden + hidden
        return 3 * gate + hidden * output_size + output_size

    @classmethod
    def from_genome(cls, genome: Genome) -> "NeuralNetwork":
        expected = cls.genome_length()
        if genome.genes.size != expected:
            raise ValueError(f"Expected genome length {expected}, got {genome.genes.size}.")

        input_size = config.NN_INPUT_SIZE
        hidden = config.NN_RNN_HIDDEN
        output_size = config.NN_OUTPUT_SIZE
        genes = genome.genes
        idx = 0

        def read_gate() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            nonlocal idx
            w = genes[idx : idx + input_size * hidden].reshape(input_size, hidden)
            idx += input_size * hidden
            u = genes[idx : idx + hidden * hidden].reshape(hidden, hidden)
            idx += hidden * hidden
            b = genes[idx : idx + hidden]
            idx += hidden
            return w, u, b

        Wz, Uz, bz = read_gate()
        Wr, Ur, br = read_gate()
        Wh, Uh, bh = read_gate()
        Wo = genes[idx : idx + hidden * output_size].reshape(hidden, output_size)
        idx += hidden * output_size
        bo = genes[idx : idx + output_size]
        idx += output_size

        return cls(
            _GRUWeights(Wz=Wz, Uz=Uz, bz=bz, Wr=Wr, Ur=Ur, br=br, Wh=Wh, Uh=Uh, bh=bh, Wo=Wo, bo=bo)
        )

    def to_genome(self) -> Genome:
        g = self._gru
        chunks = [
            g.Wz.reshape(-1),
            g.Uz.reshape(-1),
            g.bz,
            g.Wr.reshape(-1),
            g.Ur.reshape(-1),
            g.br,
            g.Wh.reshape(-1),
            g.Uh.reshape(-1),
            g.bh,
            g.Wo.reshape(-1),
            g.bo,
        ]
        return Genome(np.concatenate(chunks))

    @classmethod
    def random_genome(cls) -> Genome:
        """He-style init; smaller std on recurrent U matrices."""
        input_size = config.NN_INPUT_SIZE
        hidden = config.NN_RNN_HIDDEN
        output_size = config.NN_OUTPUT_SIZE
        chunks: list[np.ndarray] = []

        for _ in range(3):
            std_in = np.sqrt(2.0 / input_size)
            std_rec = np.sqrt(2.0 / (input_size + hidden)) * 0.5
            chunks.append(np.random.normal(0.0, std_in, size=input_size * hidden))
            chunks.append(np.random.normal(0.0, std_rec, size=hidden * hidden))
            chunks.append(np.zeros(hidden))

        std_out = np.sqrt(2.0 / hidden)
        chunks.append(np.random.normal(0.0, std_out, size=hidden * output_size))
        chunks.append(np.zeros(output_size))
        return Genome(np.concatenate(chunks))

    def forward(
        self, inputs: np.ndarray, hidden: np.ndarray
    ) -> tuple[ForwardResult, np.ndarray]:
        """
        One GRU step: update hidden state and produce direction logits.

        Returns (ForwardResult, h_new). ``hidden`` must match ``rnn_hidden_size()``.
        """
        x = np.asarray(inputs, dtype=np.float64)
        h = np.asarray(hidden, dtype=np.float64)
        g = self._gru

        z = _sigmoid(x @ g.Wz + h @ g.Uz + g.bz)
        r = _sigmoid(x @ g.Wr + h @ g.Ur + g.br)
        h_tilde = np.tanh(x @ g.Wh + (r * h) @ g.Uh + g.bh)
        h_new = (1.0 - z) * h + z * h_tilde
        outputs = h_new @ g.Wo + g.bo

        return (
            ForwardResult(
                inputs=x,
                hidden_layers=(h_new.copy(),),
                outputs=outputs,
                rnn_hidden=h_new.copy(),
            ),
            h_new,
        )
