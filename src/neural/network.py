"""
Neural network mapped to/from a Genome.

Supports feedforward MLP (default) or GRU memory. MLP: grid inputs -> hidden layers
-> 4 direction logits. GRU: grid -> GRU -> 4 logits with per-game hidden state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config
from evolution.genome import Genome


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


@dataclass(frozen=True, slots=True)
class ForwardResult:
    """Activations produced by one forward pass (used for UI and replay recording)."""

    inputs: np.ndarray
    hidden_layers: tuple[np.ndarray, ...]
    outputs: np.ndarray
    rnn_hidden: np.ndarray


@dataclass(frozen=True, slots=True)
class _MLPLayer:
    W: np.ndarray
    b: np.ndarray


@dataclass(frozen=True, slots=True)
class _GRUWeights:
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
    """Policy network (MLP or GRU) whose weights are stored in a flat Genome."""

    def __init__(
        self,
        *,
        mlp_layers: tuple[_MLPLayer, ...] | None = None,
        gru: _GRUWeights | None = None,
    ) -> None:
        if (mlp_layers is None) == (gru is None):
            raise ValueError("Specify exactly one of mlp_layers or gru.")
        self._mlp_layers = mlp_layers
        self._gru = gru

    @classmethod
    def is_recurrent(cls) -> bool:
        return config.NN_ARCH == "gru"

    @classmethod
    def layer_sizes(cls) -> tuple[int, ...]:
        """Full layer width sequence including input and output."""
        if cls.is_recurrent():
            return (config.NN_INPUT_SIZE, config.NN_RNN_HIDDEN, config.NN_OUTPUT_SIZE)
        return (config.NN_INPUT_SIZE, *config.NN_HIDDEN_SIZES, config.NN_OUTPUT_SIZE)

    @classmethod
    def architecture_label(cls) -> str:
        """Human-readable architecture string for training logs."""
        if cls.is_recurrent():
            return f"{config.NN_INPUT_SIZE}->GRU({config.NN_RNN_HIDDEN})->{config.NN_OUTPUT_SIZE}"
        hidden = "->".join(str(size) for size in config.NN_HIDDEN_SIZES)
        return f"{config.NN_INPUT_SIZE}->{hidden}->{config.NN_OUTPUT_SIZE}"

    @classmethod
    def architecture(cls) -> tuple[int, ...]:
        """Architecture tuple stored in checkpoints for compatibility checks."""
        return cls.layer_sizes()

    @classmethod
    def rnn_hidden_size(cls) -> int:
        return config.NN_RNN_HIDDEN if cls.is_recurrent() else 0

    @classmethod
    def new_hidden_state(cls) -> np.ndarray:
        """Recurrent state for a new game (zeros for MLP)."""
        size = cls.rnn_hidden_size()
        return np.zeros(size, dtype=np.float64)

    @classmethod
    def genome_length(cls) -> int:
        if cls.is_recurrent():
            return cls._gru_genome_length()
        return cls._mlp_genome_length()

    @classmethod
    def _mlp_genome_length(cls) -> int:
        return cls.genome_length_for_sizes(cls.layer_sizes())

    @staticmethod
    def genome_length_for_sizes(sizes: tuple[int, ...]) -> int:
        total = 0
        for index in range(len(sizes) - 1):
            in_dim, out_dim = sizes[index], sizes[index + 1]
            total += in_dim * out_dim + out_dim
        return total

    @classmethod
    def _gru_genome_length(cls) -> int:
        input_size = config.NN_INPUT_SIZE
        hidden = config.NN_RNN_HIDDEN
        output_size = config.NN_OUTPUT_SIZE
        gate = input_size * hidden + hidden * hidden + hidden
        return 3 * gate + hidden * output_size + output_size

    @classmethod
    def from_genome(
        cls,
        genome: Genome,
        *,
        layer_sizes: tuple[int, ...] | None = None,
    ) -> "NeuralNetwork":
        sizes = layer_sizes or cls.layer_sizes()
        expected = cls.genome_length_for_sizes(sizes)
        if genome.genes.size != expected:
            raise ValueError(
                f"Expected genome length {expected} for architecture {sizes}, "
                f"got {genome.genes.size}."
            )
        if cls.is_recurrent():
            return cls._from_genome_gru(genome)
        return cls._from_genome_mlp(genome, sizes)

    @classmethod
    def _from_genome_mlp(cls, genome: Genome, sizes: tuple[int, ...]) -> "NeuralNetwork":
        genes = genome.genes
        idx = 0
        layers: list[_MLPLayer] = []
        for layer_index in range(len(sizes) - 1):
            in_dim, out_dim = sizes[layer_index], sizes[layer_index + 1]
            weight_count = in_dim * out_dim
            w = genes[idx : idx + weight_count].reshape(in_dim, out_dim)
            idx += weight_count
            b = genes[idx : idx + out_dim]
            idx += out_dim
            layers.append(_MLPLayer(W=w, b=b))
        return cls(mlp_layers=tuple(layers))

    @classmethod
    def _from_genome_gru(cls, genome: Genome) -> "NeuralNetwork":
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
        return cls(
            gru=_GRUWeights(
                Wz=Wz, Uz=Uz, bz=bz, Wr=Wr, Ur=Ur, br=br, Wh=Wh, Uh=Uh, bh=bh, Wo=Wo, bo=bo
            )
        )

    def to_genome(self) -> Genome:
        if self._mlp_layers is not None:
            chunks: list[np.ndarray] = []
            for layer in self._mlp_layers:
                chunks.append(layer.W.reshape(-1))
                chunks.append(layer.b)
            return Genome(np.concatenate(chunks))
        g = self._gru
        assert g is not None
        return Genome(
            np.concatenate(
                [
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
            )
        )

    @classmethod
    def random_genome(cls) -> Genome:
        if cls.is_recurrent():
            return cls._random_genome_gru()
        return cls._random_genome_mlp()

    @classmethod
    def _random_genome_mlp(cls) -> Genome:
        sizes = cls.layer_sizes()
        chunks: list[np.ndarray] = []
        for index in range(len(sizes) - 1):
            in_dim, out_dim = sizes[index], sizes[index + 1]
            std = np.sqrt(2.0 / in_dim)
            chunks.append(np.random.normal(0.0, std, size=in_dim * out_dim))
            chunks.append(np.zeros(out_dim))
        return Genome(np.concatenate(chunks))

    @classmethod
    def _random_genome_gru(cls) -> Genome:
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
        self, inputs: np.ndarray, hidden: np.ndarray | None = None
    ) -> tuple[ForwardResult, np.ndarray]:
        if self._mlp_layers is not None:
            return self._forward_mlp(inputs)
        return self._forward_gru(inputs, hidden if hidden is not None else self.new_hidden_state())

    def _forward_mlp(self, inputs: np.ndarray) -> tuple[ForwardResult, np.ndarray]:
        x = np.asarray(inputs, dtype=np.float64)
        activations: list[np.ndarray] = []
        assert self._mlp_layers is not None
        for layer_index, layer in enumerate(self._mlp_layers):
            x = x @ layer.W + layer.b
            if layer_index < len(self._mlp_layers) - 1:
                x = _relu(x)
                activations.append(x.copy())
        outputs = x
        empty_hidden = self.new_hidden_state()
        return (
            ForwardResult(
                inputs=np.asarray(inputs, dtype=np.float64),
                hidden_layers=tuple(activations),
                outputs=outputs,
                rnn_hidden=empty_hidden,
            ),
            empty_hidden,
        )

    def _forward_gru(
        self, inputs: np.ndarray, hidden: np.ndarray
    ) -> tuple[ForwardResult, np.ndarray]:
        x = np.asarray(inputs, dtype=np.float64)
        h = np.asarray(hidden, dtype=np.float64)
        g = self._gru
        assert g is not None

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
