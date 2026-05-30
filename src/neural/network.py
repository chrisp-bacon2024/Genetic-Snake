"""
Feedforward neural network mapped to/from a Genome.

Topology is configurable via config.NN_INPUT_SIZE, config.NN_HIDDEN_SIZES (a tuple
of hidden-layer widths), and config.NN_OUTPUT_SIZE. Every hidden layer is ReLU; the
output layer is linear (raw direction logits). Default: 32 -> 20 -> 12 -> 4.
"""

from dataclasses import dataclass

import numpy as np

import config
from evolution.genome import Genome


@dataclass(frozen=True, slots=True)
class ForwardResult:
    """Activations produced by one forward pass (used for UI and replay recording)."""

    inputs: np.ndarray
    hidden_layers: tuple[np.ndarray, ...]
    outputs: np.ndarray


class NeuralNetwork:
    """
    Multi-layer perceptron whose weights are stored in a flat Genome.

    Layers are stored as a list of (W, b) pairs where W has shape (fan_in, fan_out).
    The genome packs them in order: W0, b0, W1, b1, ... flattened row-major.
    """

    def __init__(self, layers: list[tuple[np.ndarray, np.ndarray]]) -> None:
        self._layers = layers

    @classmethod
    def architecture(cls) -> tuple[int, ...]:
        """Full layer-size sequence: (input, *hidden, output)."""
        return (config.NN_INPUT_SIZE, *config.NN_HIDDEN_SIZES, config.NN_OUTPUT_SIZE)

    @classmethod
    def genome_length(cls) -> int:
        """Number of floats required in a Genome for this architecture."""
        sizes = cls.architecture()
        total = 0
        for fan_in, fan_out in zip(sizes, sizes[1:]):
            total += fan_in * fan_out + fan_out
        return total

    @classmethod
    def from_genome(cls, genome: Genome) -> "NeuralNetwork":
        """Unpack a flat gene array into per-layer weight matrices and bias vectors."""
        expected = cls.genome_length()
        if genome.genes.size != expected:
            raise ValueError(f"Expected genome length {expected}, got {genome.genes.size}.")

        sizes = cls.architecture()
        genes = genome.genes
        idx = 0
        layers: list[tuple[np.ndarray, np.ndarray]] = []
        for fan_in, fan_out in zip(sizes, sizes[1:]):
            w_size = fan_in * fan_out
            w = genes[idx : idx + w_size].reshape(fan_in, fan_out)
            idx += w_size
            b = genes[idx : idx + fan_out]
            idx += fan_out
            layers.append((w, b))
        return cls(layers)

    def to_genome(self) -> Genome:
        """Serialize current weights back into a Genome (same order as from_genome)."""
        chunks: list[np.ndarray] = []
        for w, b in self._layers:
            chunks.append(w.reshape(-1))
            chunks.append(b)
        return Genome(np.concatenate(chunks))

    @classmethod
    def random_genome(cls) -> Genome:
        """
        He-normal initialized genome (good defaults for ReLU layers).

        Each weight ~ N(0, sqrt(2/fan_in)); biases start at zero. This keeps initial
        activations well-scaled instead of the flat uniform range, which gives the GA
        a much healthier starting population.
        """
        sizes = cls.architecture()
        chunks: list[np.ndarray] = []
        for fan_in, fan_out in zip(sizes, sizes[1:]):
            std = np.sqrt(2.0 / fan_in)
            chunks.append(np.random.normal(0.0, std, size=fan_in * fan_out))
            chunks.append(np.zeros(fan_out))
        return Genome(np.concatenate(chunks))

    def forward(self, inputs: np.ndarray) -> ForwardResult:
        """
        Run one forward pass.

        Each hidden layer: ReLU(x @ W + b). Output layer is linear (raw logits).
        """
        x = np.asarray(inputs, dtype=np.float64)
        activation = x
        hidden_layers: list[np.ndarray] = []
        last = len(self._layers) - 1
        for i, (w, b) in enumerate(self._layers):
            z = activation @ w + b
            if i < last:
                activation = np.maximum(z, 0.0)
                hidden_layers.append(activation)
            else:
                activation = z
        return ForwardResult(
            inputs=x,
            hidden_layers=tuple(hidden_layers),
            outputs=activation,
        )
