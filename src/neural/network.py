from dataclasses import dataclass

import numpy as np

import config
from evolution.genome import Genome


@dataclass(frozen=True, slots=True)
class ForwardResult:
    inputs: np.ndarray
    hidden: np.ndarray
    outputs: np.ndarray


class NeuralNetwork:
    """
    Feedforward network: 24 -> 16 (ReLU) -> 4

    Genome layout: [W1 flat (384), b1 (16), W2 flat (64), b2 (4)] = 468 genes
    """

    def __init__(
        self,
        w1: np.ndarray,
        b1: np.ndarray,
        w2: np.ndarray,
        b2: np.ndarray,
    ) -> None:
        self._w1 = w1
        self._b1 = b1
        self._w2 = w2
        self._b2 = b2

    @classmethod
    def genome_length(cls) -> int:
        input_size = config.NN_INPUT_SIZE
        hidden_size = config.NN_HIDDEN_SIZE
        output_size = config.NN_OUTPUT_SIZE
        return (
            input_size * hidden_size
            + hidden_size
            + hidden_size * output_size
            + output_size
        )

    @classmethod
    def from_genome(cls, genome: Genome) -> "NeuralNetwork":
        expected = cls.genome_length()
        if genome.genes.size != expected:
            raise ValueError(f"Expected genome length {expected}, got {genome.genes.size}.")

        input_size = config.NN_INPUT_SIZE
        hidden_size = config.NN_HIDDEN_SIZE
        output_size = config.NN_OUTPUT_SIZE

        idx = 0
        w1_size = input_size * hidden_size
        w1 = genome.genes[idx : idx + w1_size].reshape(input_size, hidden_size)
        idx += w1_size

        b1 = genome.genes[idx : idx + hidden_size]
        idx += hidden_size

        w2_size = hidden_size * output_size
        w2 = genome.genes[idx : idx + w2_size].reshape(hidden_size, output_size)
        idx += w2_size

        b2 = genome.genes[idx : idx + output_size]

        return cls(w1, b1, w2, b2)

    def to_genome(self) -> Genome:
        genes = np.concatenate(
            [
                self._w1.reshape(-1),
                self._b1,
                self._w2.reshape(-1),
                self._b2,
            ]
        )
        return Genome(genes)

    def forward(self, inputs: np.ndarray) -> ForwardResult:
        x = np.asarray(inputs, dtype=np.float64)
        hidden_raw = x @ self._w1 + self._b1
        hidden = np.maximum(hidden_raw, 0.0)
        outputs = hidden @ self._w2 + self._b2
        return ForwardResult(inputs=x, hidden=hidden, outputs=outputs)
