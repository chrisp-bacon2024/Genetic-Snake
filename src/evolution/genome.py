"""
Genome: a serializable array of neural-network weights.

Each individual in a future genetic population will be one Genome. Crossover and
mutation methods will be added here when GA training is implemented.
"""

import numpy as np

import config


class Genome:
    """One-dimensional float array representing all network weights and biases."""

    def __init__(self, genes: np.ndarray) -> None:
        self.genes = np.asarray(genes, dtype=np.float64)

    @classmethod
    def random(cls, length: int) -> "Genome":
        """Create a genome with uniform random weights in NN_WEIGHT_INIT_RANGE."""
        low, high = config.NN_WEIGHT_INIT_RANGE
        genes = np.random.uniform(low, high, size=length)
        return cls(genes)

    def copy(self) -> "Genome":
        """Deep copy for elitism or branching individuals."""
        return Genome(self.genes.copy())

    # Future: crossover(parent_b) -> Genome
    # Future: mutate(rate, magnitude) -> None
