import numpy as np

import config


class Genome:
    """Flat gene array representing a neural network weight vector."""

    def __init__(self, genes: np.ndarray) -> None:
        self.genes = np.asarray(genes, dtype=np.float64)

    @classmethod
    def random(cls, length: int) -> "Genome":
        low, high = config.NN_WEIGHT_INIT_RANGE
        genes = np.random.uniform(low, high, size=length)
        return cls(genes)

    def copy(self) -> "Genome":
        return Genome(self.genes.copy())

    # Future: crossover(parent_b) -> Genome
    # Future: mutate(rate, magnitude) -> None
