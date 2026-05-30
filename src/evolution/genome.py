"""
Genome: a serializable array of neural-network weights.

Each individual in a genetic population is one Genome. Genetic operators live here:
SBX crossover (sbx_pair), mixed-scale Gaussian mutation (mutate), and clip.
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

    def sbx_pair(self, other: "Genome", eta: float | None = None) -> tuple["Genome", "Genome"]:
        """
        Simulated Binary Crossover: produce two children from two parents.

        SBX is the standard real-coded crossover. For each gene it samples a spread
        factor beta from the eta-controlled distribution and reflects the two parent
        values around their mean. eta large -> children near parents; small -> wider.
        """
        if eta is None:
            eta = config.SBX_ETA
        p1 = self.genes
        p2 = other.genes
        u = np.random.random(p1.shape)
        beta = np.where(
            u <= 0.5,
            (2.0 * u) ** (1.0 / (eta + 1.0)),
            (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0)),
        )
        child1 = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2)
        child2 = 0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2)
        return Genome(child1), Genome(child2)

    def mutate(self, rate: float | None = None, magnitude: float | None = None) -> None:
        """
        Mixed-scale Gaussian mutation, applied in place.

        Each gene mutates with probability `rate`. Of the genes that mutate, a
        LARGE_MUTATION_FRACTION get a large-std jump (escape local optima) and the
        rest get a small-std nudge (preserve learned behavior).
        """
        if rate is None:
            rate = config.MUTATION_RATE
        small_std = config.MUTATION_MAGNITUDE if magnitude is None else magnitude
        large_std = config.MUTATION_MAGNITUDE_LARGE

        mutate_mask = np.random.random(self.genes.shape) < rate
        if not mutate_mask.any():
            return

        large_mask = mutate_mask & (np.random.random(self.genes.shape) < config.LARGE_MUTATION_FRACTION)
        small_mask = mutate_mask & ~large_mask

        self.genes[small_mask] += np.random.normal(0.0, small_std, size=int(small_mask.sum()))
        self.genes[large_mask] += np.random.normal(0.0, large_std, size=int(large_mask.sum()))

    def clip(self, low: float | None = None, high: float | None = None) -> None:
        """Clamp all genes into the weight range so values cannot explode."""
        if low is None or high is None:
            low, high = config.NN_WEIGHT_CLIP_RANGE
        np.clip(self.genes, low, high, out=self.genes)
