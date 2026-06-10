"""Population breeding operators."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import config  # noqa: E402
import numpy as np  # noqa: E402
from evolution.genome import Genome  # noqa: E402
from evolution.population import Individual, Population  # noqa: E402


def _individual(genes: list[float], fitness: float) -> Individual:
    return Individual(genome=Genome(np.asarray(genes, dtype=np.float64)), fitness=fitness)


class ChampionAsexualTests(unittest.TestCase):
    def test_champion_fraction_biases_offspring_toward_rank_one(self) -> None:
        pop = Population(
            [
                _individual([1.0, 0.0, 0.0], 100.0),
                _individual([0.0, 1.0, 0.0], 50.0),
                _individual([0.0, 0.0, 1.0], 10.0),
            ]
        )
        with mock.patch.object(config, "ELITE_COUNT", 0), mock.patch.object(
            config, "CHAMPION_ASEXUAL_FRACTION", 1.0
        ), mock.patch.object(config, "CROSSOVER_RATE", 1.0), mock.patch.object(
            config, "MUTATION_RATE", 0.0
        ):
            child_genes = [
                ind.genome.genes.copy()
                for ind in pop.evolve_next_generation(crossover_rate=1.0).individuals
            ]
        for genes in child_genes:
            self.assertTrue(np.allclose(genes, np.array([1.0, 0.0, 0.0])))


if __name__ == "__main__":
    unittest.main()
