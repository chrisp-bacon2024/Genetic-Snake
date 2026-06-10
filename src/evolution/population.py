"""
Population of genomes and the per-generation genetic operators.

Reproduction model (elitism + tournament + SBX + mixed-scale mutation):
  1. Rank the current generation by fitness.
  2. Carry the top ELITE_COUNT genomes unchanged (their fitness is re-measured next
     generation, so a lucky elite cannot coast on a stale score).
  3. Fill the rest by selecting two parents via tournament from the top
     PARENT_POOL_FRACTION, breeding children with SBX crossover (default) or
     cloning a single parent (asexual mode), then mutating and clipping.
  4. With probability CHAMPION_ASEXUAL_FRACTION, clone the #1 genome instead
     (local search around the current champion when crossover is enabled).

Fitness is assigned externally each generation by the trainer (every individual,
elites included, is re-evaluated on the current scenario set).
"""

import random
from dataclasses import dataclass

import config
from evolution.genome import Genome
from game.game_state import DeathCause


@dataclass
class Individual:
    """A genome plus the metrics from its most recent evaluation."""

    genome: Genome
    fitness: float = 0.0
    score: int = 0
    steps: int = 0
    best_food_seed: int = 0
    death_cause: DeathCause = "wall"


class Population:
    """A fixed-size collection of Individuals."""

    def __init__(self, individuals: list[Individual]) -> None:
        self._individuals = individuals

    @classmethod
    def random(cls, size: int) -> "Population":
        """He-initialized starting population."""
        from neural.network import NeuralNetwork

        return cls([Individual(genome=NeuralNetwork.random_genome()) for _ in range(size)])

    @property
    def individuals(self) -> list[Individual]:
        return self._individuals

    @property
    def size(self) -> int:
        return len(self._individuals)

    def sorted_by_fitness(self) -> list[Individual]:
        """Individuals from fittest to least fit."""
        return sorted(self._individuals, key=lambda ind: ind.fitness, reverse=True)

    def average_fitness(self) -> float:
        if not self._individuals:
            return 0.0
        return sum(ind.fitness for ind in self._individuals) / len(self._individuals)

    def evolve_next_generation(self, *, crossover_rate: float | None = None) -> "Population":
        """Produce the next generation (elites + bred offspring)."""
        ranked = self.sorted_by_fitness()
        elite_count = min(config.ELITE_COUNT, len(ranked))
        rate = config.CROSSOVER_RATE if crossover_rate is None else crossover_rate
        parent_pool = self._parent_pool(ranked)

        next_gen: list[Individual] = [
            Individual(genome=elite.genome.copy()) for elite in ranked[:elite_count]
        ]

        low, high = config.NN_WEIGHT_CLIP_RANGE
        champion_fraction = config.CHAMPION_ASEXUAL_FRACTION if rate > 0.0 else 0.0
        champion = ranked[0].genome if ranked else None
        while len(next_gen) < self.size:
            if (
                champion is not None
                and champion_fraction > 0.0
                and random.random() < champion_fraction
            ):
                children = [champion.copy()]
            elif rate > 0.0 and random.random() < rate and len(parent_pool) >= 2:
                parent_a = self._tournament_pick(parent_pool)
                parent_b = self._tournament_pick(parent_pool)
                child_a, child_b = parent_a.genome.sbx_pair(parent_b.genome)
                children = [child_a, child_b]
            else:
                parent = self._tournament_pick(parent_pool if parent_pool else ranked)
                children = [parent.genome.copy()]

            for child in children:
                if len(next_gen) >= self.size:
                    break
                child.mutate()
                child.clip(low, high)
                next_gen.append(Individual(genome=child))

        return Population(next_gen)

    def _parent_pool(self, ranked: list[Individual]) -> list[Individual]:
        """Top fraction of the population eligible to reproduce."""
        pool_size = max(2, int(len(ranked) * config.PARENT_POOL_FRACTION))
        return ranked[:pool_size]

    def _tournament_pick(self, ranked: list[Individual]) -> Individual:
        """Sample TOURNAMENT_SIZE individuals and return the fittest."""
        k = min(config.TOURNAMENT_SIZE, len(ranked))
        contenders = random.sample(ranked, k)
        return max(contenders, key=lambda ind: ind.fitness)
