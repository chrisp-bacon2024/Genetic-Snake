"""
Population of genomes and the per-generation genetic operators.

Reproduction model (elitism + tournament + SBX + mixed-scale mutation):
  1. Rank the current generation by fitness.
  2. Carry the top ELITE_COUNT genomes unchanged (their fitness is re-measured next
     generation, so a lucky elite cannot coast on a stale score).
  3. Fill the rest by selecting parents via tournament, breeding children with SBX
     (or cloning), mutating, and clipping.

Fitness is assigned externally each generation by the trainer (every individual,
elites included, is re-evaluated on the current scenario set).
"""

import random
from dataclasses import dataclass

import config
from evolution.genome import Genome
from game.game_state import DeathCause
from neural.network import NeuralNetwork


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

    def best(self) -> Individual:
        return max(self._individuals, key=lambda ind: ind.fitness)

    def average_fitness(self) -> float:
        if not self._individuals:
            return 0.0
        return sum(ind.fitness for ind in self._individuals) / len(self._individuals)

    def evolve_next_generation(self) -> "Population":
        """Produce the next generation (elites + bred offspring)."""
        ranked = self.sorted_by_fitness()
        elite_count = min(config.ELITE_COUNT, len(ranked))

        next_gen: list[Individual] = [
            Individual(genome=elite.genome.copy()) for elite in ranked[:elite_count]
        ]

        low, high = config.NN_WEIGHT_CLIP_RANGE
        while len(next_gen) < self.size:
            if random.random() < config.CROSSOVER_RATE and len(ranked) >= 2:
                parent_a = self._tournament_pick(ranked)
                parent_b = self._tournament_pick(ranked)
                child_a, child_b = parent_a.genome.sbx_pair(parent_b.genome)
                children = [child_a, child_b]
            else:
                parent = self._tournament_pick(ranked)
                children = [parent.genome.copy()]

            for child in children:
                if len(next_gen) >= self.size:
                    break
                child.mutate()
                child.clip(low, high)
                next_gen.append(Individual(genome=child))

        return Population(next_gen)

    def _tournament_pick(self, ranked: list[Individual]) -> Individual:
        """Sample TOURNAMENT_SIZE individuals and return the fittest."""
        k = min(config.TOURNAMENT_SIZE, len(ranked))
        contenders = random.sample(ranked, k)
        return max(contenders, key=lambda ind: ind.fitness)
