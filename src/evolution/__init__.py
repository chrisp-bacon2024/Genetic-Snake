"""Genetic-algorithm primitives: genomes, fitness, and population evolution."""

from .fitness import compute_fitness
from .genome import Genome
from .population import Individual, Population

__all__ = ["Genome", "Individual", "Population", "compute_fitness"]
