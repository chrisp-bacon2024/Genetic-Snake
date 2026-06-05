"""
Fitness function for genetic-algorithm training.

Chrispresso-inspired exponential reward: surviving (steps) earns a little, eating
apples earns exponentially more (so 5 apples is worth far more than 5x one apple),
and a penalty discourages aimless wandering once the snake has started eating.

Optional distance shaping rewards moving toward food before the first apple.
"""

import config


def compute_fitness(score: int, steps: int, shaping_bonus: float = 0.0) -> float:
    """
    fitness = steps
            + (2^score + score^2.1 * FITNESS_SCORE_WEIGHT)
            - ((0.25 * steps)^1.3 * score^1.2)
            + capped distance-shaping bonus

    Clamped to a small positive floor so it is usable as a roulette weight.
    """
    score = max(0, int(score))
    steps = max(0, int(steps))
    score_reward = (2.0**score) + (score**2.1) * config.FITNESS_SCORE_WEIGHT
    step_penalty = ((0.25 * steps) ** 1.3) * (score**1.2)
    capped_shaping = min(float(shaping_bonus), config.FITNESS_SHAPING_CAP)
    fitness = float(steps) + score_reward - step_penalty + capped_shaping
    return max(fitness, 0.1)
