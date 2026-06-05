"""
Fitness function for genetic-algorithm training.

Chrispresso-inspired exponential reward: surviving (steps) earns a little, eating
apples earns exponentially more (so 5 apples is worth far more than 5x one apple),
and a penalty discourages aimless wandering once the snake has started eating.

Optional distance shaping rewards moving toward food before the first apple.
Space and win bonuses encourage tail-safe routing and full-board clears.
"""

import config


def compute_fitness(
    score: int,
    steps: int,
    shaping_bonus: float = 0.0,
    *,
    won: bool = False,
    space_ratio: float = 0.0,
    grid_cols: int | None = None,
    grid_rows: int | None = None,
) -> float:
    """
    fitness = steps
            + (2^score + score^2.1 * FITNESS_SCORE_WEIGHT)
            - ((0.25 * steps)^1.3 * score^1.2)
            + capped distance-shaping bonus
            + space_ratio * FITNESS_SPACE_WEIGHT
            + FITNESS_WIN_BONUS on full-board win

    Clamped to a small positive floor so it is usable as a roulette weight.
    """
    score = max(0, int(score))
    steps = max(0, int(steps))
    score_reward = (2.0**score) + (score**2.1) * config.FITNESS_SCORE_WEIGHT
    step_penalty = ((0.25 * steps) ** 1.3) * (score**1.2)
    capped_shaping = min(float(shaping_bonus), config.FITNESS_SHAPING_CAP)
    space_bonus = max(0.0, min(1.0, float(space_ratio))) * config.FITNESS_SPACE_WEIGHT
    fitness = float(steps) + score_reward - step_penalty + capped_shaping + space_bonus

    cols = grid_cols if grid_cols is not None else config.GRID_COLS
    rows = grid_rows if grid_rows is not None else config.GRID_ROWS
    if won:
        fitness += config.FITNESS_WIN_BONUS

    return max(fitness, 0.1)
