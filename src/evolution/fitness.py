"""
Fitness function for genetic-algorithm training.

Eating apples dominates via exponential score terms. Before the first apple,
toward-food shaping and a first-eat bonus encourage direct play; raw survival
steps do not count until the snake has started scoring. Space bonus ramps in
after a few apples so late-game routing still matters.
"""

import config


def compute_fitness(
    score: int,
    steps: int,
    shaping_bonus: float = 0.0,
    *,
    won: bool = False,
    space_ratio: float = 0.0,
    steps_to_first_eat: int | None = None,
    grid_cols: int | None = None,
    grid_rows: int | None = None,
) -> float:
    """
  fitness = (steps if score > 0 else 0)
          + (2^score + score^2.1 * FITNESS_SCORE_WEIGHT)
          - wander_penalty (score > 0)
          - scoreless_step_penalty (score == 0)
          + capped distance-shaping (higher cap before first eat)
          + first_eat_bonus + speed bonus
          + phased space_ratio * FITNESS_SPACE_WEIGHT
          + FITNESS_WIN_BONUS on full-board win

    Clamped to a small positive floor so it is usable as a roulette weight.
    """
    score = max(0, int(score))
    steps = max(0, int(steps))
    score_reward = (2.0**score) + (score**2.1) * config.FITNESS_SCORE_WEIGHT
    step_penalty = ((0.25 * steps) ** 1.3) * (score**1.2) if score > 0 else 0.0
    scoreless_penalty = config.FITNESS_SCORELESS_STEP_PENALTY * steps if score == 0 else 0.0

    shaping_cap = (
        config.FITNESS_SHAPING_CAP_SCORELESS
        if score == 0
        else config.FITNESS_SHAPING_CAP
    )
    capped_shaping = min(float(shaping_bonus), shaping_cap)

    space_ramp = min(1.0, score / float(max(1, config.FITNESS_SPACE_SCORE_RAMP)))
    space_bonus = (
        max(0.0, min(1.0, float(space_ratio)))
        * config.FITNESS_SPACE_WEIGHT
        * space_ramp
    )

    survival_credit = float(steps) if score > 0 else 0.0
    fitness = (
        survival_credit
        + score_reward
        - step_penalty
        - scoreless_penalty
        + capped_shaping
        + space_bonus
    )

    if score >= 1:
        fitness += config.FITNESS_FIRST_EAT_BONUS
        if steps_to_first_eat is not None and steps_to_first_eat > 0:
            fitness += config.FITNESS_FIRST_EAT_SPEED_WEIGHT * max(
                0.0,
                float(config.FITNESS_FIRST_EAT_BUDGET_STEPS) - float(steps_to_first_eat),
            )

    if won:
        fitness += config.FITNESS_WIN_BONUS

    return max(fitness, 0.1)
