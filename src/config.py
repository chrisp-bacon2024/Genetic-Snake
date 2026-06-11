"""
Application-wide constants for layout, colors, simulation speed, and neural topology.

All tunable values live here so gameplay, UI, and network behavior can be adjusted
without hunting through individual modules. Import this module as `config` from `src/`.
"""

PANEL_WIDTH = 340
CELL_SIZE = 24
GRID_COLS = 20
GRID_ROWS = 20
TICKS_PER_SECOND = 10
RENDER_FPS = 60

# Starvation: ticks allowed without eating before the snake dies.
# Limit scales with snake length — longer bodies need more time to navigate safely.
STARVATION_BASE_STEPS = 100
STARVATION_STEPS_PER_SEGMENT = 20
# Cap scales with board area so long endgames on big grids are not cut off early.
STARVATION_MAX_STEPS = 50_000


def starvation_limit(body_length: int, grid_cols: int, grid_rows: int) -> int:
    """
    Max ticks without food before starvation.

    Scales with snake length and board size; capped high enough for full-board runs.
    """
    grid_floor = max(grid_cols, grid_rows) * 5
    area_floor = grid_cols * grid_rows * 8
    scaled = (
        max(STARVATION_BASE_STEPS, grid_floor, area_floor)
        + body_length * STARVATION_STEPS_PER_SEGMENT
    )
    return min(scaled, STARVATION_MAX_STEPS)


def max_win_score(grid_cols: int, grid_rows: int) -> int:
    """Apples needed to fill the board (one cell starts occupied by the head)."""
    return grid_cols * grid_rows - 1


# Max grid used for curriculum sizing (ray encoder is grid-size agnostic).
MAX_GRID_COLS = GRID_COLS
MAX_GRID_ROWS = GRID_ROWS

# Ray encoder: 8 heading-relative rays x [wall, food, body].
ENCODER_RAY_COUNT = 8
# Draw encoder vision rays on the playfield (head → wall/body/food).
VISION_RAYS_ENABLED = True


def nn_input_size() -> int:
    """Ray vision + food + direction + lookahead + space/body metrics."""
    return ENCODER_RAY_COUNT * 3 + 5 + 4 + 4 + 4 + 3


WINDOW_WIDTH = PANEL_WIDTH + GRID_COLS * CELL_SIZE
WINDOW_HEIGHT = GRID_ROWS * CELL_SIZE

# Neural network topology
# Inputs: 8 rays (24) + food (5) + head/tail dir (8) + lookahead (4) + space/body (3) = 44.
# Changing encoder or NN_ARCH requires retraining; saved genomes/replays are not compatible.
NN_INPUT_SIZE = nn_input_size()
# Network architecture: feedforward MLP (rays -> hidden -> 4) or GRU memory.
NN_ARCH = "mlp"
NN_RNN_HIDDEN = 48
# Mask illegal one-step moves (lookahead) before choosing a direction.
MASK_UNSAFE_MOVES = True
# Single hidden layer — rays + fitness-aligned space cues replace the full board.
NN_HIDDEN_SIZES = (64,)
NN_OUTPUT_SIZE = 4
NN_WEIGHT_INIT_RANGE = (-1.0, 1.0)
# Genes are clamped to this range after crossover/mutation so weights cannot explode.
NN_WEIGHT_CLIP_RANGE = (-1.0, 1.0)
RESTART_NEW_GENOME = False

# --- Genetic algorithm training -------------------------------------------
POPULATION_SIZE = 1000
# Top individuals copied unchanged into the next generation.
ELITE_COUNT = 5
# Parent selection: sample TOURNAMENT_SIZE individuals, keep the fittest.
TOURNAMENT_SIZE = 8
# Fraction of offspring created via SBX crossover (rest are clone+mutate asexual).
# Default 1.0 = always crossover; use --asexual CLI to disable.
CROSSOVER_RATE = 1.0
# SBX distribution index: larger = children closer to parents, smaller = wider spread.
SBX_ETA = 15.0
# Crossover parents are chosen by tournament from this top fraction of the population.
PARENT_POOL_FRACTION = 0.25
# Fraction of non-elite offspring cloned from the #1 genome (mutate after copy).
CHAMPION_ASEXUAL_FRACTION = 0.15
# Per-gene mutation probability. Lower when eval is noisy (random 1-game boards).
MUTATION_RATE = 0.05
# Mixed-scale Gaussian mutation: most mutated genes get a small nudge, a fraction
# get a large jump to escape local optima.
MUTATION_MAGNITUDE = 0.10  # small-step std (also default for mutate())
MUTATION_MAGNITUDE_LARGE = 0.40  # large-step std
LARGE_MUTATION_FRACTION = 0.10  # fraction of mutated genes that take a large jump
# Fitness (Chrispresso-inspired exponential reward):
# score>0: steps + (2^score + score^2.1 * FITNESS_SCORE_WEIGHT) - wander_penalty + shaping + space
# score==0: no raw step credit; stronger toward-food shaping; penalty for long scoreless runs
FITNESS_SCORE_WEIGHT = 500.0
# Bonus per cell of Manhattan distance closed toward food each step.
FITNESS_DISTANCE_SHAPING = 2.0
# Higher cap before the first apple so honing in can outrank idle survival.
FITNESS_SHAPING_CAP = 50.0
FITNESS_SHAPING_CAP_SCORELESS = 180.0
# Lump reward for eating the first apple; speed bonus if steps_to_first_eat is known.
FITNESS_FIRST_EAT_BONUS = 400.0
# Extra fitness when first apple is eaten within this many steps (linear tail-off above).
FITNESS_FIRST_EAT_BUDGET_STEPS = 80
FITNESS_FIRST_EAT_SPEED_WEIGHT = 8.0
# Small per-step penalty while score is still 0 (discourages endless wandering).
FITNESS_SCORELESS_STEP_PENALTY = 0.15
# Space bonus ramps in after the snake has started eating (avoids circling at score 0).
FITNESS_SPACE_WEIGHT = 200.0
FITNESS_SPACE_SCORE_RAMP = 3
# Encoder: food should be visible at similar strength to nearby walls in the UI / MLP.
ENCODER_FOOD_RAY_ALIGN_FLOOR = 0.45
ENCODER_FOOD_RAY_BOOST = 1.25
ENCODER_FOOD_DIRECTION_SCALE = 4.0
# Huge bonus for filling the board (score == cols*rows - 1).
FITNESS_WIN_BONUS = 1.0e12
# Screening games averaged per genome; top fraction gets SELECT_EVAL_RUNS re-runs.
EVAL_RUNS_PER_GENOME = 2
# When True (or train.py --single-shot), each snake plays one board; no refine pass.
TRAINING_SINGLE_SHOT_EVAL = False
# Top SELECT_TOP_FRACTION of the population (by screening fitness) are re-evaluated with
# this many boards; selection/elites use the averaged result.
SELECT_TOP_FRACTION = 0.25
SELECT_EVAL_RUNS = 3
# When True, every snake in a generation plays the same food seeds (fair comparison).
SHARED_EVAL_SEEDS = True
# Each snake gets a fresh random food seed every evaluation when SHARED_EVAL_SEEDS is False.
RANDOM_FOOD_EVAL = True
GENERATIONS = 200
# Step budget per headless eval game. None = run until wall/body/starvation.
# Set to an int (e.g. cols * rows * 4) to cap long games during training.
MAX_EVAL_STEPS = None

# Curriculum: train on smaller grids; advance when enough snakes win the current board.
CURRICULUM_ENABLED = True
CURRICULUM_STAGES = (
    (5, 5),
    (10, 10),
    (20, 20),
)
# Fraction of the population that must win (death_cause == "win") among the top
# Fraction of population that must win to advance a stage or stop on the final board.
CURRICULUM_ADVANCE_WIN_FRACTION = 0.10
# Minimum generations on a stage before win-rate advancement (avoids lucky early jumps).
CURRICULUM_MIN_GENS_PER_STAGE = 5

# Stop training early when enough snakes win the target board (final curriculum stage
# or fixed grid when curriculum is off). Uses the same win fraction as stage advancement.
TRAINING_STOP_ON_WIN = True
TRAINING_STOP_WIN_FRACTION = CURRICULUM_ADVANCE_WIN_FRACTION
TRAINING_STOP_MIN_GENS = CURRICULUM_MIN_GENS_PER_STAGE

# Output index -> Direction mapping order
OUTPUT_DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT")

# Colors (R, G, B)
COLOR_BACKGROUND = (18, 18, 24)
COLOR_PANEL = (24, 24, 32)
COLOR_GRID_LINE = (40, 40, 52)
# Snake body gradient: dark forest green at the head → lighter mint at the tail.
COLOR_SNAKE_HEAD = (38, 118, 72)
COLOR_SNAKE_TAIL = (96, 210, 132)
COLOR_SNAKE_EYE = (232, 248, 236)
COLOR_SNAKE_PUPIL = (16, 32, 24)
# Perpendicular half-width as a fraction of cell size (head widest → tail slimmest).
SNAKE_SEGMENT_WIDTH_HEAD_FRAC = 0.44
SNAKE_SEGMENT_WIDTH_TAIL_FRAC = 0.16
# Stub from segment center toward a neighbor only (leaves gap on other sides).
SNAKE_STUB_FRAC = 0.20
SNAKE_HEAD_NOSE_FRAC = 0.28
SNAKE_JOINT_RADIUS_SCALE = 1.05
COLOR_FOOD = (220, 58, 52)
COLOR_APPLE_SHADOW = (150, 28, 32)
COLOR_APPLE_HIGHLIGHT = (255, 170, 158)
COLOR_APPLE_STEM = (92, 58, 28)
COLOR_APPLE_LEAF = (56, 138, 62)
COLOR_APPLE_OUTLINE = (120, 24, 28)
COLOR_TEXT = (220, 220, 230)
COLOR_TEXT_DIM = (120, 120, 140)
COLOR_CONTROL_INACTIVE = (45, 45, 58)
COLOR_CONTROL_BORDER = (70, 70, 90)
COLOR_CONTROL_ACTIVE = (80, 180, 255)
COLOR_CONTROL_ACTIVE_GLOW = (120, 210, 255)
COLOR_GAME_OVER = (255, 100, 100)
COLOR_WIN = (72, 220, 118)
COLOR_NEURON_INACTIVE = (50, 50, 65)
COLOR_NEURON_ACTIVE = (100, 200, 255)
COLOR_NEURON_INPUT_WALL = (180, 120, 80)
COLOR_NEURON_INPUT_BODY = (80, 180, 120)
COLOR_NEURON_INPUT_FOOD = (240, 100, 100)

# Neural network panel layout (scaled to fill sidebar height)
NETWORK_VIZ_TOP = 12
NN_VIZ_LAYER_SPACING = 14
NN_VIZ_LABEL_TO_NODES_GAP = 8
NN_INPUT_NODE_RADIUS = 5
NN_INPUT_ROW_HEIGHT = 17
NN_INPUT_COL_GAP = 7
NN_HIDDEN_NODE_RADIUS = 7
NN_HIDDEN_NODE_GAP = 5
NN_OUTPUT_ARROW_SIZE = 54
NN_OUTPUT_ARROW_GAP = 12
NN_VIZ_TITLE_SIZE = 18
NN_VIZ_LAYER_LABEL_SIZE = 11
NN_VIZ_LEGEND_SIZE = 10
NN_VIZ_ARROW_GLYPH_SIZE = 20

OUTPUT_ARROW_SIZE = NN_OUTPUT_ARROW_SIZE
OUTPUT_ARROW_GAP = NN_OUTPUT_ARROW_GAP

HINT_LINE_HEIGHT = 18

# Replay files (saved manually or by GA when keeping best-of-epoch)
REPLAYS_DIR = "replays"
