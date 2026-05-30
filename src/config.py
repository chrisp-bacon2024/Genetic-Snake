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
STARVATION_MAX_STEPS = 800


def starvation_limit(body_length: int, grid_cols: int, grid_rows: int) -> int:
    """
    Max ticks without food before starvation.

    Short snakes: ~120 steps (kills spin loops quickly).
    Length 10: ~300 steps. Length 20: ~500 steps @ 10 tps ≈ 50 s.
    """
    grid_floor = max(grid_cols, grid_rows) * 5
    scaled = max(STARVATION_BASE_STEPS, grid_floor) + body_length * STARVATION_STEPS_PER_SEGMENT
    return min(scaled, STARVATION_MAX_STEPS)

WINDOW_WIDTH = PANEL_WIDTH + GRID_COLS * CELL_SIZE
WINDOW_HEIGHT = GRID_ROWS * CELL_SIZE

# Neural network topology
# Inputs (32): 8 rays x [wall, food, body] inverse-distance (24)
#            + 4 one-hot head direction + 4 one-hot tail direction.
NN_INPUT_SIZE = 32
# Hidden layers (ReLU). Tuple so the architecture can grow/shrink in one place.
NN_HIDDEN_SIZES = (20, 12)
NN_OUTPUT_SIZE = 4
NN_WEIGHT_INIT_RANGE = (-1.0, 1.0)
# Genes are clamped to this range after crossover/mutation so weights cannot explode.
NN_WEIGHT_CLIP_RANGE = (-1.0, 1.0)
RESTART_NEW_GENOME = False

# --- Genetic algorithm training -------------------------------------------
POPULATION_SIZE = 500
# Top individuals copied unchanged into the next generation.
ELITE_COUNT = 5
# Parent selection: sample TOURNAMENT_SIZE individuals, keep the fittest.
TOURNAMENT_SIZE = 5
# Fraction of offspring created via SBX crossover (rest are clone+mutate).
CROSSOVER_RATE = 0.75
# SBX distribution index: larger = children closer to parents, smaller = wider spread.
SBX_ETA = 15.0
# Per-gene mutation probability.
MUTATION_RATE = 0.10
# Mixed-scale Gaussian mutation: most mutated genes get a small nudge, a fraction
# get a large jump to escape local optima.
MUTATION_MAGNITUDE = 0.10  # small-step std (also default for mutate())
MUTATION_MAGNITUDE_LARGE = 0.40  # large-step std
LARGE_MUTATION_FRACTION = 0.10  # fraction of mutated genes that take a large jump
# Fitness (Chrispresso-inspired exponential reward):
# fitness = steps + (2^score + score^2.1 * FITNESS_SCORE_WEIGHT)
#          - ((0.25 * steps)^1.3 * score^1.2)
FITNESS_SCORE_WEIGHT = 500.0
# Games per genome per generation; averaging reduces luck-of-the-board noise.
EVAL_RUNS_PER_GENOME = 3
# Refresh the seeded scenario set every N generations. Holding boards fixed for a
# stretch gives a stable, comparable fitness so the GA can hill-climb; refreshing
# occasionally prevents pure board memorization.
SCENARIO_RESEED_FREQUENCY = 25
GENERATIONS = 200
# Step budget per evaluation game (kills infinite loops).
MAX_EVAL_STEPS = GRID_COLS * GRID_ROWS * 4

# Output index -> Direction mapping order
OUTPUT_DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT")

# Colors (R, G, B)
COLOR_BACKGROUND = (18, 18, 24)
COLOR_PANEL = (24, 24, 32)
COLOR_GRID_LINE = (40, 40, 52)
COLOR_SNAKE_HEAD = (80, 220, 120)
COLOR_SNAKE_BODY = (50, 160, 90)
COLOR_FOOD = (240, 80, 80)
COLOR_TEXT = (220, 220, 230)
COLOR_TEXT_DIM = (120, 120, 140)
COLOR_CONTROL_INACTIVE = (45, 45, 58)
COLOR_CONTROL_BORDER = (70, 70, 90)
COLOR_CONTROL_ACTIVE = (80, 180, 255)
COLOR_CONTROL_ACTIVE_GLOW = (120, 210, 255)
COLOR_GAME_OVER = (255, 100, 100)
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
