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
NN_INPUT_SIZE = 24
NN_HIDDEN_SIZE = 16
NN_OUTPUT_SIZE = 4
NN_WEIGHT_INIT_RANGE = (-1.0, 1.0)
RESTART_NEW_GENOME = False

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
