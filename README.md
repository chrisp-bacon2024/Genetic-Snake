# Genetic Snake

An interactive Snake simulation where each snake is controlled by a small neural network. The left panel visualizes what the network "sees" and which direction it chooses each tick. Games are recorded in memory (and can be saved to JSON) for future replay on a website.

This project is designed to evolve toward **genetic-algorithm training**: populations of snakes compete, the best genomes reproduce, and top runs are saved per epoch.

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python src/main.py
```

| Key | Action |
|-----|--------|
| R | Restart (same brain unless `RESTART_NEW_GENOME` is True) |
| Esc | Quit |

The snake is **AI-controlled** — arrow keys do not move it.

---

## High-level architecture

```
┌─────────────────────────────────────────────────────────────┐
│  SnakeApp (pygame loop)                                     │
│    ├── Game          — rules, score, collisions (no pygame)   │
│    ├── AIController  — encode → network → direction          │
│    ├── GameRecorder  — stores every tick for replay         │
│    └── UI            — grid + neural net panel              │
└─────────────────────────────────────────────────────────────┘
```

**Design principle:** Game logic, encoding, and neural math live in pure Python modules with **no pygame dependency**. That lets you run headless training later while reusing the same `Game`, `GameStateEncoder`, and `NeuralNetwork` classes.

### Per-tick flow

1. `AIController.get_direction()` reads the board via `GameStateEncoder` (24 vision features).
2. `NeuralNetwork.forward()` produces hidden activations and 4 output logits.
3. The highest valid output becomes a `Direction` (180° reversals are blocked).
4. `Game.tick(direction)` moves the snake, checks collisions, updates score.
5. `GameRecorder.record_frame()` stores board state + all neuron values.
6. The UI draws the grid and live network diagram from `NetworkSnapshot`.

---

## Project layout

```
src/
  main.py                 Entry point
  config.py               All tunable constants (grid, colors, NN sizes)

  models/                 Domain objects (pure Python)
    direction.py          Direction enum + relative vision rays
    position.py           Grid coordinate
    grid.py               Bounds and empty-cell lookup
    snake.py              Body, movement, growth
    food.py               Food placement

  game/                   Simulation rules (no pygame)
    game.py               tick(), reset(), scoring
    game_state.py         TickResult, GameState

  neural/                 Brain
    encoder.py            8-ray vision → 24 inputs
    network.py            24→16→4 feedforward net + genome mapping

  evolution/              Genetic algorithm hooks (GA not implemented yet)
    genome.py             Flat weight vector (468 genes)

  controllers/            Input / decision layer
    controller.py         Abstract Controller interface
    ai_controller.py      Neural network driver + NetworkSnapshot
    keyboard_controller.py  Legacy keyboard control (unused in main app)

  replay/                 Saved games for web visualization
    frame.py              One tick of state + activations
    recorder.py             start / record_frame / save / load

  ui/                     Pygame rendering
    app.py                Main loop, wires everything together
    control_panel.py      Left sidebar orchestration
    network_visualizer.py Input / hidden / output arrow diagram
    game_renderer.py      Grid, snake, food, score
```

---

## Neural network

| Layer | Size | Description |
|-------|------|-------------|
| Input | 24 | 8 relative rays × (wall, body, food) distances |
| Hidden | 16 | ReLU activation |
| Output | 4 | Logits for UP, DOWN, LEFT, RIGHT |

**Genome:** 468 floating-point weights stored flat in `Genome.genes`. Layout:

`[W1 (384), b1 (16), W2 (64), b2 (4)]`

See `NeuralNetwork.from_genome()` in `src/neural/network.py`.

### Vision encoding

From the snake's head, eight rays are cast **relative to current heading** (forward, forward-right, right, …). Along each ray, Manhattan step counts are recorded to the first wall, body segment, and food. Values are normalized to `[0, 1]` by grid size.

---

## Replay format

Recordings are **not saved automatically** during normal play. Frames accumulate in memory; call `recorder.save()` when you want to persist (e.g. best snake per epoch during GA).

Example:

```python
from pathlib import Path
import config

recorder.save(Path(config.REPLAYS_DIR) / "epoch_0042_best.json")
```

Each JSON file contains:

- `version`, `grid`, `ticks_per_second`
- `genome` — full weight vector to reconstruct the brain
- `frames[]` — per tick: `inputs`, `hidden`, `outputs`, `direction`, `snake`, `food`, `score`, flags

Load with `GameRecorder.load(path)`. See `replays/sample_replay.json` for a real example.

---

## Configuration

All magic numbers live in `src/config.py`:

- **Grid:** `GRID_COLS`, `GRID_ROWS`, `TICKS_PER_SECOND`
- **Network:** `NN_INPUT_SIZE`, `NN_HIDDEN_SIZE`, `NN_OUTPUT_SIZE`
- **UI:** panel width, colors, neuron layout sizes
- **Replay:** `REPLAYS_DIR` (default `"replays"`, gitignored)

Set `RESTART_NEW_GENOME = True` to give the snake a new random brain on each restart.

---

## What's next (not implemented)

- Genetic algorithm: population, fitness, crossover, mutation
- Headless batch evaluation during training
- Auto-save best replay per epoch
- GitHub Pages web replay viewer

The `Genome`, `GameRecorder`, and pygame-free game stack are already structured for these features.

---

## Dependencies

- **pygame** — window, rendering, event loop
- **numpy** — neural network matrix math and genome arrays
