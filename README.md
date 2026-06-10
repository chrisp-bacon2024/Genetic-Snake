# Genetic Snake

Snake controlled by a small neural network, trained with a genetic algorithm. The pygame app shows the board and a live network panel; headless training evolves populations on configurable grid sizes with curriculum learning.

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

**Play (pygame):**

```bash
cd src
python main.py
```

| Key | Action |
|-----|--------|
| R | Restart (same brain unless `RESTART_NEW_GENOME` is True) |
| Esc | Quit |

**Train (headless):**

```bash
cd src
python train.py --dashboard --generations 500 --population 500 --workers 0
python train.py --resume --generations 200 --dashboard
python train.py --watch-only
```

Architecture or encoder changes require a **fresh** training run (old checkpoints are incompatible).

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  train.py          — GA loop, checkpoint/resume, curriculum   │
│  HeadlessSimulator — fast eval (no pygame)                    │
│  Game              — rules, scoring, win detection            │
│  GameStateEncoder  — 43 ray/food/direction features           │
│  NeuralNetwork     — MLP 43→32→4 (optional GRU in config)     │
│  Population        — tournament selection, SBX, mutation      │
└──────────────────────────────────────────────────────────────┘
```

**Design:** Game logic, encoding, and neural math have **no pygame dependency**, so training reuses the same `Game`, encoder, and network as the visual app.

### Per-tick flow (play or eval)

1. `GameStateEncoder` builds a 43-dimensional state vector (8 rays × wall/food/body, food bearing, directions, lookahead, space).
2. `NeuralNetwork` outputs 4 direction logits (masked for illegal moves).
3. `Game.tick(direction)` moves the snake, updates score, detects wall/body/starvation/win.

---

## Training

| Setting | Default | Notes |
|---------|---------|--------|
| Population | 1000 | `--population` |
| Curriculum | 5×5 → 10×10 → 20×20 | `--no-curriculum` for 20×20 only |
| Advance / stop | 25% wins | `--no-stop-on-win` to run full `--generations` |
| Eval | 2 screening + 3 refine (top 25%) | Shared food seeds per generation |
| Workers | auto (CPU−1) | `--workers 1` for serial |

**Outputs** (under `src/replays/` by default):

| File | Purpose |
|------|---------|
| `gen_XXXX.npz` | Best genome per generation + food seed for replay |
| `best.npz` / `best_score.npz` | Best by fitness / best score ever |
| `checkpoint.npz` | Full population for `--resume` |
| `training_log.jsonl` | Per-gen metrics for dashboard history |

**Replay saved best snakes:**

```bash
cd src
python train.py --watch-only
```

**Analyze training from replays:**

```bash
cd src
python analyze_training.py --show
```

---

## Neural network (default)

| Layer | Size | Description |
|-------|------|-------------|
| Input | 43 | Ray vision, food, heading, lookahead, reachable space |
| Hidden | 32 | ReLU (MLP) |
| Output | 4 | UP, DOWN, LEFT, RIGHT |

Genome size: **~1,540** floats (`NeuralNetwork.genome_length()`). Set `NN_ARCH = "gru"` in `config.py` for recurrent mode (different gene count).

---

## Project layout

```
src/
  main.py                 Pygame entry
  train.py                GA training CLI
  config.py               Grid, GA, NN, curriculum constants

  game/                   Rules (no pygame)
  models/                 Grid, snake, food, direction
  neural/                 Encoder, network, policy
  evolution/              Genome, population, fitness, curriculum, checkpoint
  simulation/             Headless eval, parallel workers
  controllers/            AIController (+ legacy KeyboardController)
  replay/                 Frame recorder (JSON, used by pygame app)
  ui/                     App, replay viewer, training dashboard

tests/
  test_game_win.py        Win detection smoke tests
```

---

## Configuration

All tunables live in `src/config.py`: grid size, `NN_HIDDEN_SIZES`, GA rates, curriculum stages, fitness weights, `REPLAYS_DIR`.

---

## Dependencies

- **numpy** — network and genomes
- **pygame** — interactive app and replay viewer
- **matplotlib** — training dashboard (`--dashboard`)

---

## Tests

```bash
cd src
python neural/verify_network.py
python -m unittest discover -s ../tests -p "test_*.py"
```
