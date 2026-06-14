# Genetic Snake

Snake controlled by a small neural network, trained with a genetic algorithm — no backpropagation, no labeled moves. The pygame app shows the board, heading-relative vision rays, and a live network panel; headless training evolves populations on configurable grid sizes with curriculum learning.

## Demo

<video src="docs/snake-gameplay-readme.mp4" controls autoplay muted loop width="100%"></video>

*Generation 215 replay — board, vision rays, and live neural panel from the portfolio site.*

**Live portfolio site:** [chrisp-bacon2024.github.io/Genetic-Snake](https://chrisp-bacon2024.github.io/Genetic-Snake/) — replay demos, training chart, and architecture walkthrough. Local dev: [`site/README.md`](site/README.md).

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

**Pygame demo (random AI):** launches the board, vision rays, and network panel with an **untrained random brain** — useful for debugging the UI, not for watching evolved snakes. Human keyboard control is not wired (`KeyboardController` is legacy only).

```bash
cd src
python main.py
```

| Key | Action |
|-----|--------|
| R | Restart (same brain unless `RESTART_NEW_GENOME` is True) |
| Esc | Quit |

**Watch trained snakes (pygame replay viewer):** needs `replays/gen_*.npz` from a training run.

```bash
cd src
python train.py --watch-only
```

**Train (headless):**

```bash
cd src
python train.py --dashboard --generations 500 --population 500 --workers 0
python train.py --resume --generations 200 --dashboard
```

Architecture or encoder changes require a **fresh** training run (old checkpoints are incompatible).

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  train.py          — GA loop, checkpoint/resume, curriculum   │
│  HeadlessSimulator — fast eval (no pygame)                    │
│  Game              — rules, scoring, win detection            │
│  GameStateEncoder  — 44 ray/feature inputs                    │
│  NeuralNetwork     — MLP 44→64→4 (optional GRU in config)     │
│  Population        — tournament selection, SBX, mutation      │
└──────────────────────────────────────────────────────────────┘
```

**Design:** Game logic, encoding, and neural math have **no pygame dependency**, so training reuses the same `Game`, encoder, and network as the visual app.

### Per-tick flow (pygame demo or headless eval)

1. `GameStateEncoder` builds a 44-dimensional state vector (8 rays × wall/food/body, food cues, head/tail direction, lookahead, space metrics).
2. `NeuralNetwork` outputs 4 direction logits (illegal moves masked before argmax).
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

**Analyze training from replays:**

```bash
cd src
python analyze_training.py --show
```

**Export data for the portfolio site:**

```bash
python scripts/export_site_data.py --replays-dir src/replays
cd site && npm install && npm run dev
```

---

## Neural network (default)

| Layer | Size | Description |
|-------|------|-------------|
| Input | 44 | Ray vision, food, heading, lookahead, reachable space |
| Hidden | 64 | ReLU (MLP) |
| Output | 4 | UP, DOWN, LEFT, RIGHT |

Genome size: **~3,140** floats (`NeuralNetwork.genome_length()`). Set `NN_ARCH = "gru"` in `config.py` for recurrent mode (different gene count).

---

## Project layout

```
src/
  main.py                 Pygame demo (random AI; not trained replays)
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

site/                     Portfolio demo (Vite + TypeScript)
scripts/export_site_data.py   NPZ replays → site JSON + chart

tests/
  test_game_win.py        Win detection smoke tests
docs/
  snake-gameplay-readme.mp4   Compressed demo video (README)
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
