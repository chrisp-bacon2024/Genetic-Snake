"""
Manual verification demos for NeuralNetwork, Genome operators, and the encoder.

Run from repo root or src/:
    python src/neural/verify_network.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow running as a script from repo root or src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from evolution.genome import Genome
from game.game import Game
from models.direction import Direction
from models.grid import Grid
from neural.encoder import GameStateEncoder
from neural.network import NeuralNetwork
from neural.policy import decide_step, new_rnn_hidden


def section(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def test_genome_length() -> None:
    section("TEST 1: Genome length matches architecture")
    sizes = NeuralNetwork.layer_sizes()
    manual = 0
    for index in range(len(sizes) - 1):
        in_dim, out_dim = sizes[index], sizes[index + 1]
        manual += in_dim * out_dim + out_dim
    print(f"  Architecture: {NeuralNetwork.architecture_label()}")
    print(f"  Manual total: {manual}")
    print(f"  NeuralNetwork.genome_length(): {NeuralNetwork.genome_length()}")
    assert NeuralNetwork.genome_length() == manual
    print(f"  PASS: genome length matches manual count ({manual})")


def test_genome_unpack_layout() -> None:
    section("TEST 2: Genome unpack layout (known gene values)")
    length = NeuralNetwork.genome_length()
    genes = np.zeros(length, dtype=np.float64)
    sizes = NeuralNetwork.layer_sizes()
    idx = 0
    for layer_index in range(len(sizes) - 1):
        in_dim, out_dim = sizes[layer_index], sizes[layer_index + 1]
        tag = float(layer_index + 1)
        genes[idx : idx + in_dim * out_dim] = tag
        idx += in_dim * out_dim
        genes[idx : idx + out_dim] = tag + 0.5
        idx += out_dim

    net = NeuralNetwork.from_genome(Genome(genes))
    if NeuralNetwork.is_recurrent():
        g = net._gru
        assert g is not None
        assert np.all(g.Wz == 1.0) and np.all(g.Uz == 2.0) and np.all(g.bz == 3.0)
    else:
        assert net._mlp_layers is not None
        assert np.all(net._mlp_layers[0].W == 1.0) and np.all(net._mlp_layers[0].b == 1.5)
        assert np.all(net._mlp_layers[-1].W == float(len(sizes) - 1))
    print("  PASS: from_genome unpacks layers at correct offsets")


def test_genome_round_trip() -> None:
    section("TEST 3: Genome round-trip (to_genome o from_genome)")
    np.random.seed(0)
    original = NeuralNetwork.random_genome()
    recovered = NeuralNetwork.from_genome(original).to_genome()
    max_diff = float(np.max(np.abs(original.genes - recovered.genes)))
    print(f"  length {len(original.genes)}, max abs diff {max_diff}")
    assert np.allclose(original.genes, recovered.genes)
    print("  PASS: round-trip preserves every gene exactly")


def test_forward_pass() -> None:
    section("TEST 4: Forward pass produces valid outputs")
    np.random.seed(1)
    net = NeuralNetwork.from_genome(NeuralNetwork.random_genome())
    x = np.random.uniform(0.0, 1.0, size=config.NN_INPUT_SIZE)
    result, hidden = net.forward(x, new_rnn_hidden())
    assert result.outputs.shape == (config.NN_OUTPUT_SIZE,)
    assert len(result.hidden_layers) == (
        1 if NeuralNetwork.is_recurrent() else len(config.NN_HIDDEN_SIZES)
    )
    if NeuralNetwork.is_recurrent():
        assert hidden.shape == (config.NN_RNN_HIDDEN,)
        assert not np.allclose(hidden, 0.0)
    else:
        assert hidden.size == 0
    print(f"  outputs={result.outputs.tolist()}")
    print("  PASS: forward pass shape and activations are valid")


def test_decide_step() -> None:
    section("TEST 5: decide_step picks a direction")
    net = NeuralNetwork.from_genome(NeuralNetwork.random_genome())
    encoder = GameStateEncoder()
    game = Game(Grid(10, 10), food_seed=42)
    hidden = new_rnn_hidden()
    direction, hidden, _ = decide_step(net, game, encoder, hidden, game.snake.direction)
    game.tick(direction)
    assert direction is not None
    print("  PASS: decide_step returns a valid direction")


def test_encoder_shape_and_range() -> None:
    section("TEST 6: Encoder produces ray features in [0, 1]")
    encoder = GameStateEncoder()
    game = Game(Grid(config.GRID_COLS, config.GRID_ROWS), food_seed=123)
    features = encoder.encode(game)
    print(f"  feature count: {features.shape[0]} (expected {config.NN_INPUT_SIZE})")
    assert features.shape[0] == config.NN_INPUT_SIZE == encoder.input_size()
    assert features.min() >= 0.0 and features.max() <= 1.0
    mask = encoder.safe_move_mask(game)
    assert mask.shape == (4,)
    print("  PASS: encoder shape, mask, and value range are correct")


def test_genetic_operators() -> None:
    section("TEST 7: SBX crossover, mutation, and clip")
    np.random.seed(1)
    length = NeuralNetwork.genome_length()
    p1 = Genome(np.full(length, -0.5))
    p2 = Genome(np.full(length, 0.5))
    c1, c2 = p1.sbx_pair(p2, eta=15.0)
    assert np.allclose(c1.genes + c2.genes, p1.genes + p2.genes)

    base = Genome(np.zeros(length))
    base.mutate(rate=0.10, magnitude=0.1)
    changed = int(np.count_nonzero(base.genes))
    frac = changed / length
    assert 0.03 < frac < 0.20

    wild = Genome(np.full(length, 5.0))
    wild.clip()
    low, high = config.NN_WEIGHT_CLIP_RANGE
    assert wild.genes.max() <= high and wild.genes.min() >= low
    print("  PASS: genetic operators behave as expected")


def test_win_detection() -> None:
    section("TEST 8: Full board triggers win")
    grid = Grid(2, 2)
    game = Game(grid, food_seed=0)
    directions = (
        Direction.UP,
        Direction.LEFT,
        Direction.DOWN,
        Direction.RIGHT,
        Direction.UP,
        Direction.LEFT,
        Direction.DOWN,
    )
    for direction in directions:
        if not game.alive:
            break
        game.tick(direction)
    assert game.won, f"expected win, got cause={game.death_cause} score={game.score}"
    assert game.death_cause == "win"
    assert game.score >= config.max_win_score(2, 2)
    print(f"  score={game.score} cause={game.death_cause}")
    print("  PASS: win detected on full 2x2 board")


def main() -> None:
    print(f"NeuralNetwork ({config.NN_ARCH}) + Genome + Encoder verification")
    test_genome_length()
    test_genome_unpack_layout()
    test_genome_round_trip()
    test_forward_pass()
    test_decide_step()
    test_encoder_shape_and_range()
    test_genetic_operators()
    test_win_detection()
    print()
    print("All tests passed.")


if __name__ == "__main__":
    main()
