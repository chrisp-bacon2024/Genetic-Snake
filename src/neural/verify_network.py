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
    section("TEST 1: Genome length matches GRU architecture")
    input_size, hidden, output_size = NeuralNetwork.architecture()
    gate = input_size * hidden + hidden * hidden + hidden
    manual = 3 * gate + hidden * output_size + output_size
    print(f"  Architecture: {input_size} -> GRU({hidden}) -> {output_size}")
    print(f"  Manual total: {manual}")
    print(f"  NeuralNetwork.genome_length(): {NeuralNetwork.genome_length()}")
    assert NeuralNetwork.genome_length() == manual
    print(f"  PASS: genome length matches manual count ({manual})")


def test_genome_unpack_layout() -> None:
    section("TEST 2: Genome unpack layout (known gene values)")
    length = NeuralNetwork.genome_length()
    genes = np.zeros(length, dtype=np.float64)
    input_size = config.NN_INPUT_SIZE
    hidden = config.NN_RNN_HIDDEN

    idx = 0
    for gate_no in range(3):
        w_tag = float(gate_no * 3 + 1)
        u_tag = float(gate_no * 3 + 2)
        b_tag = float(gate_no * 3 + 3)
        genes[idx : idx + input_size * hidden] = w_tag
        idx += input_size * hidden
        genes[idx : idx + hidden * hidden] = u_tag
        idx += hidden * hidden
        genes[idx : idx + hidden] = b_tag
        idx += hidden

    genes[idx : idx + hidden * config.NN_OUTPUT_SIZE] = 99.0
    idx += hidden * config.NN_OUTPUT_SIZE
    genes[idx : idx + config.NN_OUTPUT_SIZE] = 100.0

    net = NeuralNetwork.from_genome(Genome(genes))
    g = net._gru
    assert np.all(g.Wz == 1.0) and np.all(g.Uz == 2.0) and np.all(g.bz == 3.0)
    assert np.all(g.Wr == 4.0) and np.all(g.Ur == 5.0) and np.all(g.br == 6.0)
    assert np.all(g.Wh == 7.0) and np.all(g.Uh == 8.0) and np.all(g.bh == 9.0)
    assert np.all(g.Wo == 99.0) and np.all(g.bo == 100.0)
    print("  PASS: from_genome unpacks GRU gates and output layer at correct offsets")


def test_genome_round_trip() -> None:
    section("TEST 3: Genome round-trip (to_genome o from_genome)")
    np.random.seed(0)
    original = NeuralNetwork.random_genome()
    recovered = NeuralNetwork.from_genome(original).to_genome()
    max_diff = float(np.max(np.abs(original.genes - recovered.genes)))
    print(f"  length {len(original.genes)}, max abs diff {max_diff}")
    assert np.allclose(original.genes, recovered.genes)
    print("  PASS: round-trip preserves every gene exactly")


def test_gru_forward_state() -> None:
    section("TEST 4: GRU forward updates hidden state")
    np.random.seed(1)
    net = NeuralNetwork.from_genome(NeuralNetwork.random_genome())
    h0 = new_rnn_hidden()
    x = np.random.uniform(0.0, 1.0, size=config.NN_INPUT_SIZE)
    result, h1 = net.forward(x, h0)
    assert result.outputs.shape == (config.NN_OUTPUT_SIZE,)
    assert h1.shape == (config.NN_RNN_HIDDEN,)
    assert result.rnn_hidden.shape == h1.shape
    assert not np.allclose(h0, h1)
    result2, h2 = net.forward(x, h1)
    assert not np.allclose(h1, h2)
    print(f"  outputs={result.outputs.tolist()}")
    print(f"  |h0|={np.linalg.norm(h0):.3f} |h1|={np.linalg.norm(h1):.3f} |h2|={np.linalg.norm(h2):.3f}")
    print("  PASS: GRU hidden state changes each step")


def test_decide_step_resets_per_game() -> None:
    section("TEST 5: decide_step with fresh hidden each game")
    net = NeuralNetwork.from_genome(NeuralNetwork.random_genome())
    encoder = GameStateEncoder()
    game = Game(Grid(10, 10), food_seed=42)
    hidden = new_rnn_hidden()
    direction, hidden, _ = decide_step(net, game, encoder, hidden, game.snake.direction)
    game.tick(direction)
    assert hidden.shape == (config.NN_RNN_HIDDEN,)
    hidden2 = new_rnn_hidden()
    assert np.allclose(hidden2, 0.0)
    print("  PASS: new_rnn_hidden() is zero; decide_step returns updated state")


def test_encoder_shape_and_range() -> None:
    section("TEST 6: Encoder produces 41 features in [0, 1]")
    game = Game(Grid(config.GRID_COLS, config.GRID_ROWS), food_seed=123)
    features = GameStateEncoder().encode(game)
    print(f"  feature count: {features.shape[0]} (expected {config.NN_INPUT_SIZE})")
    assert features.shape[0] == config.NN_INPUT_SIZE == 41
    assert features.min() >= 0.0 and features.max() <= 1.0
    print("  PASS: encoder shape and value range are correct")


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


def main() -> None:
    print("NeuralNetwork (GRU) + Genome + Encoder verification")
    test_genome_length()
    test_genome_unpack_layout()
    test_genome_round_trip()
    test_gru_forward_state()
    test_decide_step_resets_per_game()
    test_encoder_shape_and_range()
    test_genetic_operators()
    print()
    print("All tests passed.")


if __name__ == "__main__":
    main()
