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


def section(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def test_genome_length() -> None:
    section("TEST 1: Genome length matches architecture")
    sizes = NeuralNetwork.architecture()
    total = sum(fan_in * fan_out + fan_out for fan_in, fan_out in zip(sizes, sizes[1:]))
    print(f"  Architecture: {sizes}")
    for fan_in, fan_out in zip(sizes, sizes[1:]):
        print(f"  layer {fan_in} -> {fan_out}: {fan_in * fan_out} weights + {fan_out} biases")
    print(f"  Manual total: {total}")
    print(f"  NeuralNetwork.genome_length(): {NeuralNetwork.genome_length()}")
    assert NeuralNetwork.genome_length() == total
    print(f"  PASS: genome length matches manual count ({total})")


def test_genome_unpack_layout() -> None:
    section("TEST 2: Genome unpack layout (known gene values)")
    sizes = NeuralNetwork.architecture()
    length = NeuralNetwork.genome_length()
    genes = np.zeros(length, dtype=np.float64)

    # Tag each layer's weights and biases with distinct values.
    idx = 0
    tags: list[tuple[float, float]] = []
    for layer_no, (fan_in, fan_out) in enumerate(zip(sizes, sizes[1:])):
        w_tag = float(layer_no * 2 + 1)
        b_tag = float(layer_no * 2 + 2)
        tags.append((w_tag, b_tag))
        genes[idx : idx + fan_in * fan_out] = w_tag
        idx += fan_in * fan_out
        genes[idx : idx + fan_out] = b_tag
        idx += fan_out

    net = NeuralNetwork.from_genome(Genome(genes))
    for layer_no, ((w, b), (fan_in, fan_out)) in enumerate(zip(net._layers, zip(sizes, sizes[1:]))):
        w_tag, b_tag = tags[layer_no]
        print(f"  layer {layer_no}: W{w.shape} all=={w_tag}? {np.all(w == w_tag)}  "
              f"b{b.shape} all=={b_tag}? {np.all(b == b_tag)}")
        assert w.shape == (fan_in, fan_out)
        assert b.shape == (fan_out,)
        assert np.all(w == w_tag)
        assert np.all(b == b_tag)
    print("  PASS: from_genome unpacks every layer at the correct offset")


def test_genome_round_trip() -> None:
    section("TEST 3: Genome round-trip (to_genome o from_genome)")
    np.random.seed(0)
    original = NeuralNetwork.random_genome()
    recovered = NeuralNetwork.from_genome(original).to_genome()
    max_diff = float(np.max(np.abs(original.genes - recovered.genes)))
    print(f"  length {len(original.genes)}, max abs diff {max_diff}")
    assert np.allclose(original.genes, recovered.genes)
    print("  PASS: round-trip preserves every gene exactly")


def test_forward_hand_computed() -> None:
    section("TEST 4: Forward pass (hand-computed, 3->2->2 net)")
    # Build a tiny traceable network directly (independent of config topology).
    w0 = np.zeros((3, 2))
    w0[0, 0] = 1.0
    b0 = np.array([0.5, 0.0])
    w1 = np.zeros((2, 2))
    w1[0, 1] = 2.0
    b1 = np.array([0.0, 1.0])
    net = NeuralNetwork([(w0, b0), (w1, b1)])

    x = np.array([1.0, 0.0, 0.0])
    # hidden = ReLU([1*1+0.5, 0]) = [1.5, 0]
    # outputs = [1.5*0+0, 1.5*2+1] = [0, 4]
    result = net.forward(x)
    print(f"  hidden_layers = {[h.tolist() for h in result.hidden_layers]}")
    print(f"  outputs       = {result.outputs.tolist()}  argmax={int(np.argmax(result.outputs))}")
    assert np.allclose(result.hidden_layers[0], [1.5, 0.0])
    assert np.allclose(result.outputs, [0.0, 4.0])
    assert int(np.argmax(result.outputs)) == 1
    print("  PASS: two-layer ReLU forward matches manual calculation")


def test_encoder_shape_and_range() -> None:
    section("TEST 5: Encoder produces 37 features in [0, 1]")
    game = Game(Grid(config.GRID_COLS, config.GRID_ROWS), food_seed=123)
    features = GameStateEncoder().encode(game)
    print(f"  feature count: {features.shape[0]} (expected {config.NN_INPUT_SIZE})")
    print(f"  min={features.min():.3f}  max={features.max():.3f}")
    food_block = features[24:29]
    one_hot_tail = features[-8:]
    print(f"  food block (dist + dir): {food_block.tolist()}")
    print(f"  head+tail one-hot block sums: {one_hot_tail[:4].sum():.0f}, {one_hot_tail[4:].sum():.0f}")
    assert features.shape[0] == config.NN_INPUT_SIZE == 37
    assert features.min() >= 0.0 and features.max() <= 1.0
    assert food_block[0] > 0.0  # inverse manhattan distance
    assert food_block[1:].sum() > 0.0  # direction offsets
    assert np.isclose(one_hot_tail[:4].sum(), 1.0)
    assert np.isclose(one_hot_tail[4:].sum(), 1.0)
    print("  PASS: encoder shape and value range are correct")


def test_encoder_dense_food_off_ray() -> None:
    section("TEST 6: Food distance + direction when food is off all rays")
    from models.direction import Direction, relative_ray_deltas
    from models.position import Position

    encoder = GameStateEncoder()
    head = Position(10, 10)
    food = Position(12, 13)
    food_feats = encoder._food_features(Direction.RIGHT, head, food)
    print(f"  head={head} food={food} facing=RIGHT  features={food_feats}")
    assert food_feats[0] == 1.0 / (2 + 3 + 1)  # inverse manhattan = 1/6
    assert food_feats[1] > 0.0  # forward (2 cells)
    assert food_feats[2] > 0.0  # right (3 cells)
    assert food_feats[3] == 0.0  # no food behind
    assert food_feats[4] == 0.0  # no food to left
    print("  PASS: off-ray food has clear distance and direction signals")


def test_encoder_angular_ray_food() -> None:
    section("TEST 7: Angular ray-food active on all rays when food is off-ray")
    from models.direction import Direction, relative_ray_deltas
    from models.position import Position

    encoder = GameStateEncoder()
    head = Position(10, 10)
    food = Position(12, 13)
    ray_food = [
        encoder._ray_food_alignment(head, food, dx, dy)
        for dx, dy in relative_ray_deltas(Direction.RIGHT)
    ]
    print(f"  off-ray food ray signals: {[round(v, 3) for v in ray_food]}")
    assert len(ray_food) == 8
    assert all(value > 0.0 for value in ray_food)
    assert max(ray_food) > min(ray_food)  # strongest on aligned rays

    game = Game(Grid(config.GRID_COLS, config.GRID_ROWS), food_seed=456)
    features = encoder.encode(game)
    ray_food_block = features[1::3][:8]
    print(f"  full encode ray-food: min={ray_food_block.min():.3f} max={ray_food_block.max():.3f}")
    assert np.all(ray_food_block > 0.0)
    print("  PASS: all 8 ray-food inputs are non-zero")


def test_genetic_operators() -> None:
    section("TEST 8: SBX crossover, mutation, and clip")
    np.random.seed(1)
    length = NeuralNetwork.genome_length()
    p1 = Genome(np.full(length, -0.5))
    p2 = Genome(np.full(length, 0.5))
    c1, c2 = p1.sbx_pair(p2, eta=15.0)
    # SBX preserves the per-gene mean of the two parents.
    mean_preserved = np.allclose(c1.genes + c2.genes, p1.genes + p2.genes)
    print(f"  SBX preserves parent mean per gene? {mean_preserved}")
    assert mean_preserved

    base = Genome(np.zeros(length))
    base.mutate(rate=0.10, magnitude=0.1)
    changed = int(np.count_nonzero(base.genes))
    frac = changed / length
    print(f"  mutate(rate=0.10): {changed}/{length} genes changed ({frac:.3f})")
    assert 0.03 < frac < 0.20  # roughly the configured rate

    wild = Genome(np.full(length, 5.0))
    wild.clip()
    low, high = config.NN_WEIGHT_CLIP_RANGE
    print(f"  clip(): max gene now {wild.genes.max():.2f} (limit {high})")
    assert wild.genes.max() <= high and wild.genes.min() >= low
    print("  PASS: genetic operators behave as expected")


def main() -> None:
    print("NeuralNetwork + Genome + Encoder verification")
    test_genome_length()
    test_genome_unpack_layout()
    test_genome_round_trip()
    test_forward_hand_computed()
    test_encoder_shape_and_range()
    test_encoder_dense_food_off_ray()
    test_encoder_angular_ray_food()
    test_genetic_operators()
    print()
    print("All tests passed.")


if __name__ == "__main__":
    main()
