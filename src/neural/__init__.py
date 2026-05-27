"""Neural network and vision encoding for snake decision-making."""

from .encoder import GameStateEncoder
from .network import ForwardResult, NeuralNetwork

__all__ = ["GameStateEncoder", "ForwardResult", "NeuralNetwork"]
