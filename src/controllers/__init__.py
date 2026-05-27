"""
Input controllers that decide snake direction each tick.

Controller is the abstraction shared by keyboard play, AI play, and (future) replay
playback. The UI reads get_active_direction() to highlight the chosen move.
"""

from .controller import Controller
from .keyboard_controller import KeyboardController
from .ai_controller import AIController, NetworkSnapshot

__all__ = ["Controller", "KeyboardController", "AIController", "NetworkSnapshot"]
