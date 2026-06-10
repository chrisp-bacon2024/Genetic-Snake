"""Single-line terminal progress for screening/refining eval phases."""

from __future__ import annotations

import sys

_CLEAR_EOL = "\033[K"


class InlineProgress:
    """
    Overwrite one terminal line while eval runs, then replace it with the gen log.

    Falls back to normal line printing when stdout is not a TTY.
    """

    _active = False

    @classmethod
    def _use_inline(cls) -> bool:
        return sys.stdout.isatty()

    @classmethod
    def update(cls, message: str) -> None:
        text = message.strip()
        if not cls._use_inline():
            print(text, flush=True)
            return
        cls._active = True
        sys.stdout.write(f"\r{text}{_CLEAR_EOL}")
        sys.stdout.flush()

    @classmethod
    def finish(cls, message: str) -> None:
        text = message.strip()
        if not cls._use_inline():
            print(text, flush=True)
            cls._active = False
            return
        if cls._active:
            sys.stdout.write(f"\r{text}{_CLEAR_EOL}\n")
        else:
            sys.stdout.write(f"{text}\n")
        sys.stdout.flush()
        cls._active = False
