"""Small terminal-only presentation helpers for operator-facing status lines."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable


_RESET = "\033[0m"
_COLORS = {
    "green": "\033[32m",
    "red": "\033[31m",
}


def colorize(message: str, color: str | None, *, stream: object | None = None) -> str:
    """Color a message only for an interactive terminal unless explicitly requested."""
    if color not in _COLORS:
        return message
    output = sys.stdout if stream is None else stream
    if os.environ.get("NO_COLOR") is not None:
        return message
    isatty = getattr(output, "isatty", None)
    if not callable(isatty) or not isatty():
        return message
    return f"{_COLORS[color]}{message}{_RESET}"


def print_status(message: str, color: str | None = None, *, printer: Callable[[str], object] = print) -> None:
    """Print a colored operator status without changing injected test printers."""
    if printer is print:
        print(colorize(message, color), flush=True)
    else:
        printer(message)
