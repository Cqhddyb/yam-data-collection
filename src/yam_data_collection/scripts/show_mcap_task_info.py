#!/usr/bin/env python3
"""Display JSON values stored on an MCAP /task_info topic."""

import os
import sys
from pathlib import Path

# Permit direct execution with the system Python by using this checkout's
# environment, which provides the MCAP and ROS2 decoder dependencies.
VENV_PYTHON = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python"
if VENV_PYTHON.is_file() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), __file__, *sys.argv[1:]])

from mcap_task_info import show_main  # noqa: E402 - import after the venv handoff


if __name__ == "__main__":
    raise SystemExit(show_main())
