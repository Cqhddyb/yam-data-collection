#!/usr/bin/env python3
"""Update task_description and task_id stored on an MCAP /task_info topic."""

from mcap_task_info import update_main


if __name__ == "__main__":
    raise SystemExit(update_main())
