#!/usr/bin/env python3
"""Release a YAM follower from position hold into gravity-compensation idle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import portal
import tyro

_FOLLOWER_PORTS = {
    "left": 1235,
    "right": 1234,
}


@dataclass
class Args:
    side: Literal["left", "right"] = "left"
    """Follower side to release."""
    timeout_s: float = 2.0
    """Timeout for connecting and sending the release request."""


def main(args: Args) -> None:
    if args.timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    port = _FOLLOWER_PORTS[args.side]
    client = portal.Client(f"127.0.0.1:{port}", autoconn=False)
    try:
        if not client.connect(timeout=args.timeout_s):
            raise TimeoutError(f"cannot connect to {args.side} follower port {port}")
        client.enter_gravity_comp_idle().result(timeout=args.timeout_s)
        print(f"{args.side} follower released to gravity-compensation idle.")
    finally:
        client.close(timeout=1.0)


if __name__ == "__main__":
    try:
        main(tyro.cli(Args))
    except (TimeoutError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
