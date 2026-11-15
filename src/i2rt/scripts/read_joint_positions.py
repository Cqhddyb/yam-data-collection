#!/usr/bin/env python3
"""Read left and right YAM follower joint positions without commanding hardware."""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from typing import Literal

import portal
import tyro

_PORTS = {
    "left": 1235,
    "right": 1234,
}


@dataclass
class Args:
    side: Literal["both", "left", "right"] = "both"
    """Arm side to read."""
    watch: bool = False
    """Keep reading until Ctrl+C is pressed."""
    interval_s: float = 1.0
    """Interval between reads in watch mode."""
    timeout_s: float = 2.0
    """Timeout for connecting and reading each telemetry endpoint."""


def _connect(side: str, timeout_s: float) -> portal.Client | None:
    port = _PORTS[side]
    client = portal.Client(f"127.0.0.1:{port}", autoconn=False)
    try:
        if not client.connect(timeout=timeout_s):
            raise TimeoutError(f"connection to {side} follower telemetry port {port} timed out")
        return client
    except Exception as exc:
        print(f"[{side}] ERROR: {exc}", file=sys.stderr)
        client.close(timeout=1.0)
        return None


def _print_state(side: str, state: dict[str, object]) -> None:
    position = [float(value) for value in state["position"]]  # type: ignore[index]
    joints_rad = position[:6]
    joints_deg = [math.degrees(value) for value in joints_rad]
    gripper = position[6] if len(position) > 6 else None
    age_ms = (time.time_ns() - int(state["timestamp_ns"])) / 1e6

    print(f"[{side}] sequence={int(state['sequence'])} sample_age={age_ms:.1f} ms")
    print("  joints_rad:", "[" + ", ".join(f"{value:.6f}" for value in joints_rad) + "]")
    print("  joints_deg:", "[" + ", ".join(f"{value:.2f}" for value in joints_deg) + "]")
    if gripper is None:
        print("  gripper_normalized: unavailable")
    else:
        print(f"  gripper_normalized: {gripper:.6f}")


def main(args: Args) -> None:
    if args.interval_s <= 0:
        raise ValueError("interval_s must be positive")
    if args.timeout_s <= 0:
        raise ValueError("timeout_s must be positive")

    sides = ["left", "right"] if args.side == "both" else [args.side]
    clients = {side: _connect(side, args.timeout_s) for side in sides}
    if not any(clients.values()):
        raise SystemExit("no telemetry endpoint could be reached")

    try:
        while True:
            if args.watch:
                print(time.strftime("%Y-%m-%d %H:%M:%S"))
            for side, client in clients.items():
                if client is None:
                    continue
                try:
                    state = client.get_telemetry().result(timeout=args.timeout_s)
                    _print_state(side, state)
                except Exception as exc:
                    print(f"[{side}] ERROR: {exc}", file=sys.stderr)
            if not args.watch:
                break
            print()
            time.sleep(args.interval_s)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        for client in clients.values():
            if client is not None:
                client.close(timeout=1.0)


if __name__ == "__main__":
    main(tyro.cli(Args))
