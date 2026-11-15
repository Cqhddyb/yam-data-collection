#!/usr/bin/env python3
"""Read the leader command and follower state for one gripper without commanding hardware."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import portal
import tyro


@dataclass
class Args:
    side: Literal["left", "right"] = "right"
    """Arm side to inspect."""
    timeout_s: float = 1.0
    """Maximum wait for each local telemetry reply."""


def main(args: Args) -> None:
    ports = {
        "left": (1245, 1235),
        "right": (1244, 1234),
    }
    leader_port, follower_port = ports[args.side]
    leader_client = portal.Client(f"127.0.0.1:{leader_port}")
    follower_client = portal.Client(f"127.0.0.1:{follower_port}")
    try:
        leader = leader_client.get_telemetry().result(timeout=args.timeout_s)
        follower = follower_client.get_telemetry().result(timeout=args.timeout_s)
        print(f"{args.side}_leader gripper command: {leader['position'][6]:.6f}")
        print(f"{args.side}_follower gripper actual: {follower['position'][6]:.6f}")
        print(f"{args.side}_follower gripper effort: {follower['effort'][6]:.6f}")
    finally:
        leader_client.close()
        follower_client.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
