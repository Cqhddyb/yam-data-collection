#!/usr/bin/env python3
"""Move one YAM follower through a preprogrammed joint-space motion.

The script is read-only unless both ``--execute`` and ``--confirm MOVE`` are
provided. It talks to the existing follower Portal server and sends complete
``[joint1..joint6, gripper]`` position targets in command space:

* arm joints are radians;
* gripper is normalized to [0, 1] (closed to open, according to telemetry).

Example plan JSON::

    {
      "side": "left",
      "start": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
      "target": [0.2, 0.1, 0.0, 0.3, 0.0, 0.0, 0.2],
      "duration_s": 5.0
    }

The placeholder values above are examples only. Replace them with measured
values from ``read_joint_positions.py`` before using ``--execute``.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import portal
import tyro

_PORTS = {
    "left": {"follower": 1235, "leader_telemetry": 1245},
    "right": {"follower": 1234, "leader_telemetry": 1244},
}
_NUM_ARM_JOINTS = 6
_NUM_DOFS_WITH_GRIPPER = 7
_SYNC_CHECK_PERIOD_S = 0.5


@dataclass
class Args:
    plan: Path
    """JSON plan containing side, start, target, and duration_s."""
    side: Literal["left", "right"] | None = None
    """Override the side in the plan JSON."""
    execute: bool = False
    """Actually send motion commands. Default is a no-command preview."""
    confirm: str = ""
    """Must be exactly MOVE together with --execute."""
    timeout_s: float = 2.0
    """Timeout for each Portal connection or request."""
    rate_hz: float = 25.0
    """Setpoint stream rate during the interpolated move."""
    position_tolerance_rad: float = 0.12
    """Allowed start mismatch for each arm joint."""
    gripper_tolerance: float = 0.12
    """Allowed start mismatch for the normalized gripper value."""
    ignore_start_check: bool = False
    """Skip the measured-start safety check. Use only when the plan is intentionally approximate."""
    ignore_sync_check: bool = False
    """Skip the leader synchronization safety check."""


def _connect(port: int, timeout_s: float) -> portal.Client:
    client = portal.Client(f"127.0.0.1:{port}", autoconn=False)
    if not client.connect(timeout=timeout_s):
        client.close(timeout=1.0)
        raise TimeoutError(f"cannot connect to Portal port {port}")
    return client


def _rpc(client: portal.Client, method: str, timeout_s: float, *args: Any) -> Any:
    return getattr(client, method)(*args).result(timeout=timeout_s)


def _read_plan(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            plan = json.load(stream)
    except OSError as exc:
        raise ValueError(f"cannot read plan {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(plan, dict):
        raise ValueError("plan must be a JSON object")
    return plan


def _vector(plan: dict[str, Any], key: str) -> np.ndarray:
    raw = plan.get(key)
    if not isinstance(raw, list) or len(raw) != _NUM_DOFS_WITH_GRIPPER:
        raise ValueError(f"plan.{key} must contain exactly 7 values: 6 joints plus gripper")
    try:
        result = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"plan.{key} must contain numeric values") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"plan.{key} contains non-finite values")
    if not 0.0 <= result[6] <= 1.0:
        raise ValueError(f"plan.{key}[6] gripper value must be in [0, 1]")
    return result


def _validate_plan(plan: dict[str, Any], side_override: str | None) -> tuple[str, np.ndarray, np.ndarray, float]:
    side = side_override or plan.get("side")
    if side not in _PORTS:
        raise ValueError("side must be left or right, either in the plan or via --side")
    start = _vector(plan, "start")
    target = _vector(plan, "target")
    duration_s = plan.get("duration_s", 5.0)
    if not isinstance(duration_s, (int, float)) or not math.isfinite(float(duration_s)) or float(duration_s) <= 0:
        raise ValueError("plan.duration_s must be a positive number")
    return side, start, target, float(duration_s)


def _print_vector(label: str, values: np.ndarray) -> None:
    print(f"{label} joints_rad: [" + ", ".join(f"{value:.6f}" for value in values[:6]) + "]")
    print(f"{label} gripper_normalized: {values[6]:.6f}")


def _check_start(current: np.ndarray, expected: np.ndarray, args: Args) -> None:
    arm_error = np.abs(current[:_NUM_ARM_JOINTS] - expected[:_NUM_ARM_JOINTS])
    gripper_error = abs(float(current[6] - expected[6]))
    max_arm_error = float(np.max(arm_error))
    if max_arm_error > args.position_tolerance_rad or gripper_error > args.gripper_tolerance:
        raise RuntimeError(
            "current pose is not close to plan.start: "
            f"max_arm_error={max_arm_error:.4f} rad, gripper_error={gripper_error:.4f}; "
            "move the arm to the planned start pose or use --ignore-start-check"
        )


def _check_leader_not_synchronized(client: portal.Client, args: Args, side: str) -> None:
    telemetry = _rpc(client, "get_telemetry", args.timeout_s)
    if "synchronized" not in telemetry:
        raise RuntimeError(
            f"{side} leader telemetry does not expose synchronization state; "
            "use --ignore-sync-check only after confirming that no leader is sending commands"
        )
    if bool(telemetry["synchronized"]):
        raise RuntimeError(
            f"{side} leader is synchronized with the follower; disable teleoperation before preprogrammed motion"
        )


def _release_to_gravity_comp(client: portal.Client, args: Args) -> None:
    """Leave the follower in gravity-compensation idle instead of position hold."""
    try:
        _rpc(client, "enter_gravity_comp_idle", args.timeout_s)
        print("Follower released to gravity-compensation idle; manual movement is enabled.")
    except Exception as exc:
        print(
            "Could not release follower to gravity-compensation idle; it may still hold the last position "
            f"target: {exc}",
            file=sys.stderr,
        )


def _execute_move(
    follower: portal.Client,
    leader: portal.Client | None,
    current: np.ndarray,
    target: np.ndarray,
    duration_s: float,
    args: Args,
    side: str,
) -> None:
    steps = max(1, math.ceil(duration_s * args.rate_hz))
    period_s = duration_s / steps
    started = time.monotonic()
    last_sync_check = started
    try:
        for step in range(1, steps + 1):
            if leader is not None and time.monotonic() - last_sync_check >= _SYNC_CHECK_PERIOD_S:
                _check_leader_not_synchronized(leader, args, side)
                last_sync_check = time.monotonic()
            alpha = step / steps
            command = (1.0 - alpha) * current + alpha * target
            _rpc(follower, "command_joint_pos", args.timeout_s, command)
            deadline = started + step * period_s
            time.sleep(max(0.0, deadline - time.monotonic()))
    except KeyboardInterrupt:
        _release_to_gravity_comp(follower, args)
        raise
    except Exception:
        _release_to_gravity_comp(follower, args)
        raise
    else:
        _release_to_gravity_comp(follower, args)


def main(args: Args) -> None:
    if args.timeout_s <= 0 or args.rate_hz <= 0:
        raise ValueError("timeout_s and rate_hz must be positive")
    if args.position_tolerance_rad < 0 or args.gripper_tolerance < 0:
        raise ValueError("position tolerances must be non-negative")

    plan = _read_plan(args.plan)
    side, start, target, duration_s = _validate_plan(plan, args.side)
    _print_vector("plan.start", start)
    _print_vector("plan.target", target)
    print(f"side={side} duration_s={duration_s:.2f} rate_hz={args.rate_hz:.1f}")

    follower = _connect(_PORTS[side]["follower"], args.timeout_s)
    leader: portal.Client | None = None
    try:
        state = _rpc(follower, "get_telemetry", args.timeout_s)
        current = np.asarray(state["position"], dtype=np.float64)
        if current.shape != (_NUM_DOFS_WITH_GRIPPER,):
            raise RuntimeError(f"expected 7 telemetry positions, received {current.shape}")
        _print_vector("current", current)
        if not args.ignore_start_check:
            _check_start(current, start, args)

        if not args.ignore_sync_check:
            try:
                leader = _connect(_PORTS[side]["leader_telemetry"], args.timeout_s)
            except TimeoutError as exc:
                raise RuntimeError(
                    f"{side} leader telemetry port {_PORTS[side]['leader_telemetry']} is unavailable; "
                    "start that leader service, or use --ignore-sync-check only after confirming "
                    "that no leader or other controller can send commands"
                ) from exc
            _check_leader_not_synchronized(leader, args, side)

        if not args.execute:
            print("Preview only: no motion command was sent. Add --execute --confirm MOVE to execute.")
            return
        if args.confirm != "MOVE":
            raise SystemExit("Execution requires --confirm MOVE")

        print("Executing preprogrammed joint-space move...")
        _execute_move(follower, leader, current, target, duration_s, args, side)
        print("Move command stream completed; follower will hold the target pose.")
    finally:
        if leader is not None:
            leader.close(timeout=1.0)
        follower.close(timeout=1.0)


if __name__ == "__main__":
    try:
        main(tyro.cli(Args))
    except (RuntimeError, TimeoutError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
