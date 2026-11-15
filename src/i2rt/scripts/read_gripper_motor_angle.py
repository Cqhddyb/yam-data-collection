#!/usr/bin/env python3
"""Passively read a gripper motor's DM feedback position over SocketCAN.

This tool listens for feedback frames emitted during normal motor control. It
does not create a robot, send CAN frames, or connect to a Portal command server.
The reported position is the motor feedback angle in the DM protocol range; it
is not adjusted by the robot's software zero offset or direction.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Literal

import can
import tyro

from i2rt.motor_drivers.utils import MotorType, uint_to_float

MotorTypeName = Literal["DM4310", "DM3507"]
_NORMAL_ERROR_CODE = 0x1
_RECEIVE_ID_OFFSET = 16


@dataclass
class Args:
    channel: str = "can0"
    """SocketCAN interface already used by the arm, for example can0."""
    motor_id: int = 7
    """Gripper motor's configured CAN ID."""
    motor_type: MotorTypeName = "DM4310"
    """Gripper motor model from its i2rt gripper configuration."""
    timeout_s: float = 2.0
    """Seconds to wait for a feedback frame before reporting a timeout."""
    watch: bool = False
    """Keep listening and print the latest angle at the report interval."""
    report_interval_s: float = 0.1
    """Minimum time between lines in watch mode."""


def decode_position_rad(data: bytes, motor_type: MotorTypeName) -> float:
    """Decode the position field from one normal DM p16 feedback frame."""
    if len(data) != 8:
        raise ValueError(f"expected an 8-byte DM feedback frame, got {len(data)} bytes")
    error_code = (data[0] >> 4) & 0x0F
    if error_code != _NORMAL_ERROR_CODE:
        raise RuntimeError(f"motor feedback reports error code 0x{error_code:X}")

    position_int = (data[1] << 8) | data[2]
    constants = MotorType.get_motor_constants(motor_type)
    return float(uint_to_float(position_int, constants.POSITION_MIN, constants.POSITION_MAX, 16))


def _validate_args(args: Args) -> None:
    if not args.channel:
        raise ValueError("channel must not be empty")
    if not 0 <= args.motor_id <= 0x7FF - _RECEIVE_ID_OFFSET:
        raise ValueError("motor_id must map to a standard CAN receive ID")
    if args.timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    if args.report_interval_s <= 0:
        raise ValueError("report_interval_s must be positive")


def main(args: Args) -> None:
    _validate_args(args)
    feedback_id = args.motor_id + _RECEIVE_ID_OFFSET
    constants = MotorType.get_motor_constants(args.motor_type)
    bus = can.Bus(
        interface="socketcan",
        channel=args.channel,
        receive_own_messages=False,
    )
    try:
        bus.set_filters([{"can_id": feedback_id, "can_mask": 0x7FF, "extended": False}])
        print(
            f"Listening only: channel={args.channel} motor_id=0x{args.motor_id:X} "
            f"feedback_id=0x{feedback_id:X} type={args.motor_type}"
        )
        print(
            f"Angle is motor feedback in [{constants.POSITION_MIN:g}, {constants.POSITION_MAX:g}] rad; "
            "software zero offset/direction are not applied."
        )

        last_report = 0.0
        while True:
            message = bus.recv(timeout=args.timeout_s)
            if message is None:
                if args.watch:
                    continue
                raise TimeoutError(
                    f"no feedback on CAN ID 0x{feedback_id:X} within {args.timeout_s:.2f}s; "
                    "check the channel, motor ID, and gripper motor type"
                )
            if message.is_extended_id or message.arbitration_id != feedback_id:
                continue
            angle_rad = decode_position_rad(bytes(message.data), args.motor_type)
            now = time.monotonic()
            if not args.watch or now - last_report >= args.report_interval_s:
                print(f"gripper_motor_feedback_rad: {angle_rad:+.6f}", flush=True)
                last_report = now
            if not args.watch:
                return
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        bus.shutdown()


if __name__ == "__main__":
    try:
        main(tyro.cli(Args))
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
