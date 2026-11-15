"""Guarded joint-space moves and coordinated arm-pair prepositioning."""

from __future__ import annotations

import concurrent.futures
import json
import math
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import portal

from yam_data_collection.config import DatasetConfig


_ARM_JOINTS = 6
_DOFS = 7
_POSITION_TOLERANCE_RAD = 0.12
_GRIPPER_TOLERANCE = 0.12
_MAX_JOINT_SPEED_RAD_S = 1.0
_MAX_GRIPPER_SPEED_S = 0.25
_RATE_HZ = 25.0
_TELEMETRY_STALE_S = 1.0
_ROOT = Path(__file__).resolve().parent.parent
_YAM_MODEL = _ROOT / "i2rt/i2rt/robot_models/arm/yam/v1/yam.xml"
_PORTS = {
    "left": {"follower": 1235, "leader": 1245},
    "right": {"follower": 1234, "leader": 1244},
}


class PreprogrammedMotionError(RuntimeError):
    """A requested preprogrammed move is unsafe or was not completed."""


@dataclass(frozen=True)
class MotionPlan:
    side: str
    start: tuple[float, ...] | None
    target: tuple[float, ...]
    duration_s: float
    leader_start: tuple[float, ...] | None
    digest: str
    from_current: bool = False


def _vector(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != _DOFS:
        raise ValueError(f"plan.{name} must contain exactly seven values: six joint angles in radians and one gripper")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"plan.{name} must contain numeric joint positions")
    values = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in values):
        raise ValueError(f"plan.{name} must contain finite joint positions")
    if not 0.0 <= values[-1] <= 1.0:
        raise ValueError(f"plan.{name}[6] gripper must be normalized to [0, 1]")
    return values


def load_motion_plan(path: str | Path, *, side: str, pair: bool, from_current: bool = False) -> MotionPlan:
    plan_path = Path(path).expanduser().resolve()
    try:
        content = plan_path.read_bytes()
        raw = json.loads(content)
    except OSError as exc:
        raise ValueError(f"cannot read motion plan {plan_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"motion plan is not valid JSON: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("motion plan must be a JSON object")
    if raw.get("side") != side:
        raise ValueError(f"motion plan side must be {side!r}; do not reuse another arm's measurements")
    if pair and not from_current and "leader_start" not in raw:
        raise ValueError("paired preposition plan must include a separately measured leader_start array")
    if pair and not from_current and "follower_start" not in raw:
        raise ValueError("paired preposition plan must include a separately measured follower_start array")
    start_field = "follower_start" if pair else "start"
    start = None if from_current else _vector(raw.get(start_field), start_field)
    target = _vector(raw.get("target"), "target")
    leader_start = _vector(raw["leader_start"], "leader_start") if pair and not from_current else None
    duration = raw.get("duration_s")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(float(duration)):
        raise ValueError("plan.duration_s must be a finite number")
    duration_s = float(duration)
    if not 0.5 <= duration_s <= 120.0:
        raise ValueError("plan.duration_s must be between 0.5 and 120 seconds")
    if leader_start is not None and abs(leader_start[-1] - target[-1]) > _GRIPPER_TOLERANCE:
        raise ValueError(
            "the leader teaching-handle gripper is passive: manually place it near target[6] before prepositioning"
        )
    _validate_joint_limits(target[:_ARM_JOINTS])
    if start is not None:
        _validate_joint_limits(start[:_ARM_JOINTS])
        _validate_motion_speed(start, target, duration_s, "follower", include_gripper=True)
    if leader_start is not None:
        _validate_joint_limits(leader_start[:_ARM_JOINTS])
        _validate_motion_speed(leader_start, target, duration_s, "leader", include_gripper=False)
    return MotionPlan(
        side=side,
        start=start,
        target=target,
        duration_s=duration_s,
        leader_start=leader_start,
        digest=sha256(content).hexdigest(),
        from_current=from_current,
    )


def _resolve_motion_start(
    plan: MotionPlan, follower_state: Mapping[str, Any], leader_state: Mapping[str, Any] | None = None
) -> MotionPlan:
    if not plan.from_current:
        return plan
    start = _vector(follower_state["position"].tolist(), "current_follower")
    _validate_joint_limits(start[:_ARM_JOINTS], name="current follower")
    leader_start = None
    if leader_state is not None:
        leader_start = _vector(leader_state["position"].tolist(), "current_leader")
        _validate_joint_limits(leader_start[:_ARM_JOINTS], name="current leader")
    return replace(plan, start=start, leader_start=leader_start)


def _validate_joint_limits(target: tuple[float, ...], *, name: str = "target") -> None:
    root = ET.parse(_YAM_MODEL).getroot()
    limits: dict[str, tuple[float, float]] = {}
    for joint in root.iter("joint"):
        joint_name, range_text = joint.get("name"), joint.get("range")
        if joint_name in {f"joint{index}" for index in range(1, _ARM_JOINTS + 1)} and range_text:
            lower, upper = (float(item) for item in range_text.split())
            limits[joint_name] = (lower - 0.15, upper + 0.15)
    if set(limits) != {f"joint{index}" for index in range(1, _ARM_JOINTS + 1)}:
        raise RuntimeError(f"could not load all six YAM joint limits from {_YAM_MODEL}")
    invalid = [
        f"joint{index}={value:.4f} rad (allowed {limits[f'joint{index}'][0]:.4f}..{limits[f'joint{index}'][1]:.4f})"
        for index, value in enumerate(target, 1)
        if not limits[f"joint{index}"][0] <= value <= limits[f"joint{index}"][1]
    ]
    if invalid:
        raise ValueError(f"{name} exceeds configured YAM joint limits: " + "; ".join(invalid))


def _validate_motion_speed(
    start: tuple[float, ...] | np.ndarray,
    target: tuple[float, ...] | np.ndarray,
    duration_s: float,
    name: str,
    *,
    include_gripper: bool,
) -> None:
    start_values, target_values = np.asarray(start), np.asarray(target)
    max_joint_speed = float(np.max(np.abs(target_values[:_ARM_JOINTS] - start_values[:_ARM_JOINTS]))) / duration_s
    if max_joint_speed > _MAX_JOINT_SPEED_RAD_S:
        raise ValueError(
            f"plan.duration_s is too short for {name}: joint speed {max_joint_speed:.3f} rad/s exceeds "
            f"the {_MAX_JOINT_SPEED_RAD_S:.2f} rad/s limit"
        )
    if include_gripper:
        gripper_speed = abs(float(target_values[-1] - start_values[-1])) / duration_s
        if gripper_speed > _MAX_GRIPPER_SPEED_S:
            raise ValueError(
                f"plan.duration_s is too short for {name}: gripper speed {gripper_speed:.3f}/s exceeds "
                f"the {_MAX_GRIPPER_SPEED_S:.2f}/s limit"
            )


def _connect(port: int, timeout_s: float) -> portal.Client:
    client = portal.Client(f"127.0.0.1:{port}", autoconn=False)
    try:
        if not client.connect(timeout=timeout_s):
            raise TimeoutError(f"could not connect to Portal port {port}")
        return client
    except Exception:
        client.close(timeout=1.0)
        raise


def _rpc(client: portal.Client, method: str, timeout_s: float, *args: Any) -> Any:
    return getattr(client, method)(*args).result(timeout=timeout_s)


def _telemetry(client: portal.Client, timeout_s: float) -> dict[str, Any]:
    result = _rpc(client, "get_telemetry", timeout_s)
    if not isinstance(result, Mapping):
        raise PreprogrammedMotionError("telemetry response must be a mapping")
    position = np.asarray(result.get("position"), dtype=np.float64)
    if position.shape != (_DOFS,) or not np.isfinite(position).all():
        raise PreprogrammedMotionError("telemetry must contain seven finite position values")
    sequence = int(result.get("sequence", -1))
    timestamp_ns = int(result.get("timestamp_ns", 0))
    if sequence < 0 or timestamp_ns <= 0:
        raise PreprogrammedMotionError("telemetry sample is missing its sequence or timestamp")
    age_s = (time.time_ns() - timestamp_ns) / 1e9
    if not 0.0 <= age_s <= _TELEMETRY_STALE_S:
        raise PreprogrammedMotionError(f"telemetry is stale ({age_s:.3f}s old)")
    return {
        **result,
        "position": position,
        "sequence": sequence,
        "timestamp_ns": timestamp_ns,
    }


def _check_leader_idle(state: Mapping[str, Any]) -> None:
    if "synchronized" not in state:
        raise PreprogrammedMotionError("leader telemetry lacks synchronization status; refusing to move")
    if bool(state["synchronized"]):
        raise PreprogrammedMotionError("leader is synchronized; disable teleoperation before moving")
    buttons = np.asarray(state.get("button_state", []), dtype=np.float64)
    if buttons.shape != (2,) or not np.isfinite(buttons).all():
        raise PreprogrammedMotionError("leader telemetry lacks its two valid button states")
    if buttons[0] > 0.5:
        raise PreprogrammedMotionError("leader button 0 is pressed; release it before moving")


def _check_start(actual: np.ndarray, expected: tuple[float, ...], name: str) -> None:
    arm_error = float(np.max(np.abs(actual[:_ARM_JOINTS] - np.asarray(expected[:_ARM_JOINTS]))))
    gripper_error = abs(float(actual[-1]) - expected[-1])
    if arm_error > _POSITION_TOLERANCE_RAD or gripper_error > _GRIPPER_TOLERANCE:
        raise PreprogrammedMotionError(
            f"{name} is not at its measured start pose "
            f"(max joint error {arm_error:.4f} rad, gripper error {gripper_error:.4f})"
        )


def validate_motion_endpoints(config: DatasetConfig) -> str:
    side = str(config.raw["single_arm"]["side"])
    for role in ("follower", "leader"):
        endpoint = config.telemetry[f"{side}_{role}"]
        if endpoint.host not in {"localhost", "127.0.0.1"} or endpoint.port != _PORTS[side][role]:
            raise PreprogrammedMotionError(
                "motion commands require the documented local arm ports in the configuration"
            )
    return side


def _settle_follower(
    client: portal.Client,
    leader_telemetry: portal.Client,
    target: tuple[float, ...],
    timeout_s: float,
    cancelled: threading.Event,
) -> np.ndarray:
    desired = np.asarray(target)
    deadline = time.monotonic() + 5.0
    next_command = 0.0
    while time.monotonic() < deadline:
        if cancelled.is_set():
            raise PreprogrammedMotionError("follower move was cancelled")
        _check_leader_idle(_telemetry(leader_telemetry, timeout_s))
        state = _telemetry(client, timeout_s)
        position = state["position"]
        if (
            float(np.max(np.abs(position[:_ARM_JOINTS] - desired[:_ARM_JOINTS]))) <= _POSITION_TOLERANCE_RAD
            and abs(float(position[-1] - desired[-1])) <= _GRIPPER_TOLERANCE
        ):
            return position.copy()
        if time.monotonic() >= next_command:
            _rpc(client, "command_joint_pos", timeout_s, desired)
            next_command = time.monotonic() + 0.1
        time.sleep(0.02)
    raise PreprogrammedMotionError("follower did not reach its target within the position tolerance")


def _move_follower(
    client: portal.Client,
    leader_telemetry: portal.Client,
    start: tuple[float, ...],
    target: tuple[float, ...],
    duration_s: float,
    timeout_s: float,
    cancelled: threading.Event,
) -> np.ndarray:
    state = _telemetry(client, timeout_s)
    _check_start(state["position"], start, "follower")
    _validate_motion_speed(state["position"], target, duration_s, "follower", include_gripper=True)
    leader_state = _telemetry(leader_telemetry, timeout_s)
    _check_leader_idle(leader_state)
    current = state["position"].copy()
    destination = np.asarray(target)
    steps = max(1, math.ceil(duration_s * _RATE_HZ))
    period_s = duration_s / steps
    last_leader_check = time.monotonic()
    last_sequence = state["sequence"]
    for step in range(1, steps + 1):
        if cancelled.is_set():
            raise PreprogrammedMotionError("follower move was cancelled")
        if step == 1 or time.monotonic() - last_leader_check >= 0.25:
            leader_state = _telemetry(leader_telemetry, timeout_s)
            _check_leader_idle(leader_state)
            follower_state = _telemetry(client, timeout_s)
            if follower_state["sequence"] == last_sequence and step > 1:
                raise PreprogrammedMotionError("follower telemetry stopped advancing during motion")
            last_sequence = follower_state["sequence"]
            last_leader_check = time.monotonic()
        if step == 1 or step % 25 == 0:
            print(f"Follower move progress: {step}/{steps} setpoints")
        alpha = step / steps
        setpoint = (1.0 - alpha) * current + alpha * destination
        _rpc(client, "command_joint_pos", timeout_s, setpoint)
        if cancelled.wait(period_s):
            raise PreprogrammedMotionError("follower move was cancelled")
    return _settle_follower(client, leader_telemetry, target, timeout_s, cancelled)


def _move_leader(
    client: portal.Client,
    plan: MotionPlan,
    timeout_s: float,
    request_id: int,
    cancelled: threading.Event,
) -> np.ndarray:
    assert plan.leader_start is not None
    if cancelled.is_set():
        raise PreprogrammedMotionError("leader move was cancelled before starting")
    state = _telemetry(client, timeout_s)
    _check_leader_idle(state)
    _check_start(state["position"], plan.leader_start, "leader")
    if abs(float(state["position"][-1] - plan.target[-1])) > _GRIPPER_TOLERANCE:
        raise PreprogrammedMotionError(
            "leader teaching-handle gripper must be manually placed at the target opening before the paired move"
        )
    _validate_motion_speed(state["position"], plan.target, plan.duration_s, "leader", include_gripper=False)
    if cancelled.is_set():
        raise PreprogrammedMotionError("leader move was cancelled before starting")
    result = _rpc(
        client,
        "preprogrammed_move",
        plan.duration_s + 15.0,
        request_id,
        np.asarray(plan.leader_start[:_ARM_JOINTS]),
        np.asarray(plan.target[:_ARM_JOINTS]),
        plan.duration_s,
        _POSITION_TOLERANCE_RAD,
    )
    if cancelled.is_set():
        raise PreprogrammedMotionError("leader move was cancelled")
    if not isinstance(result, Mapping) or int(result.get("request_id", -1)) != request_id:
        raise PreprogrammedMotionError("leader controller returned an invalid motion result")
    position = np.asarray(result.get("position"), dtype=np.float64)
    if position.shape != (_DOFS,) or not np.isfinite(position).all():
        raise PreprogrammedMotionError("leader controller returned an invalid final telemetry sample")
    return position


def _cancel_leader(side: str, request_id: int, timeout_s: float) -> None:
    client: portal.Client | None = None
    try:
        client = _connect(_PORTS[side]["leader"], timeout_s)
        _rpc(client, "cancel_preprogrammed_move", timeout_s, request_id)
    except Exception as exc:
        print(f"Warning: leader cancellation request failed: {exc}")
    finally:
        if client is not None:
            client.close(timeout=1.0)


def move_follower(plan: MotionPlan, *, execute: bool, timeout_s: float = 2.0) -> dict[str, Any]:
    """Preview or move one follower A-to-B while guarding leader synchronization."""
    follower: portal.Client | None = None
    leader: portal.Client | None = None
    try:
        follower = _connect(_PORTS[plan.side]["follower"], timeout_s)
        leader = _connect(_PORTS[plan.side]["leader"], timeout_s)
        follower_state, leader_state = (
            _telemetry(follower, timeout_s),
            _telemetry(leader, timeout_s),
        )
        plan = _resolve_motion_start(plan, follower_state)
        assert plan.start is not None
        _check_start(follower_state["position"], plan.start, "follower")
        _validate_motion_speed(
            follower_state["position"], plan.target, plan.duration_s, "follower", include_gripper=True
        )
        _check_leader_idle(leader_state)
        print(f"Preview follower {plan.side}: {plan.start} -> {plan.target}; {plan.duration_s:.2f} s.")
        if plan.from_current:
            print("Start mode: current telemetry; recorded start fields are ignored.")
        if not execute:
            print("Preview only: no robot position command was sent.")
            return {
                "executed": False,
                "follower_start": follower_state["position"].tolist(),
                "start_mode": "current" if plan.from_current else "recorded",
            }
        cancelled = threading.Event()
        try:
            final = _move_follower(
                follower,
                leader,
                plan.start,
                plan.target,
                plan.duration_s,
                timeout_s,
                cancelled,
            )
            return {
                "executed": True,
                "start_mode": "current" if plan.from_current else "recorded",
                "follower_start": follower_state["position"].tolist(),
                "follower_final": final.tolist(),
            }
        finally:
            try:
                _rpc(follower, "enter_gravity_comp_idle", timeout_s)
            except Exception as exc:
                print(f"Warning: could not release the follower to gravity-compensation idle: {exc}")
    finally:
        if leader is not None:
            leader.close(timeout=1.0)
        if follower is not None:
            follower.close(timeout=1.0)


def preposition_pair(
    config: DatasetConfig,
    plan_path: str | Path,
    *,
    timeout_s: float = 2.0,
    execute: bool = True,
    from_current: bool = False,
) -> dict[str, Any]:
    """Move the selected leader/follower pair together before cameras start."""
    side = validate_motion_endpoints(config)
    plan = load_motion_plan(plan_path, side=side, pair=True, from_current=from_current)
    follower: portal.Client | None = None
    leader: portal.Client | None = None
    leader_command: portal.Client | None = None
    execution_started = False
    released = False
    cancelled = threading.Event()
    request_id = max(1, time.time_ns())
    try:
        follower = _connect(_PORTS[side]["follower"], timeout_s)
        leader = _connect(_PORTS[side]["leader"], timeout_s)
        follower_before = _telemetry(follower, timeout_s)
        leader_before = _telemetry(leader, timeout_s)
        plan = _resolve_motion_start(plan, follower_before, leader_before)
        assert plan.start is not None and plan.leader_start is not None
        if not bool(leader_before.get("preprogrammed_motion_available", False)):
            raise PreprogrammedMotionError(
                "leader motion API is disabled or outdated; in a safe maintenance window, restart it using "
                "yam_single_arm.sh leader-motion --side SIDE --confirm-motion"
            )
        readiness = _rpc(leader, "preprogrammed_motion_ready", timeout_s)
        if not isinstance(readiness, Mapping) or readiness.get("version") != 1 or not readiness.get("available"):
            raise PreprogrammedMotionError("selected leader does not expose the preprogrammed-motion API")
        _check_start(follower_before["position"], plan.start, "follower")
        _check_start(leader_before["position"], plan.leader_start, "leader")
        _check_leader_idle(leader_before)
        _validate_motion_speed(
            follower_before["position"], plan.target, plan.duration_s, "follower", include_gripper=True
        )
        _validate_motion_speed(
            leader_before["position"], plan.target, plan.duration_s, "leader", include_gripper=False
        )
        if abs(float(leader_before["position"][-1] - plan.target[-1])) > _GRIPPER_TOLERANCE:
            raise PreprogrammedMotionError(
                "leader teaching-handle gripper is passive; manually place it at target[6] before prepositioning"
            )
        if not execute:
            return {
                "executed": False,
                "start_mode": "current" if plan.from_current else "recorded",
                "side": side,
                "target": list(plan.target),
                "duration_s": plan.duration_s,
                "plan_sha256": plan.digest,
                "follower_start": follower_before["position"].tolist(),
                "leader_start": leader_before["position"].tolist(),
            }
        leader_command = _connect(_PORTS[side]["leader"], timeout_s)
        print(
            f"Prepositioning {side} leader and follower to the same six-joint target over "
            f"{plan.duration_s:.2f} s. Keep hands clear and the emergency stop ready."
        )
        execution_started = True
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            follower_future = executor.submit(
                _move_follower,
                follower,
                leader,
                plan.start,
                plan.target,
                plan.duration_s,
                timeout_s,
                cancelled,
            )
            leader_future = executor.submit(_move_leader, leader_command, plan, timeout_s, request_id, cancelled)
            futures = (follower_future, leader_future)
            try:
                failure: BaseException | None = None
                while not all(future.done() for future in futures):
                    for future in futures:
                        if future.done() and future.exception() is not None:
                            failure = future.exception()
                            break
                    if failure is not None:
                        raise PreprogrammedMotionError(
                            f"paired prepositioning failed; collection will not start: {failure}"
                        )
                    time.sleep(0.01)
                follower_after = follower_future.result()
                leader_after = leader_future.result()
            except BaseException:
                cancelled.set()
                _cancel_leader(side, request_id, timeout_s)
                for future in futures:
                    try:
                        future.result(timeout=8.0)
                    except BaseException:
                        pass
                raise
        _rpc(follower, "enter_gravity_comp_idle", timeout_s)
        released = True
        follower_final_check = _telemetry(follower, timeout_s)["position"]
        leader_final_state = _telemetry(leader, timeout_s)
        _check_leader_idle(leader_final_state)
        leader_final_check = leader_final_state["position"]
        for label, position in (
            ("follower", follower_final_check),
            ("leader", leader_final_check),
        ):
            arm_error = float(np.max(np.abs(position[:_ARM_JOINTS] - np.asarray(plan.target[:_ARM_JOINTS]))))
            if arm_error > _POSITION_TOLERANCE_RAD:
                raise PreprogrammedMotionError(f"{label} drifted from the shared target by {arm_error:.4f} rad")
        if abs(float(follower_final_check[-1] - leader_final_check[-1])) > _GRIPPER_TOLERANCE:
            raise PreprogrammedMotionError("leader and follower gripper telemetry do not match after prepositioning")
        if (
            float(np.max(np.abs(follower_final_check[:_ARM_JOINTS] - leader_final_check[:_ARM_JOINTS])))
            > _POSITION_TOLERANCE_RAD
        ):
            raise PreprogrammedMotionError("leader and follower arm joints do not match after prepositioning")
        if any(
            abs(float(position[-1]) - plan.target[-1]) > _GRIPPER_TOLERANCE
            for position in (follower_final_check, leader_final_check)
        ):
            raise PreprogrammedMotionError("gripper telemetry does not match target[6] after prepositioning")
        return {
            "executed": True,
            "start_mode": "current" if plan.from_current else "recorded",
            "side": side,
            "plan_sha256": plan.digest,
            "duration_s": plan.duration_s,
            "target": list(plan.target),
            "leader_before": leader_before["position"].tolist(),
            "follower_before": follower_before["position"].tolist(),
            "leader_final": leader_after.tolist(),
            "follower_final": follower_after.tolist(),
            "verified_leader_final": leader_final_check.tolist(),
            "verified_follower_final": follower_final_check.tolist(),
            "leader_synchronized": False,
        }
    except BaseException:
        if execution_started and leader is not None:
            cancelled.set()
            _cancel_leader(side, request_id, timeout_s)
        raise
    finally:
        if execution_started and not released and follower is not None:
            try:
                _rpc(follower, "enter_gravity_comp_idle", timeout_s)
            except Exception as exc:
                print(f"Warning: could not release the follower to gravity-compensation idle: {exc}")
        if leader is not None:
            leader.close(timeout=1.0)
        if leader_command is not None:
            leader_command.close(timeout=1.0)
        if follower is not None:
            follower.close(timeout=1.0)


def read_pair_poses(config: DatasetConfig, *, timeout_s: float = 2.0) -> dict[str, Any]:
    """Read the selected leader and follower for a measured joint-space plan."""
    side = validate_motion_endpoints(config)
    follower: portal.Client | None = None
    leader: portal.Client | None = None
    try:
        follower = _connect(_PORTS[side]["follower"], timeout_s)
        leader = _connect(_PORTS[side]["leader"], timeout_s)
        follower_state = _telemetry(follower, timeout_s)
        leader_state = _telemetry(leader, timeout_s)
        payload = {
            "side": side,
            "follower": {
                "position": follower_state["position"].tolist(),
                "joint_unit": "radian",
                "gripper_unit": "normalized_0_to_1",
                "sequence": follower_state["sequence"],
                "timestamp_ns": follower_state["timestamp_ns"],
            },
            "leader": {
                "position": leader_state["position"].tolist(),
                "joint_unit": "radian",
                "gripper_unit": "normalized_0_to_1",
                "sequence": leader_state["sequence"],
                "timestamp_ns": leader_state["timestamp_ns"],
                "synchronized": bool(leader_state.get("synchronized", False)),
                "preprogrammed_motion_available": bool(leader_state.get("preprogrammed_motion_available", False)),
                "button_state": np.asarray(leader_state.get("button_state", []), dtype=np.float64).tolist(),
            },
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload
    finally:
        if leader is not None:
            leader.close(timeout=1.0)
        if follower is not None:
            follower.close(timeout=1.0)
