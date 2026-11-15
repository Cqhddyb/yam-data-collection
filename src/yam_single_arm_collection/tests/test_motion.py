from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from yam_single_arm_collection import cli, motion


def _values(joints: float, gripper: float = 0.5) -> list[float]:
    return [joints] * 6 + [gripper]


def _plan_file(tmp_path: Path, **overrides: Any) -> Path:
    plan = {
        "side": "right",
        "start": _values(0.0),
        "target": _values(0.1),
        "duration_s": 2.0,
    }
    plan.update(overrides)
    path = tmp_path / "move.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def test_single_move_plan_requires_side_specific_measured_start_and_target(
    tmp_path: Path,
) -> None:
    plan = motion.load_motion_plan(_plan_file(tmp_path), side="right", pair=False)
    assert plan.start == tuple(_values(0.0))
    assert plan.target == tuple(_values(0.1))
    assert len(plan.digest) == 64
    with pytest.raises(ValueError, match="do not reuse"):
        motion.load_motion_plan(_plan_file(tmp_path), side="left", pair=False)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"start": _values(0.0)[:-1]}, "exactly"),
        ({"target": _values(0.1, gripper=1.1)}, "normalized"),
        ({"target": _values(0.1, gripper=float("nan"))}, "finite"),
        ({"duration_s": 0.1}, "between 0.5 and 120"),
        ({"target": _values(0.5005), "duration_s": 0.5}, "joint speed.*1.00 rad/s limit"),
        ({"target": _values(0.5, gripper=0.626), "duration_s": 0.5}, "gripper speed.*0.25/s limit"),
        ({"target": _values(10.0)}, "joint limits"),
    ],
)
def test_single_move_plan_rejects_invalid_values(tmp_path: Path, overrides: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        motion.load_motion_plan(_plan_file(tmp_path, **overrides), side="right", pair=False)


@pytest.mark.parametrize("pair", [False, True])
@pytest.mark.parametrize("joint_speed", [0.368, 1.0])
def test_motion_plan_accepts_joint_speeds_up_to_one_rad_per_second(
    tmp_path: Path, pair: bool, joint_speed: float
) -> None:
    target = _values(joint_speed * 0.5)
    path = _plan_file(
        tmp_path,
        follower_start=_values(0.0),
        leader_start=_values(0.0),
        target=target,
        duration_s=0.5,
    )
    plan = motion.load_motion_plan(path, side="right", pair=pair)
    assert plan.target == tuple(target)


def test_paired_plan_requires_separate_leader_start_and_matching_passive_gripper(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="leader_start"):
        motion.load_motion_plan(_plan_file(tmp_path), side="right", pair=True)
    with pytest.raises(ValueError, match="passive"):
        motion.load_motion_plan(
            _plan_file(
                tmp_path,
                follower_start=_values(0.0),
                leader_start=_values(0.0, gripper=0.1),
            ),
            side="right",
            pair=True,
        )
    plan = motion.load_motion_plan(
        _plan_file(tmp_path, follower_start=_values(0.0), leader_start=_values(0.0)),
        side="right",
        pair=True,
    )
    assert plan.leader_start == tuple(_values(0.0))


class _Future:
    def __init__(self, value: Any = None, error: BaseException | None = None) -> None:
        self.value = value
        self.error = error

    def result(self, timeout: float) -> Any:
        del timeout
        if self.error is not None:
            raise self.error
        return self.value


class _Client:
    def __init__(self, port: int, arm_state: list[float], *, leader: bool = False) -> None:
        self.port = port
        self.position = np.asarray(arm_state, dtype=np.float64)
        self.leader = leader
        self.sequence = 1
        self.commands: list[np.ndarray] = []
        self.releases = 0
        self.cancelled: list[int] = []

    def connect(self, timeout: float) -> bool:
        del timeout
        return True

    def close(self, timeout: float) -> None:
        del timeout

    def get_telemetry(self) -> _Future:
        self.sequence += 1
        state = {
            "position": self.position.tolist(),
            "sequence": self.sequence,
            "timestamp_ns": time.time_ns(),
        }
        if self.leader:
            state.update(synchronized=False, button_state=[0.0, 0.0], preprogrammed_motion_available=True)
        return _Future(state)

    def preprogrammed_motion_ready(self) -> _Future:
        return _Future({"version": 1, "available": True})

    def command_joint_pos(self, target: np.ndarray) -> _Future:
        command = np.asarray(target, dtype=np.float64)
        self.commands.append(command.copy())
        self.position = command.copy()
        return _Future()

    def enter_gravity_comp_idle(self) -> _Future:
        self.releases += 1
        return _Future()

    def preprogrammed_move(
        self,
        request_id: int,
        start: np.ndarray,
        target: np.ndarray,
        duration: float,
        tolerance: float,
    ) -> _Future:
        del start, duration, tolerance
        self.position[:6] = np.asarray(target, dtype=np.float64)
        return _Future({"request_id": request_id, "position": self.position.tolist()})

    def cancel_preprogrammed_move(self, request_id: int) -> _Future:
        self.cancelled.append(request_id)
        return _Future(True)


@pytest.fixture
def fake_clients(monkeypatch: pytest.MonkeyPatch) -> dict[int, _Client]:
    clients = {
        1234: _Client(1234, _values(0.0)),
        1244: _Client(1244, _values(0.0), leader=True),
    }
    monkeypatch.setattr(
        motion.portal,
        "Client",
        lambda address, **kwargs: clients[int(address.rsplit(":", 1)[1])],
    )
    return clients


def test_single_follower_move_previews_without_command(tmp_path: Path, fake_clients: dict[int, _Client]) -> None:
    plan = motion.load_motion_plan(_plan_file(tmp_path), side="right", pair=False)

    result = motion.move_follower(plan, execute=False)

    assert not result["executed"]
    assert fake_clients[1234].commands == []
    assert fake_clients[1234].releases == 0


def test_single_follower_move_sends_only_after_execute_and_releases(
    tmp_path: Path, fake_clients: dict[int, _Client], monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = motion.load_motion_plan(_plan_file(tmp_path), side="right", pair=False)
    final = np.asarray(_values(0.1))

    def fake_move(
        client: _Client,
        leader: _Client,
        start: tuple[float, ...],
        target: tuple[float, ...],
        duration: float,
        timeout: float,
        cancelled: threading.Event,
    ) -> np.ndarray:
        del leader, start, duration, timeout, cancelled
        client.position = np.asarray(target)
        return final

    monkeypatch.setattr(motion, "_move_follower", fake_move)

    result = motion.move_follower(plan, execute=True)

    assert result["follower_final"] == final.tolist()
    assert fake_clients[1234].commands == []
    assert fake_clients[1234].releases == 1


def test_paired_preposition_moves_both_arms_to_same_target_before_returning(
    tmp_path: Path,
    config_factory: Any,
    fake_clients: dict[int, _Client],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = config_factory("right")
    path = _plan_file(
        tmp_path,
        follower_start=_values(0.0),
        leader_start=_values(0.0),
    )
    plan = motion.load_motion_plan(path, side="right", pair=True)
    targets: list[tuple[tuple[float, ...], tuple[float, ...]]] = []

    def move_follower(
        follower: _Client,
        leader: _Client,
        start: tuple[float, ...],
        target: tuple[float, ...],
        duration: float,
        timeout: float,
        cancelled: threading.Event,
    ) -> np.ndarray:
        del leader, start, duration, timeout, cancelled
        follower.position = np.asarray(target)
        targets.append((tuple(follower.position), tuple(target)))
        return follower.position.copy()

    def move_leader(
        leader: _Client,
        paired_plan: motion.MotionPlan,
        timeout: float,
        request_id: int,
        cancelled: threading.Event,
    ) -> np.ndarray:
        del timeout, request_id, cancelled
        leader.position = np.asarray(paired_plan.target)
        targets.append((tuple(leader.position), paired_plan.target))
        return leader.position.copy()

    monkeypatch.setattr(motion, "_move_follower", move_follower)
    monkeypatch.setattr(motion, "_move_leader", move_leader)

    result = motion.preposition_pair(config, path)

    assert len(targets) == 2
    assert all(actual == expected == plan.target for actual, expected in targets)
    assert result["verified_leader_final"] == list(plan.target)
    assert result["verified_follower_final"] == list(plan.target)
    assert result["leader_synchronized"] is False
    assert fake_clients[1234].releases == 1


def test_failed_follower_move_cancels_leader_and_refuses_preposition(
    tmp_path: Path,
    config_factory: Any,
    fake_clients: dict[int, _Client],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = config_factory("right")
    path = _plan_file(tmp_path, follower_start=_values(0.0), leader_start=_values(0.0))
    entered_leader = threading.Event()

    def fail_follower(*args: Any, **kwargs: Any) -> np.ndarray:
        del args, kwargs
        raise motion.PreprogrammedMotionError("synthetic follower fault")

    def wait_for_cancellation(
        leader: _Client,
        plan: motion.MotionPlan,
        timeout: float,
        request_id: int,
        cancelled: threading.Event,
    ) -> np.ndarray:
        del leader, plan, timeout, request_id
        entered_leader.set()
        assert cancelled.wait(timeout=2.0)
        raise motion.PreprogrammedMotionError("leader cancelled")

    monkeypatch.setattr(motion, "_move_follower", fail_follower)
    monkeypatch.setattr(motion, "_move_leader", wait_for_cancellation)

    with pytest.raises(motion.PreprogrammedMotionError, match="synthetic follower fault"):
        motion.preposition_pair(config, path)

    assert entered_leader.is_set()
    assert fake_clients[1234].releases == 1


def test_leader_cancellation_uses_a_separate_portal_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active_client = _Client(1244, _values(0.0), leader=True)
    cancellation_client = _Client(1244, _values(0.0), leader=True)
    monkeypatch.setattr(motion, "_connect", lambda port, timeout: cancellation_client)

    motion._cancel_leader("right", 42, 0.5)

    assert active_client.cancelled == []
    assert cancellation_client.cancelled == [42]


def test_read_pair_poses_reads_selected_follower_and_unsynchronized_leader(
    config_factory: Any,
    fake_clients: dict[int, _Client],
    capsys: Any,
) -> None:
    config = config_factory("right")

    result = motion.read_pair_poses(config)

    assert result["follower"]["position"] == _values(0.0)
    assert result["leader"]["position"] == _values(0.0)
    assert result["leader"]["synchronized"] is False
    assert '"side": "right"' in capsys.readouterr().out


@pytest.mark.parametrize("buttons", [np.array([0.0, 1.0]), [np.array(0.0), np.array(1.0)]])
def test_read_pair_poses_serializes_numpy_button_states(
    buttons: Any,
    config_factory: Any,
    fake_clients: dict[int, _Client],
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
) -> None:
    leader = fake_clients[1244]
    original = leader.get_telemetry

    def numpy_telemetry() -> _Future:
        state = original().value
        state["button_state"] = buttons
        return _Future(state)

    monkeypatch.setattr(leader, "get_telemetry", numpy_telemetry)

    result = motion.read_pair_poses(config_factory("right"))

    assert json.loads(capsys.readouterr().out) == result
    assert result["leader"]["button_state"] == [0.0, 1.0]
    assert fake_clients[1234].releases == 0
    assert fake_clients[1244].releases == 0
    assert not fake_clients[1234].commands
    assert not fake_clients[1244].commands


def test_paired_preview_only_reads_without_releasing_or_moving(
    tmp_path: Path, config_factory: Any, fake_clients: dict[int, _Client]
) -> None:
    path = _plan_file(tmp_path, follower_start=_values(0.0), leader_start=_values(0.0))
    result = motion.preposition_pair(config_factory("right"), path, execute=False)
    assert result["executed"] is False
    assert all(client.releases == 0 and not client.commands for client in fake_clients.values())


def test_disabled_leader_api_fails_before_control_requests(
    tmp_path: Path, config_factory: Any, fake_clients: dict[int, _Client], monkeypatch: pytest.MonkeyPatch
) -> None:
    leader = fake_clients[1244]
    original = leader.get_telemetry

    def old_telemetry() -> _Future:
        state = original().value
        state.pop("preprogrammed_motion_available")
        return _Future(state)

    monkeypatch.setattr(leader, "get_telemetry", old_telemetry)
    monkeypatch.setattr(leader, "preprogrammed_motion_ready", lambda: pytest.fail("unknown RPC must not be probed"))
    path = _plan_file(tmp_path, follower_start=_values(0.0), leader_start=_values(0.0))
    with pytest.raises(motion.PreprogrammedMotionError, match="leader-motion"):
        motion.preposition_pair(config_factory("right"), path)
    assert all(client.releases == 0 and not client.commands for client in fake_clients.values())


def test_preposition_rejects_pair_disagreement_even_with_both_near_target(
    tmp_path: Path, config_factory: Any, fake_clients: dict[int, _Client], monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _plan_file(tmp_path, follower_start=_values(0.0), leader_start=_values(0.0))

    def follower_move(*args: Any) -> np.ndarray:
        fake_clients[1234].position = np.asarray(_values(0.0))
        return fake_clients[1234].position.copy()

    def leader_move(*args: Any) -> np.ndarray:
        fake_clients[1244].position = np.asarray(_values(0.2))
        return fake_clients[1244].position.copy()

    monkeypatch.setattr(motion, "_move_follower", follower_move)
    monkeypatch.setattr(motion, "_move_leader", leader_move)
    with pytest.raises(motion.PreprogrammedMotionError, match="arm joints do not match"):
        motion.preposition_pair(config_factory("right"), path)


def test_real_follower_interpolation_reaches_target_and_limits_each_step(
    tmp_path: Path, fake_clients: dict[int, _Client], monkeypatch: pytest.MonkeyPatch
) -> None:
    class ImmediateEvent:
        def is_set(self) -> bool:
            return False

        def wait(self, timeout: float) -> bool:
            return False

    plan = motion.load_motion_plan(_plan_file(tmp_path), side="right", pair=False)
    final = motion._move_follower(
        fake_clients[1234], fake_clients[1244], plan.start, plan.target, plan.duration_s, 0.5, ImmediateEvent()
    )
    positions = np.vstack([plan.start, *fake_clients[1234].commands])
    assert final == pytest.approx(plan.target)
    assert np.max(np.abs(np.diff(positions[:, :6], axis=0))) <= motion._MAX_JOINT_SPEED_RAD_S / motion._RATE_HZ


def test_cancelled_follower_never_sends_a_position_command(fake_clients: dict[int, _Client]) -> None:
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(motion.PreprogrammedMotionError, match="cancelled"):
        motion._move_follower(
            fake_clients[1234], fake_clients[1244], tuple(_values(0)), tuple(_values(0.1)), 2.0, 0.5, cancelled
        )
    assert not fake_clients[1234].commands


def test_motion_rejects_configuration_with_a_different_command_endpoint(config_factory: Any) -> None:
    config = config_factory("right")
    config.telemetry["right_follower"] = replace(config.telemetry["right_follower"], port=1235)
    with pytest.raises(motion.PreprogrammedMotionError, match="local arm ports"):
        motion.validate_motion_endpoints(config)


@pytest.mark.parametrize("pair", [False, True])
def test_from_current_plan_can_omit_recorded_starts(tmp_path: Path, pair: bool) -> None:
    path = tmp_path / "target.json"
    path.write_text(json.dumps({"side": "right", "target": _values(0.1), "duration_s": 2.0}), encoding="utf-8")
    plan = motion.load_motion_plan(path, side="right", pair=pair, from_current=True)
    assert plan.from_current
    assert plan.start is None and plan.leader_start is None
    with pytest.raises(ValueError):
        motion.load_motion_plan(path, side="right", pair=pair)


def test_from_current_follower_uses_new_telemetry_each_time(
    tmp_path: Path, fake_clients: dict[int, _Client], monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _plan_file(tmp_path, start=_values(1.5), duration_s=4.0)
    original_contents = path.read_bytes()
    plan = motion.load_motion_plan(path, side="right", pair=False, from_current=True)
    starts: list[tuple[float, ...]] = []

    def fake_move(
        follower: _Client,
        leader: _Client,
        start: tuple[float, ...],
        target: tuple[float, ...],
        duration: float,
        timeout: float,
        cancelled: threading.Event,
    ) -> np.ndarray:
        starts.append(start)
        assert tuple(follower.position) == start
        follower.position = np.asarray(target)
        return follower.position.copy()

    monkeypatch.setattr(motion, "_move_follower", fake_move)
    for actual in (_values(0.4, 0.9), _values(-0.1, 0.7)):
        fake_clients[1234].position = np.asarray(actual)
        result = motion.move_follower(plan, execute=True)
        assert result["start_mode"] == "current"
        assert result["follower_start"] == actual
    assert starts == [tuple(_values(0.4, 0.9)), tuple(_values(-0.1, 0.7))]
    assert path.read_bytes() == original_contents


@pytest.mark.parametrize("side", ["left", "right"])
def test_move_cli_without_start_flags_uses_current_pose_on_every_execution(
    tmp_path: Path, fake_clients: dict[int, _Client], monkeypatch: pytest.MonkeyPatch, side: str
) -> None:
    path = _plan_file(tmp_path, side=side, start="ignored old start", duration_s=4.0)
    original_contents = path.read_bytes()
    if side == "left":
        fake_clients[1235], fake_clients[1245] = fake_clients[1234], fake_clients[1244]
    starts: list[tuple[float, ...]] = []

    def fake_move(
        follower: _Client,
        leader: _Client,
        start: tuple[float, ...],
        target: tuple[float, ...],
        duration: float,
        timeout: float,
        cancelled: threading.Event,
    ) -> np.ndarray:
        starts.append(start)
        assert tuple(follower.position) == start
        follower.position = np.asarray(target)
        return follower.position.copy()

    monkeypatch.setattr(motion, "_move_follower", fake_move)
    actual_starts = (_values(0.4, 0.9), _values(-0.1, 0.7))
    for actual in actual_starts:
        fake_clients[1234].position = np.asarray(actual)
        assert cli.main(["move", "--side", side, "--plan", str(path), "--execute", "--confirm", "MOVE"]) == 0
    assert starts == [tuple(actual) for actual in actual_starts]
    assert path.read_bytes() == original_contents


def test_move_cli_default_current_start_still_rejects_excessive_speed(
    tmp_path: Path, fake_clients: dict[int, _Client]
) -> None:
    path = _plan_file(tmp_path, duration_s=0.5)
    fake_clients[1234].position = np.asarray(_values(0.8))
    with pytest.raises(SystemExit) as error:
        cli.main(["move", "--side", "right", "--plan", str(path), "--execute", "--confirm", "MOVE"])
    assert error.value.code == 1
    assert not fake_clients[1234].commands and fake_clients[1234].releases == 0


def test_from_current_pair_uses_each_arm_own_start(
    tmp_path: Path, config_factory: Any, fake_clients: dict[int, _Client], monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _plan_file(tmp_path, follower_start=_values(1.5), leader_start=_values(-1.5), duration_s=4.0)
    follower_start, leader_start = _values(0.3, 0.7), _values(-0.1)
    fake_clients[1234].position = np.asarray(follower_start)
    fake_clients[1244].position = np.asarray(leader_start)

    def fake_follower(*args: Any) -> np.ndarray:
        assert args[2] == tuple(follower_start)
        fake_clients[1234].position = np.asarray(args[3])
        return fake_clients[1234].position.copy()

    def fake_leader(client: _Client, plan: motion.MotionPlan, *args: Any) -> np.ndarray:
        assert plan.leader_start == tuple(leader_start)
        client.position = np.asarray(plan.target)
        return client.position.copy()

    monkeypatch.setattr(motion, "_move_follower", fake_follower)
    monkeypatch.setattr(motion, "_move_leader", fake_leader)
    result = motion.preposition_pair(config_factory("right"), path, from_current=True)
    assert result["start_mode"] == "current"
    assert result["follower_before"] == follower_start
    assert result["leader_before"] == leader_start
    assert result["verified_follower_final"] == result["verified_leader_final"] == _values(0.1)


@pytest.mark.parametrize("pair,fast_role", [(False, "follower"), (True, "follower"), (True, "leader")])
def test_from_current_still_rejects_excessive_joint_speed(
    tmp_path: Path, config_factory: Any, fake_clients: dict[int, _Client], pair: bool, fast_role: str
) -> None:
    path = _plan_file(tmp_path, duration_s=0.5)
    fake_clients[1234 if fast_role == "follower" else 1244].position = np.asarray(_values(0.8))
    with pytest.raises(ValueError, match="joint speed"):
        if pair:
            motion.preposition_pair(config_factory("right"), path, from_current=True)
        else:
            plan = motion.load_motion_plan(path, side="right", pair=False, from_current=True)
            motion.move_follower(plan, execute=True)
    assert not fake_clients[1234].commands and fake_clients[1234].releases == 0


def test_from_current_pair_still_requires_passive_leader_gripper_at_target(
    tmp_path: Path, config_factory: Any, fake_clients: dict[int, _Client]
) -> None:
    fake_clients[1244].position = np.asarray(_values(0.2, 0.9))
    with pytest.raises(motion.PreprogrammedMotionError, match="passive"):
        motion.preposition_pair(config_factory("right"), _plan_file(tmp_path), from_current=True)
    assert all(client.releases == 0 and not client.commands for client in fake_clients.values())


def test_from_current_follower_preview_does_not_send_control_commands(
    tmp_path: Path, fake_clients: dict[int, _Client]
) -> None:
    fake_clients[1234].position = np.asarray(_values(0.4, 0.8))
    plan = motion.load_motion_plan(_plan_file(tmp_path, duration_s=4.0), side="right", pair=False, from_current=True)
    result = motion.move_follower(plan, execute=False)
    assert not result["executed"] and result["start_mode"] == "current"
    assert result["follower_start"] == _values(0.4, 0.8)
    assert all(client.releases == 0 and not client.commands for client in fake_clients.values())


@pytest.mark.parametrize("pair", [False, True])
def test_from_current_still_rejects_synchronized_leader(
    tmp_path: Path,
    config_factory: Any,
    fake_clients: dict[int, _Client],
    monkeypatch: pytest.MonkeyPatch,
    pair: bool,
) -> None:
    leader = fake_clients[1244]
    original = leader.get_telemetry

    def synchronized_telemetry() -> _Future:
        state = original().value
        state["synchronized"] = True
        return _Future(state)

    monkeypatch.setattr(leader, "get_telemetry", synchronized_telemetry)
    path = _plan_file(tmp_path)
    with pytest.raises(motion.PreprogrammedMotionError, match="leader is synchronized"):
        if pair:
            motion.preposition_pair(config_factory("right"), path, from_current=True)
        else:
            plan = motion.load_motion_plan(path, side="right", pair=False, from_current=True)
            motion.move_follower(plan, execute=True)
    assert all(client.releases == 0 and not client.commands for client in fake_clients.values())
