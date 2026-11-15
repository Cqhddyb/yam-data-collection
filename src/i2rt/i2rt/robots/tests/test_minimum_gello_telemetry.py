"""Tests for the read-only ``minimum_gello`` Portal telemetry interfaces.

The example is loaded by path because it lives outside the importable package.
The Portal server tests use ephemeral local test ports and do not open CAN devices
or create hardware robots.
"""

import importlib.util
import pathlib
import queue as _queue
import threading
import time
from types import SimpleNamespace

import numpy as np
import portal
import pytest

_MG_PATH = pathlib.Path(__file__).resolve().parents[3] / "examples" / "minimum_gello" / "minimum_gello.py"
_spec = importlib.util.spec_from_file_location("minimum_gello_telemetry", _MG_PATH)
assert _spec is not None and _spec.loader is not None
mg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mg)


def test_telemetry_snapshot_is_coherent_and_copied() -> None:
    """A client-visible snapshot must keep all components from one published sample."""
    telemetry_shared = mg._make_telemetry_shared_state()
    position = np.array([1.0, 2.0, 3.0])
    velocity = np.array([4.0, 5.0, 6.0])
    effort = np.array([7.0, 8.0, 9.0])
    try:
        before_ns = time.time_ns()
        mg._publish_telemetry(telemetry_shared, position, velocity, effort)
        snapshot = mg._get_telemetry_snapshot(telemetry_shared)

        # Mutating source arrays after publication must not change the copied snapshot.
        position[:] = -1.0
        velocity[:] = -1.0
        effort[:] = -1.0

        assert snapshot["timestamp_ns"] >= before_ns
        assert snapshot["sequence"] == 1
        assert snapshot["position"] == [1.0, 2.0, 3.0]
        assert snapshot["velocity"] == [4.0, 5.0, 6.0]
        assert snapshot["effort"] == [7.0, 8.0, 9.0]
        assert snapshot["button_state"] == [0.0, 0.0]
        assert snapshot["synchronized"] is False

        mg._publish_telemetry(telemetry_shared, [10.0], [0.0], [0.0])
        next_snapshot = mg._get_telemetry_snapshot(telemetry_shared)
        assert next_snapshot["sequence"] == 2
        assert next_snapshot["position"] == [10.0]
        assert next_snapshot["velocity"] == [0.0]
        assert next_snapshot["effort"] == [0.0]

        mg._publish_telemetry(
            telemetry_shared,
            [11.0],
            [1.0],
            [2.0],
            button_state=[1.0, 0.0],
            synchronized=True,
        )
        diagnostic_snapshot = mg._get_telemetry_snapshot(telemetry_shared)
        assert diagnostic_snapshot["button_state"] == [1.0, 0.0]
        assert diagnostic_snapshot["synchronized"] is True
    finally:
        telemetry_shared.joint_state.close()
        telemetry_shared.button_state.close()


def test_leader_gripper_endpoint_calibration_maps_measured_handle_range() -> None:
    """A side-specific handle range maps its measured endpoints to [closed, open]."""

    class FakeLeaderRobot:
        def __init__(self, handle_position: float) -> None:
            self.motor_chain = SimpleNamespace(
                get_same_bus_device_states=lambda: [SimpleNamespace(position=handle_position, io_inputs=[0, 0])]
            )

        def get_observations(self) -> dict[str, np.ndarray]:
            return {"joint_pos": np.zeros(6)}

    closed = 0.068655
    opened = 0.960555
    closed_robot = mg.YAMLeaderRobot(
        FakeLeaderRobot(1.0 - closed), gripper_command_closed=closed, gripper_command_open=opened
    )
    open_robot = mg.YAMLeaderRobot(
        FakeLeaderRobot(1.0 - opened), gripper_command_closed=closed, gripper_command_open=opened
    )

    assert closed_robot.get_info()[0][-1] == pytest.approx(0.0)
    assert open_robot.get_info()[0][-1] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="gripper command endpoints"):
        mg.YAMLeaderRobot(FakeLeaderRobot(0.5), gripper_command_closed=0.5, gripper_command_open=0.5)


def test_follower_server_exposes_telemetry_snapshot() -> None:
    """The existing follower Portal server serves read-only telemetry on its normal port."""
    port = portal.free_port()
    pos_shared = portal.SharedArray(shape=(mg._MAX_DOFS,), dtype=np.float64)
    telemetry_shared = mg._make_telemetry_shared_state()
    n_dofs_value = portal.mp.Value("i", 7)
    cmd_queue: _queue.Queue = _queue.Queue(maxsize=1)
    server = None
    client = None
    try:
        mg._publish_telemetry(
            telemetry_shared,
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            [1.0] * 7,
            [2.0] * 7,
        )
        server = mg._build_follower_server(port, n_dofs_value, pos_shared, telemetry_shared, cmd_queue)
        server.start(block=False)
        client = portal.Client(f"127.0.0.1:{port}")

        snapshot = client.get_telemetry().result(timeout=5.0)

        assert snapshot["sequence"] == 1
        assert snapshot["position"] == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        assert snapshot["velocity"] == [1.0] * 7
        assert snapshot["effort"] == [2.0] * 7

        client.enter_gravity_comp_idle().result(timeout=5.0)
        assert cmd_queue.get(timeout=1.0) == ("enter_gravity_comp_idle",)
    finally:
        if client is not None:
            client.close()
        if server is not None and server.running:
            server.close()
        pos_shared.close()
        telemetry_shared.joint_state.close()
        telemetry_shared.button_state.close()


def test_leader_telemetry_server_is_nonblocking_and_includes_handle_gripper() -> None:
    """The optional leader server streams seven positions and zeroed unavailable fields."""
    port = portal.free_port()
    telemetry_shared = mg._make_telemetry_shared_state()
    server = None
    client = None
    try:
        leader_position = np.arange(7, dtype=float) / 10.0
        mg._publish_telemetry(
            telemetry_shared,
            leader_position,
            np.zeros_like(leader_position),
            np.zeros_like(leader_position),
        )
        server = mg._build_telemetry_server(port, telemetry_shared)

        start = time.monotonic()
        server.start(block=False)
        assert time.monotonic() - start < 1.0
        client = portal.Client(f"127.0.0.1:{port}")
        snapshot = client.get_telemetry().result(timeout=5.0)

        assert snapshot["position"] == pytest.approx(leader_position.tolist())
        assert len(snapshot["position"]) == 7
        assert snapshot["position"][-1] == pytest.approx(0.6)
        assert snapshot["velocity"] == [0.0] * 7
        assert snapshot["effort"] == [0.0] * 7
    finally:
        if client is not None:
            client.close()
        if server is not None and server.running:
            server.close()
        telemetry_shared.joint_state.close()
        telemetry_shared.button_state.close()


def test_leader_sync_button_keeps_telemetry_heartbeat() -> None:
    """The three-second follower synchronization move must not freeze telemetry."""

    class FakeLeader:
        _kp = np.ones(6)
        is_running = True

        def __init__(self) -> None:
            self.motor_chain = SimpleNamespace(
                get_same_bus_device_states=lambda: [SimpleNamespace(position=0.5, io_inputs=[1, 0])]
            )

        def get_observations(self) -> dict[str, np.ndarray]:
            return {
                "joint_pos": np.zeros(6),
                "joint_vel": np.zeros(6),
                "joint_eff": np.zeros(6),
            }

        def command_joint_pos(self, _joint_pos: np.ndarray) -> None:
            return None

        def update_kp_kd(self, kp: np.ndarray, kd: np.ndarray) -> None:
            return None

    telemetry_shared = mg._make_telemetry_shared_state()
    pos_shared = SimpleNamespace(array=np.zeros(7))
    stop_event = portal.mp.Event()
    command_queue: _queue.Queue = _queue.Queue(maxsize=1)
    worker = threading.Thread(
        target=mg._run_leader_control_loop,
        args=(FakeLeader(), pos_shared, telemetry_shared, command_queue, stop_event, 0.0),
        daemon=True,
    )
    worker.start()
    try:
        deadline = time.monotonic() + 0.8
        sequence_values: list[int] = []
        while time.monotonic() < deadline:
            sequence_values.append(mg._get_telemetry_snapshot(telemetry_shared)["sequence"])
            if len(sequence_values) >= 3 and sequence_values[-1] > sequence_values[0]:
                break
            time.sleep(0.01)
        assert sequence_values[-1] > sequence_values[0]
        snapshot = mg._get_telemetry_snapshot(telemetry_shared)
        assert snapshot["button_state"] == [1.0, 0.0]
        assert snapshot["synchronized"] is False
    finally:
        stop_event.set()
        worker.join(timeout=2.0)
        assert not worker.is_alive()
        telemetry_shared.joint_state.close()
        telemetry_shared.button_state.close()
