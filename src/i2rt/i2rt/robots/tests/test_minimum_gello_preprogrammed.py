"""Tests for leader preprogrammed movement without opening any motor bus."""

from __future__ import annotations

import importlib.util
import pathlib
import queue
import threading
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import portal
import pytest

_MG_PATH = pathlib.Path(__file__).resolve().parents[3] / "examples" / "minimum_gello" / "minimum_gello.py"
_SPEC = importlib.util.spec_from_file_location("minimum_gello_preprogrammed", _MG_PATH)
assert _SPEC is not None and _SPEC.loader is not None
mg = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mg)


class _Rate:
    def __init__(self, **kwargs: Any) -> None:
        del kwargs

    def start(self) -> None:
        pass

    def track(self) -> None:
        pass


class _FakeLeader:
    def __init__(self) -> None:
        self._kp = np.ones(6)
        self._kd = np.ones(6)
        self.position = np.zeros(6)
        self.commands: list[np.ndarray] = []
        self.gains: list[tuple[np.ndarray, np.ndarray]] = []
        self.releases = 0
        self.is_running = True
        self.motor_chain = SimpleNamespace(
            get_same_bus_device_states=lambda: [SimpleNamespace(position=0.5, io_inputs=[0.0, 0.0])]
        )

    def get_observations(self) -> dict[str, np.ndarray]:
        return {"joint_pos": self.position.copy()}

    def command_joint_pos(self, position: np.ndarray) -> None:
        self.position = np.asarray(position, dtype=np.float64).copy()
        self.commands.append(self.position.copy())

    def update_kp_kd(self, kp: np.ndarray, kd: np.ndarray) -> None:
        self._kp, self._kd = kp.copy(), kd.copy()
        self.gains.append((kp.copy(), kd.copy()))

    def enter_gravity_comp_idle(self) -> None:
        self.releases += 1

    def _clip_robot_joint_pos_command(self, position: np.ndarray) -> np.ndarray:
        return position.copy()


class _ResultQueue:
    def __init__(self, stop: Any) -> None:
        self.stop = stop
        self.values: list[Any] = []

    def put(self, value: Any, timeout: float) -> None:
        del timeout
        self.values.append(value)
        self.stop.set()


@pytest.mark.parametrize("joint_speed", [0.1, 0.368, 1.0])
def test_leader_worker_interpolates_target_checks_settle_and_releases(
    monkeypatch: pytest.MonkeyPatch,
    joint_speed: float,
) -> None:
    monkeypatch.setattr(mg, "RateRecorder", _Rate)
    monkeypatch.setattr(
        mg,
        "time",
        SimpleNamespace(monotonic=time.monotonic, sleep=lambda duration: None, time_ns=time.time_ns),
    )
    yam = _FakeLeader()
    telemetry = mg._make_telemetry_shared_state()
    stop = portal.mp.Event()
    request_queue: queue.Queue = queue.Queue()
    target = np.full(6, joint_speed * 0.5)
    request_queue.put((11, [0.0] * 6, target.tolist(), 0.5, 0.12))
    results = _ResultQueue(stop)
    try:
        mg._run_leader_control_loop(
            yam,
            SimpleNamespace(array=np.zeros(7)),
            telemetry,
            queue.Queue(maxsize=1),
            stop,
            0.0,
            preprogrammed_move_queue=request_queue,
            preprogrammed_move_result_queue=results,
            cancelled_move_id=SimpleNamespace(value=0),
        )

        assert len(yam.commands) == 13
        assert np.allclose(yam.position, target)
        assert results.values == [(11, True, "", target.tolist() + [0.5])]
        assert len(yam.gains) == 2
        assert yam._kp == pytest.approx(np.ones(6))
        assert yam._kd == pytest.approx(np.ones(6))
        assert yam.releases == 1
        snapshot = mg._get_telemetry_snapshot(telemetry)
        assert snapshot["position"] == pytest.approx(target.tolist() + [0.5])
        assert snapshot["synchronized"] is False
    finally:
        telemetry.joint_state.close()
        telemetry.button_state.close()


@pytest.mark.parametrize("cancel_before_start", [False, True])
def test_leader_worker_cancellation_stops_setpoints_and_restores_gains(
    cancel_before_start: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    cancelled = SimpleNamespace(value=41 if cancel_before_start else 0)
    yam = _FakeLeader()

    def cancel_during_sleep(duration: float) -> None:
        cancelled.value = 41

    monkeypatch.setattr(mg, "RateRecorder", _Rate)
    monkeypatch.setattr(
        mg, "time", SimpleNamespace(monotonic=time.monotonic, time_ns=time.time_ns, sleep=cancel_during_sleep)
    )
    telemetry = mg._make_telemetry_shared_state()
    stop = portal.mp.Event()
    requests: queue.Queue = queue.Queue()
    requests.put((41, [0.0] * 6, [0.05] * 6, 0.5, 0.12))
    results = _ResultQueue(stop)
    try:
        mg._run_leader_control_loop(
            yam,
            SimpleNamespace(array=np.zeros(7)),
            telemetry,
            queue.Queue(maxsize=1),
            stop,
            0.0,
            preprogrammed_move_queue=requests,
            preprogrammed_move_result_queue=results,
            cancelled_move_id=cancelled,
        )
        assert results.values[0][1] is False
        assert "cancelled" in results.values[0][2]
        assert len(yam.commands) == (0 if cancel_before_start else 1)
        assert yam.releases == (0 if cancel_before_start else 1)
        assert yam._kp == pytest.approx(np.ones(6))
        assert yam._kd == pytest.approx(np.ones(6))
    finally:
        telemetry.joint_state.close()
        telemetry.button_state.close()


@pytest.mark.parametrize(
    "move_request,match",
    [
        ((1, [0.0] * 5, [0.1] * 6, 1.0, 0.1), "six arm joints"),
        ((1, [0.0] * 6, [0.1] * 6, 0.1, 0.1), "duration_s"),
        ((1, [0.0] * 6, [0.1] * 6, 1.0, 1.0), "tolerance"),
        ((2, [0.0] * 6, [0.5005] * 6, 0.5, 0.1), "exceeds the 1.00 rad/s limit"),
    ],
)
def test_leader_worker_rejects_invalid_move_without_changing_motor_gains(
    monkeypatch: pytest.MonkeyPatch,
    move_request: tuple,
    match: str,
) -> None:
    monkeypatch.setattr(mg, "RateRecorder", _Rate)
    monkeypatch.setattr(
        mg,
        "time",
        SimpleNamespace(monotonic=time.monotonic, sleep=lambda duration: None, time_ns=time.time_ns),
    )
    yam = _FakeLeader()
    telemetry = mg._make_telemetry_shared_state()
    stop = portal.mp.Event()
    requests: queue.Queue = queue.Queue()
    requests.put(move_request)
    results = _ResultQueue(stop)
    try:
        mg._run_leader_control_loop(
            yam,
            SimpleNamespace(array=np.zeros(7)),
            telemetry,
            queue.Queue(maxsize=1),
            stop,
            0.0,
            preprogrammed_move_queue=requests,
            preprogrammed_move_result_queue=results,
            cancelled_move_id=SimpleNamespace(value=0),
        )

        assert results.values[0][0] == move_request[0]
        assert results.values[0][1] is False
        assert match in results.values[0][2]
        assert yam.commands == []
        assert yam.gains == []
        assert yam.releases == 0
    finally:
        telemetry.joint_state.close()
        telemetry.button_state.close()


def test_optional_leader_move_rpc_keeps_read_only_leader_telemetry_available() -> None:
    port = portal.free_port()
    telemetry = mg._make_telemetry_shared_state()
    requests: queue.Queue = queue.Queue(maxsize=1)
    results: queue.Queue = queue.Queue(maxsize=1)
    cancelled = portal.mp.Value("q", 0)
    server = client = observer = cancel_client = None
    try:
        mg._publish_telemetry(
            telemetry,
            [0.0] * 6 + [0.5],
            [0.0] * 7,
            [0.0] * 7,
            button_state=[0.0, 0.0],
            synchronized=False,
        )
        server = mg._build_telemetry_server(port, telemetry, requests, results, cancelled)
        server.start(block=False)
        client = portal.Client(f"127.0.0.1:{port}")
        observer = portal.Client(f"127.0.0.1:{port}")
        assert client.preprogrammed_motion_ready().result(timeout=2.0) == {"version": 1, "available": True}

        request_id = 1234
        motion_result: list[Any] = []

        def call_motion() -> None:
            motion_result.append(
                client.preprogrammed_move(request_id, np.zeros(6), np.full(6, 0.05), 0.5, 0.12).result(timeout=4.0)
            )

        worker = threading.Thread(target=call_motion)
        worker.start()
        queued = requests.get(timeout=2.0)
        assert queued == (request_id, [0.0] * 6, [0.05] * 6, 0.5, 0.12)

        snapshot = observer.get_telemetry().result(timeout=1.0)
        assert snapshot["position"] == [0.0] * 6 + [0.5]
        results.put((request_id, True, "", [0.05] * 6 + [0.5]), timeout=1.0)
        worker.join(timeout=2.0)
        assert not worker.is_alive()
        assert motion_result == [{"request_id": request_id, "position": [0.05] * 6 + [0.5]}]

        cancel_request_id = request_id + 1
        cancellation_errors: list[BaseException] = []

        def call_pending_motion() -> None:
            try:
                client.preprogrammed_move(cancel_request_id, np.zeros(6), np.full(6, 0.05), 0.5, 0.12).result(
                    timeout=4.0
                )
            except BaseException as exc:
                cancellation_errors.append(exc)

        pending_worker = threading.Thread(target=call_pending_motion)
        pending_worker.start()
        assert requests.get(timeout=2.0)[0] == cancel_request_id
        cancel_client = portal.Client(f"127.0.0.1:{port}")
        assert bool(cancel_client.cancel_preprogrammed_move(cancel_request_id).result(timeout=2.0))
        assert cancelled.value == cancel_request_id
        results.put((cancel_request_id, False, "preprogrammed move was cancelled", []), timeout=1.0)
        pending_worker.join(timeout=2.0)
        assert not pending_worker.is_alive()
        assert cancellation_errors and "cancelled" in str(cancellation_errors[0])

        assert mg._cancel_preprogrammed_move(cancelled, [request_id], threading.Lock(), request_id) is True
        assert cancelled.value == request_id
        assert mg._cancel_preprogrammed_move(cancelled, [0], threading.Lock(), request_id + 2) is True
        assert cancelled.value == request_id + 2
        assert observer.get_telemetry().result(timeout=1.0)["position"] == [0.0] * 6 + [0.5]
    finally:
        for connected in (cancel_client, observer, client):
            if connected is not None:
                connected.close(timeout=1.0)
        if server is not None and server.running:
            server.close(timeout=1.0)
        telemetry.joint_state.close()
        telemetry.button_state.close()
