"""Fail-closed worker supervision tests for ``minimum_gello``."""

import importlib.util
import pathlib
import queue
import threading
import time
from types import SimpleNamespace
from typing import Any, NoReturn

import numpy as np
import pytest

_MG_PATH = pathlib.Path(__file__).resolve().parents[3] / "examples" / "minimum_gello" / "minimum_gello.py"
_spec = importlib.util.spec_from_file_location("minimum_gello_health", _MG_PATH)
assert _spec is not None and _spec.loader is not None
mg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mg)


class _StopEvent:
    def __init__(self) -> None:
        self.set_count = 0

    def set(self) -> None:
        self.set_count += 1

    def is_set(self) -> bool:
        return self.set_count > 0


class _FakeRobot:
    """Small robot double used to exercise worker-owned cleanup paths."""

    def __init__(self, running: bool = True) -> None:
        self.is_running = running
        self.close_calls = 0
        self._kp = np.zeros(6)
        self.motor_chain = SimpleNamespace()

    def num_dofs(self) -> int:
        return 1

    def get_observations(self) -> dict[str, np.ndarray]:
        return {
            "joint_pos": np.zeros(1),
            "joint_vel": np.zeros(1),
            "joint_eff": np.zeros(1),
        }

    def close(self) -> None:
        self.close_calls += 1


@pytest.mark.parametrize(
    "worker_name,impl_name,args",
    [
        ("_yam_polling_worker", "_yam_polling_worker_impl", (None, None, None, None, None, "test")),
        ("_leader_control_worker", "_leader_control_worker_impl", (None, None, None, None, None)),
    ],
)
def test_fatal_hardware_worker_sets_shared_stop_event(
    monkeypatch: Any, worker_name: str, impl_name: str, args: tuple[Any, ...]
) -> None:
    stop_event = _StopEvent()

    def fail(*_args: Any) -> NoReturn:
        raise RuntimeError("loss communication")

    monkeypatch.setattr(mg, impl_name, fail)
    worker = getattr(mg, worker_name)

    with pytest.raises(RuntimeError, match="loss communication"):
        worker(*args[:-1], stop_event, args[-1]) if worker_name == "_yam_polling_worker" else worker(
            *args[:-1], stop_event
        )

    assert stop_event.set_count == 1


def test_wait_for_dof_aborts_when_worker_stops() -> None:
    stop_event = _StopEvent()
    stop_event.set()
    with pytest.raises(RuntimeError, match="before publishing"):
        mg._wait_for_dof(type("Value", (), {"value": 0})(), stop_event=stop_event)


@pytest.mark.parametrize("worker_impl", [mg._yam_polling_worker_impl, mg._leader_control_worker_impl])
@pytest.mark.parametrize("running", [True, False])
def test_hardware_worker_impl_closes_robot_on_exit(monkeypatch: Any, worker_impl: Any, running: bool) -> None:
    """Robot-owned CAN resources are closed on both normal and fatal worker exits."""
    robot = _FakeRobot(running=running)
    monkeypatch.setattr(mg, "get_yam_robot", lambda **_kwargs: robot)
    args = mg.Args(sim=True)
    stop_event = _StopEvent()
    if running:
        # Exercise the normal stop path without entering a polling iteration.
        stop_event.set()
    pos_shared = SimpleNamespace(array=np.zeros(mg._MAX_DOFS))
    n_dofs_value = SimpleNamespace(value=0)

    call_args = (
        args,
        pos_shared,
        None,
        n_dofs_value,
        None,
        stop_event,
        "test-worker",
    )
    if worker_impl is mg._leader_control_worker_impl:
        call_args = (args, pos_shared, None, None, stop_event)

    if running:
        worker_impl(*call_args)
    else:
        with pytest.raises(RuntimeError, match="underlying motor control loop"):
            worker_impl(*call_args)
    assert robot.close_calls == 1
    assert stop_event.is_set()


class _Worker:
    def __init__(self, exitcode: int | None) -> None:
        self.exitcode = exitcode
        self.join_calls = 0

    def join(self, timeout: float) -> None:
        self.join_calls += 1


@pytest.mark.parametrize("exitcode,should_raise", [(0, False), (1, True), (None, True)])
def test_worker_exit_check_rejects_failed_or_unfinished_worker(exitcode: int | None, should_raise: bool) -> None:
    worker = _Worker(exitcode)
    if should_raise:
        with pytest.raises(RuntimeError, match="stopped unexpectedly"):
            mg._require_clean_worker_exit(worker, "test worker")
    else:
        mg._require_clean_worker_exit(worker, "test worker")
    assert worker.join_calls == 1


class _FakeServer:
    def __init__(self) -> None:
        self.running = True
        self.close_calls = 0
        self.close_timeout: float | None = None

    def close(self, timeout: float | None = None) -> None:
        self.close_calls += 1
        self.close_timeout = timeout
        self.running = False


class _FakeProcess:
    def __init__(self, exits_after_join: bool) -> None:
        self.exitcode: int | None = None
        self.exits_after_join = exits_after_join
        self.join_calls = 0
        self.kill_calls = 0

    def join(self, timeout: float) -> None:
        self.join_calls += 1
        if self.exits_after_join:
            self.exitcode = 0

    def kill(self, timeout: float = 1.0) -> None:
        self.kill_calls += 1
        self.exitcode = -9


class _FakeShared:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def test_cleanup_waits_for_workers_before_kill_fallback() -> None:
    """Cleanup gives robot workers a chance to run their ``finally`` blocks."""
    stop_event = _StopEvent()
    server = _FakeServer()
    graceful = _FakeProcess(exits_after_join=True)
    stuck = _FakeProcess(exits_after_join=False)
    pos_shared = _FakeShared()
    telemetry_shared = SimpleNamespace(joint_state=_FakeShared())
    resources = mg.Resources(
        processes=[graceful, stuck],
        servers=[server],
        pos_shared=pos_shared,
        telemetry_shared=telemetry_shared,
        stop_event=stop_event,
    )

    mg.cleanup(resources)

    assert stop_event.set_count == 1
    assert server.close_calls == 1
    assert server.close_timeout == mg._WORKER_SHUTDOWN_TIMEOUT_S
    assert graceful.join_calls == 1
    assert graceful.kill_calls == 0
    assert stuck.join_calls == 2  # graceful wait plus the post-kill reap
    assert stuck.kill_calls == 1
    assert pos_shared.close_calls == 1
    assert telemetry_shared.joint_state.close_calls == 1


class _FakeFuture:
    def __init__(self, value: Any = None, error: BaseException | None = None) -> None:
        self.value = value
        self.error = error
        self.timeouts: list[float | None] = []

    def result(self, timeout: float | None = None) -> Any:
        self.timeouts.append(timeout)
        if self.error is not None:
            raise self.error
        return self.value


class _FakePortalClient:
    def __init__(self, connects: bool = True) -> None:
        self.connects = connects
        self.connect_timeouts: list[float | None] = []
        self.num_dofs_future = _FakeFuture(7)
        self.joint_pos_future = _FakeFuture(np.arange(7, dtype=np.float64))
        self.command_pos_future = _FakeFuture()
        self.command_state_future = _FakeFuture()
        self.observations_future = _FakeFuture({"joint_pos": np.zeros(7)})
        self.close_timeouts: list[float | None] = []

    def connect(self, timeout: float | None = None) -> bool:
        self.connect_timeouts.append(timeout)
        return self.connects

    def num_dofs(self) -> _FakeFuture:
        return self.num_dofs_future

    def get_joint_pos(self) -> _FakeFuture:
        return self.joint_pos_future

    def command_joint_pos(self, _joint_pos: np.ndarray) -> _FakeFuture:
        return self.command_pos_future

    def command_joint_state(self, _joint_state: dict[str, np.ndarray]) -> _FakeFuture:
        return self.command_state_future

    def get_observations(self) -> _FakeFuture:
        return self.observations_future

    def close(self, timeout: float | None = None) -> None:
        self.close_timeouts.append(timeout)


def test_client_robot_bounds_every_portal_operation(monkeypatch: Any) -> None:
    portal_client = _FakePortalClient()
    client_args: list[tuple[str, dict[str, Any]]] = []

    def make_client(address: str, **kwargs: Any) -> _FakePortalClient:
        client_args.append((address, kwargs))
        return portal_client

    monkeypatch.setattr(mg.portal, "Client", make_client)
    client_robot = mg.ClientRobot(port=1234, host="127.0.0.1")

    assert client_robot.num_dofs() == 7
    np.testing.assert_array_equal(client_robot.get_joint_pos(), np.arange(7, dtype=np.float64))
    client_robot.command_joint_pos(np.zeros(7))
    client_robot.command_joint_state({"position": np.zeros(7)})
    np.testing.assert_array_equal(client_robot.get_observations()["joint_pos"], np.zeros(7))
    client_robot.close()

    futures = [
        portal_client.num_dofs_future,
        portal_client.joint_pos_future,
        portal_client.command_pos_future,
        portal_client.command_state_future,
        portal_client.observations_future,
    ]
    assert [future.timeouts for future in futures] == [[mg._PORTAL_RPC_TIMEOUT_S]] * len(futures)
    assert portal_client.connect_timeouts == [mg._PORTAL_RPC_TIMEOUT_S] * len(futures)
    assert portal_client.close_timeouts == [mg._PORTAL_RPC_TIMEOUT_S]
    assert client_args == [("127.0.0.1:1234", {"autoconn": False})]


def test_client_robot_propagates_portal_result_timeout_and_replaces_stale_client(monkeypatch: Any) -> None:
    portal_client = _FakePortalClient()
    portal_client.joint_pos_future = _FakeFuture(error=TimeoutError())
    replacement_client = _FakePortalClient()
    clients = [portal_client, replacement_client]
    monkeypatch.setattr(mg.portal, "Client", lambda *_args, **_kwargs: clients.pop(0))
    client_robot = mg.ClientRobot()

    with pytest.raises(TimeoutError):
        client_robot.get_joint_pos()
    assert portal_client.joint_pos_future.timeouts == [mg._PORTAL_RPC_TIMEOUT_S]
    assert portal_client.close_timeouts == [mg._PORTAL_RPC_TIMEOUT_S]
    assert client_robot._client is replacement_client


def test_client_robot_bounds_unavailable_portal_connection(monkeypatch: Any) -> None:
    portal_client = _FakePortalClient(connects=False)
    monkeypatch.setattr(mg.portal, "Client", lambda *_args, **_kwargs: portal_client)
    client_robot = mg.ClientRobot()

    with pytest.raises(TimeoutError, match="timed out connecting"):
        client_robot.get_joint_pos()
    assert portal_client.connect_timeouts == [mg._PORTAL_RPC_TIMEOUT_S]
    assert portal_client.joint_pos_future.timeouts == []


def test_client_robot_explicit_bounded_connection_works_with_portal_server() -> None:
    """Disabling Portal auto-reconnect still permits a normal local RPC."""
    port = mg.portal.free_port()
    server = mg.portal.Server(port)
    client_robot = None
    try:
        server.bind("get_joint_pos", lambda: np.arange(7, dtype=np.float64))
        server.start(block=False)
        client_robot = mg.ClientRobot(port=port, host="127.0.0.1")

        np.testing.assert_array_equal(client_robot.get_joint_pos(), np.arange(7, dtype=np.float64))
    finally:
        if client_robot is not None:
            client_robot.close()
        if server.running:
            server.close()


def test_leader_io_stops_after_continuous_follower_rpc_failure(monkeypatch: Any) -> None:
    class FailingFollower:
        def get_joint_pos(self) -> NoReturn:
            raise TimeoutError("follower did not respond")

    stop_event = threading.Event()
    monkeypatch.setattr(mg, "_LEADER_RPC_FAILURE_TIMEOUT_S", 0.01)
    monkeypatch.setattr(mg, "_LEADER_RPC_RETRY_PERIOD_S", 0.001)

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="follower RPC unavailable"):
        mg._run_leader_io_loop(
            FailingFollower(),
            queue.Queue(),
            SimpleNamespace(array=np.zeros(7)),
            stop_event,
            SimpleNamespace(track=lambda: None),
        )

    assert stop_event.is_set()
    assert time.monotonic() - started < 0.5
