from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest

from yam_data_collection.web_console import (
    ArmCommand,
    ArmProcessSupervisor,
    ArmStartupError,
    PreviewBuffer,
    ReadinessProbe,
    WebCaptureController,
    WebConsoleCallbacks,
    WebConsoleApplication,
    WebConsoleState,
    create_server,
    encode_camera_preview,
    load_arm_commands,
)
from yam_data_collection.web_runtime import build_default_arm_commands


class _FakeProcess:
    def __init__(self, *, fail: bool = False) -> None:
        self.pid = None
        self.returncode = None
        self.terminated = False
        self.fail = fail
        self.stdout = None

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.terminated = True
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def test_preview_buffer_is_bounded_and_encodes_without_cv2() -> None:
    preview = PreviewBuffer(camera_roles=("cam_high",), arm_roles=("left_leader",), joint_history=2)
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    assert preview.submit_camera(type("Frame", (), {"role": "cam_high", "image_bgr": image, "pixel_format": "bgr24", "timestamp_ns": 1})())
    assert preview.get_camera("cam_high") is not None
    preview.put_joints({"left_leader": [1, 2]}, timestamp_ns=1)
    preview.put_joints({"left_leader": [2, 3]}, timestamp_ns=2)
    preview.put_joints({"left_leader": [3, 4]}, timestamp_ns=3)
    assert len(preview.get_joints()["left_leader"]) == 2
    encoded, content_type = encode_camera_preview(image, "bgr24", max_width=4, max_height=4)
    assert encoded.startswith(b"\x89PNG") or encoded.startswith(b"\xff\xd8")
    assert content_type in {"image/png", "image/jpeg"}


def test_web_server_rejects_non_loopback_bind() -> None:
    with pytest.raises(ValueError, match="loopback"):
        create_server("0.0.0.0", 0)


def test_arm_supervisor_uses_documented_order_and_reverse_cleanup() -> None:
    started: list[str] = []
    processes: dict[str, _FakeProcess] = {}

    def popen(argv: list[str], **kwargs: object) -> _FakeProcess:
        del kwargs
        role = argv[0]
        started.append(role)
        process = _FakeProcess()
        processes[role] = process
        return process

    commands = [ArmCommand(role, (role,)) for role in ("left_leader", "right_follower", "left_follower", "right_leader")]
    supervisor = ArmProcessSupervisor(commands, popen=popen)
    assert supervisor.ordered_roles == ("left_follower", "right_follower", "right_leader", "left_leader")
    supervisor.start_all()
    assert started == ["left_follower", "right_follower", "right_leader", "left_leader"]
    supervisor.stop_all()
    assert all(process.terminated for process in processes.values())


def test_default_arm_commands_match_documented_ports_and_order() -> None:
    commands = build_default_arm_commands(Path(__file__).resolve().parents[2] / "i2rt")
    assert [command.role for command in commands] == [
        "left_follower",
        "right_follower",
        "right_leader",
        "left_leader",
    ]
    assert [probe.port for command in commands for probe in command.readiness] == [1235, 1234, 1234, 1244, 1235, 1245]
    right_leader = commands[2].argv
    assert "--gripper-command-closed" in right_leader
    assert "--gripper-command-open" in right_leader


def test_arm_status_reports_every_process() -> None:
    processes: list[_FakeProcess] = []

    def popen(argv: list[str], **kwargs: object) -> _FakeProcess:
        del argv, kwargs
        process = _FakeProcess()
        processes.append(process)
        return process

    commands = [ArmCommand(role, (role,)) for role in ("left_follower", "right_follower", "right_leader", "left_leader")]
    supervisor = ArmProcessSupervisor(commands, popen=popen)
    supervisor.start_all()
    assert set(supervisor.statuses()) == {command.role for command in commands}
    supervisor.stop_all()


def test_capture_start_failure_drops_closed_collector_for_retry() -> None:
    class FailingCollector:
        def __init__(self, sink: object) -> None:
            del sink
            self.state = type("State", (), {"value": "idle"})()

        @property
        def active_episode_snapshot(self) -> dict[str, object]:
            return {"state": self.state.value}

        def start(self) -> None:
            raise RuntimeError("synthetic start failure")

        def close(self) -> None:
            self.state.value = "closed"

    created: list[FailingCollector] = []

    def factory(*, preview_sink: object) -> FailingCollector:
        collector = FailingCollector(preview_sink)
        created.append(collector)
        return collector

    preview = PreviewBuffer(camera_roles=("cam_high",), arm_roles=("left_leader",))
    controller = WebCaptureController(factory, preview)
    controller.start_capture()
    assert controller.wait_idle(1.0)
    assert controller.collector is None
    controller.start_capture()
    assert controller.wait_idle(1.0)
    assert len(created) == 2
    controller.close()


def test_web_application_rejects_concurrent_actions() -> None:
    entered = threading.Event()
    release = threading.Event()

    def slow_action() -> None:
        entered.set()
        release.wait(1.0)

    app = WebConsoleApplication(
        WebConsoleState(preview=PreviewBuffer(camera_roles=("cam_high",), arm_roles=("left_leader",))),
        callbacks=WebConsoleCallbacks(start_arms=slow_action),
    )
    thread = threading.Thread(target=lambda: app.action("arms/start"))
    thread.start()
    assert entered.wait(1.0)
    with pytest.raises(RuntimeError, match="another control action"):
        app.action("arms/start")
    release.set()
    thread.join(1.0)


def test_web_application_requires_ready_arms_before_capture() -> None:
    preview = PreviewBuffer(camera_roles=("cam_high",), arm_roles=("left_leader",))
    fake_process = _FakeProcess()
    supervisor = ArmProcessSupervisor([ArmCommand("left_leader", ("leader",))], popen=lambda *args, **kwargs: fake_process)
    app = WebConsoleApplication(WebConsoleState(preview=preview), arm_supervisor=supervisor)
    app.callbacks = type(app.callbacks)(start_capture=lambda: {"accepted": True})
    with pytest.raises(RuntimeError, match="all four arm processes"):
        app.action("capture/start")


def test_arm_supervisor_rolls_back_on_readiness_failure() -> None:
    processes: list[_FakeProcess] = []

    def popen(argv: list[str], **kwargs: object) -> _FakeProcess:
        del argv, kwargs
        process = _FakeProcess()
        processes.append(process)
        return process

    probes = iter([True, False, False])
    supervisor = ArmProcessSupervisor(
        [
            ArmCommand("left_follower", ("left",), (ReadinessProbe("127.0.0.1", 1235),), startup_timeout_s=.01),
            ArmCommand("right_follower", ("right",), (ReadinessProbe("127.0.0.1", 1234),), startup_timeout_s=.01),
        ],
        popen=popen,
        connect=lambda probe: next(probes),
        sleeper=lambda delay: None,
    )
    with pytest.raises(ArmStartupError):
        supervisor.start_all()
    assert processes and all(process.terminated for process in processes)


def test_arm_command_loader_rejects_non_list_and_preserves_order(tmp_path: Path) -> None:
    path = tmp_path / "arms.json"
    path.write_text(json.dumps({"commands": [{"role": "left_leader", "argv": ["a"]}]}), encoding="utf-8")
    commands = load_arm_commands(path)
    assert commands[0].role == "left_leader"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        load_arm_commands(path)


def test_capture_controller_serializes_actions() -> None:
    class FakeCollector:
        def __init__(self, sink: object) -> None:
            self.state = type("State", (), {"value": "idle"})()
            self.sink = sink
            self.started = False

        @property
        def active_episode_snapshot(self) -> dict[str, object]:
            return {"state": self.state.value}

        def start(self) -> None:
            self.started = True

        def start_episode(self) -> None:
            self.state.value = "recording"

        def save_episode(self) -> None:
            self.state.value = "idle"

        def discard_episode(self) -> None:
            self.state.value = "idle"

        def close(self) -> None:
            self.state.value = "closed"

    holder: list[FakeCollector] = []

    def factory(*, preview_sink: object) -> FakeCollector:
        collector = FakeCollector(preview_sink)
        holder.append(collector)
        return collector

    preview = PreviewBuffer(camera_roles=("cam_high",), arm_roles=("left_leader",))
    controller = WebCaptureController(factory, preview)
    accepted = controller.start_capture()
    assert accepted["accepted"]
    controller._operation_thread.join(timeout=1)  # type: ignore[union-attr]
    assert holder[0].started
    assert controller.state.capture_state == "recording"
    accepted = controller.stop_capture()
    assert accepted["accepted"]
    controller._operation_thread.join(timeout=1)  # type: ignore[union-attr]
    assert controller.state.capture_state == "idle"
