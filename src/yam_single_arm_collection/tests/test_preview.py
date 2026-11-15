from __future__ import annotations

import queue
import threading
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from yam_data_collection.live_preview import LivePreviewConfig, _CameraUpdate
from yam_data_collection.models import CameraFrame

from yam_single_arm_collection import cli
from yam_single_arm_collection.collector import SingleArmCollector
from yam_single_arm_collection.live_preview import WristCameraPreview, _WristPreviewApplication


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("setup", ["wrist", "wrist-top"])
def test_preview_only_receives_the_selected_devices(
    config_factory: Any, collector_factory: Any, monkeypatch: pytest.MonkeyPatch, side: str, setup: str
) -> None:
    config = config_factory(side, setup)
    monkeypatch.setattr(SingleArmCollector, "_available_preview_cpu_set", lambda self: (2, 3))
    collector = collector_factory(config, enable_preview=True, preview_fps=2)
    preview = collector._live_preview
    assert preview is not None
    assert preview.config.camera_roles == (f"cam_{side}",)
    assert set(preview.config.arm_roles) == set(config.telemetry)
    wrist = config.cameras[f"cam_{side}"]
    assert (preview.config.camera_width, preview.config.camera_height) == (wrist.width, wrist.height)
    assert preview.config.camera_fps == 2
    assert preview.config.cpu_set == (2, 3)
    assert preview.config.queue_size == 1
    assert not preview.running


def test_capture_without_preview_keeps_the_gui_disabled(config_factory: Any, collector_factory: Any) -> None:
    collector = collector_factory(config_factory())
    assert collector._live_preview is None


def test_preview_can_be_closed_or_fail_without_closing_collection(
    config_factory: Any, collector_factory: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    collector = collector_factory(config_factory())

    class UnavailablePreview:
        def __init__(self) -> None:
            self.stopped = False

        def start(self) -> None:
            raise RuntimeError("no desktop display")

        def stop(self, timeout_s: float = 2) -> None:
            self.stopped = True

    preview = UnavailablePreview()
    collector._live_preview = preview
    collector._start_live_preview()
    assert preview.stopped
    assert collector.background_error is None
    assert "capture continues" in capsys.readouterr().out


def test_preview_does_not_share_exclusive_recorder_cpus(
    config_factory: Any, collector_factory: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = config_factory()
    monkeypatch.setattr(SingleArmCollector, "_available_preview_cpu_set", lambda self: ())
    collector = collector_factory(config, enable_preview=True)
    assert collector._live_preview is None
    assert "no CPU is available" in capsys.readouterr().out


@pytest.mark.parametrize("fps", ["0", "-1", "11", "nan", "inf"])
def test_bad_preview_rate_fails_before_preposition_or_devices(fps: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "preposition_pair", lambda *args, **kwargs: pytest.fail("unexpected motion"))
    monkeypatch.setattr(cli, "SingleArmCollector", lambda *args, **kwargs: pytest.fail("unexpected hardware"))
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "capture",
                "--side",
                "left",
                "--preview",
                "--preview-fps",
                fps,
                "--preposition",
                "target.json",
                "--from-current",
                "--confirm",
                "PREPOSITION",
            ]
        )
    assert error.value.code == 2


@pytest.mark.parametrize("setup", ["wrist", "wrist-top"])
@pytest.mark.parametrize("fps_args,expected_fps", [([], 10), (["--preview-fps", "2"], 2)])
def test_capture_with_preposition_can_enable_preview(
    setup: str, fps_args: list[str], expected_fps: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[Any] = []
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli, "preposition_pair", lambda *args, **kwargs: events.append("preposition") or {})

    class PreviewCollector:
        def __init__(self, config: Any, *, enable_preview: bool, preview_fps: float) -> None:
            events.append((set(config.cameras), enable_preview, preview_fps))

        def run_interactive(self) -> None:
            events.append("capture")

    monkeypatch.setattr(cli, "SingleArmCollector", PreviewCollector)
    assert (
        cli.main(
            [
                "capture",
                "--side",
                "left",
                "--cameras",
                setup,
                "--preview",
                *fps_args,
                "--preposition",
                "target.json",
                "--from-current",
                "--confirm",
                "PREPOSITION",
            ]
        )
        == 0
    )
    expected_cameras = {"cam_left"} if setup == "wrist" else {"cam_left", "cam_high"}
    assert events == ["preposition", (expected_cameras, True, expected_fps), "capture"]


@pytest.mark.parametrize("side", ["left", "right"])
def test_wrist_preview_filters_other_cameras_and_never_queues_joints(side: str) -> None:
    class PreviewProcess:
        def __init__(self, **kwargs: Any) -> None:
            self.stop_event = kwargs["args"][-1]

        def start(self) -> None:
            pass

        def is_alive(self) -> bool:
            return not self.stop_event.is_set()

        def join(self, **kwargs: Any) -> None:
            pass

    context = SimpleNamespace(Queue=queue.Queue, Event=threading.Event, Process=PreviewProcess)
    now_ns = 1_000_000_000
    preview = WristCameraPreview(
        LivePreviewConfig(camera_roles=(f"cam_{side}",), camera_width=848, camera_height=480, camera_fps=10),
        _context=context,
        _clock_ns=lambda: now_ns,
    )
    image = np.zeros((480, 848, 3), dtype=np.uint8)
    preview.start()
    try:
        assert preview.running
        assert set(preview._camera_queues) == {f"cam_{side}"}
        assert preview._joint_queue is None
        for role in ("cam_high", "cam_left", "cam_right"):
            frame = CameraFrame(role, now_ns, image)
            assert preview.submit_camera(frame) == (role == f"cam_{side}")
        update = preview._camera_queues[f"cam_{side}"].get_nowait()
        assert update.image.shape == (480, 848, 3)
        assert not preview.submit_joints({f"{side}_leader": (0.0,) * 7})
        frame = CameraFrame(f"cam_{side}", now_ns, image)
        assert not preview.submit_camera(frame)
        now_ns += 100_000_000
        assert preview.submit_camera(frame)
        now_ns += 100_000_000
        assert not preview.submit_camera(frame)
    finally:
        preview.stop()
    assert not preview.running


@pytest.mark.parametrize("side", ["left", "right"])
def test_wrist_window_contains_one_large_camera_without_joint_plots(side: str) -> None:
    widgets: list[tuple[str, dict[str, Any]]] = []

    def widget(kind: str, parent: Any, **kwargs: Any) -> Any:
        widgets.append((kind, kwargs))
        return SimpleNamespace(pack=lambda **kwargs: None, create_text=lambda *args, **kwargs: None)

    callbacks: list[Any] = []
    destroyed: list[bool] = []
    root = SimpleNamespace(
        title=lambda title: None,
        configure=lambda **kwargs: None,
        protocol=lambda *args: None,
        after=lambda *args: callbacks.append(args),
        destroy=lambda: destroyed.append(True),
    )
    toolkit = SimpleNamespace(
        Frame=lambda *args, **kwargs: widget("Frame", *args, **kwargs),
        Label=lambda *args, **kwargs: widget("Label", *args, **kwargs),
        Canvas=lambda *args, **kwargs: widget("Canvas", *args, **kwargs),
    )
    config = LivePreviewConfig(camera_roles=(f"cam_{side}",), camera_width=848, camera_height=480)
    frames: queue.Queue[Any] = queue.Queue()
    stop_event = threading.Event()
    app = _WristPreviewApplication(toolkit, root, config, {f"cam_{side}": frames}, None, stop_event)
    assert set(app._camera_canvases) == {f"cam_{side}"}
    assert not app._plot_canvases
    canvases = [options for kind, options in widgets if kind == "Canvas"]
    assert len(canvases) == 1
    assert (canvases[0]["width"], canvases[0]["height"]) == (848, 480)
    labels = [options["text"] for kind, options in widgets if kind == "Label"]
    assert labels == [f"Cam {side.title()} (wrist)"]
    updates: list[Any] = []
    app._draw_camera = lambda role, update: updates.append((role, update))
    older = _CameraUpdate(f"cam_{side}", "rgb24", np.zeros((480, 848, 3), dtype=np.uint8))
    latest = _CameraUpdate(f"cam_{side}", "rgb24", np.ones((480, 848, 3), dtype=np.uint8))
    frames.put_nowait(older)
    frames.put_nowait(latest)
    app._refresh()
    app._refresh()
    assert len(updates) == 1
    assert updates[0][0] == f"cam_{side}"
    assert updates[0][1] is latest
    assert len(callbacks) == 2
    stop_event.set()
    app._refresh()
    assert destroyed == [True]
