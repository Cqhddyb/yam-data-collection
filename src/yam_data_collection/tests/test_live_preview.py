from __future__ import annotations

import queue

import numpy as np
import pytest

from yam_data_collection.live_preview import (
    LivePreview,
    LivePreviewConfig,
    _CameraUpdate,
    _TkPreviewApplication,
    _camera_update_to_rgb,
    _downsample_camera,
    _plot_geometry,
)
from yam_data_collection.models import CameraFrame, RobotSnapshot


class _FakeEvent:
    def __init__(self) -> None:
        self.value = False

    def is_set(self) -> bool:
        return self.value

    def set(self) -> None:
        self.value = True


class _FakeProcess:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.started = False
        self.terminated = False

    def start(self) -> None:
        self.started = True

    def is_alive(self) -> bool:
        return self.started and not self.terminated

    def join(self, timeout: float | None = None) -> None:
        del timeout
        if self.kwargs["args"][-1].is_set():
            self.started = False

    def terminate(self) -> None:
        self.terminated = True


class _FakeQueue(queue.Queue):
    def cancel_join_thread(self) -> None:
        pass

    def close(self) -> None:
        pass


class _FakeContext:
    def __init__(self) -> None:
        self.queues: list[_FakeQueue] = []
        self.process: _FakeProcess | None = None

    def Queue(self, maxsize: int) -> _FakeQueue:
        result = _FakeQueue(maxsize=maxsize)
        self.queues.append(result)
        return result

    def Event(self) -> _FakeEvent:
        return _FakeEvent()

    def Process(self, **kwargs: object) -> _FakeProcess:
        self.process = _FakeProcess(**kwargs)
        return self.process


def _frame(role: str = "cam_left", pixel_format: str = "bgr24") -> CameraFrame:
    channels = 2 if pixel_format == "yuyv422" else 3
    return CameraFrame(
        role=role,
        timestamp_ns=1,
        image_bgr=np.zeros((480, 640, channels), dtype=np.uint8),
        pixel_format=pixel_format,
    )


def _snapshot(role: str) -> RobotSnapshot:
    return RobotSnapshot(
        role=role,
        source_timestamp_ns=1,
        sequence=1,
        position=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        velocity=(),
        effort=(),
    )


def test_live_preview_uses_bounded_queues_and_never_waits_to_publish() -> None:
    now_ns = 1_000_000_000
    context = _FakeContext()
    preview = LivePreview(
        LivePreviewConfig(camera_fps=5.0, joint_fps=10.0),
        _context=context,
        _clock_ns=lambda: now_ns,
    )
    preview.start()

    assert context.process is not None and context.process.started
    assert all(source.maxsize == 1 for source in context.queues)
    assert preview.submit_camera(_frame())
    assert not preview.submit_camera(_frame())
    assert preview.stats.camera_rate_limited["cam_left"] == 1

    now_ns += 200_000_000
    assert not preview.submit_camera(_frame())
    assert preview.stats.camera_queue_dropped["cam_left"] == 1

    preview.stop()
    assert context.process.kwargs["target"].__name__ == "_run_preview"
    assert context.process.kwargs["args"][-1].is_set()


def test_joint_submission_is_rate_limited_and_copies_positions() -> None:
    now_ns = 1_000_000_000
    context = _FakeContext()
    preview = LivePreview(
        LivePreviewConfig(camera_roles=("cam_left",), arm_roles=("left_leader",)),
        _context=context,
        _clock_ns=lambda: now_ns,
    )
    preview.start()

    snapshot = _snapshot("left_leader")
    assert preview.submit_joints({"left_leader": snapshot})
    update = context.queues[-1].get_nowait()
    assert update.timestamp_ns == now_ns
    assert update.positions == {"left_leader": snapshot.position}
    assert not preview.submit_joints({"left_leader": snapshot})
    assert preview.stats.joint_rate_limited == 1

    preview.stop()


def test_bgr_preview_is_downsampled_before_rgb_conversion() -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[..., 0] = 10
    image[..., 1] = 20
    image[..., 2] = 30

    compact = _downsample_camera(image, "bgr24", 320, 240)
    rgb = _camera_update_to_rgb(_CameraUpdate("cam_left", "bgr24", compact))

    assert compact.shape == (240, 320, 3)
    assert compact.flags.c_contiguous
    assert tuple(rgb[0, 0]) == (30, 20, 10)


def test_yuyv_preview_preserves_pairs_and_converts_neutral_gray() -> None:
    image = np.empty((480, 640, 2), dtype=np.uint8)
    image[..., 0] = 126
    image[:, 0::2, 1] = 128
    image[:, 1::2, 1] = 128

    compact = _downsample_camera(image, "yuyv422", 320, 240)
    rgb = _camera_update_to_rgb(_CameraUpdate("cam_left", "yuyv422", compact))

    assert compact.shape == (240, 320, 2)
    assert compact.flags.c_contiguous
    assert np.max(np.abs(rgb[..., 0].astype(int) - rgb[..., 1].astype(int))) == 0
    assert np.max(np.abs(rgb[..., 1].astype(int) - rgb[..., 2].astype(int))) == 0


@pytest.mark.parametrize(
    ("width", "height", "expected_shape"),
    [
        (848, 480, (181, 320, 2)),
        (960, 540, (180, 320, 2)),
        (1280, 720, (180, 320, 2)),
    ],
)
def test_widescreen_yuyv_preview_preserves_source_aspect_ratio(
    width: int,
    height: int,
    expected_shape: tuple[int, int, int],
) -> None:
    image = np.zeros((height, width, 2), dtype=np.uint8)

    compact = _downsample_camera(image, "yuyv422", 320, 240)

    assert compact.shape == expected_shape
    assert compact.shape[1] / compact.shape[0] == pytest.approx(width / height, abs=0.01)


def test_camera_canvas_resizes_to_the_aspect_fitted_frame() -> None:
    class FakeCanvas:
        def __init__(self) -> None:
            self.size: tuple[int, int] | None = None
            self.image_position: tuple[float, float] | None = None

        def configure(self, *, width: int, height: int) -> None:
            self.size = (width, height)

        def delete(self, tag: str) -> None:
            assert tag == "camera"

        def create_image(self, x: float, y: float, **kwargs: object) -> None:
            assert kwargs["anchor"] == "center"
            self.image_position = (x, y)

    class FakeTk:
        class TclError(Exception):
            pass

        @staticmethod
        def PhotoImage(**kwargs: object) -> object:
            assert kwargs["format"] == "PPM"
            return object()

    canvas = FakeCanvas()
    app = object.__new__(_TkPreviewApplication)
    app._tk = FakeTk
    app._root = object()
    app._camera_canvases = {"cam_left": canvas}
    app._camera_photos = {}

    app._draw_camera("cam_left", _CameraUpdate("cam_left", "rgb24", np.zeros((181, 320, 3), dtype=np.uint8)))

    assert canvas.size == (320, 181)
    assert canvas.image_position == (160.0, 90.5)


def test_plot_geometry_maps_all_joint_series_into_canvas() -> None:
    geometry = _plot_geometry(
        (
            (1_000_000_000, (0.0, 1.0, 2.0)),
            (2_000_000_000, (1.0, 0.0, 3.0)),
        ),
        width=475,
        height=190,
    )

    assert geometry is not None
    assert len(geometry.lines) == 3
    assert all(len(line) == 4 for line in geometry.lines)
    assert geometry.y_min < 0.0
    assert geometry.y_max > 3.0


@pytest.mark.parametrize(
    "override",
    [
        {"camera_fps": 0.0},
        {"joint_fps": 0.0},
        {"camera_width": 0},
        {"queue_size": 0},
        {"cpu_set": ()},
    ],
)
def test_live_preview_config_rejects_invalid_limits(override: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        LivePreviewConfig(**override)
