from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from yam_data_collection.cameras import (
    CameraError,
    RealSenseCamera,
    _copy_yuyv,
    _yuyv_to_bgr,
    ffmpeg_input_pixel_format,
)
from yam_data_collection.config import CameraConfig, CameraIntrinsics
from yam_data_collection.models import CameraFrame


class _FakeColor:
    def __init__(
        self,
        *,
        timestamp_ms: float = 1.0,
        timestamp_domain: str = "hardware_clock",
        frame_number: int = 42,
    ) -> None:
        self.timestamp_ms = timestamp_ms
        self.timestamp_domain = timestamp_domain
        self.frame_number = frame_number

    def get_data(self) -> np.ndarray:
        return np.zeros((480, 848, 3), dtype=np.uint8)

    def get_timestamp(self) -> float:
        return self.timestamp_ms

    def get_frame_number(self) -> int:
        return self.frame_number

    def get_frame_timestamp_domain(self) -> str:
        return self.timestamp_domain

    def __bool__(self) -> bool:
        return True


class _FakeFrameset:
    def __init__(self, color: _FakeColor) -> None:
        self.color = color

    def as_frameset(self) -> "_FakeFrameset":
        return self

    def get_color_frame(self) -> _FakeColor:
        return self.color


class _FakeQueue:
    def __init__(self, capacity: int, keep_frames: bool, colors: list[_FakeColor]) -> None:
        self.capacity = capacity
        self.keep_frames = keep_frames
        self._colors = iter(colors)
        self.timeouts: list[int] = []

    def wait_for_frame(self, timeout_ms: int) -> _FakeFrameset:
        self.timeouts.append(timeout_ms)
        return _FakeFrameset(next(self._colors))


class _FakeConfig:
    def enable_device(self, serial: str) -> None:
        self.serial = serial

    def enable_stream(self, *args: object) -> None:
        self.stream = args


class _FakeSensor:
    def __init__(self, *, supports_global_time: bool, set_error: Exception | None = None) -> None:
        self.supports_global_time = supports_global_time
        self.set_error = set_error
        self.supported_options: list[object] = []
        self.set_options: list[tuple[object, float]] = []

    def supports(self, option: object) -> bool:
        self.supported_options.append(option)
        return self.supports_global_time

    def set_option(self, option: object, value: float) -> None:
        if self.set_error is not None:
            raise self.set_error
        self.set_options.append((option, value))


class _FakeProfile:
    def __init__(self, sensors: list[_FakeSensor]) -> None:
        self._sensors = sensors

    def get_device(self) -> SimpleNamespace:
        return SimpleNamespace(query_sensors=lambda: self._sensors)


class _FakePipeline:
    def __init__(self, profile: _FakeProfile) -> None:
        self.profile = profile
        self.started_with: tuple[_FakeConfig, _FakeQueue] | None = None
        self.stopped = False

    def start(self, config: _FakeConfig, queue: _FakeQueue) -> _FakeProfile:
        self.started_with = (config, queue)
        return self.profile

    def stop(self) -> None:
        self.stopped = True


def _fake_realsense(
    colors: list[_FakeColor],
    sensors: list[_FakeSensor] | None = None,
    *,
    expose_global_time_option: bool = True,
) -> tuple[SimpleNamespace, _FakePipeline]:
    """Build the subset of the librealsense API used by ``RealSenseCamera``."""
    pipeline = _FakePipeline(_FakeProfile(sensors or []))
    rs = SimpleNamespace(
        format=SimpleNamespace(bgr8="bgr8"),
        stream=SimpleNamespace(color="color"),
        pipeline=lambda: pipeline,
        config=_FakeConfig,
        frame_queue=lambda capacity, keep_frames: _FakeQueue(capacity, keep_frames, colors),
    )
    if expose_global_time_option:
        rs.option = SimpleNamespace(global_time_enabled="global_time_enabled")
    return rs, pipeline


def _camera_config() -> CameraConfig:
    return CameraConfig(
        "cam_high", "serial", 848, 480, 60, "frame", CameraIntrinsics.uncalibrated(), stream_format="bgr8"
    )


def test_camera_frame_preserves_pixel_format_positional_slot() -> None:
    frame = CameraFrame("cam_high", 1, np.zeros((1, 1, 3), dtype=np.uint8), False, None, "yuyv422")

    assert frame.pixel_format == "yuyv422"
    assert frame.device_frame_number is None
    assert frame.device_timestamp_domain is None


def test_yuyv_conversion_preserves_black_and_white_luma() -> None:
    # One YUYV pair: Y0/U/Y1/V. Neutral chroma must decode to gray BGR pixels.
    packed = np.array([[[16, 128], [235, 128]]], dtype=np.uint8)
    raw = packed.reshape(1, 4).view(np.uint16).reshape(1, 2)

    bgr = _yuyv_to_bgr(raw)

    assert bgr.shape == (1, 2, 3)
    assert bgr.dtype == np.uint8
    assert bgr[0, 0].tolist() == [0, 0, 0]
    assert bgr[0, 1].tolist() == [255, 255, 255]


def test_yuyv_copy_normalizes_realsense_uint16_storage_for_ffmpeg() -> None:
    packed = np.array([[[16, 128], [235, 128]]], dtype=np.uint8)
    raw = packed.reshape(1, 4).view(np.uint16).reshape(1, 2)

    normalized = _copy_yuyv(raw)

    assert normalized.shape == (1, 2, 2)
    assert normalized.dtype == np.uint8
    assert normalized.flags.c_contiguous
    assert normalized.tolist() == packed.tolist()


def test_stream_formats_map_to_ffmpeg_raw_video_formats() -> None:
    assert ffmpeg_input_pixel_format("bgr8") == "bgr24"
    assert ffmpeg_input_pixel_format("yuyv") == "yuyv422"


def test_realsense_camera_uses_a_bounded_native_frame_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_rs, pipeline = _fake_realsense([_FakeColor()])
    monkeypatch.setattr("yam_data_collection.cameras._load_realsense", lambda: fake_rs)
    camera = RealSenseCamera(_camera_config())

    camera.start()
    frame = camera.read(timeout_ms=321)
    camera.stop()

    assert pipeline.started_with is not None
    queue = pipeline.started_with[1]
    assert queue.capacity == 8
    assert queue.keep_frames
    assert queue.timeouts == [321]
    assert frame.image_bgr.shape == (480, 848, 3)
    assert frame.device_frame_number == 42
    assert frame.device_timestamp_domain == "hardware_clock"
    assert frame.capture_timestamp_source == "host_receive"
    assert pipeline.stopped


def test_realsense_camera_enables_global_time_and_uses_trusted_global_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host_timestamp_ns = 1_700_000_000_123_000_000
    global_timestamp_ms = host_timestamp_ns / 1_000_000
    supported = _FakeSensor(supports_global_time=True)
    unsupported = _FakeSensor(supports_global_time=False)
    fake_rs, _pipeline = _fake_realsense(
        [_FakeColor(timestamp_ms=global_timestamp_ms, timestamp_domain="timestamp_domain.global_time")],
        [supported, unsupported],
    )
    monkeypatch.setattr("yam_data_collection.cameras._load_realsense", lambda: fake_rs)
    monkeypatch.setattr("yam_data_collection.cameras.time.time_ns", lambda: host_timestamp_ns)
    camera = RealSenseCamera(_camera_config())

    camera.start()
    frame = camera.read()
    camera.stop()

    assert supported.set_options == [("global_time_enabled", 1.0)]
    assert unsupported.set_options == []
    # Librealsense exposes timestamps as floating-point milliseconds, so an
    # epoch-scale global clock can round by a few tens of nanoseconds.
    assert abs(frame.timestamp_ns - host_timestamp_ns) < 1_000
    assert frame.host_timestamp_ns == host_timestamp_ns
    assert frame.device_timestamp_ns == frame.timestamp_ns
    assert frame.capture_timestamp_source == "realsense_global_time"


def test_realsense_camera_discards_startup_host_frame_before_global_time_is_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host_timestamp_ns = 1_700_000_000_123_000_000
    global_timestamp_ms = host_timestamp_ns / 1_000_000
    fake_rs, _pipeline = _fake_realsense(
        [
            _FakeColor(timestamp_ms=1.0, timestamp_domain="timestamp_domain.hardware_clock"),
            _FakeColor(timestamp_ms=global_timestamp_ms, timestamp_domain="timestamp_domain.global_time"),
        ],
        [_FakeSensor(supports_global_time=True)],
    )
    monkeypatch.setattr("yam_data_collection.cameras._load_realsense", lambda: fake_rs)
    monkeypatch.setattr("yam_data_collection.cameras.time.time_ns", lambda: host_timestamp_ns)
    camera = RealSenseCamera(_camera_config())

    camera.start()
    frame = camera.read()
    camera.stop()

    assert frame.capture_timestamp_source == "realsense_global_time"
    assert abs(frame.timestamp_ns - host_timestamp_ns) < 1_000


def test_realsense_camera_falls_back_to_host_timestamp_without_global_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host_timestamp_ns = 1_700_000_000_123_000_000
    fake_rs, _pipeline = _fake_realsense(
        [_FakeColor(timestamp_ms=123.0, timestamp_domain="timestamp_domain.hardware_clock")]
    )
    monkeypatch.setattr("yam_data_collection.cameras._load_realsense", lambda: fake_rs)
    monkeypatch.setattr("yam_data_collection.cameras.time.time_ns", lambda: host_timestamp_ns)
    camera = RealSenseCamera(_camera_config())

    camera.start()
    frame = camera.read()
    camera.stop()

    assert frame.timestamp_ns == host_timestamp_ns
    assert frame.host_timestamp_ns == host_timestamp_ns
    assert frame.device_timestamp_ns == 123_000_000
    assert frame.capture_timestamp_source == "host_receive"


def test_realsense_camera_falls_back_when_global_time_option_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host_timestamp_ns = 1_700_000_000_123_000_000
    fake_rs, _pipeline = _fake_realsense(
        [_FakeColor(timestamp_ms=123.0, timestamp_domain="timestamp_domain.hardware_clock")],
        expose_global_time_option=False,
    )
    monkeypatch.setattr("yam_data_collection.cameras._load_realsense", lambda: fake_rs)
    monkeypatch.setattr("yam_data_collection.cameras.time.time_ns", lambda: host_timestamp_ns)
    camera = RealSenseCamera(_camera_config())

    camera.start()
    frame = camera.read()
    camera.stop()

    assert frame.timestamp_ns == host_timestamp_ns
    assert frame.capture_timestamp_source == "host_receive"


def test_realsense_camera_keeps_host_capture_time_when_non_global_device_timestamp_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MissingTimestampColor(_FakeColor):
        def get_timestamp(self) -> float:
            raise RuntimeError("timestamp metadata unavailable")

    host_timestamp_ns = 1_700_000_000_123_000_000
    fake_rs, _pipeline = _fake_realsense(
        [MissingTimestampColor(timestamp_domain="timestamp_domain.hardware_clock")]
    )
    monkeypatch.setattr("yam_data_collection.cameras._load_realsense", lambda: fake_rs)
    monkeypatch.setattr("yam_data_collection.cameras.time.time_ns", lambda: host_timestamp_ns)
    camera = RealSenseCamera(_camera_config())

    camera.start()
    frame = camera.read()
    camera.stop()

    assert frame.timestamp_ns == host_timestamp_ns
    assert frame.device_timestamp_ns is None
    assert frame.capture_timestamp_source == "host_receive"


def test_realsense_camera_rejects_a_global_frame_without_timestamp_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MissingTimestampColor(_FakeColor):
        def get_timestamp(self) -> float:
            raise RuntimeError("timestamp metadata unavailable")

    fake_rs, _pipeline = _fake_realsense([MissingTimestampColor(timestamp_domain="global_time")])
    monkeypatch.setattr("yam_data_collection.cameras._load_realsense", lambda: fake_rs)
    camera = RealSenseCamera(_camera_config())

    camera.start()
    with pytest.raises(CameraError, match="global timestamp is unavailable"):
        camera.read()
    camera.stop()


@pytest.mark.parametrize(
    ("colors", "host_timestamps", "error"),
    [
        (
            [_FakeColor(timestamp_ms=1.0, timestamp_domain="global_time")],
            [3_000_000_000],
            "differs from host time",
        ),
        (
            [
                _FakeColor(timestamp_ms=1_000.0, timestamp_domain="global_time"),
                _FakeColor(timestamp_ms=1_000.0, timestamp_domain="global_time"),
            ],
            [1_000_000_000, 1_000_000_000],
            "did not advance",
        ),
    ],
)
def test_realsense_camera_rejects_untrustworthy_global_timestamps(
    monkeypatch: pytest.MonkeyPatch,
    colors: list[_FakeColor],
    host_timestamps: list[int],
    error: str,
) -> None:
    fake_rs, _pipeline = _fake_realsense(colors)
    host_timestamp_iterator = iter(host_timestamps)
    monkeypatch.setattr("yam_data_collection.cameras._load_realsense", lambda: fake_rs)
    monkeypatch.setattr("yam_data_collection.cameras.time.time_ns", lambda: next(host_timestamp_iterator))
    camera = RealSenseCamera(_camera_config())

    camera.start()
    if error == "did not advance":
        camera.read()
    with pytest.raises(CameraError, match=error):
        camera.read()
    camera.stop()


def test_realsense_camera_rejects_a_clock_domain_switch_that_regresses_capture_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_rs, _pipeline = _fake_realsense(
        [
            _FakeColor(timestamp_ms=1_000.0, timestamp_domain="hardware_clock"),
            _FakeColor(timestamp_ms=900.0, timestamp_domain="global_time"),
        ]
    )
    host_timestamps = iter((1_000_000_000, 900_000_000))
    monkeypatch.setattr("yam_data_collection.cameras._load_realsense", lambda: fake_rs)
    monkeypatch.setattr("yam_data_collection.cameras.time.time_ns", lambda: next(host_timestamps))
    camera = RealSenseCamera(_camera_config())

    camera.start()
    camera.read()
    with pytest.raises(CameraError, match="capture timestamp did not advance"):
        camera.read()
    camera.stop()
