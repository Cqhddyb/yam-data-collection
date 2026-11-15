"""RealSense capture backend keyed by physical camera serial number."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .config import CameraConfig
from .models import CameraFrame


_FRAME_QUEUE_CAPACITY = 8
_GLOBAL_TIME_MAX_HOST_OFFSET_NS = 1_000_000_000
_GLOBAL_TIME_SETTLE_S = 0.25


def _frame_number(color: Any) -> int | None:
    """Read the device frame counter when the RealSense backend exposes it."""
    try:
        return int(color.get_frame_number())
    except Exception:
        return None


def _frame_timestamp_domain(color: Any) -> str | None:
    """Make the RealSense timestamp domain safe to persist in diagnostics."""
    try:
        return str(color.get_frame_timestamp_domain())
    except Exception:
        return None


def _uses_global_time(timestamp_domain: str | None) -> bool:
    """Return whether librealsense mapped the frame clock to system time."""
    return bool(timestamp_domain and "global_time" in timestamp_domain.lower())


def _enable_global_time(rs: Any, pipeline_profile: Any) -> bool:
    """Explicitly enable global time on every sensor that exposes the option."""
    if pipeline_profile is None:
        return False
    try:
        device = pipeline_profile.get_device()
        option = rs.option.global_time_enabled
        sensors = device.query_sensors()
    except (AttributeError, RuntimeError, TypeError):
        return False
    configured = False
    for sensor in sensors:
        try:
            if sensor.supports(option):
                sensor.set_option(option, 1.0)
                configured = True
        except Exception as exc:
            raise CameraError("could not enable RealSense global time") from exc
    return configured


class CameraError(RuntimeError):
    """A configured camera could not be discovered or deliver a color frame."""


class Camera(Protocol):
    role: str

    def start(self) -> None: ...

    def read(self, timeout_ms: int = 1000) -> CameraFrame: ...

    def stop(self) -> None: ...


def _copy_yuyv(image: np.ndarray) -> np.ndarray:
    """Normalize a RealSense YUYV frame to packed uint8 YUYV422 bytes."""
    if image.ndim == 2 and image.dtype == np.uint16:
        height, width = image.shape
        packed = image.view(np.uint8).reshape(height, width, 2)
    elif image.ndim == 3 and image.dtype == np.uint8 and image.shape[2] == 2:
        packed = image
    else:
        raise CameraError(f"unexpected YUYV frame shape/dtype: {image.shape}/{image.dtype}")
    return packed.copy(order="C")


def _yuyv_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert packed YUYV 4:2:2 to BGR for callers that explicitly need BGR."""
    packed = _copy_yuyv(image)
    height, width, _ = packed.shape
    if width % 2:
        raise CameraError(f"YUYV frame width must be even, got {width}")

    y = packed[..., 0].astype(np.int32)
    u = np.repeat(packed[:, 0::2, 1], 2, axis=1).astype(np.int32)
    v = np.repeat(packed[:, 1::2, 1], 2, axis=1).astype(np.int32)
    luminance = np.maximum(y - 16, 0)
    chroma_u = u - 128
    chroma_v = v - 128
    bgr = np.empty((height, width, 3), dtype=np.uint8)
    bgr[..., 0] = np.clip((298 * luminance + 516 * chroma_u + 128) >> 8, 0, 255)
    bgr[..., 1] = np.clip((298 * luminance - 100 * chroma_u - 208 * chroma_v + 128) >> 8, 0, 255)
    bgr[..., 2] = np.clip((298 * luminance + 409 * chroma_v + 128) >> 8, 0, 255)
    return bgr


def ffmpeg_input_pixel_format(stream_format: str) -> str:
    """Map configured RealSense stream formats to ffmpeg raw-video names."""
    formats = {"bgr8": "bgr24", "yuyv": "yuyv422"}
    try:
        return formats[stream_format]
    except KeyError as exc:
        raise CameraError(f"unsupported stream format for ffmpeg: {stream_format}") from exc


def _load_realsense() -> Any:
    try:
        import pyrealsense2 as rs
    except ImportError as exc:  # pragma: no cover - requires hardware dependency
        raise CameraError(
            "pyrealsense2 is unavailable; install the camera extra with `uv sync --extra camera`"
        ) from exc
    return rs


def _device_timestamp_ns(color: Any) -> int | None:
    """Read RealSense milliseconds without turning metadata failure into a fake clock."""
    try:
        return int(round(float(color.get_timestamp()) * 1_000_000))
    except (AttributeError, RuntimeError, TypeError, ValueError, OverflowError):
        return None


@dataclass
class RealSenseCamera:
    """One independent RealSense color pipeline with BGR output frames."""

    config: CameraConfig

    def __post_init__(self) -> None:
        self.role = self.config.role
        self._rs: Any = None
        self._pipeline: Any = None
        self._frame_queue: Any = None
        self._running = False
        self._global_time_requested = False
        self._global_time_wait_deadline: float | None = None
        self._capture_timestamp_source: str | None = None
        self._last_global_timestamp_ns: int | None = None
        self._last_capture_timestamp_ns: int | None = None

    def start(self) -> None:
        rs = _load_realsense()
        pipeline = rs.pipeline()
        stream_config = rs.config()
        stream_config.enable_device(self.config.serial)
        stream_format = getattr(rs.format, self.config.stream_format)
        stream_config.enable_stream(
            rs.stream.color, self.config.width, self.config.height, stream_format, self.config.fps
        )
        # ``pipeline.wait_for_frames`` drops frames produced while Python is
        # writing MCAP metadata or queueing a video frame. This bounded native
        # queue preserves short scheduling stalls without adding unbounded RAM.
        frame_queue = rs.frame_queue(_FRAME_QUEUE_CAPACITY, keep_frames=True)
        try:
            pipeline_profile = pipeline.start(stream_config, frame_queue)
            self._global_time_requested = _enable_global_time(rs, pipeline_profile)
        except Exception as exc:  # pragma: no cover - requires hardware
            try:
                pipeline.stop()
            except Exception:
                pass
            raise CameraError(
                f"could not start {self.role} (RealSense serial {self.config.serial}) at "
                f"{self.config.width}x{self.config.height}@{self.config.fps}: {exc}"
            ) from exc
        self._rs = rs
        self._pipeline = pipeline
        self._frame_queue = frame_queue
        self._global_time_wait_deadline = time.monotonic() + _GLOBAL_TIME_SETTLE_S
        self._running = True

    def read(self, timeout_ms: int = 1000) -> CameraFrame:
        if not self._running or self._frame_queue is None:
            raise CameraError(f"camera {self.role} is not running")
        deadline = time.monotonic() + timeout_ms / 1_000
        while True:
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                raise CameraError(f"camera {self.role} did not produce a frame within {timeout_ms} ms")
            try:
                frames = self._frame_queue.wait_for_frame(max(1, round(remaining_s * 1_000))).as_frameset()
                color = frames.get_color_frame()
            except Exception as exc:  # pragma: no cover - requires hardware
                raise CameraError(f"camera {self.role} did not produce a frame: {exc}") from exc
            if not color:
                raise CameraError(f"camera {self.role} produced a frameset without a color frame")
            host_timestamp_ns = time.time_ns()
            device_timestamp_ns = _device_timestamp_ns(color)
            timestamp_domain = _frame_timestamp_domain(color)
            uses_global_time = _uses_global_time(timestamp_domain)

            # A sensor can emit one or two startup frames before the option
            # takes effect. Consume those rather than mixing host and global
            # timestamps in an episode. If it never switches, lock host time
            # for the session so the fallback remains monotonic and explicit.
            if (
                self._capture_timestamp_source is None
                and self._global_time_requested
                and not uses_global_time
                and self._global_time_wait_deadline is not None
                and time.monotonic() < self._global_time_wait_deadline
            ):
                continue

            if self._capture_timestamp_source == "realsense_global_time" and not uses_global_time:
                raise CameraError(f"camera {self.role} stopped reporting RealSense global time")
            if self._capture_timestamp_source == "host_receive":
                uses_global_time = False

            if uses_global_time:
                if device_timestamp_ns is None:
                    raise CameraError(f"camera {self.role} global timestamp is unavailable")
                offset_ns = abs(host_timestamp_ns - device_timestamp_ns)
                if offset_ns > _GLOBAL_TIME_MAX_HOST_OFFSET_NS:
                    raise CameraError(
                        f"camera {self.role} global timestamp differs from host time by "
                        f"{offset_ns / 1_000_000:.1f} ms"
                    )
                if self._last_global_timestamp_ns is not None and device_timestamp_ns <= self._last_global_timestamp_ns:
                    raise CameraError(f"camera {self.role} global timestamp did not advance")
                timestamp_ns = device_timestamp_ns
                timestamp_source = "realsense_global_time"
            else:
                timestamp_ns = host_timestamp_ns
                timestamp_source = "host_receive"
            if self._last_capture_timestamp_ns is not None and timestamp_ns <= self._last_capture_timestamp_ns:
                raise CameraError(f"camera {self.role} capture timestamp did not advance")
            if self._capture_timestamp_source is None:
                self._capture_timestamp_source = timestamp_source
            break
        # RealSense owns ``color`` only until this method returns. Preserve the
        # D435's lower-bandwidth packed YUYV form and let ffmpeg convert it in
        # optimized native code instead of converting every frame in Python.
        raw_image = np.asanyarray(color.get_data())
        is_yuyv = self.config.stream_format == "yuyv"
        image = _copy_yuyv(raw_image) if is_yuyv else raw_image.copy(order="C")
        if uses_global_time:
            self._last_global_timestamp_ns = timestamp_ns
        self._last_capture_timestamp_ns = timestamp_ns
        return CameraFrame(
            role=self.role,
            timestamp_ns=timestamp_ns,
            image_bgr=image,
            owns_image=True,
            device_timestamp_ns=device_timestamp_ns,
            device_frame_number=_frame_number(color),
            device_timestamp_domain=timestamp_domain,
            host_timestamp_ns=host_timestamp_ns,
            capture_timestamp_source=timestamp_source,
            pixel_format=ffmpeg_input_pixel_format(self.config.stream_format),
        )

    def stop(self) -> None:
        if self._running and self._pipeline is not None:
            self._pipeline.stop()
        self._running = False
        self._pipeline = None
        self._frame_queue = None
        self._global_time_requested = False
        self._global_time_wait_deadline = None
        self._capture_timestamp_source = None
        self._last_global_timestamp_ns = None
        self._last_capture_timestamp_ns = None


def discover_realsense() -> list[dict[str, str]]:
    """List connected RealSense devices for serial-based configuration."""
    rs = _load_realsense()
    context = rs.context()
    devices: list[dict[str, str]] = []
    for device in context.query_devices():  # pragma: no cover - requires hardware
        devices.append(
            {
                "serial": device.get_info(rs.camera_info.serial_number),
                "name": device.get_info(rs.camera_info.name),
                "product_line": device.get_info(rs.camera_info.product_line)
                if device.supports(rs.camera_info.product_line)
                else "unknown",
            }
        )
    return devices
