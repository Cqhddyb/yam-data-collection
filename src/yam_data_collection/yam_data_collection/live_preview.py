"""Best-effort camera and joint preview isolated from the capture hot path.

The recorder may call :class:`LivePreview` from camera and telemetry worker
threads.  Submission is rate limited, uses bounded multiprocessing queues, and
never waits for the GUI.  A slow or closed preview therefore loses preview
updates instead of delaying the recorded data stream.
"""

from __future__ import annotations

import math
import os
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .models import CameraFrame, RobotSnapshot


DEFAULT_CAMERA_ROLES = ("cam_high", "cam_left", "cam_right")
DEFAULT_ARM_ROLES = ("left_leader", "right_leader", "left_follower", "right_follower")


@dataclass(frozen=True)
class LivePreviewConfig:
    """Low-bandwidth presentation settings independent of capture rates."""

    camera_roles: tuple[str, ...] = DEFAULT_CAMERA_ROLES
    arm_roles: tuple[str, ...] = DEFAULT_ARM_ROLES
    camera_fps: float = 5.0
    joint_fps: float = 10.0
    refresh_fps: float = 10.0
    camera_width: int = 320
    camera_height: int = 240
    joint_history_s: float = 10.0
    queue_size: int = 1
    cpu_set: tuple[int, ...] | None = None
    window_title: str = "YAM live capture preview"

    def __post_init__(self) -> None:
        if not self.camera_roles:
            raise ValueError("camera_roles must not be empty")
        if not self.arm_roles:
            raise ValueError("arm_roles must not be empty")
        if self.camera_fps <= 0 or self.joint_fps <= 0 or self.refresh_fps <= 0:
            raise ValueError("preview frame rates must be positive")
        if self.camera_width <= 0 or self.camera_height <= 0:
            raise ValueError("preview camera dimensions must be positive")
        if self.joint_history_s <= 0:
            raise ValueError("joint_history_s must be positive")
        if self.queue_size <= 0:
            raise ValueError("queue_size must be positive")
        if self.cpu_set is not None and (not self.cpu_set or any(cpu < 0 for cpu in self.cpu_set)):
            raise ValueError("cpu_set must contain non-negative CPU indices")


@dataclass(frozen=True)
class LivePreviewStats:
    """Producer-side counters; queue drops never invalidate an episode."""

    camera_submitted: Mapping[str, int]
    camera_rate_limited: Mapping[str, int]
    camera_queue_dropped: Mapping[str, int]
    joint_submitted: int
    joint_rate_limited: int
    joint_queue_dropped: int


@dataclass(frozen=True)
class _CameraUpdate:
    role: str
    pixel_format: str
    image: np.ndarray


@dataclass(frozen=True)
class _JointUpdate:
    timestamp_ns: int
    positions: Mapping[str, tuple[float, ...]]


@dataclass(frozen=True)
class _PlotGeometry:
    lines: tuple[tuple[float, ...], ...]
    y_min: float
    y_max: float


class _RateLimiter:
    def __init__(self, fps: float) -> None:
        self._interval_ns = max(1, round(1_000_000_000 / fps))
        self._last_ns: dict[str, int] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, now_ns: int) -> bool:
        with self._lock:
            last_ns = self._last_ns.get(key)
            if last_ns is not None and 0 <= now_ns - last_ns < self._interval_ns:
                return False
            self._last_ns[key] = now_ns
            return True


def _packed_yuyv(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2 and image.dtype == np.uint16:
        height, width = image.shape
        return image.view(np.uint8).reshape(height, width, 2)
    if image.ndim == 3 and image.dtype == np.uint8 and image.shape[2] == 2:
        return image
    raise ValueError(f"unexpected YUYV preview shape/dtype: {image.shape}/{image.dtype}")


def _aspect_fit_dimensions(
    width: int,
    height: int,
    max_width: int,
    max_height: int,
    *,
    even_width: bool = False,
) -> tuple[int, int]:
    """Fit source dimensions inside a preview box without changing aspect ratio."""
    if min(width, height, max_width, max_height) <= 0:
        raise ValueError("preview dimensions must be positive")
    scale = min(1.0, max_width / width, max_height / height)
    target_width = max(1, min(width, max_width, round(width * scale)))
    if even_width:
        target_width -= target_width % 2
        if target_width < 2:
            raise ValueError("YUYV preview width must be at least 2 pixels")
    target_height = max(1, min(height, max_height, round(height * target_width / width)))
    return target_width, target_height


def _sample_indices(source_size: int, target_size: int) -> np.ndarray:
    if source_size == target_size:
        return np.arange(source_size, dtype=np.intp)
    return np.rint(np.linspace(0, source_size - 1, target_size)).astype(np.intp)


def _downsample_camera(
    image: np.ndarray,
    pixel_format: str,
    max_width: int,
    max_height: int,
) -> np.ndarray:
    """Make a small owned frame before crossing the process boundary."""
    frame = np.asarray(image)
    normalized_format = pixel_format.lower()
    if normalized_format == "yuyv422":
        packed = _packed_yuyv(frame)
        height, width, _ = packed.shape
        if width % 2:
            raise ValueError(f"YUYV preview width must be even, got {width}")
        target_width, target_height = _aspect_fit_dimensions(
            width,
            height,
            max_width,
            max_height,
            even_width=True,
        )
        pairs = packed.reshape(height, width // 2, 4)
        rows = _sample_indices(height, target_height)
        pair_columns = _sample_indices(width // 2, target_width // 2)
        sampled = np.ascontiguousarray(pairs[rows[:, None], pair_columns[None, :], :])
        return sampled.reshape(sampled.shape[0], sampled.shape[1] * 2, 2)

    if normalized_format in {"bgr24", "rgb24"}:
        if frame.ndim != 3 or frame.dtype != np.uint8 or frame.shape[2] != 3:
            raise ValueError(f"unexpected {normalized_format} preview shape/dtype: {frame.shape}/{frame.dtype}")
        height, width, _ = frame.shape
        target_width, target_height = _aspect_fit_dimensions(width, height, max_width, max_height)
        rows = _sample_indices(height, target_height)
        columns = _sample_indices(width, target_width)
        return np.ascontiguousarray(frame[rows[:, None], columns[None, :], :])

    raise ValueError(f"unsupported preview pixel format: {pixel_format}")


def _camera_update_to_rgb(update: _CameraUpdate) -> np.ndarray:
    """Convert a compact worker-side frame to RGB for Tk's PPM decoder."""
    image = update.image
    if update.pixel_format == "rgb24":
        return np.ascontiguousarray(image)
    if update.pixel_format == "bgr24":
        return np.ascontiguousarray(image[..., ::-1])
    if update.pixel_format != "yuyv422":
        raise ValueError(f"unsupported preview pixel format: {update.pixel_format}")

    packed = _packed_yuyv(image)
    height, width, _ = packed.shape
    if width % 2:
        raise ValueError(f"YUYV preview width must be even, got {width}")
    y = packed[..., 0].astype(np.int32)
    u = np.repeat(packed[:, 0::2, 1], 2, axis=1).astype(np.int32) - 128
    v = np.repeat(packed[:, 1::2, 1], 2, axis=1).astype(np.int32) - 128
    luminance = np.maximum(y - 16, 0)
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip((298 * luminance + 409 * v + 128) >> 8, 0, 255)
    rgb[..., 1] = np.clip((298 * luminance - 100 * u - 208 * v + 128) >> 8, 0, 255)
    rgb[..., 2] = np.clip((298 * luminance + 516 * u + 128) >> 8, 0, 255)
    return rgb


def _ppm_bytes(rgb: np.ndarray) -> bytes:
    height, width, channels = rgb.shape
    if rgb.dtype != np.uint8 or channels != 3:
        raise ValueError("PPM input must be an HxWx3 uint8 RGB array")
    return f"P6\n{width} {height}\n255\n".encode("ascii") + np.ascontiguousarray(rgb).tobytes()


def _drain_latest(source: Any) -> Any | None:
    """Drain a bounded queue without waiting and return only its newest item."""
    latest = None
    while True:
        try:
            latest = source.get_nowait()
        except queue.Empty:
            return latest
        except (EOFError, OSError, ValueError):
            return latest


def _plot_geometry(
    samples: Sequence[tuple[int, tuple[float, ...]]],
    width: int,
    height: int,
    *,
    left: int = 42,
    right: int = 10,
    top: int = 24,
    bottom: int = 24,
) -> _PlotGeometry | None:
    """Return canvas coordinates for all joints without depending on Tk."""
    if not samples:
        return None
    joint_count = min((len(values) for _, values in samples), default=0)
    if joint_count == 0:
        return None
    timestamps = np.asarray([timestamp for timestamp, _ in samples], dtype=np.float64)
    values = np.asarray([positions[:joint_count] for _, positions in samples], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        return None

    y_min = float(values.min())
    y_max = float(values.max())
    if y_max - y_min < 1e-9:
        y_min -= 0.5
        y_max += 0.5
    else:
        padding = (y_max - y_min) * 0.08
        y_min -= padding
        y_max += padding

    plot_width = max(1, width - left - right)
    plot_height = max(1, height - top - bottom)
    first_timestamp = timestamps[0]
    duration = max(1.0, timestamps[-1] - first_timestamp)
    x_values = left + (timestamps - first_timestamp) * plot_width / duration
    lines: list[tuple[float, ...]] = []
    for index in range(joint_count):
        y_values = top + (y_max - values[:, index]) * plot_height / (y_max - y_min)
        coordinates = np.column_stack((x_values, y_values)).reshape(-1)
        lines.append(tuple(float(value) for value in coordinates))
    return _PlotGeometry(tuple(lines), y_min, y_max)


class LivePreview:
    """Publish compact best-effort updates to an isolated Tk process."""

    def __init__(
        self,
        config: LivePreviewConfig | None = None,
        *,
        _context: Any | None = None,
        _clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.config = config or LivePreviewConfig()
        self._context = _context
        self._clock_ns = _clock_ns
        self._camera_limiter = _RateLimiter(self.config.camera_fps)
        self._joint_limiter = _RateLimiter(self.config.joint_fps)
        self._camera_queues: dict[str, Any] = {}
        self._joint_queue: Any | None = None
        self._stop_event: Any | None = None
        self._process: Any | None = None
        self._started = False
        self._closed = False
        self._lifecycle_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._camera_submitted = {role: 0 for role in self.config.camera_roles}
        self._camera_rate_limited = {role: 0 for role in self.config.camera_roles}
        self._camera_queue_dropped = {role: 0 for role in self.config.camera_roles}
        self._joint_submitted = 0
        self._joint_rate_limited = 0
        self._joint_queue_dropped = 0

    def start(self) -> None:
        """Spawn the GUI without importing or running Tk in the recorder."""
        with self._lifecycle_lock:
            if self._started:
                return
            if self._closed:
                raise RuntimeError("a stopped live preview cannot be restarted")
            if self._context is None:
                import multiprocessing

                self._context = multiprocessing.get_context("spawn")
            self._camera_queues = {
                role: self._context.Queue(maxsize=self.config.queue_size) for role in self.config.camera_roles
            }
            self._joint_queue = self._context.Queue(maxsize=self.config.queue_size)
            self._stop_event = self._context.Event()
            self._process = self._context.Process(
                target=_run_preview,
                args=(self.config, self._camera_queues, self._joint_queue, self._stop_event),
                name="yam-live-preview",
                daemon=True,
            )
            try:
                self._process.start()
            except BaseException:
                self._close_queues()
                self._process = None
                self._stop_event = None
                raise
            self._started = True

    @property
    def running(self) -> bool:
        process = self._process
        stop_event = self._stop_event
        return bool(
            self._started
            and not self._closed
            and process is not None
            and process.is_alive()
            and stop_event is not None
            and not stop_event.is_set()
        )

    @property
    def stats(self) -> LivePreviewStats:
        with self._stats_lock:
            return LivePreviewStats(
                camera_submitted=dict(self._camera_submitted),
                camera_rate_limited=dict(self._camera_rate_limited),
                camera_queue_dropped=dict(self._camera_queue_dropped),
                joint_submitted=self._joint_submitted,
                joint_rate_limited=self._joint_rate_limited,
                joint_queue_dropped=self._joint_queue_dropped,
            )

    def submit_camera(self, frame: CameraFrame) -> bool:
        """Offer one camera frame without waiting for the GUI."""
        role = frame.role
        destination = self._camera_queues.get(role)
        stop_event = self._stop_event
        if destination is None or stop_event is None or stop_event.is_set():
            return False
        if not self._camera_limiter.allow(role, self._clock_ns()):
            with self._stats_lock:
                self._camera_rate_limited[role] += 1
            return False
        try:
            image = _downsample_camera(
                frame.image_bgr,
                frame.pixel_format,
                self.config.camera_width,
                self.config.camera_height,
            )
            destination.put_nowait(_CameraUpdate(role, frame.pixel_format.lower(), image))
        except (queue.Full, EOFError, OSError, ValueError):
            with self._stats_lock:
                self._camera_queue_dropped[role] += 1
            return False
        except Exception:
            # Preview conversion must never turn a malformed optional frame
            # into a recorder or camera-worker failure.
            with self._stats_lock:
                self._camera_queue_dropped[role] += 1
            return False
        with self._stats_lock:
            self._camera_submitted[role] += 1
        return True

    def submit_joints(
        self,
        snapshots: Mapping[str, RobotSnapshot | Sequence[float]],
        *,
        timestamp_ns: int | None = None,
    ) -> bool:
        """Offer one four-arm position snapshot without waiting for the GUI."""
        destination = self._joint_queue
        stop_event = self._stop_event
        if destination is None or stop_event is None or stop_event.is_set():
            return False
        now_ns = self._clock_ns()
        if not self._joint_limiter.allow("joints", now_ns):
            with self._stats_lock:
                self._joint_rate_limited += 1
            return False

        positions: dict[str, tuple[float, ...]] = {}
        try:
            for role in self.config.arm_roles:
                snapshot = snapshots.get(role)
                if snapshot is None:
                    continue
                values = snapshot.position if isinstance(snapshot, RobotSnapshot) else snapshot
                normalized = tuple(float(value) for value in values)
                if normalized and all(math.isfinite(value) for value in normalized):
                    positions[role] = normalized
            if not positions:
                raise ValueError("joint preview received no finite positions")
            destination.put_nowait(_JointUpdate(timestamp_ns if timestamp_ns is not None else now_ns, positions))
        except (queue.Full, EOFError, OSError, TypeError, ValueError):
            with self._stats_lock:
                self._joint_queue_dropped += 1
            return False
        except Exception:
            # Keep the preview strictly best effort if a caller hands us an
            # incomplete or otherwise unexpected snapshot mapping.
            with self._stats_lock:
                self._joint_queue_dropped += 1
            return False
        with self._stats_lock:
            self._joint_submitted += 1
        return True

    def stop(self, timeout_s: float = 2.0) -> None:
        """Request GUI shutdown; terminate it if a toolkit call is wedged."""
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            stop_event = self._stop_event
            process = self._process
            if stop_event is not None:
                stop_event.set()
            if process is not None:
                process.join(timeout=max(0.0, timeout_s))
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=0.5)
            self._close_queues()

    def _close_queues(self) -> None:
        sources = [*self._camera_queues.values()]
        if self._joint_queue is not None:
            sources.append(self._joint_queue)
        for source in sources:
            try:
                source.cancel_join_thread()
            except (AttributeError, OSError, ValueError):
                pass
            try:
                source.close()
            except (AttributeError, OSError, ValueError):
                pass
        self._camera_queues = {}
        self._joint_queue = None

    def __enter__(self) -> "LivePreview":
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()


_JOINT_COLORS = ("#e15759", "#4e79a7", "#59a14f", "#f28e2b", "#b07aa1", "#76b7b2", "#edc948")


class _TkPreviewApplication:
    def __init__(
        self,
        tk: Any,
        root: Any,
        config: LivePreviewConfig,
        camera_queues: Mapping[str, Any],
        joint_queue: Any,
        stop_event: Any,
    ) -> None:
        self._tk = tk
        self._root = root
        self._config = config
        self._camera_queues = camera_queues
        self._joint_queue = joint_queue
        self._stop_event = stop_event
        self._camera_canvases: dict[str, Any] = {}
        self._camera_photos: dict[str, Any] = {}
        self._plot_canvases: dict[str, Any] = {}
        self._joint_history: dict[str, deque[tuple[int, tuple[float, ...]]]] = {
            role: deque() for role in config.arm_roles
        }
        self._closed = False
        self._refresh_ms = max(20, round(1_000 / config.refresh_fps))
        self._build()

    def _build(self) -> None:
        self._root.title(self._config.window_title)
        self._root.configure(background="#15171a")
        self._root.protocol("WM_DELETE_WINDOW", self.close)

        camera_row = self._tk.Frame(self._root, background="#15171a")
        camera_row.pack(fill="x", padx=8, pady=(8, 4))
        for column, role in enumerate(self._config.camera_roles):
            panel = self._tk.Frame(camera_row, background="#202328")
            panel.grid(row=0, column=column, padx=4, sticky="n")
            self._tk.Label(
                panel,
                text=role.replace("_", " ").title(),
                foreground="#f1f3f5",
                background="#202328",
                font=("TkDefaultFont", 10, "bold"),
            ).pack(fill="x", pady=(4, 2))
            canvas = self._tk.Canvas(
                panel,
                width=self._config.camera_width,
                height=self._config.camera_height,
                background="#08090a",
                highlightthickness=0,
            )
            canvas.pack(padx=4, pady=(0, 4))
            canvas.create_text(
                self._config.camera_width / 2,
                self._config.camera_height / 2,
                text="Waiting for frames",
                fill="#8b929a",
                tags=("camera",),
            )
            self._camera_canvases[role] = canvas

        plots = self._tk.Frame(self._root, background="#15171a")
        plots.pack(fill="both", expand=True, padx=8, pady=(4, 8))
        for index, role in enumerate(self._config.arm_roles):
            panel = self._tk.Frame(plots, background="#202328")
            panel.grid(row=index // 2, column=index % 2, padx=4, pady=4, sticky="nsew")
            self._tk.Label(
                panel,
                text=role.replace("_", " ").title(),
                foreground="#f1f3f5",
                background="#202328",
                font=("TkDefaultFont", 10, "bold"),
            ).pack(fill="x", pady=(4, 0))
            canvas = self._tk.Canvas(
                panel,
                width=475,
                height=190,
                background="#101215",
                highlightthickness=0,
            )
            canvas.pack(padx=4, pady=4, fill="both", expand=True)
            self._plot_canvases[role] = canvas
            self._draw_plot(role)
        for index in range(2):
            plots.columnconfigure(index, weight=1)
            plots.rowconfigure(index, weight=1)

    def run(self) -> None:
        self._root.after(self._refresh_ms, self._refresh)
        self._root.mainloop()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        try:
            self._root.destroy()
        except Exception:
            pass

    def _refresh(self) -> None:
        if self._closed:
            return
        if self._stop_event.is_set():
            self.close()
            return
        for role, source in self._camera_queues.items():
            update = _drain_latest(source)
            if isinstance(update, _CameraUpdate):
                self._draw_camera(role, update)
        joint_update = _drain_latest(self._joint_queue)
        if isinstance(joint_update, _JointUpdate):
            self._add_joint_update(joint_update)
        self._root.after(self._refresh_ms, self._refresh)

    def _draw_camera(self, role: str, update: _CameraUpdate) -> None:
        canvas = self._camera_canvases.get(role)
        if canvas is None:
            return
        try:
            rgb = _camera_update_to_rgb(update)
            photo = self._tk.PhotoImage(master=self._root, data=_ppm_bytes(rgb), format="PPM")
        except (ValueError, self._tk.TclError):
            return
        height, width, _ = rgb.shape
        canvas.configure(width=width, height=height)
        canvas.delete("camera")
        canvas.create_image(
            width / 2,
            height / 2,
            image=photo,
            anchor="center",
            tags=("camera",),
        )
        self._camera_photos[role] = photo

    def _add_joint_update(self, update: _JointUpdate) -> None:
        cutoff_ns = update.timestamp_ns - round(self._config.joint_history_s * 1_000_000_000)
        for role, positions in update.positions.items():
            history = self._joint_history.get(role)
            if history is None:
                continue
            history.append((update.timestamp_ns, positions))
            while history and history[0][0] < cutoff_ns:
                history.popleft()
            self._draw_plot(role)

    def _draw_plot(self, role: str) -> None:
        canvas = self._plot_canvases[role]
        width = max(100, int(float(canvas.cget("width"))))
        height = max(80, int(float(canvas.cget("height"))))
        canvas.delete("all")
        canvas.create_line(42, 24, 42, height - 24, fill="#59616b")
        canvas.create_line(42, height - 24, width - 10, height - 24, fill="#59616b")
        geometry = _plot_geometry(tuple(self._joint_history[role]), width, height)
        if geometry is None:
            canvas.create_text(width / 2, height / 2, text="Waiting for joints", fill="#8b929a")
            return
        canvas.create_text(4, 14, text=f"{geometry.y_max:.2f}", fill="#a9b0b7", anchor="w")
        canvas.create_text(4, height - 25, text=f"{geometry.y_min:.2f}", fill="#a9b0b7", anchor="w")
        for index, coordinates in enumerate(geometry.lines):
            color = _JOINT_COLORS[index % len(_JOINT_COLORS)]
            if len(coordinates) >= 4:
                canvas.create_line(*coordinates, fill=color, width=1.5)
            canvas.create_text(
                48 + index * 51,
                12,
                text=f"J{index + 1}",
                fill=color,
                anchor="w",
                font=("TkDefaultFont", 8),
            )


def _run_preview(
    config: LivePreviewConfig,
    camera_queues: Mapping[str, Any],
    joint_queue: Any,
    stop_event: Any,
) -> None:
    """Multiprocessing entry point; all Tk imports and calls stay here."""
    try:
        if config.cpu_set is not None and hasattr(os, "sched_setaffinity"):
            try:
                os.sched_setaffinity(0, set(config.cpu_set))
            except OSError as exc:
                print(f"Live preview CPU affinity was not applied: {exc}", file=sys.stderr)
        import tkinter as tk

        root = tk.Tk()
        _TkPreviewApplication(tk, root, config, camera_queues, joint_queue, stop_event).run()
    except Exception as exc:
        print(f"Live preview unavailable: {exc}", file=sys.stderr)
    finally:
        stop_event.set()
