"""Production wiring for the local YAM web console.

The HTTP primitives live in :mod:`web_console` so they can be tested without
hardware.  This module is the deployment adapter: it loads the host config,
feeds the collector's bounded preview callbacks, starts the four i2rt
processes in the documented order, and maps the browser buttons to the
collector lifecycle.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    from .collector import EpisodeCollector
from .live_preview import _CameraUpdate, _camera_update_to_rgb, _downsample_camera
from .web_console import (
    DEFAULT_CAMERA_ROLES,
    DEFAULT_ARM_ROLES,
    ArmCommand,
    ArmProcessSupervisor,
    PreviewBuffer,
    ReadinessProbe,
    WebCaptureController,
    WebConsoleApplication,
    WebConsoleCallbacks,
    WebConsoleState,
    encode_camera_preview,
)


class WebPreviewSink:
    """Bounded, asynchronous bridge from collector callbacks to the browser."""

    def __init__(
        self,
        buffer: PreviewBuffer,
        *,
        max_width: int = 640,
        max_height: int = 360,
        camera_fps: float = 5.0,
        joint_fps: float = 10.0,
    ) -> None:
        if max_width <= 0 or max_height <= 0:
            raise ValueError("preview dimensions must be positive")
        if camera_fps <= 0 or joint_fps <= 0:
            raise ValueError("preview rates must be positive")
        self.buffer = buffer
        self.max_width = int(max_width)
        self.max_height = int(max_height)
        self.camera_interval_ns = max(1, round(1_000_000_000 / float(camera_fps)))
        self.joint_interval_ns = max(1, round(1_000_000_000 / float(joint_fps)))
        # Keep one latest pending item per role. A global FIFO of role names
        # prevents a busy camera from evicting another camera's preview frame.
        self._camera_queue: queue.Queue[str] = queue.Queue(
            maxsize=max(1, len(buffer.camera_roles))
        )
        self._camera_pending: dict[str, tuple[str, str, Any, int]] = {}
        self._queued_roles: set[str] = set()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._last_camera_ns: dict[str, int] = {}
        self._last_joint_ns = 0

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._encode_loop, name="yam-web-preview", daemon=True)
            self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(max(0.0, timeout_s))
        self._thread = None
        with self._lock:
            self._camera_pending.clear()
            self._queued_roles.clear()
        while True:
            try:
                self._camera_queue.get_nowait()
            except queue.Empty:
                break

    def submit_camera(self, frame: Any) -> bool:
        role = str(getattr(frame, "role", ""))
        if role not in self.buffer.camera_roles:
            return False
        now_ns = time.monotonic_ns()
        with self._lock:
            previous = self._last_camera_ns.get(role)
            if previous is not None and now_ns - previous < self.camera_interval_ns:
                return False
            self._last_camera_ns[role] = now_ns
        try:
            pixel_format = str(getattr(frame, "pixel_format", "bgr24")).lower()
            compact = _downsample_camera(
                getattr(frame, "image_bgr"),
                pixel_format,
                self.max_width,
                self.max_height,
            )
            item = (role, pixel_format, compact, int(getattr(frame, "timestamp_ns", time.time_ns())))
            with self._lock:
                self._camera_pending[role] = item
                if role in self._queued_roles:
                    return True
                self._queued_roles.add(role)
                try:
                    self._camera_queue.put_nowait(role)
                except queue.Full:
                    self._queued_roles.discard(role)
                    self._camera_pending.pop(role, None)
                    return False
            return True
        except (AttributeError, OSError, TypeError, ValueError, queue.Full):
            return False
        except Exception:
            return False

    def submit_joints(self, snapshots: Mapping[str, Any], *, timestamp_ns: int | None = None) -> bool:
        now_ns = time.monotonic_ns()
        with self._lock:
            if self._last_joint_ns and now_ns - self._last_joint_ns < self.joint_interval_ns:
                return False
            self._last_joint_ns = now_ns
        values: dict[str, Sequence[float]] = {}
        for role, snapshot in snapshots.items():
            position = getattr(snapshot, "position", snapshot)
            try:
                values[str(role)] = tuple(float(value) for value in position)
            except (TypeError, ValueError):
                continue
        if not values:
            return False
        self.buffer.put_joints(values, timestamp_ns=timestamp_ns)
        return True

    def _encode_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                role = self._camera_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            with self._lock:
                item = self._camera_pending.pop(role, None)
                self._queued_roles.discard(role)
            if item is None:
                continue
            _, pixel_format, image, timestamp_ns = item
            try:
                rgb = _camera_update_to_rgb(_CameraUpdate(role, pixel_format, image))
                data, content_type = encode_camera_preview(
                    rgb,
                    "rgb24",
                    max_width=self.max_width,
                    max_height=self.max_height,
                )
                self.buffer.put_camera(role, data, content_type=content_type, timestamp_ns=timestamp_ns)
            except Exception:
                # Optional preview failures must not affect formal recording.
                continue
            finally:
                # A newer frame may have replaced the item while encoding.
                # Requeue that role so latest-only delivery does not strand it.
                with self._lock:
                    if (
                        role in self._camera_pending
                        and role not in self._queued_roles
                        and not self._stop_event.is_set()
                    ):
                        self._queued_roles.add(role)
                        try:
                            self._camera_queue.put_nowait(role)
                        except queue.Full:
                            self._queued_roles.discard(role)


def build_default_arm_commands(
    i2rt_dir: str | Path,
    *,
    cpu_set: str = "0-15",
    python_executable: str | Path | None = None,
    startup_timeout_s: float = 20.0,
) -> tuple[ArmCommand, ...]:
    """Build the four documented minimum_gello commands in safe order."""
    root = Path(i2rt_dir).expanduser().resolve()
    script = root / "examples/minimum_gello/minimum_gello.py"
    if not script.is_file():
        raise ValueError(f"minimum_gello.py was not found under {root}")
    if python_executable is None:
        candidate = root / ".venv/bin/python"
        python_executable = candidate if candidate.exists() else Path(sys.executable)
    python_path = str(Path(python_executable).expanduser())

    def base(*args: str) -> tuple[str, ...]:
        return ("taskset", "-c", str(cpu_set), python_path, str(script), *args)

    return (
        ArmCommand(
            "left_follower",
            base(
                "--mode",
                "follower",
                "--can-channel",
                "can_follower_l",
                "--gripper",
                "linear_4310",
                "--server-port",
                "1235",
            ),
            (ReadinessProbe("127.0.0.1", 1235),),
            cwd=root,
            startup_timeout_s=startup_timeout_s,
        ),
        ArmCommand(
            "right_follower",
            base(
                "--mode",
                "follower",
                "--can-channel",
                "can_follower_r",
                "--gripper",
                "linear_4310",
                "--server-port",
                "1234",
            ),
            (ReadinessProbe("127.0.0.1", 1234),),
            cwd=root,
            startup_timeout_s=startup_timeout_s,
        ),
        ArmCommand(
            "right_leader",
            base(
                "--mode",
                "leader",
                "--can-channel",
                "can_leader_r",
                "--gripper",
                "yam_teaching_handle",
                "--server-host",
                "127.0.0.1",
                "--server-port",
                "1234",
                "--bilateral-kp",
                "0.0",
                "--telemetry-port",
                "1244",
                "--gripper-command-closed",
                "0.068655",
                "--gripper-command-open",
                "0.960555",
            ),
            (
                ReadinessProbe("127.0.0.1", 1234),
                ReadinessProbe("127.0.0.1", 1244),
            ),
            cwd=root,
            startup_timeout_s=startup_timeout_s,
        ),
        ArmCommand(
            "left_leader",
            base(
                "--mode",
                "leader",
                "--can-channel",
                "can_leader_l",
                "--gripper",
                "yam_teaching_handle",
                "--server-host",
                "127.0.0.1",
                "--server-port",
                "1235",
                "--bilateral-kp",
                "0.0",
                "--telemetry-port",
                "1245",
            ),
            (
                ReadinessProbe("127.0.0.1", 1235),
                ReadinessProbe("127.0.0.1", 1245),
            ),
            cwd=root,
            startup_timeout_s=startup_timeout_s,
        ),
    )


class WebCaptureRuntime:
    """Own the web-facing collector session and arm process supervisor."""

    def __init__(
        self,
        config_path: str | Path,
        capture_mode: str | None,
        *,
        i2rt_dir: str | Path,
        arm_cpu_set: str = "0-15",
        arm_control: bool = True,
        arm_commands: Sequence[ArmCommand] | None = None,
    ) -> None:
        # Keep arm-command generation and the standalone preview sink usable
        # on hosts where optional MCAP/ROS dependencies are not installed.
        # The collector imports are deferred until a real capture runtime is
        # constructed.
        from .collector import CaptureState, EpisodeCollector
        from .config import load_config

        self.config_path = Path(config_path).expanduser().resolve()
        self.capture_mode = capture_mode
        self.config = load_config(self.config_path, capture_mode=capture_mode)
        self._capture_state_type = CaptureState
        self._episode_collector_type = EpisodeCollector
        configured_cameras = tuple(self.config.cameras)
        # Keep four stable tiles even when a host config exposes fewer sources;
        # missing roles remain explicit placeholders rather than fake frames.
        display_cameras = tuple(dict.fromkeys((*configured_cameras, *DEFAULT_CAMERA_ROLES)))[:4]
        self.preview = PreviewBuffer(display_cameras, DEFAULT_ARM_ROLES, joint_history=240)
        self.preview_sink = WebPreviewSink(self.preview)
        self.state = WebConsoleState(preview=self.preview)
        self._lock = threading.RLock()
        self._collector: EpisodeCollector | None = None
        self._collector_started = False
        self._closed = False
        self._last_collector_error: str | None = None
        self._monitor_stop = threading.Event()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, name="yam-web-runtime", daemon=True)

        self.arm_supervisor = (
            ArmProcessSupervisor(
                tuple(arm_commands)
                if arm_commands is not None
                else build_default_arm_commands(i2rt_dir, cpu_set=arm_cpu_set)
            )
            if arm_control
            else None
        )
        self.capture_controller = WebCaptureController(
            self._collector_factory,
            self.preview,
            state=self.state,
            preview_sink=self.preview_sink,
        )
        callbacks = WebConsoleCallbacks(
            start_arms=self.start_arms,
            stop_arms=self.stop_arms,
            start_capture=self.start_capture,
            stop_capture=self.stop_capture,
        )
        # WebConsoleApplication installs the arm event handler and serializes
        # button actions.  The runtime monitor updates capture state when a
        # camera/telemetry worker fails asynchronously.
        self.application = WebConsoleApplication(
            self.state,
            callbacks=callbacks,
            arm_supervisor=self.arm_supervisor,
            capture_controller=self.capture_controller,
        )
        self._monitor_thread.start()
        self.state.message("info", "Web console ready; start arms before capture")
        self.state.message("info", "机械臂就绪后，请分别按左右 teaching handle 顶部同步按钮，再开始采集")
        self.state.message("info", "开始采集后才会打开相机并显示网页预览")
        if len(configured_cameras) < 4:
            self.state.message("info", f"当前配置包含 {len(configured_cameras)} 路相机，其余画面为未配置占位")

    @property
    def collector(self) -> EpisodeCollector | None:
        return self.capture_controller.collector

    def _collector_factory(self, *, preview_sink: object) -> EpisodeCollector:
        return self._episode_collector_type(
            self.config,
            enable_preview=False,
            preview_sink=preview_sink,
        )

    def start_arms(self) -> Any:
        supervisor = self.arm_supervisor
        if supervisor is None:
            raise RuntimeError("arm control is disabled")
        collector = self.collector
        with self._lock:
            if collector is not None and collector.state in {
                self._capture_state_type.ARMING,
                self._capture_state_type.RECORDING,
                self._capture_state_type.FINALIZING,
            }:
                raise RuntimeError("end the active capture before starting arms")
            if collector is not None and collector.state is not self._capture_state_type.CLOSED:
                self.capture_controller.close_collector()
        return supervisor.start_all()

    def stop_arms(self) -> Any:
        supervisor = self.arm_supervisor
        if supervisor is None:
            raise RuntimeError("arm control is disabled")
        collector = self.collector
        with self._lock:
            if collector is not None and collector.state in {
                self._capture_state_type.ARMING,
                self._capture_state_type.RECORDING,
                self._capture_state_type.FINALIZING,
            }:
                raise RuntimeError("end or discard the active capture before stopping arms")
            if collector is not None and collector.state is not self._capture_state_type.CLOSED:
                self.capture_controller.close_collector()
                self._collector_started = False
        return supervisor.stop_all()

    def start_capture(self) -> Any:
        supervisor = self.arm_supervisor
        if supervisor is not None:
            statuses = supervisor.statuses()
            if not statuses or not all(item.get("ready") for item in statuses.values()):
                raise RuntimeError("all four arm processes must be ready before capture")
        with self._lock:
            if self._closed:
                raise RuntimeError("web runtime is closed")
        result = self.capture_controller.start_capture()
        with self._lock:
            self._collector_started = True
        return result

    def stop_capture(self) -> Any:
        return self.capture_controller.stop_capture()

    def status_extra(self) -> Mapping[str, Any]:
        collector = self.collector
        with self._lock:
            started = self._collector_started
        if collector is None:
            return {"collector_ready": False, "collector": None}
        try:
            snapshot = collector.active_episode_snapshot
            state = str(snapshot.get("state", collector.state.value))
            self.state.set_capture_state(state)
            error = snapshot.get("background_error")
            if error:
                self._report_collector_error(str(error))
            return {"collector_ready": started, "collector": snapshot, "capture_state": state}
        except Exception as exc:
            self._report_collector_error(str(exc))
            return {"collector_ready": started, "collector": None}

    def _report_collector_error(self, message: str) -> None:
        if message and message != self._last_collector_error:
            self._last_collector_error = message
            self.state.message("error", message)

    def _monitor_loop(self) -> None:
        while not self._monitor_stop.wait(0.5):
            try:
                self.status_extra()
            except Exception:
                pass

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._monitor_stop.set()
        self._monitor_thread.join(timeout=2.0)
        try:
            self.capture_controller.close()
        except Exception as exc:
            self.state.message("error", f"Collector close failed: {exc}")
        self.preview_sink.stop()
        if self.arm_supervisor is not None:
            try:
                self.arm_supervisor.stop_all()
            except Exception as exc:
                self.state.message("error", f"Arm shutdown failed: {exc}")


__all__ = [
    "WebCaptureRuntime",
    "WebPreviewSink",
    "build_default_arm_commands",
]
