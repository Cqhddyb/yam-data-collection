"""Episode coordinator: four Portal streams, three cameras, one MCAP, and MP4 assets."""

from __future__ import annotations

import concurrent.futures
import copy
import enum
import json
import os
import select
import shutil
import sys
import termios
import threading
import time
import tty
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from .cameras import Camera, RealSenseCamera, ffmpeg_input_pixel_format
from .config import DatasetConfig, cpu_set_members
from .dataset_lock import DatasetOperationLock
from .mcap_writer import EpisodeMcapWriter
from .live_preview import LivePreview, LivePreviewConfig
from .models import CameraFrame, CameraTimingSample, RobotSnapshot
from .quality import build_validation_policy, print_validation, validate_mcap, validate_video_assets
from .runtime_diagnostics import RuntimeDiagnosticsMonitor
from .telemetry import PortalTelemetryClient, TelemetryError
from .terminal import colorize
from .video import AsyncFfmpegWriter, TimestampedVideoWriter, close_video_writers
from .video_pipeline import (
    final_codec_family,
    live_video_dir,
    pending_episode_dir,
    pending_mcap_path,
    publish_live_episode,
    remove_episode_assets,
    timestamped_mcap_filename,
    uses_direct_h264_publication,
    validate_live_episode,
)


class CaptureError(RuntimeError):
    """The active episode cannot be safely recorded or finalized."""


class CaptureState(enum.Enum):
    IDLE = "idle"
    ARMING = "arming"
    RECORDING = "recording"
    FINALIZING = "finalizing"
    CLOSED = "closed"


def _set_event() -> threading.Event:
    event = threading.Event()
    event.set()
    return event


@dataclass
class _FrameTiming:
    """A compact timing summary that does not retain image data."""

    frame_count: int = 0
    first_timestamp_ns: int = 0
    last_timestamp_ns: int = 0
    max_gap_ns: int = 0
    device_frame_number_gaps: int = 0
    device_frame_number_regressions: int = 0
    _last_device_frame_number: int | None = None

    def add(self, timestamp_ns: int) -> None:
        if self.frame_count:
            self.max_gap_ns = max(self.max_gap_ns, timestamp_ns - self.last_timestamp_ns)
        else:
            self.first_timestamp_ns = timestamp_ns
        self.last_timestamp_ns = timestamp_ns
        self.frame_count += 1

    def add_device_frame_number(self, frame_number: int | None) -> None:
        """Record skipped RealSense frame counters without changing validation."""
        if frame_number is None:
            return
        if self._last_device_frame_number is not None:
            delta = frame_number - self._last_device_frame_number
            if delta > 1:
                self.device_frame_number_gaps += delta - 1
            elif delta <= 0:
                self.device_frame_number_regressions += 1
        self._last_device_frame_number = frame_number

    @property
    def duration_s(self) -> float:
        if self.frame_count < 2:
            return 0.0
        return (self.last_timestamp_ns - self.first_timestamp_ns) / 1_000_000_000

    @property
    def average_fps(self) -> float:
        duration_s = self.duration_s
        return (self.frame_count - 1) / duration_s if duration_s > 0 else 0.0

    @property
    def max_gap_ms(self) -> float:
        return self.max_gap_ns / 1_000_000


@dataclass
class _Episode:
    episode_id: int
    partial_mcap_path: Path
    final_mcap_path: Path
    video_dir: Path
    live_video_dir: Path
    pending_dir: Path
    mcap: EpisodeMcapWriter
    videos: dict[str, object]
    started_at_ns: int
    arming_started_monotonic_ns: int = 0
    recording_started_at_ns: int | None = None
    recording_stopped_at_ns: int | None = None
    state_samples: int = 0
    recording_state_samples: int = 0
    last_state_sample_monotonic_ns: int = 0
    max_state_sample_gap_ns: int = 0
    telemetry_last_source_timestamp_ns: dict[str, int] = field(default_factory=dict)
    telemetry_max_source_gap_ns: dict[str, int] = field(default_factory=dict)
    camera_frames: dict[str, int] = field(default_factory=dict)
    camera_timing: dict[str, _FrameTiming] = field(default_factory=dict)
    camera_host_timing: dict[str, _FrameTiming] = field(default_factory=dict)
    camera_timing_samples: dict[str, list[CameraTimingSample]] = field(default_factory=dict)
    # Camera workers run continuously before an episode begins, so their native
    # queues can contain frames captured before the first fresh robot snapshot.
    # Do not write those frames: source timestamps, not callback order, define
    # the joint-camera synchronization contract.
    camera_start_timestamp_ns: int | None = None
    active_operations: int = 0
    operations_drained: threading.Event = field(default_factory=_set_event)
    # HEVC recordings keep this at the pending MCAP. Direct-H.264 recordings
    # replace it with the root-level MCAP after verified publication.
    committed_mcap_path: Path | None = None


@dataclass
class _TelemetryProgress:
    sequence: int
    source_timestamp_ns: int
    last_progress_monotonic_ns: int


class _CameraWorker:
    """Run one camera independently so sequential reads cannot desynchronize the trio."""

    def __init__(
        self,
        camera: Camera,
        on_frame: Callable[[CameraFrame], None],
        on_error: Callable[[str, BaseException], None],
        stop_event: threading.Event,
    ) -> None:
        self._camera = camera
        self._on_frame = on_frame
        self._on_error = on_error
        self._stop_event = stop_event
        self._thread = threading.Thread(target=self._run, name=f"camera-{camera.role}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float = 2.0) -> None:
        self._thread.join(timeout)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame = self._camera.read(timeout_ms=1000)
            except BaseException as exc:
                if not self._stop_event.is_set():
                    self._on_error(self._camera.role, exc)
                return
            try:
                self._on_frame(frame)
            except BaseException as exc:
                self._on_error(self._camera.role, exc)
                return


class _TerminalKeys:
    """Minimal raw-terminal keyboard reader used by the operator workflow."""

    def __init__(self) -> None:
        self._fd: Optional[int] = None
        self._saved: Optional[list[object]] = None

    def __enter__(self) -> "_TerminalKeys":
        if not sys.stdin.isatty():
            raise CaptureError("interactive capture requires a TTY")
        self._fd = sys.stdin.fileno()
        self._saved = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def poll(self, timeout_s: float) -> Optional[str]:
        if self._fd is None:
            raise RuntimeError("terminal reader is not active")
        readable, _, _ = select.select([sys.stdin], [], [], timeout_s)
        return sys.stdin.read(1) if readable else None

    def __exit__(self, *args: object) -> None:
        if self._fd is not None and self._saved is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)


class EpisodeCollector:
    """Coordinates a live bimanual episode without opening a CAN interface."""

    _FINALIZATION_WAIT_S = 70.0
    _CAMERA_START_MAX_ATTEMPTS = 3
    _CAMERA_START_RETRY_DELAY_S = 1.0
    _CAMERA_STARTUP_PREFLIGHT_TIMEOUT_S = 8.0
    _CAMERA_STARTUP_PREFLIGHT_POLL_S = 0.1
    _CAMERA_PREFLIGHT_WINDOW_S = 2.0
    _CAMERA_PREFLIGHT_MIN_DURATION_S = 1.0
    # This tiny runtime margin avoids rejecting a profile-boundary value
    # because a short rolling window or binary floating-point arithmetic lands
    # just outside the displayed threshold.
    _CAMERA_PREFLIGHT_FPS_EPSILON_HZ = 0.01
    _LEGACY_CAMERA_PREFLIGHT_MAX_GAP_FACTOR = 2.1
    _TELEMETRY_PROBE_DURATION_S = 1.0
    _ARMING_DURATION_S = 2.0
    _ARMING_POLL_S = 0.02
    _ARMING_MIN_STATE_SAMPLES = 10
    _ARMING_MIN_CAMERA_FRAMES = 10
    _ARMING_MAX_QUEUE_DEPTH = 2
    _ARMING_MAX_PIPE_WRITE_MS = 50.0
    _DELETED_DURATION_FILENAME = "deleted_episode_durations.txt"
    _DELETED_DURATION_REASONS = frozenset({"validation_failed", "recording_error", "user_deleted"})
    _DELETED_DURATION_FILE_LOCK = threading.Lock()

    def __init__(
        self,
        config: DatasetConfig,
        *,
        enable_runtime_diagnostics: bool = True,
        on_episode_saved: Callable[[Path], bool] | None = None,
        enable_preview: bool = False,
        preview_sink: Any | None = None,
        auto_delete_failed_validation: bool = False,
        enable_post_save_review: bool = False,
        video_timeline: str = "fixed",
    ) -> None:
        self.config = config
        self._on_episode_saved = on_episode_saved
        self._auto_delete_failed_validation = bool(auto_delete_failed_validation)
        self._enable_post_save_review = bool(enable_post_save_review)
        if video_timeline not in {"fixed", "source"}:
            raise ValueError("video_timeline must be 'fixed' or 'source'")
        self._video_timeline = video_timeline
        self._stop_after_saved_episode = threading.Event()
        self._state = CaptureState.IDLE
        self._state_lock = threading.RLock()
        self._dataset_lock: DatasetOperationLock | None = None
        self._stop_event = threading.Event()
        self._finalization_done = threading.Event()
        self._finalization_done.set()
        self._episode: Optional[_Episode] = None
        self._allowed_cpu_set: tuple[int, ...] = ()
        self._effective_recorder_cpu_set: tuple[int, ...] | None = None
        # Apply this before Portal/RealSense construction. Any worker those
        # libraries create during initialization must inherit the recorder
        # mask just like the Python threads started later in ``start``.
        self._configure_recorder_affinity()
        self._telemetry_clients = {
            role: PortalTelemetryClient(endpoint) for role, endpoint in config.telemetry.items()
        }
        self._cameras: dict[str, Camera] = {role: RealSenseCamera(camera) for role, camera in config.cameras.items()}
        self._camera_workers: list[_CameraWorker] = []
        self._camera_arrival_lock = threading.Lock()
        # This window uses the same capture clock that is written to CameraInfo
        # and later validated in MCAP.  Host delivery timing is retained in
        # episode diagnostics, but must not turn a scheduling delay into a
        # false source-rate failure before recording can begin.
        self._camera_arrivals: dict[str, deque[tuple[int, int]]] = {role: deque() for role in config.cameras}
        self._state_thread: Optional[threading.Thread] = None
        self._last_telemetry_success_ns = 0
        self._telemetry_progress: dict[str, _TelemetryProgress] = {}
        self._background_error: Optional[CaptureError] = None
        self._last_episode_diagnostics: dict[str, object] | None = None
        self._last_post_validation: dict[str, object] | None = None
        self._last_saved_episode_path: Path | None = None
        self._last_saved_episode_id: int | None = None
        self._last_saved_episode_duration_s: float | None = None
        # The experiment runner owns a case-level monitor, while interactive
        # capture owns its dataset-level monitor.  Keeping this switch here
        # avoids two 200 ms diagnostic threads competing with a 60 Hz trial.
        self._runtime_diagnostics_enabled = enable_runtime_diagnostics
        self._runtime_diagnostics: object | None = None
        self._live_preview: LivePreview | None = None
        # Optional non-GUI preview publisher (for example the local Web UI).
        # It receives the same low-rate, best-effort updates as the Tk preview;
        # failures are isolated from camera/state recording.
        self._preview_sink = preview_sink
        self._preview_error_reported = False
        if enable_preview:
            preview_cpus = self._available_preview_cpu_set()
            if preview_cpus:
                self._live_preview = LivePreview(LivePreviewConfig(cpu_set=preview_cpus))
            else:
                self._print(
                    "Warning: live preview disabled because no CPU remains outside the recorder and live FFmpeg sets."
                )
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self._telemetry_clients), thread_name_prefix="telemetry"
        )

    @property
    def state(self) -> CaptureState:
        with self._state_lock:
            return self._state

    @property
    def background_error(self) -> CaptureError | None:
        """Return the asynchronous error that invalidated the active capture.

        Camera and telemetry workers run independently of the caller that is
        timing an episode.  Exposing this read-only marker lets bounded
        runners stop waiting as soon as those workers discard an episode,
        without reaching into the collector's synchronization state.
        """
        with self._state_lock:
            return self._background_error

    @property
    def last_episode_diagnostics(self) -> dict[str, object] | None:
        """Return the latest finalized episode's writer diagnostics.

        Failed episodes intentionally delete their partial MCAP and MP4 files.
        This small metadata snapshot remains available to bounded diagnostic
        runners after that cleanup, without exposing live writer objects.
        """
        with self._state_lock:
            return copy.deepcopy(getattr(self, "_last_episode_diagnostics", None))

    @property
    def last_post_validation(self) -> dict[str, object] | None:
        """Return the latest post-save validation summary for UI adapters."""
        with self._state_lock:
            value = getattr(self, "_last_post_validation", None)
            return copy.deepcopy(value) if isinstance(value, dict) else None

    @property
    def active_episode_snapshot(self) -> dict[str, object]:
        """Return a non-mutating, non-blocking view of the active recording.

        This is intentionally limited to small counters and the video writers'
        own diagnostic snapshots.  It is used by the experiment runner's
        periodic host monitor; it never reads image data, waits for an encoder,
        or changes the capture state.  Writer snapshots are collected after
        releasing the collector lock because they may briefly take their own
        diagnostics lock.
        """
        with self._state_lock:
            state = self._state.value
            error = str(self._background_error) if self._background_error is not None else None
            episode = self._episode
            if episode is None:
                snapshot: dict[str, object] = {
                    "snapshot_unix_ns": time.time_ns(),
                    "snapshot_monotonic_ns": time.monotonic_ns(),
                    "state": state,
                    "episode_id": None,
                    "state_samples": 0,
                    "camera_frames": {},
                    "video_writers": {},
                    "recorder_cpu_set": list(getattr(self, "_effective_recorder_cpu_set", None) or ()),
                    "background_error": error,
                }
                # A failed episode is detached before asynchronous cleanup, so
                # the next 200 ms runtime-diagnostics sample would otherwise
                # lose the offending timing peak. Keep its small final record
                # alongside the sticky error without exposing live writers.
                if error is not None:
                    diagnostics = getattr(self, "_last_episode_diagnostics", None)
                    if isinstance(diagnostics, dict):
                        snapshot["last_episode_diagnostics"] = copy.deepcopy(diagnostics)
                return snapshot

            # State and camera callbacks update these inexpensive counters.
            # A diagnostic sample may land between two callbacks, which is
            # expected; copying them is preferable to taking any I/O path
            # under the collector lock.
            state_samples = int(episode.state_samples)
            recording_state_samples = int(getattr(episode, "recording_state_samples", 0))
            max_state_sample_gap_ns = int(getattr(episode, "max_state_sample_gap_ns", 0))
            try:
                camera_frames = dict(episode.camera_frames)
            except RuntimeError:
                # A first frame for a role can be added concurrently with this
                # one-Hz diagnostic copy. Returning a partial sample is safer
                # than making a monitor observation affect capture.
                camera_frames = {}
            videos = dict(episode.videos)
            episode_id = episode.episode_id
            started_at_ns = episode.started_at_ns
            active_operations = episode.active_operations

        writers = {role: self._video_writer_snapshot(writer) for role, writer in videos.items()}
        return {
            "snapshot_unix_ns": time.time_ns(),
            "snapshot_monotonic_ns": time.monotonic_ns(),
            "state": state,
            "episode_id": episode_id,
            "started_at_unix_ns": started_at_ns,
            "state_samples": state_samples,
            "recording_state_samples": recording_state_samples,
            "max_state_sample_gap_ms": round(max_state_sample_gap_ns / 1_000_000, 3),
            "camera_frames": camera_frames,
            "active_operations": active_operations,
            "video_writers": writers,
            "recorder_cpu_set": list(getattr(self, "_effective_recorder_cpu_set", None) or ()),
            "background_error": error,
        }

    def _print(self, message: str, *, color: str | None = None) -> None:
        print(colorize(message, color), flush=True)

    def _next_episode_id(self) -> int:
        dataset_dir = self.config.collection.dataset_dir
        ids = [int(path.stem) for path in dataset_dir.glob("*.mcap") if path.stem.isdigit()]
        for path in dataset_dir.iterdir() if dataset_dir.is_dir() else ():
            if path.is_dir() and path.name.isdigit() and any(path.glob("*.mcap")):
                ids.append(int(path.name))
        for path in dataset_dir.glob("*.partial.mcap"):
            episode_name = path.name.removesuffix(".partial.mcap")
            if episode_name.isdigit():
                ids.append(int(episode_name))
        for root in (dataset_dir / "videos", dataset_dir / "videos_live", dataset_dir / ".pending"):
            if not root.is_dir():
                continue
            for path in root.iterdir():
                if not path.is_dir():
                    continue
                episode_name = path.name.removesuffix(".partial")
                if episode_name.isdigit():
                    ids.append(int(episode_name))
        return max(ids) + 1 if ids else 0

    def _prepare_dataset_dir(self) -> None:
        dataset_dir = self.config.collection.dataset_dir
        dataset_dir.mkdir(parents=True, exist_ok=True)
        record_config = dataset_dir / "record_config.yaml"
        saved_config = dict(self.config.raw)
        saved_collection = dict(saved_config.get("collection_config", {}))
        saved_collection["validation_profile"] = self.config.collection.validation_profile
        if hasattr(self.config.collection, "state_save_freq"):
            saved_collection["state_save_freq"] = self.config.collection.state_save_freq
        if hasattr(self.config.collection, "target_fps"):
            saved_collection["target_fps"] = self.config.collection.target_fps
        saved_config["collection_config"] = saved_collection
        capture_mode = getattr(self.config, "capture_mode", None)
        if capture_mode is not None:
            saved_config["selected_capture_mode"] = capture_mode
        with record_config.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(saved_config, stream, allow_unicode=True, sort_keys=False)
        for camera in self.config.cameras.values():
            if not camera.intrinsics.is_calibrated:
                self._print(
                    f"Warning: {camera.role} has no calibrated intrinsics configured; "
                    "capture will be rejected unless allow_uncalibrated_intrinsics is explicitly enabled."
                )

    def _acquire_dataset_lock(self) -> None:
        """Prevent offline x265 from running beside a live camera recorder."""
        if self._dataset_lock is not None:
            return
        lock = DatasetOperationLock(self.config.collection.dataset_dir, "capture")
        lock.acquire()
        self._dataset_lock = lock

    def _release_dataset_lock(self) -> None:
        lock = getattr(self, "_dataset_lock", None)
        if lock is None:
            return
        self._dataset_lock = None
        lock.release()

    def _configure_recorder_affinity(self) -> None:
        """Reserve the configured collector CPUs before input workers exist.

        Linux threads created after this call inherit the main thread's mask,
        so camera workers, Portal RPC workers, and the state recorder stay on
        the recorder set while each live FFmpeg child receives its separate
        ``taskset`` mask in :class:`AsyncFfmpegWriter`.
        """
        if getattr(self, "_effective_recorder_cpu_set", None) is not None:
            return
        get_affinity = getattr(os, "sched_getaffinity", None)
        set_affinity = getattr(os, "sched_setaffinity", None)
        requested_text = getattr(self.config.collection, "recorder_cpu_set", None)
        if not callable(get_affinity):
            if requested_text is None:
                self._effective_recorder_cpu_set = ()
                return
            raise CaptureError("configured recorder_cpu_set requires Linux CPU-affinity support")
        try:
            available = set(get_affinity(0))
        except OSError as exc:
            if requested_text is None:
                self._effective_recorder_cpu_set = ()
                return
            raise CaptureError(f"could not inspect recorder CPU affinity: {exc}") from exc
        self._allowed_cpu_set = tuple(sorted(available))
        encoder_text = getattr(self.config.collection, "live_video_cpu_set", None)
        if encoder_text is not None:
            encoder_cpus = set(cpu_set_members(encoder_text, "collection_config.live_video_cpu_set"))
            unavailable_encoder = sorted(encoder_cpus - available)
            if unavailable_encoder:
                raise CaptureError(
                    "configured live_video_cpu_set is outside this process's allowed CPU set: "
                    + ", ".join(str(cpu) for cpu in unavailable_encoder)
                    + "; start the collector with an allowed CPU set that includes both recorder and encoder CPUs"
                )
        if requested_text is None:
            self._effective_recorder_cpu_set = tuple(sorted(available))
            return
        if not callable(set_affinity):
            raise CaptureError("configured recorder_cpu_set requires Linux CPU-affinity support")
        requested = set(cpu_set_members(requested_text, "collection_config.recorder_cpu_set"))
        unavailable = sorted(requested - available)
        if unavailable:
            raise CaptureError(
                "configured recorder_cpu_set is outside this shell's allowed CPU set: "
                + ", ".join(str(cpu) for cpu in unavailable)
            )
        try:
            set_affinity(0, requested)
            effective = tuple(sorted(get_affinity(0)))
        except OSError as exc:
            raise CaptureError(f"could not apply recorder CPU affinity {requested_text}: {exc}") from exc
        if set(effective) != requested:
            raise CaptureError(
                f"recorder CPU affinity did not take effect: requested {requested_text}, "
                f"effective {','.join(str(cpu) for cpu in effective)}"
            )
        self._effective_recorder_cpu_set = effective

    def _available_preview_cpu_set(self) -> tuple[int, ...]:
        """Return CPUs available to the best-effort preview process.

        ``_allowed_cpu_set`` is captured before recorder affinity is applied.
        A spawned preview process inherits the narrowed mask, so it must
        explicitly move back to the selected set in its own process. An
        explicitly configured recorder mask is exclusive and is removed here.
        An unpinned recorder, as used by the qualified 30 Hz profile, has no
        exclusive reservation; in that case the preview may use the original
        allowed mask instead of being incorrectly disabled.
        """
        available = set(getattr(self, "_allowed_cpu_set", ()) or ())
        if not available:
            get_affinity = getattr(os, "sched_getaffinity", None)
            if callable(get_affinity):
                try:
                    available = set(get_affinity(0))
                except OSError:
                    available = set()
        config = getattr(self, "config", None)
        collection = getattr(config, "collection", None)
        recorder: set[int] = set()
        if getattr(collection, "recorder_cpu_set", None) is not None:
            recorder = set(getattr(self, "_effective_recorder_cpu_set", ()) or ())
        encoder_text = getattr(collection, "live_video_cpu_set", None)
        encoder: set[int] = set()
        if encoder_text is not None:
            try:
                encoder = set(cpu_set_members(encoder_text, "collection_config.live_video_cpu_set"))
            except (TypeError, ValueError):
                encoder = set()
        return tuple(sorted(available - recorder - encoder))

    def _start_live_preview(self) -> None:
        preview = getattr(self, "_live_preview", None)
        sink = getattr(self, "_preview_sink", None)
        for name, target in (("live preview", preview), ("web preview", sink)):
            if target is None:
                continue
            try:
                start = getattr(target, "start", None)
                if callable(start):
                    start()
                self._print(
                    f"{name.capitalize()} requested: cameras=5 Hz, joints=10 Hz; "
                    "best-effort display (capture is independent)."
                )
            except BaseException as exc:
                self._report_preview_error(f"could not start {name}", exc)
                try:
                    stop = getattr(target, "stop", None)
                    if callable(stop):
                        stop(timeout_s=0.1)
                except BaseException:
                    pass

    def _stop_live_preview(self) -> None:
        preview = getattr(self, "_live_preview", None)
        sink = getattr(self, "_preview_sink", None)
        for name, target in (("live preview", preview), ("web preview", sink)):
            if target is None:
                continue
            try:
                stop = getattr(target, "stop", None)
                if callable(stop):
                    stop()
            except BaseException as exc:
                self._report_preview_error(f"could not stop {name}", exc)

    def _report_preview_error(self, prefix: str, exc: BaseException) -> None:
        if getattr(self, "_preview_error_reported", False):
            return
        self._preview_error_reported = True
        self._print(f"Warning: {prefix}; capture continues: {exc}")

    def _submit_preview_camera(self, frame: CameraFrame) -> None:
        preview = getattr(self, "_live_preview", None)
        sink = getattr(self, "_preview_sink", None)
        if preview is None and sink is None:
            return
        if preview is not None:
            try:
                preview.submit_camera(frame)
            except BaseException as exc:
                self._report_preview_error("camera preview failed", exc)
        if sink is not None:
            try:
                submit = getattr(sink, "submit_camera", None)
                if callable(submit):
                    submit(frame)
            except BaseException as exc:
                self._report_preview_error("web preview failed", exc)

    def _submit_preview_joints(self, snapshots: dict[str, RobotSnapshot], timestamp_ns: int) -> None:
        preview = getattr(self, "_live_preview", None)
        sink = getattr(self, "_preview_sink", None)
        if preview is None and sink is None:
            return
        if preview is not None:
            try:
                preview.submit_joints(snapshots, timestamp_ns=timestamp_ns)
            except BaseException as exc:
                self._report_preview_error("joint preview failed", exc)
        if sink is not None:
            try:
                submit = getattr(sink, "submit_joints", None)
                if callable(submit):
                    submit(snapshots, timestamp_ns=timestamp_ns)
            except BaseException as exc:
                self._report_preview_error("web joint preview failed", exc)

    def _telemetry_gap_budget_ns(self) -> int:
        """Return the runtime continuity budget derived from the selected contract."""
        profile_name = getattr(self.config.collection, "validation_profile", "piper_compatible")
        policy = build_validation_policy(getattr(self.config, "cameras", {}), profile_name)
        # This is an early, fail-closed runtime guard. The formal contract and
        # its 17/34 ms validation threshold remain unchanged.
        return int(round(2.0 * policy.joint_camera_sync_ms * 1_000_000))

    def _probe_telemetry_health(self, *, duration_s: float | None = None) -> None:
        """Bounded live telemetry check used before writers or teleoperation begin.

        The recording path uses separate progress state, so this short probe
        cannot make the first in-episode state sample appear stale. A Portal
        RPC, source-time jump, or stalled sequence fails before an operator is
        asked to move the arm.
        """
        # A few focused unit tests construct only the video-setup portion of
        # a collector with ``object.__new__``. A real EpisodeCollector always
        # has all configured Portal clients, so this compatibility escape does
        # not weaken production capture.
        if not getattr(self, "_telemetry_clients", None):
            return
        probe_duration_s = self._TELEMETRY_PROBE_DURATION_S if duration_s is None else duration_s
        if probe_duration_s <= 0:
            raise ValueError("telemetry probe duration must be positive")
        gap_budget_ns = self._telemetry_gap_budget_ns()
        sample_period_s = 1.0 / self.config.collection.state_save_freq
        deadline = time.monotonic() + probe_duration_s
        next_sample_at = time.monotonic()
        previous: dict[str, RobotSnapshot] = {}
        last_sequence_progress_ns: dict[str, int] = {}
        last_timestamp_progress_ns: dict[str, int] = {}
        samples = 0
        try:
            while True:
                now = time.monotonic()
                if samples and now >= deadline:
                    break
                if now < next_sample_at:
                    time.sleep(min(next_sample_at - now, 0.005))
                    continue
                sample_started_ns = time.monotonic_ns()
                snapshots = self._read_snapshots()
                sample_finished_ns = time.monotonic_ns()
                sample_elapsed_ns = sample_finished_ns - sample_started_ns
                if sample_elapsed_ns > gap_budget_ns:
                    raise TelemetryError(
                        f"telemetry RPC batch took {sample_elapsed_ns / 1_000_000:.1f} ms "
                        f"(runtime budget {gap_budget_ns / 1_000_000:.1f} ms)"
                    )
                for role, snapshot in snapshots.items():
                    earlier = previous.get(role)
                    if earlier is None:
                        last_sequence_progress_ns[role] = sample_finished_ns
                        last_timestamp_progress_ns[role] = sample_finished_ns
                    else:
                        if snapshot.sequence < earlier.sequence or snapshot.source_timestamp_ns < earlier.source_timestamp_ns:
                            raise TelemetryError(f"{role} telemetry sequence or source timestamp moved backwards")
                        if snapshot.sequence > earlier.sequence:
                            last_sequence_progress_ns[role] = sample_finished_ns
                        if snapshot.source_timestamp_ns > earlier.source_timestamp_ns:
                            source_gap_ns = snapshot.source_timestamp_ns - earlier.source_timestamp_ns
                            if source_gap_ns > gap_budget_ns:
                                raise TelemetryError(
                                    f"{role} telemetry source gap was {source_gap_ns / 1_000_000:.1f} ms "
                                    f"(runtime budget {gap_budget_ns / 1_000_000:.1f} ms)"
                                )
                            last_timestamp_progress_ns[role] = sample_finished_ns
                        if sample_finished_ns - last_sequence_progress_ns[role] > gap_budget_ns:
                            raise TelemetryError(
                                f"{role} telemetry sequence did not advance for "
                                f"{(sample_finished_ns - last_sequence_progress_ns[role]) / 1_000_000:.1f} ms"
                            )
                        if sample_finished_ns - last_timestamp_progress_ns[role] > gap_budget_ns:
                            raise TelemetryError(
                                f"{role} telemetry source timestamp did not advance for "
                                f"{(sample_finished_ns - last_timestamp_progress_ns[role]) / 1_000_000:.1f} ms"
                            )
                    previous[role] = snapshot
                samples += 1
                next_sample_at += sample_period_s
                if next_sample_at <= time.monotonic():
                    next_sample_at = time.monotonic()
        except BaseException as exc:
            if isinstance(exc, CaptureError):
                raise
            raise CaptureError(f"telemetry health probe failed; do not record: {exc}") from exc
        if samples < 2 or len(previous) != len(self._telemetry_clients):
            raise CaptureError("telemetry health probe did not observe every configured endpoint twice")

    def _start_runtime_diagnostics(self) -> None:
        """Start best-effort 200 ms host evidence for regular interactive capture."""
        if getattr(self, "_runtime_diagnostics", None) is not None:
            return
        recorder_cpus = self._effective_recorder_cpu_set or ()
        path = (
            self.config.collection.dataset_dir
            / "runtime_diagnostics"
            / f"capture-{time.time_ns()}.jsonl"
        )
        monitor = RuntimeDiagnosticsMonitor(path, self, recorder_cpus)
        if monitor.start():
            self._runtime_diagnostics = monitor
            self._print(f"Runtime diagnostics: {monitor.path}")
        else:
            self._print(f"Warning: runtime diagnostics disabled: {monitor.error}")

    def _stop_runtime_diagnostics(self) -> None:
        monitor = getattr(self, "_runtime_diagnostics", None)
        self._runtime_diagnostics = None
        if monitor is None:
            return
        try:
            monitor.stop()
        except BaseException as exc:
            self._print(f"Warning: could not stop runtime diagnostics: {exc}")

    def check(self, start_cameras: bool = False) -> None:
        """Validate endpoints, and optionally start/stop configured cameras once."""
        self._configure_recorder_affinity()
        snapshots = self._read_snapshots()
        descriptions = ", ".join(f"{role}={len(snapshot.position)} DoF" for role, snapshot in snapshots.items())
        self._print(f"Telemetry ready: {descriptions}")
        if start_cameras:
            started: list[Camera] = []
            try:
                for camera in self._cameras.values():
                    camera.start()
                    started.append(camera)
                frames = self._read_camera_startup_frames(started)
                for camera in started:
                    frame = frames[camera.role]
                    self._print(f"Camera ready: {camera.role} {frame.image_bgr.shape[1]}x{frame.image_bgr.shape[0]}")
            finally:
                for camera in reversed(started):
                    camera.stop()

    def start(self) -> None:
        """Open all read-only inputs before any episode file is created."""
        started_cameras: list[Camera] = []
        try:
            self._configure_recorder_affinity()
            self._acquire_dataset_lock()
            self._prepare_dataset_dir()
            if self._runtime_diagnostics_enabled:
                self._start_runtime_diagnostics()
            self.check(start_cameras=False)
            started_cameras = self._start_cameras_with_readiness()
            self._camera_workers = [
                _CameraWorker(camera, self._on_camera_frame, self._on_camera_error, self._stop_event)
                for camera in self._cameras.values()
            ]
            # Spawn the GUI before starting recorder threads. The child gets a
            # clean process snapshot and then moves to its selected CPU set;
            # configured recorder/encoder reservations remain disjoint, and
            # camera workers can immediately publish idle previews.
            self._start_live_preview()
            for worker in self._camera_workers:
                worker.start()
            self._wait_for_camera_preflight()
            self._probe_telemetry_health()
            self._state_thread = threading.Thread(target=self._state_loop, name="state-recorder", daemon=True)
            self._state_thread.start()
        except BaseException:
            self._stop_event.set()
            for camera in reversed(started_cameras):
                try:
                    camera.stop()
                except Exception:
                    pass
            for worker in self._camera_workers:
                worker.join()
            self._stop_live_preview()
            self._stop_runtime_diagnostics()
            self._release_dataset_lock()
            raise
        recorder_cpus = self._effective_recorder_cpu_set or ()
        encoder_cpus = getattr(self.config.collection, "live_video_cpu_set", None)
        if getattr(self.config.collection, "recorder_cpu_set", None) is not None:
            self._print(
                "CPU isolation ready: recorder="
                + ",".join(str(cpu) for cpu in recorder_cpus)
                + f"; live FFmpeg={encoder_cpus}."
            )
        self._print(
            "Inputs ready. Press Space to arm an episode; wait for Recording episode before teleoperating. "
            "Enter saves; q discards an active episode or deletes the last passed episode."
        )

    def _read_camera_startup_frames(self, cameras: list[Camera]) -> dict[str, CameraFrame]:
        """Keep the backend's bounded first-frame wait and camera-specific error."""
        frames: dict[str, CameraFrame] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(cameras)) as executor:
            pending = {camera.role: executor.submit(camera.read, timeout_ms=1000) for camera in cameras}
            for camera in cameras:
                try:
                    # read() enforces its own timeout. A matching Future timeout
                    # can win the race and hide the backend error behind an empty
                    # TimeoutError; executor shutdown still waits for read anyway.
                    frames[camera.role] = pending[camera.role].result()
                except Exception as exc:
                    detail = str(exc) or type(exc).__name__
                    raise CaptureError(f"camera {camera.role} startup read failed: {detail}") from exc
        return frames

    def _start_cameras_with_readiness(self) -> list[Camera]:
        """Start all streams and prove that each delivers a frame before workers run.

        A D4xx device can report a successful ``pipeline.start`` immediately
        after a previous process exits, then need a short USB/firmware recovery
        before its first frame arrives. Retrying here is safe because no
        episode files or robot samples exist yet.
        """
        last_error: BaseException | None = None
        for attempt in range(1, self._CAMERA_START_MAX_ATTEMPTS + 1):
            started: list[Camera] = []
            try:
                for camera in self._cameras.values():
                    camera.start()
                    started.append(camera)
                self._read_camera_startup_frames(started)
                return started
            except BaseException as exc:
                last_error = exc
                detail = str(exc) or type(exc).__name__
                self._print(f"Camera startup attempt {attempt}/{self._CAMERA_START_MAX_ATTEMPTS} failed: {detail}")
                for camera in reversed(started):
                    try:
                        camera.stop()
                    except Exception:
                        pass
                if attempt == self._CAMERA_START_MAX_ATTEMPTS or self._stop_event.wait(
                    self._CAMERA_START_RETRY_DELAY_S
                ):
                    break
        detail = str(last_error) or type(last_error).__name__
        raise CaptureError(
            f"could not make all camera streams ready after {self._CAMERA_START_MAX_ATTEMPTS} attempts: {detail}"
        ) from last_error

    def _wait_for_camera_preflight(self) -> None:
        """Do not expose the record key until the selected quality gate passes."""
        deadline = time.monotonic() + self._CAMERA_STARTUP_PREFLIGHT_TIMEOUT_S
        last_error: CaptureError | None = None
        while not self._stop_event.is_set():
            with self._state_lock:
                background_error = self._background_error
            if background_error is not None:
                raise background_error
            try:
                self._validate_camera_preflight()
                return
            except CaptureError as exc:
                last_error = exc
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                break
            self._stop_event.wait(min(self._CAMERA_STARTUP_PREFLIGHT_POLL_S, remaining_s))
        detail = str(last_error) if last_error is not None else "camera startup was interrupted"
        raise CaptureError(f"camera startup preflight did not pass: {detail}")

    def _read_snapshots(self, require_fresh: bool = False) -> dict[str, RobotSnapshot]:
        futures = {role: self._executor.submit(client.read) for role, client in self._telemetry_clients.items()}
        snapshots: dict[str, RobotSnapshot] = {}
        deadline = time.monotonic() + self.config.collection.timeout_threshold_s
        for role, future in futures.items():
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                raise TelemetryError("timed out waiting for telemetry snapshots")
            snapshots[role] = future.result(timeout=remaining_s)
        if require_fresh:
            self._validate_telemetry_progress(snapshots)
        return snapshots

    def _validate_telemetry_progress(self, snapshots: dict[str, RobotSnapshot]) -> None:
        """Reject snapshots that keep replaying after an i2rt worker has stopped."""
        now_ns = time.monotonic_ns()
        stale_roles: list[str] = []
        for role, snapshot in snapshots.items():
            previous = self._telemetry_progress.get(role)
            if previous is None:
                self._telemetry_progress[role] = _TelemetryProgress(
                    sequence=snapshot.sequence,
                    source_timestamp_ns=snapshot.source_timestamp_ns,
                    last_progress_monotonic_ns=now_ns,
                )
                continue
            if snapshot.sequence < previous.sequence or snapshot.source_timestamp_ns < previous.source_timestamp_ns:
                raise TelemetryError(f"{role} telemetry sequence or source timestamp moved backwards")
            if snapshot.sequence > previous.sequence:
                previous.sequence = snapshot.sequence
                previous.source_timestamp_ns = snapshot.source_timestamp_ns
                previous.last_progress_monotonic_ns = now_ns
            elapsed_s = (now_ns - previous.last_progress_monotonic_ns) / 1_000_000_000
            if elapsed_s >= self.config.collection.timeout_threshold_s:
                stale_roles.append(role)
        if stale_roles:
            raise TelemetryError("telemetry stopped advancing for " + ", ".join(stale_roles))

    def _episode_telemetry_timing_error(
        self,
        episode: _Episode,
        snapshots: dict[str, RobotSnapshot],
        sample_monotonic_ns: int,
    ) -> CaptureError | None:
        """Reject a scheduler or Portal stall before it reaches formal validation.

        The formal validator still decides whether a saved MCAP meets its
        contract. This runtime check simply prevents a confirmed joint-state
        hole from being recorded after the operator has begun moving.
        """
        gap_budget_ns = self._telemetry_gap_budget_ns()
        last_sample_ns = int(getattr(episode, "last_state_sample_monotonic_ns", 0))
        if last_sample_ns:
            state_gap_ns = sample_monotonic_ns - last_sample_ns
            if state_gap_ns > gap_budget_ns:
                episode.max_state_sample_gap_ns = max(
                    int(getattr(episode, "max_state_sample_gap_ns", 0)), state_gap_ns
                )
                return CaptureError(
                    f"state sampling gap was {state_gap_ns / 1_000_000:.1f} ms "
                    f"(runtime budget {gap_budget_ns / 1_000_000:.1f} ms)"
                )
        last_timestamps = getattr(episode, "telemetry_last_source_timestamp_ns", {})
        last_sequences = getattr(episode, "telemetry_last_sequence", {})
        last_progress = getattr(episode, "telemetry_last_progress_monotonic_ns", {})
        for role, snapshot in snapshots.items():
            earlier_timestamp = last_timestamps.get(role)
            earlier_sequence = last_sequences.get(role)
            if earlier_timestamp is None:
                continue
            if snapshot.sequence < earlier_sequence or snapshot.source_timestamp_ns < earlier_timestamp:
                return CaptureError(f"{role} telemetry sequence or source timestamp moved backwards during recording")
            if snapshot.source_timestamp_ns > earlier_timestamp:
                source_gap_ns = snapshot.source_timestamp_ns - earlier_timestamp
                if source_gap_ns > gap_budget_ns:
                    max_source_gaps = getattr(episode, "telemetry_max_source_gap_ns", None)
                    if max_source_gaps is None:
                        max_source_gaps = {}
                        episode.telemetry_max_source_gap_ns = max_source_gaps
                    max_source_gaps[role] = max(int(max_source_gaps.get(role, 0)), source_gap_ns)
                    return CaptureError(
                        f"{role} telemetry source gap was {source_gap_ns / 1_000_000:.1f} ms "
                        f"(runtime budget {gap_budget_ns / 1_000_000:.1f} ms)"
                    )
            else:
                progress_ns = int(last_progress.get(role, last_sample_ns))
                if progress_ns and sample_monotonic_ns - progress_ns > gap_budget_ns:
                    return CaptureError(
                        f"{role} telemetry source timestamp did not advance for "
                        f"{(sample_monotonic_ns - progress_ns) / 1_000_000:.1f} ms"
                    )
        return None

    @staticmethod
    def _remember_episode_telemetry_timing(
        episode: _Episode,
        snapshots: dict[str, RobotSnapshot],
        sample_monotonic_ns: int,
    ) -> None:
        """Commit timing baselines only after all state messages were written."""
        previous_state_sample_ns = int(getattr(episode, "last_state_sample_monotonic_ns", 0))
        if previous_state_sample_ns:
            episode.max_state_sample_gap_ns = max(
                int(getattr(episode, "max_state_sample_gap_ns", 0)),
                sample_monotonic_ns - previous_state_sample_ns,
            )
        episode.last_state_sample_monotonic_ns = sample_monotonic_ns
        timestamps = getattr(episode, "telemetry_last_source_timestamp_ns", None)
        if timestamps is None:
            timestamps = {}
            episode.telemetry_last_source_timestamp_ns = timestamps
        sequences = getattr(episode, "telemetry_last_sequence", None)
        if sequences is None:
            sequences = {}
            episode.telemetry_last_sequence = sequences
        progress = getattr(episode, "telemetry_last_progress_monotonic_ns", None)
        if progress is None:
            progress = {}
            episode.telemetry_last_progress_monotonic_ns = progress
        max_source_gaps = getattr(episode, "telemetry_max_source_gap_ns", None)
        if max_source_gaps is None:
            max_source_gaps = {}
            episode.telemetry_max_source_gap_ns = max_source_gaps
        for role, snapshot in snapshots.items():
            earlier_timestamp = timestamps.get(role)
            if earlier_timestamp is not None and snapshot.source_timestamp_ns > earlier_timestamp:
                max_source_gaps[role] = max(
                    int(max_source_gaps.get(role, 0)), snapshot.source_timestamp_ns - earlier_timestamp
                )
                progress[role] = sample_monotonic_ns
            elif earlier_timestamp is None:
                progress[role] = sample_monotonic_ns
            timestamps[role] = snapshot.source_timestamp_ns
            sequences[role] = snapshot.sequence

    def _state_loop(self) -> None:
        period_s = 1.0 / self.config.collection.state_save_freq
        next_deadline = time.monotonic()
        while not self._stop_event.is_set():
            next_deadline += period_s
            if self.state in {CaptureState.ARMING, CaptureState.RECORDING}:
                self._record_state_sample()
            wait_s = next_deadline - time.monotonic()
            if wait_s <= 0:
                next_deadline = time.monotonic()
            else:
                self._stop_event.wait(wait_s)

    def _record_state_sample(self) -> None:
        with self._state_lock:
            error = self._background_error
        if error is not None:
            self._discard_for_background_error(error)
            return
        preview_snapshots: dict[str, RobotSnapshot] | None = None
        preview_timestamp_ns: int | None = None
        try:
            snapshots = self._read_snapshots(require_fresh=True)
            capture_time_ns = time.time_ns()
            sample_monotonic_ns = time.monotonic_ns()
            self._last_telemetry_success_ns = capture_time_ns
        except Exception as exc:
            self._handle_telemetry_failure(exc)
            return
        error = None
        should_finalize = False
        episode = self._begin_episode_operation()
        if episode is None:
            # Telemetry polling already runs while idle. Reuse that snapshot so
            # the operator can inspect all four arms before pressing Space;
            # the preview's limiter and bounded queue keep this best effort.
            self._submit_preview_joints(snapshots, sample_monotonic_ns)
            return
        try:
            timing_error = self._episode_telemetry_timing_error(episode, snapshots, sample_monotonic_ns)
            if timing_error is not None:
                error = timing_error
                with self._state_lock:
                    self._background_error = error
            else:
                try:
                    for role in self.config.telemetry:
                        episode.mcap.write_joint_state(
                            snapshots[role],
                            capture_time_ns,
                            header_timestamp_ns=snapshots[role].source_timestamp_ns,
                        )
                except Exception as exc:
                    error = CaptureError(f"cannot write robot state: {exc}")
                    with self._state_lock:
                        self._background_error = error
                else:
                    self._remember_episode_telemetry_timing(episode, snapshots, sample_monotonic_ns)
                    # The first completed four-arm snapshot establishes an explicit
                    # camera watermark. Camera callbacks discard queued frames from
                    # before it, so an episode cannot start with camera samples that
                    # predate every recorded joint state. Use the earliest hardware
                    # timestamp because the offline contract pairs a camera frame
                    # with its nearest state from the four recorded streams.
                    camera_start_timestamp_ns = min(snapshot.source_timestamp_ns for snapshot in snapshots.values())
                    with self._state_lock:
                        if getattr(episode, "camera_start_timestamp_ns", None) is None:
                            episode.camera_start_timestamp_ns = camera_start_timestamp_ns
                        recording = self._state is CaptureState.RECORDING
                    episode.state_samples += 1
                    if recording:
                        episode.recording_state_samples = getattr(episode, "recording_state_samples", 0) + 1
                    preview_snapshots = snapshots
                    preview_timestamp_ns = sample_monotonic_ns
                    should_finalize = (
                        recording
                        and getattr(episode, "recording_state_samples", 0)
                        >= self.config.collection.total_sampled_frames
                    )
        finally:
            self._end_episode_operation(episode)
        if error is None and preview_snapshots is not None and preview_timestamp_ns is not None:
            self._submit_preview_joints(preview_snapshots, preview_timestamp_ns)
        if error is not None:
            self._discard_for_background_error(error)
        elif should_finalize:
            self._print("Configured state-frame limit reached; saving episode.")
            self.save_episode()

    def _handle_telemetry_failure(self, exc: BaseException) -> None:
        now_ns = time.time_ns()
        if self._last_telemetry_success_ns == 0:
            self._last_telemetry_success_ns = now_ns
        elapsed_s = (now_ns - self._last_telemetry_success_ns) / 1_000_000_000
        if elapsed_s < self.config.collection.timeout_threshold_s:
            return
        error = CaptureError(f"telemetry unavailable for {elapsed_s:.2f}s: {exc}")
        with self._state_lock:
            active = self._state in {CaptureState.ARMING, CaptureState.RECORDING}
            if active:
                self._background_error = error
        if active:
            self._discard_for_background_error(error)

    def _discard_for_background_error(self, error: CaptureError) -> None:
        with self._state_lock:
            if self._background_error is None:
                self._background_error = error
            active = self._state in {CaptureState.ARMING, CaptureState.RECORDING}
            active_error = self._background_error
        if active:
            self._print(colorize(f"Recording error: {active_error}; discarding active episode.", "red"))
            self.discard_episode(deletion_reason="recording_error")

    def _observe_camera_arrival(self, frame: CameraFrame) -> None:
        """Keep a short capture-time window for the next-recording gate."""
        timestamp_ns = frame.timestamp_ns
        received_monotonic_ns = time.monotonic_ns()
        cutoff_ns = received_monotonic_ns - int(self._CAMERA_PREFLIGHT_WINDOW_S * 1_000_000_000)
        with self._camera_arrival_lock:
            arrivals = self._camera_arrivals[frame.role]
            # A RealSense global clock is monotonic by construction. If a host
            # fallback clock ever moves backwards, restart this small preflight
            # window instead of calculating timing from mixed ordering.
            if arrivals:
                previous = arrivals[-1]
                previous_capture_ns = previous[0] if isinstance(previous, tuple) else previous
                if timestamp_ns <= previous_capture_ns:
                    arrivals.clear()
            arrivals.append((timestamp_ns, received_monotonic_ns))
            while arrivals and arrivals[0][1] < cutoff_ns:
                arrivals.popleft()

    def _camera_preflight_timings(self) -> dict[str, _FrameTiming]:
        cutoff_ns = time.monotonic_ns() - int(self._CAMERA_PREFLIGHT_WINDOW_S * 1_000_000_000)
        timings: dict[str, _FrameTiming] = {}
        with self._camera_arrival_lock:
            for role in self.config.cameras:
                arrivals = self._camera_arrivals.setdefault(role, deque())
                # Integer entries are accepted for old unit-test fixtures; the
                # live recorder always stores (capture, receive-monotonic).
                while arrivals:
                    first = arrivals[0]
                    received_ns = first[1] if isinstance(first, tuple) else first
                    if received_ns >= cutoff_ns:
                        break
                    arrivals.popleft()
                timing = _FrameTiming()
                for arrival in arrivals:
                    timestamp_ns = arrival[0] if isinstance(arrival, tuple) else arrival
                    timing.add(timestamp_ns)
                timings[role] = timing
        return timings

    def _validate_camera_preflight(self) -> None:
        """Reject a known-bad source stream before it creates an episode artifact."""
        profile_name = getattr(self.config.collection, "validation_profile", "piper_compatible")
        policy = build_validation_policy(self.config.cameras, profile_name)
        issues: list[str] = []
        configured_gap_limit = policy.camera_max_gap_ms
        expected_ranges: list[tuple[float, float]] = []
        for role, timing in self._camera_preflight_timings().items():
            expected_fps = self.config.cameras[role].fps
            minimum_fps, maximum_fps = policy.camera_rate_bounds_for(expected_fps)
            expected_ranges.append((minimum_fps, maximum_fps))
            # Formal mode contracts use the table's explicit 69 ms allowance
            # at both rates. Keep the old frame-period cap for legacy configs.
            max_gap_ms = configured_gap_limit
            if getattr(self.config, "capture_mode", None) is None:
                max_gap_ms = min(
                    max_gap_ms,
                    self._LEGACY_CAMERA_PREFLIGHT_MAX_GAP_FACTOR * 1_000 / expected_fps,
                )
            if timing.duration_s < self._CAMERA_PREFLIGHT_MIN_DURATION_S:
                issues.append(f"{role}=warming up ({timing.frame_count} frames)")
                continue
            rate_outside_bounds = (
                timing.average_fps < minimum_fps - self._CAMERA_PREFLIGHT_FPS_EPSILON_HZ
                or timing.average_fps > maximum_fps + self._CAMERA_PREFLIGHT_FPS_EPSILON_HZ
            )
            if rate_outside_bounds or timing.max_gap_ms > max_gap_ms:
                issues.append(f"{role}={timing.average_fps:.2f}Hz, max_gap={timing.max_gap_ms:.1f}/{max_gap_ms:.1f}ms")
        if issues:
            if expected_ranges:
                minimum_text = min(item[0] for item in expected_ranges)
                maximum_text = max(item[1] for item in expected_ranges)
            else:
                minimum_text = maximum_text = float("nan")
            raise CaptureError(
                "camera preflight failed; do not record: "
                + "; ".join(issues)
                + f". Expected each camera within {minimum_text:g}-{maximum_text:g} Hz, with gaps <= "
                + f"{configured_gap_limit:g} ms for profile {policy.profile_name}."
            )

    def _begin_episode_operation(self) -> _Episode | None:
        """Pin an episode while a camera or state callback uses its writers."""
        with self._state_lock:
            episode = self._episode
            if self._state not in {CaptureState.ARMING, CaptureState.RECORDING} or episode is None:
                return None
            if self._background_error is not None:
                return None
            episode.active_operations += 1
            episode.operations_drained.clear()
            return episode

    @staticmethod
    def _video_writer_snapshot(writer: object) -> dict[str, object]:
        """Extract diagnostics without making lightweight test doubles strict."""
        snapshot = getattr(writer, "diagnostic_snapshot", None)
        if callable(snapshot):
            try:
                data = snapshot()
            except BaseException as exc:
                return {"diagnostic_error": f"{type(exc).__name__}: {exc}"}
            if isinstance(data, dict):
                return dict(data)
            return {"diagnostic_error": "writer diagnostic snapshot was not a dictionary"}
        return {
            "accepted_frames": getattr(writer, "accepted_frames", None),
            "frames_submitted_to_ffmpeg": getattr(writer, "frames_submitted_to_ffmpeg", None),
            "dropped_frames": getattr(writer, "dropped_frames", None),
            "max_queue_depth": getattr(writer, "max_queue_depth", None),
            "queue_capacity": getattr(writer, "queue_size", None),
            "max_pipe_write_ms": getattr(writer, "max_pipe_write_ms", None),
        }

    def _remember_episode_diagnostics(
        self,
        episode: _Episode,
        *,
        before_close: dict[str, dict[str, object]] | None = None,
        after_close: dict[str, dict[str, object]] | None = None,
    ) -> None:
        """Retain small encoder diagnostics after an episode's assets are closed."""
        with self._state_lock:
            previous = getattr(self, "_last_episode_diagnostics", None) or {}
            diagnostics: dict[str, object] = dict(previous)
            diagnostics.update(
                {
                    "episode_id": episode.episode_id,
                    "state_samples": episode.state_samples,
                    "recording_state_samples": getattr(episode, "recording_state_samples", 0),
                    "max_state_sample_gap_ms": round(
                        int(getattr(episode, "max_state_sample_gap_ns", 0)) / 1_000_000, 3
                    ),
                    "telemetry_max_source_gap_ms": {
                        role: round(int(gap_ns) / 1_000_000, 3)
                        for role, gap_ns in getattr(episode, "telemetry_max_source_gap_ns", {}).items()
                    },
                    "recorder_cpu_set": list(getattr(self, "_effective_recorder_cpu_set", None) or ()),
                    "camera_frames": dict(episode.camera_frames),
                    "background_error": str(self._background_error) if self._background_error is not None else None,
                }
            )
            if before_close is not None:
                diagnostics["video_writers_before_close"] = before_close
            if after_close is not None:
                diagnostics["video_writers_after_close"] = after_close
            self._last_episode_diagnostics = diagnostics

    def _end_episode_operation(self, episode: _Episode) -> None:
        with self._state_lock:
            episode.active_operations -= 1
            if episode.active_operations < 0:
                raise RuntimeError("episode operation accounting underflow")
            if episode.active_operations == 0:
                episode.operations_drained.set()

    def _on_camera_frame(self, frame: CameraFrame) -> None:
        self._observe_camera_arrival(frame)
        error: CaptureError | None = None
        episode = self._begin_episode_operation()
        if episode is None:
            # Camera workers run while idle so the operator can compose the
            # preview before pressing Space. This is deliberately after the
            # cheap arrival bookkeeping and never waits for the GUI process.
            self._submit_preview_camera(frame)
            return
        try:
            # This is the only short critical section in the callback. Do not
            # hold the global state lock while enqueueing video or writing MCAP.
            with self._state_lock:
                camera_start_timestamp_ns = getattr(episode, "camera_start_timestamp_ns", None)
            if camera_start_timestamp_ns is None or frame.timestamp_ns < camera_start_timestamp_ns:
                # A frame can have been captured before the operator pressed
                # Space but still be waiting in librealsense's native queue.
                # It has no in-episode joint state to pair with, so consume it
                # without writing an MCAP record or a video frame.
                return
            video = episode.videos[frame.role]
            if getattr(self, "_video_timeline", "fixed") == "source":
                accepted = video.enqueue(
                    frame.image_bgr,
                    timestamp_ns=frame.timestamp_ns,
                    copy_frame=not frame.owns_image,
                )
            else:
                accepted = video.enqueue(frame.image_bgr, copy_frame=not frame.owns_image)
            if not accepted:
                error = CaptureError(f"video queue overflow on {frame.role}")
            else:
                episode.mcap.write_camera_info(self.config.cameras[frame.role], frame.timestamp_ns)
                episode.camera_frames[frame.role] = episode.camera_frames.get(frame.role, 0) + 1
                episode.camera_timing[frame.role].add(frame.timestamp_ns)
                host_timestamp_ns = frame.host_timestamp_ns or frame.timestamp_ns
                episode.camera_host_timing[frame.role].add(host_timestamp_ns)
                episode.camera_timing[frame.role].add_device_frame_number(frame.device_frame_number)
                episode.camera_timing_samples[frame.role].append(
                    CameraTimingSample(
                        role=frame.role,
                        capture_timestamp_ns=frame.timestamp_ns,
                        host_receive_timestamp_ns=host_timestamp_ns,
                        device_timestamp_ns=frame.device_timestamp_ns,
                        device_frame_number=frame.device_frame_number,
                        device_timestamp_domain=frame.device_timestamp_domain,
                        capture_timestamp_source=frame.capture_timestamp_source,
                    )
                )
        except Exception as exc:
            error = CaptureError(f"cannot process camera frame for {frame.role}: {exc}")
        finally:
            # Mark the episode unhealthy before the finalizer can observe that
            # this callback has drained and commit it.
            if error is not None:
                with self._state_lock:
                    self._background_error = error
            self._end_episode_operation(episode)
            # Formal video/MCAP work is complete before any preview downsample
            # or IPC attempt, keeping the display outside the capture hot path.
            self._submit_preview_camera(frame)
        if error is not None:
            self._discard_for_background_error(error)

    def _on_camera_error(self, role: str, exc: BaseException) -> None:
        error = CaptureError(f"camera {role} failed: {exc}")
        self._print(colorize(str(error), "red"))
        self._discard_for_background_error(error)

    def start_episode(self) -> int:
        with self._state_lock:
            if self._state is not CaptureState.IDLE:
                raise CaptureError(f"cannot start an episode while state={self._state.value}")
            if self._background_error is not None:
                raise self._background_error
        uncalibrated = [
            role for role, camera in self.config.cameras.items() if not camera.intrinsics.is_calibrated
        ]
        if uncalibrated and not self.config.collection.allow_uncalibrated_intrinsics:
            raise CaptureError(
                "cannot record without calibrated intrinsics for "
                + ", ".join(uncalibrated)
                + "; set collection_config.allow_uncalibrated_intrinsics: true only for a smoke test"
            )
        # Keep this bounded check outside the state lock. Camera callbacks can
        # continue to refresh their source-time preflight window while Portal
        # continuity is being tested.
        self._validate_camera_preflight()
        self._probe_telemetry_health()
        with self._state_lock:
            if self._state is not CaptureState.IDLE:
                raise CaptureError(f"cannot start an episode while state={self._state.value}")
            if self._background_error is not None:
                raise self._background_error
            episode_id = self._next_episode_id()
            dataset_dir = self.config.collection.dataset_dir
            started_at_ns = time.time_ns()
            mcap_filename = timestamped_mcap_filename(started_at_ns)
            pending_dir = pending_episode_dir(dataset_dir, episode_id)
            final_live_video_dir = live_video_dir(dataset_dir, episode_id)
            video_dir = final_live_video_dir.with_name(f"{episode_id}.partial")
            pending_dir.mkdir(parents=True, exist_ok=False)
            try:
                video_dir.mkdir(parents=True, exist_ok=False)
            except BaseException:
                shutil.rmtree(pending_dir, ignore_errors=True)
                raise
            partial_mcap_path = pending_dir / mcap_filename.replace(".mcap", ".partial.mcap")
            final_mcap_path = pending_mcap_path(dataset_dir, episode_id, filename=mcap_filename)
            collection = self.config.collection
            live_codec = getattr(collection, "live_video_codec", collection.video_codec)
            live_crf = getattr(collection, "live_video_crf", collection.video_crf)
            live_preset = getattr(collection, "live_video_preset", collection.video_preset)
            live_threads = getattr(collection, "live_video_threads", collection.video_threads)
            live_queue_size = getattr(collection, "live_video_queue_size", 120)
            videos: dict[str, object] = {}
            mcap: EpisodeMcapWriter | None = None
            try:
                mcap = EpisodeMcapWriter(partial_mcap_path, self.config.task_info)
                for role, camera in self.config.cameras.items():
                    writer_type = (
                        TimestampedVideoWriter
                        if getattr(self, "_video_timeline", "fixed") == "source"
                        else AsyncFfmpegWriter
                    )
                    videos[role] = writer_type(
                        video_dir / f"{role}.mp4",
                        camera.width,
                        camera.height,
                        camera.fps,
                        live_codec,
                        live_crf,
                        live_preset,
                        live_threads,
                        queue_size=live_queue_size,
                        input_pixel_format=ffmpeg_input_pixel_format(camera.stream_format),
                        encoder_cpu_set=getattr(collection, "live_video_cpu_set", None),
                    )
            except BaseException:
                for writer in videos.values():
                    try:
                        writer.close(commit=False)
                    except BaseException:
                        pass
                if mcap is not None:
                    try:
                        mcap.close()
                    except BaseException:
                        pass
                if partial_mcap_path.exists():
                    partial_mcap_path.unlink()
                shutil.rmtree(pending_dir, ignore_errors=True)
                shutil.rmtree(video_dir, ignore_errors=True)
                raise
            self._episode = _Episode(
                episode_id=episode_id,
                partial_mcap_path=partial_mcap_path,
                final_mcap_path=final_mcap_path,
                video_dir=video_dir,
                live_video_dir=final_live_video_dir,
                pending_dir=pending_dir,
                mcap=mcap,
                videos=videos,
                started_at_ns=started_at_ns,
                arming_started_monotonic_ns=time.monotonic_ns(),
                camera_timing={role: _FrameTiming() for role in self.config.cameras},
                camera_host_timing={role: _FrameTiming() for role in self.config.cameras},
                camera_timing_samples={role: [] for role in self.config.cameras},
            )
            self._last_episode_diagnostics = None
            self._last_telemetry_success_ns = time.time_ns()
            self._telemetry_progress.clear()
            self._state = CaptureState.ARMING
        self._print(
            f"Arming episode {episode_id}; running {self._ARMING_DURATION_S:g} s full-load health check. "
            "Wait for Recording episode before teleoperating."
        )
        self._wait_for_episode_arming(episode_id)
        return episode_id

    def _arming_health_issues(self, episode: _Episode) -> list[str]:
        """Return full-load issues observed while an episode is arming."""
        issues: list[str] = []
        gap_budget_ns = self._telemetry_gap_budget_ns()
        profile_name = getattr(self.config.collection, "validation_profile", "piper_compatible")
        camera_policy = build_validation_policy(self.config.cameras, profile_name)
        state_samples = int(getattr(episode, "state_samples", 0))
        if state_samples < self._ARMING_MIN_STATE_SAMPLES:
            issues.append(f"only {state_samples} state samples")
        max_state_gap_ns = int(getattr(episode, "max_state_sample_gap_ns", 0))
        if max_state_gap_ns > gap_budget_ns:
            issues.append(
                f"state gap {max_state_gap_ns / 1_000_000:.1f}/{gap_budget_ns / 1_000_000:.1f} ms"
            )
        for role, gap_ns in getattr(episode, "telemetry_max_source_gap_ns", {}).items():
            if int(gap_ns) > gap_budget_ns:
                issues.append(
                    f"{role} telemetry gap {int(gap_ns) / 1_000_000:.1f}/{gap_budget_ns / 1_000_000:.1f} ms"
                )
        try:
            camera_frames = dict(episode.camera_frames)
        except RuntimeError:
            camera_frames = {}
        for role in self.config.cameras:
            count = int(camera_frames.get(role, 0))
            if count < self._ARMING_MIN_CAMERA_FRAMES:
                issues.append(f"{role} has only {count} in-episode frames")
            timing = getattr(episode, "camera_timing", {}).get(role)
            if timing is not None:
                if timing.max_gap_ms > camera_policy.camera_max_gap_ms:
                    issues.append(
                        f"{role} source gap {timing.max_gap_ms:.1f}/{camera_policy.camera_max_gap_ms:.1f} ms"
                    )
                # The startup preflight already checks the source rate over a
                # rolling window before an episode is created. ARMING is a
                # short full-load liveness gate: keep checking frame count and
                # gaps, but do not reject a healthy stream on a noisy two-
                # second average that can sit just below the 0.5 Hz boundary.
        for role, writer in episode.videos.items():
            snapshot = self._video_writer_snapshot(writer)
            if snapshot.get("writer_error") or snapshot.get("diagnostic_error"):
                issues.append(f"{role} writer error")
            if snapshot.get("process_returncode") is not None:
                issues.append(f"{role} ffmpeg exited ({snapshot.get('process_returncode')})")
            dropped = snapshot.get("dropped_frames")
            if isinstance(dropped, int) and dropped:
                issues.append(f"{role} dropped {dropped} frame(s)")
            max_queue = snapshot.get("max_queue_depth")
            if isinstance(max_queue, int) and max_queue > self._ARMING_MAX_QUEUE_DEPTH:
                issues.append(f"{role} queue {max_queue}/{self._ARMING_MAX_QUEUE_DEPTH}")
            max_pipe_ms = snapshot.get("max_pipe_write_ms")
            if isinstance(max_pipe_ms, (int, float)) and max_pipe_ms > self._ARMING_MAX_PIPE_WRITE_MS:
                issues.append(f"{role} pipe {max_pipe_ms:.1f}/{self._ARMING_MAX_PIPE_WRITE_MS:.1f} ms")
        return issues

    def _wait_for_episode_arming(self, episode_id: int) -> None:
        """Require a healthy full-load window before authorizing teleoperation."""
        deadline = time.monotonic() + self._ARMING_DURATION_S
        while True:
            with self._state_lock:
                state = self._state
                episode = self._episode
                error = self._background_error
            if error is not None:
                self.wait_for_episode_finalization()
                raise error
            if state is not CaptureState.ARMING or episode is None or episode.episode_id != episode_id:
                raise CaptureError(f"episode {episode_id} stopped while arming")
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                issues = self._arming_health_issues(episode)
                if not issues:
                    with self._state_lock:
                        if self._state is CaptureState.ARMING and self._episode is episode:
                            self._state = CaptureState.RECORDING
                            episode.recording_started_at_ns = time.time_ns()
                            self._print(colorize(f"Recording episode {episode_id}.", "green"))
                            return
                    raise CaptureError(f"episode {episode_id} stopped while arming")
                error = CaptureError("arming health check failed; episode discarded: " + "; ".join(issues))
                self._print(colorize(str(error), "red"))
                self.discard_episode()
                raise error
            time.sleep(min(self._ARMING_POLL_S, remaining_s))

    def _detach_active_episode(self) -> Optional[_Episode]:
        with self._state_lock:
            if self._state not in {CaptureState.ARMING, CaptureState.RECORDING} or self._episode is None:
                return None
            was_recording = self._state is CaptureState.RECORDING
            self._state = CaptureState.FINALIZING
            episode = self._episode
            if was_recording and episode.recording_started_at_ns is not None:
                episode.recording_stopped_at_ns = time.time_ns()
            self._episode = None
            self._finalization_done.clear()
            return episode

    @staticmethod
    def _recording_duration_s(episode: _Episode, *, now_ns: int | None = None) -> float:
        """Return operator-visible recording time, excluding arming/finalization."""
        start_ns = getattr(episode, "recording_started_at_ns", None)
        if start_ns is None:
            return 0.0
        stop_ns = getattr(episode, "recording_stopped_at_ns", None) or now_ns or time.time_ns()
        return max(0.0, (stop_ns - start_ns) / 1_000_000_000)

    def _record_deleted_episode_duration(self, episode_id: int | None, duration_s: float, reason: str) -> None:
        """Append deletion duration accounting without affecting capture cleanup."""
        header = (
            "deleted_at\tepisode_id\treason\tduration_seconds\t"
            "reason_total_seconds\tall_deleted_total_seconds\n"
        )
        try:
            if reason not in self._DELETED_DURATION_REASONS:
                raise ValueError(f"unsupported deleted episode duration reason: {reason}")
            duration_s = max(0.0, float(duration_s))
            path = Path(self.config.collection.dataset_dir) / self._DELETED_DURATION_FILENAME
            path.parent.mkdir(parents=True, exist_ok=True)
            with self._DELETED_DURATION_FILE_LOCK, path.open("a+", encoding="utf-8") as stream:
                stream.seek(0)
                totals = {item: 0.0 for item in self._DELETED_DURATION_REASONS}
                for line in stream:
                    columns = line.rstrip("\n").split("\t")
                    if len(columns) < 4 or columns[2] not in totals:
                        continue
                    try:
                        totals[columns[2]] += float(columns[3])
                    except ValueError:
                        continue
                totals[reason] += duration_s
                all_deleted_s = sum(totals.values())
                stream.seek(0, os.SEEK_END)
                if stream.tell() == 0:
                    stream.write(header)
                timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                episode_label = "unknown" if episode_id is None else str(episode_id)
                stream.write(
                    f"{timestamp}\t{episode_label}\t{reason}\t{duration_s:.3f}\t"
                    f"{totals[reason]:.3f}\t{all_deleted_s:.3f}\n"
                )
                stream.flush()
        except (AttributeError, OSError, TypeError, ValueError) as exc:
            self._print(f"Warning: could not record deleted episode duration: {exc}")
            return
        self._print(
            f"Recorded deleted episode {episode_id} duration: {duration_s:.1f}s "
            f"({reason}); totals: {totals[reason]:.1f}s for this reason, {all_deleted_s:.1f}s overall."
        )

    @staticmethod
    def _write_source_timeline_manifest(episode: _Episode, origin_ns: int) -> None:
        """Persist the source-time mapping beside a source-timestamp episode."""
        cameras: dict[str, dict[str, int | None]] = {}
        for role, samples in episode.camera_timing_samples.items():
            timestamps = [int(sample.capture_timestamp_ns) for sample in samples]
            cameras[role] = {
                "frame_count": len(timestamps),
                "first_timestamp_ns": timestamps[0] if timestamps else None,
                "last_timestamp_ns": timestamps[-1] if timestamps else None,
                "first_pts_us": ((timestamps[0] - origin_ns) // 1_000) if timestamps else None,
                "last_pts_us": ((timestamps[-1] - origin_ns) // 1_000) if timestamps else None,
            }
        payload = {
            "format": 1,
            "mode": "source",
            "origin_timestamp_ns": int(origin_ns),
            "time_base": "1/1000000",
            "pts_definition": "pts_us = (capture_timestamp_ns - origin_timestamp_ns) // 1000",
            "cameras": cameras,
        }
        manifest_path = episode.video_dir / "source_timeline.json"
        manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _finish_episode(self, episode: _Episode, commit: bool) -> bool:
        errors: list[BaseException] = []
        saved = False
        cached_post_validation: dict[str, object] | None = None
        timeline_origin_ns: int | None = None
        # Keep the identity of the error that invalidated this episode.  The
        # error is cleared only after all writers and partial files have been
        # finalized, and only if no newer worker error replaced it meanwhile.
        with self._state_lock:
            finalized_background_error = self._background_error
        with self._state_lock:
            self._last_post_validation = None
        writer_stats_before_close: dict[str, dict[str, object]] | None = None
        try:
            if not episode.operations_drained.wait(timeout=self._FINALIZATION_WAIT_S):
                self._print("Waiting for in-flight camera/state writes before finalization.")
                episode.operations_drained.wait()
            with self._state_lock:
                if self._background_error is not None:
                    commit = False
            # This has to happen before discard-mode writer shutdown empties a
            # full queue and kills ffmpeg, otherwise an overflow loses the
            # evidence needed to distinguish encoding from camera delivery.
            writer_stats_before_close = {
                role: self._video_writer_snapshot(writer) for role, writer in episode.videos.items()
            }
            self._remember_episode_diagnostics(episode, before_close=writer_stats_before_close)
            if commit:
                try:
                    for role, samples in episode.camera_timing_samples.items():
                        episode.mcap.write_camera_timing_diagnostics(role, samples)
                except BaseException as exc:
                    errors.append(exc)
            try:
                episode.mcap.close()
            except BaseException as exc:
                errors.append(exc)
            if commit and getattr(self, "_video_timeline", "fixed") == "source":
                timeline_origin_ns = getattr(episode, "camera_start_timestamp_ns", None)
                if timeline_origin_ns is None:
                    timestamps = [
                        sample.capture_timestamp_ns
                        for samples in episode.camera_timing_samples.values()
                        for sample in samples
                    ]
                    timeline_origin_ns = min(timestamps) if timestamps else None
                if timeline_origin_ns is None:
                    errors.append(CaptureError("source-timestamp video mode recorded no camera timestamps"))
                else:
                    for writer in episode.videos.values():
                        set_origin = getattr(writer, "set_timeline_origin_ns", None)
                        if callable(set_origin):
                            try:
                                set_origin(timeline_origin_ns)
                            except BaseException as exc:
                                errors.append(exc)
            source_timeline = getattr(self, "_video_timeline", "fixed") == "source"
            errors.extend(
                close_video_writers(
                    episode.videos.values(),
                    commit=commit,
                    parallel=source_timeline,
                )
            )
            if commit and getattr(self, "_video_timeline", "fixed") == "source" and not errors:
                if timeline_origin_ns is None:
                    errors.append(CaptureError("source-timestamp video mode recorded no camera timestamps"))
                else:
                    try:
                        self._write_source_timeline_manifest(episode, timeline_origin_ns)
                    except BaseException as exc:
                        errors.append(exc)
            if commit and not errors:
                try:
                    # Both publication modes first establish a verified
                    # pending MCAP plus three closed live MP4s. HEVC leaves
                    # this pair for offline x265; H.264 promotes the already
                    # verified live bytes directly without re-encoding.
                    episode.partial_mcap_path.replace(episode.final_mcap_path)
                    episode.video_dir.replace(episode.live_video_dir)
                    if uses_direct_h264_publication(self.config):
                        published = publish_live_episode(episode.final_mcap_path, self.config)
                        episode.committed_mcap_path = published.final_mcap_path
                        if source_timeline:
                            cached_post_validation = dict(published.final_validation)
                    else:
                        validate_live_episode(episode.final_mcap_path, self.config)
                        episode.committed_mcap_path = episode.final_mcap_path
                    saved = True
                except BaseException as exc:
                    errors.append(exc)
            if saved:
                recording_duration_s = self._recording_duration_s(episode)
                frames = ", ".join(f"{role}={episode.camera_frames.get(role, 0)}" for role in self.config.cameras)
                timing = ", ".join(
                    f"{role}=capture {episode.camera_timing[role].average_fps:.2f}Hz/"
                    f"{episode.camera_timing[role].max_gap_ms:.1f}ms, "
                    f"host {episode.camera_host_timing[role].max_gap_ms:.1f}ms"
                    f" (device skipped={episode.camera_timing[role].device_frame_number_gaps}, "
                    f"regressions={episode.camera_timing[role].device_frame_number_regressions})"
                    for role in self.config.cameras
                )
                video_load = ", ".join(
                    f"{role}=queue {int(stats.get('max_queue_depth', 0))}/"
                    f"{int(stats.get('queue_capacity', 0))}, "
                    f"pipe {float(stats.get('max_pipe_write_ms', 0.0)):.1f}ms"
                    for role, writer in episode.videos.items()
                    for stats in (self._video_writer_snapshot(writer),)
                )
                self._print(
                    f"Saved {'H.264 final' if uses_direct_h264_publication(self.config) else 'live'} "
                    f"episode {episode.episode_id}: {episode.state_samples} state samples, "
                    f"{frames}, {recording_duration_s:.1f}s."
                )
                self._print(f"Episode {episode.episode_id} recorded duration: {recording_duration_s:.1f}s.")
                self._print(f"Camera timing: {timing}. Video writer load: {video_load}.")
                committed_mcap_path = episode.committed_mcap_path or episode.final_mcap_path
                if uses_direct_h264_publication(self.config):
                    if cached_post_validation is not None and not self._report_cached_post_episode_validation(
                        cached_post_validation,
                        video_label="Video asset validation",
                    ):
                        # A recovery/fake publication result without the
                        # embedded report should remain diagnosable rather
                        # than silently skipping the normal validation path.
                        self._run_post_episode_validation(committed_mcap_path)
                    elif cached_post_validation is None:
                        self._run_post_episode_validation(committed_mcap_path)
                    self._print(
                        f"Published H.264 final episode {episode.episode_id}: {committed_mcap_path}. "
                        "No offline transcode is required."
                    )
                else:
                    self._run_post_episode_validation(committed_mcap_path, live_video_dir=episode.live_video_dir)
                    mode = getattr(self.config, "capture_mode", None)
                    mode_arg = f" --mode {mode}" if mode is not None else ""
                    self._print(
                        "Final HEVC assets are pending. Run: "
                        f"yam-data-collect transcode {committed_mcap_path} --config {self.config.source_path}{mode_arg}"
                    )
            else:
                live_dir = getattr(episode, "live_video_dir", None)
                pending_mcap = getattr(episode, "final_mcap_path", None)
                keep_invalid_live_episode = (
                    commit
                    and isinstance(live_dir, Path)
                    and live_dir.is_dir()
                    and isinstance(pending_mcap, Path)
                    and pending_mcap.is_file()
                )
                if keep_invalid_live_episode:
                    detail = "; ".join(str(error) for error in errors) or "unknown live-asset validation failure"
                    self._print(
                        f"Episode {episode.episode_id} was not published because live video integrity validation "
                        f"failed; pending MCAP and videos_live assets were retained for diagnosis: {detail}"
                    )
                else:
                    try:
                        if episode.partial_mcap_path.exists():
                            episode.partial_mcap_path.unlink()
                    except OSError as exc:
                        errors.append(exc)
                    shutil.rmtree(getattr(episode, "pending_dir", episode.partial_mcap_path.parent), ignore_errors=True)
                    shutil.rmtree(episode.video_dir, ignore_errors=True)
                    if isinstance(live_dir, Path):
                        shutil.rmtree(live_dir, ignore_errors=True)
                if errors:
                    detail = "; ".join(str(error) for error in errors)
                    if not keep_invalid_live_episode:
                        self._print(
                            colorize(
                                f"Episode {episode.episode_id} was discarded after finalization errors: {detail}",
                                "red",
                            )
                        )
                else:
                    self._print(f"Discarded episode {episode.episode_id}.")
            return saved
        finally:
            if writer_stats_before_close is not None:
                writer_stats_after_close = {
                    role: self._video_writer_snapshot(writer) for role, writer in episode.videos.items()
                }
                self._remember_episode_diagnostics(episode, after_close=writer_stats_after_close)
            with self._state_lock:
                if self._state is not CaptureState.CLOSED:
                    self._state = CaptureState.IDLE
                if (
                    finalized_background_error is not None
                    and self._background_error is finalized_background_error
                ):
                    self._background_error = None
            self._finalization_done.set()

    def _store_post_episode_validation(
        self,
        mcap_result: dict[str, object],
        video_result: dict[str, object],
        *,
        video_label: str,
    ) -> None:
        """Print and retain a completed MCAP/video validation report."""
        print_validation(mcap_result)
        asset_status = "PASS" if video_result["passed"] else "FAIL"
        self._print(f"{video_label}: {asset_status}")
        with self._state_lock:
            self._last_post_validation = {
                "passed": bool(mcap_result["output_mcap_passed"] and video_result["passed"]),
                "mcap_passed": bool(mcap_result["output_mcap_passed"]),
                "video_passed": bool(video_result["passed"]),
                "failed_checks": list(mcap_result.get("failed_checks", [])),
                "informational_failures": list(mcap_result.get("informational_failures", [])),
                "video_label": video_label,
            }

    def _report_cached_post_episode_validation(self, validation: dict[str, object], *, video_label: str) -> bool:
        """Report validation already completed before direct source publication."""
        mcap_result = validation.get("mcap_validation")
        if not isinstance(mcap_result, dict):
            return False
        required_mcap_keys = {"output_mcap_passed", "failed_checks", "passed_checks", "total_checks"}
        if not required_mcap_keys.issubset(mcap_result):
            return False
        video_result = {
            "passed": bool(validation.get("passed")),
        }
        self._store_post_episode_validation(mcap_result, video_result, video_label=video_label)
        return True

    def _run_post_episode_validation(self, mcap_path: Path, *, live_video_dir: Path | None = None) -> None:
        """Report MCAP and live/final asset status without changing acceptance."""
        policy = build_validation_policy(self.config.cameras, self.config.collection.validation_profile)
        try:
            mcap_result = validate_mcap(mcap_path, policy)
            if live_video_dir is None:
                video_result = validate_video_assets(
                    mcap_path,
                    policy,
                    expected_video_codecs=final_codec_family(self.config),
                    require_fps_match=getattr(self, "_video_timeline", "fixed") != "source",
                )
                label = "Video asset validation"
            else:
                live_codec = getattr(self.config.collection, "live_video_codec", self.config.collection.video_codec)
                expected_codecs = ("h264",) if live_codec in {"libx264", "h264"} else ("hevc",)
                video_result = validate_video_assets(
                    mcap_path,
                    policy,
                    video_dir=live_video_dir,
                    expected_video_codecs=expected_codecs,
                    require_fps_match=getattr(self, "_video_timeline", "fixed") != "source",
                )
                label = "Live video asset validation"
            self._store_post_episode_validation(mcap_result, video_result, video_label=label)
        except Exception as exc:
            self._print(f"Post-episode validation could not run: {exc}")
            with self._state_lock:
                self._last_post_validation = {"passed": False, "error": str(exc)}

    def delete_saved_episode(self, *, deletion_reason: str | None = None) -> bool:
        """Delete the most recently published episode, if one exists."""
        with self._state_lock:
            if self._state is not CaptureState.IDLE:
                self._print("Cannot delete an episode while capture is active.", color="red")
                return False
            path = self._last_saved_episode_path
            episode_id = self._last_saved_episode_id
            duration_s = getattr(self, "_last_saved_episode_duration_s", None)
        if path is None:
            return False
        try:
            dataset_dir = Path(self.config.collection.dataset_dir).resolve()
            resolved_path = path.resolve()
            if dataset_dir not in resolved_path.parents:
                raise ValueError(f"episode path is outside configured dataset: {path}")
            remove_episode_assets(path)
        except (OSError, ValueError) as exc:
            self._print(f"Could not delete episode {episode_id}: {exc}", color="red")
            return False
        with self._state_lock:
            if self._last_saved_episode_path == path:
                self._last_saved_episode_path = None
                self._last_saved_episode_id = None
                self._last_saved_episode_duration_s = None
        label = episode_id if episode_id is not None else path.stem
        self._print(f"Deleted episode {label}.", color="red")
        if deletion_reason is not None and duration_s is not None:
            self._record_deleted_episode_duration(episode_id, duration_s, deletion_reason)
        return True

    def _review_saved_episode(self, keys: _TerminalKeys) -> None:
        """Wait for Enter/q after a passed terminal episode validation."""
        with self._state_lock:
            validation = copy.deepcopy(self._last_post_validation)
            episode_id = self._last_saved_episode_id
        if not isinstance(validation, dict) or not validation.get("passed"):
            return
        self._print(
            f"Episode {episode_id} passed validation. Press Enter/Space to keep it, or q to delete it."
        )
        while self.state is CaptureState.IDLE:
            key = keys.poll(0.2)
            if key is None:
                continue
            if key in {"\n", "\r"}:
                self._print(f"Kept episode {episode_id}.", color="green")
                with self._state_lock:
                    self._last_saved_episode_path = None
                    self._last_saved_episode_id = None
                    self._last_saved_episode_duration_s = None
                return
            if key == " ":
                self._print(f"Kept episode {episode_id}.", color="green")
                with self._state_lock:
                    self._last_saved_episode_path = None
                    self._last_saved_episode_id = None
                    self._last_saved_episode_duration_s = None
                try:
                    self.start_episode()
                except CaptureError as exc:
                    self._print(str(exc), color="red")
                return
            if key.lower() == "q":
                self.delete_saved_episode(deletion_reason="user_deleted")
                return

    def save_episode(self) -> bool:
        if self.state is CaptureState.ARMING:
            self._print("Episode is still arming; it cannot be saved before the full-load health check passes.")
            return False
        episode = self._detach_active_episode()
        if episode is None:
            return False
        with self._state_lock:
            error = self._background_error
        if error is not None:
            self._print(
                colorize(
                    f"Recording error: {error}; discarding active episode instead of saving it.",
                    "red",
                )
            )
        recording_duration_s = self._recording_duration_s(episode)
        saved = self._finish_episode(episode, commit=error is None)
        if not saved:
            with self._state_lock:
                final_error = self._background_error
            if error is not None or final_error is not None:
                self._record_deleted_episode_duration(
                    getattr(episode, "episode_id", None), recording_duration_s, "recording_error"
                )
            return False
        committed_mcap_path = getattr(episode, "committed_mcap_path", None) or episode.final_mcap_path
        with self._state_lock:
            self._last_saved_episode_path = Path(committed_mcap_path)
            self._last_saved_episode_id = episode.episode_id
            self._last_saved_episode_duration_s = recording_duration_s
            validation = copy.deepcopy(getattr(self, "_last_post_validation", None))
        if getattr(self, "_auto_delete_failed_validation", False) and (
            not isinstance(validation, dict) or not validation.get("passed")
        ):
            failed = validation.get("failed_checks") if isinstance(validation, dict) else None
            detail = ", ".join(str(item) for item in failed) if failed else "validation did not pass"
            self._print(
                f"Episode {episode.episode_id} failed validation ({detail}); deleting it automatically.",
                color="red",
            )
            self.delete_saved_episode(deletion_reason="validation_failed")
            return False
        callback = getattr(self, "_on_episode_saved", None)
        if callback is not None:
            try:
                stop_inputs = bool(callback(committed_mcap_path))
            except Exception as exc:
                self._print(
                    f"Post-save processing check failed for episode {episode.episode_id}; "
                    f"raw episode remains saved: {exc}"
                )
            else:
                if stop_inputs:
                    self._stop_after_saved_episode.set()
        return True

    def discard_episode(self, *, deletion_reason: str | None = None) -> bool:
        episode = self._detach_active_episode()
        if episode is None:
            return False
        recording_duration_s = self._recording_duration_s(episode)
        self._finish_episode(episode, commit=False)
        if deletion_reason is not None:
            self._record_deleted_episode_duration(episode.episode_id, recording_duration_s, deletion_reason)
        return True

    def wait_for_episode_finalization(self, timeout_s: float | None = None) -> bool:
        """Wait for asynchronous failure cleanup without reaching into internals."""
        with self._state_lock:
            finalizing = self._state is CaptureState.FINALIZING
        if not finalizing:
            return True
        wait_s = self._FINALIZATION_WAIT_S if timeout_s is None else timeout_s
        return self._finalization_done.wait(wait_s)

    def run_interactive(self) -> None:
        try:
            self.start()
            with _TerminalKeys() as keys:
                while self.state is not CaptureState.CLOSED:
                    if self._stop_after_saved_episode.is_set():
                        break
                    key = keys.poll(0.2)
                    if key is None:
                        continue
                    if key == " ":
                        if self.state is CaptureState.IDLE:
                            try:
                                self.start_episode()
                            except CaptureError as exc:
                                self._print(str(exc))
                    elif key in {"\n", "\r"}:
                        if self.state is CaptureState.RECORDING:
                            if self.save_episode() and getattr(self, "_enable_post_save_review", False):
                                self._review_saved_episode(keys)
                    elif key.lower() == "q":
                        if self.state in {CaptureState.ARMING, CaptureState.RECORDING}:
                            self.discard_episode(deletion_reason="user_deleted")
                        elif self.state is CaptureState.IDLE and getattr(self, "_enable_post_save_review", False):
                            self.delete_saved_episode(deletion_reason="user_deleted")
        except KeyboardInterrupt:
            self._print("Interrupted by operator.")
        finally:
            self.close()

    def close(self) -> None:
        with self._state_lock:
            if self._state is CaptureState.CLOSED:
                self._stop_live_preview()
                self._release_dataset_lock()
                return
        try:
            self.discard_episode()
            if self.state is CaptureState.FINALIZING:
                self._print("Waiting for active episode finalization.")
                if not self._finalization_done.wait(timeout=self._FINALIZATION_WAIT_S):
                    self._print("Finalization did not finish in time; leaving partial artifacts for safe recovery.")
            self._stop_event.set()
            for camera in self._cameras.values():
                try:
                    camera.stop()
                except Exception as exc:
                    self._print(f"Warning: could not stop camera {camera.role}: {exc}")
            for worker in self._camera_workers:
                worker.join()
            if self._state_thread is not None:
                self._state_thread.join(timeout=2.0)
            self._stop_live_preview()
            self._executor.shutdown(wait=False, cancel_futures=True)
            for client in self._telemetry_clients.values():
                client.close()
            with self._state_lock:
                self._state = CaptureState.CLOSED
        finally:
            self._stop_runtime_diagnostics()
            self._release_dataset_lock()
