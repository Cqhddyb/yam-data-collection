"""Standalone mixed-frame-rate RealSense qualification with optional MP4 artifacts.

This module deliberately has no Portal, CAN, MCAP, or robot dependency. It is
used to certify a camera installation on a host before the full recorder uses
the same serial-based camera definitions.
"""

from __future__ import annotations

import bisect
import json
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from .cameras import Camera, RealSenseCamera, ffmpeg_input_pixel_format
from .config import CameraConfig, CameraTestConfig
from .models import CameraFrame
from .quality import ValidationPolicy, build_validation_policy
from .video import AsyncFfmpegWriter


class CameraTestError(RuntimeError):
    """A camera qualification run could not be completed."""


@dataclass
class _Timing:
    timestamps_ns: list[int] = field(default_factory=list)
    regressions: int = 0

    def add(self, timestamp_ns: int) -> None:
        if self.timestamps_ns and timestamp_ns <= self.timestamps_ns[-1]:
            self.regressions += 1
            return
        self.timestamps_ns.append(timestamp_ns)

    def report(self) -> dict[str, Any]:
        if len(self.timestamps_ns) < 2:
            return {
                "frames": len(self.timestamps_ns),
                "duration_s": 0.0,
                "fps": 0.0,
                "max_gap_ms": 0.0,
                "timestamp_regressions": self.regressions,
            }
        duration_s = (self.timestamps_ns[-1] - self.timestamps_ns[0]) / 1_000_000_000
        gaps_ns = [later - earlier for earlier, later in zip(self.timestamps_ns, self.timestamps_ns[1:])]
        return {
            "frames": len(self.timestamps_ns),
            "duration_s": round(duration_s, 3),
            "fps": round((len(self.timestamps_ns) - 1) / duration_s, 3) if duration_s > 0 else 0.0,
            "max_gap_ms": round(max(gaps_ns) / 1_000_000, 3),
            "timestamp_regressions": self.regressions,
        }


@dataclass
class _CameraRuntime:
    camera: CameraConfig
    host_timing: _Timing = field(default_factory=_Timing)
    device_timing: _Timing = field(default_factory=_Timing)
    capture_timing: _Timing = field(default_factory=_Timing)
    captured_frames: int = 0
    accepted_frames: int = 0
    error: str | None = None
    writer: AsyncFfmpegWriter | None = None


@dataclass(frozen=True)
class CameraTestRun:
    """Result metadata for one finished qualification directory."""

    output_dir: Path
    report: Mapping[str, Any]

    @property
    def passed(self) -> bool:
        return bool(self.report["passed"])


def _parse_frame_rate(value: str) -> float:
    numerator, denominator = value.split("/", maxsplit=1)
    divisor = int(denominator)
    if divisor == 0:
        raise ValueError(f"invalid frame rate {value!r}")
    return int(numerator) / divisor


def _probe_video(path: Path) -> dict[str, Any]:
    """Decode enough stream metadata to verify the generated MP4 asset."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,nb_frames,nb_read_frames",
        "-count_frames",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return {"passed": False, "error": "ffprobe is not installed or not on PATH"}
    if completed.returncode != 0:
        return {"passed": False, "error": completed.stderr.strip() or "ffprobe failed"}
    try:
        streams = json.loads(completed.stdout).get("streams", [])
        stream = streams[0]
        frame_count = int(stream.get("nb_read_frames") or stream.get("nb_frames") or 0)
        return {
            "passed": True,
            "codec": str(stream["codec_name"]),
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "fps": round(_parse_frame_rate(str(stream["avg_frame_rate"])), 3),
            "frame_count": frame_count,
        }
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return {"passed": False, "error": "ffprobe returned incomplete stream metadata"}


def _nearest(timestamp_ns: int, candidates: list[int]) -> int | None:
    index = bisect.bisect_left(candidates, timestamp_ns)
    nearby: list[int] = []
    if index < len(candidates):
        nearby.append(candidates[index])
    if index:
        nearby.append(candidates[index - 1])
    return min(nearby, key=lambda value: abs(value - timestamp_ns)) if nearby else None


def _mutual_nearest_summary(first: list[int], second: list[int]) -> dict[str, float | int | None]:
    """Report the same mutual-nearest camera pairing used by the Piper checker."""
    a, b = sorted(first), sorted(second)
    if not a or not b:
        return {"max_offset_ms": None, "valid_pairs": 0, "orphans": len(a) + len(b)}
    nearest_b = {timestamp_ns: _nearest(timestamp_ns, b) for timestamp_ns in a}
    nearest_a = {timestamp_ns: _nearest(timestamp_ns, a) for timestamp_ns in b}
    offsets_ms = [
        abs(timestamp_ns - paired_ns) / 1_000_000
        for timestamp_ns, paired_ns in nearest_b.items()
        if paired_ns is not None and nearest_a.get(paired_ns) == timestamp_ns
    ]
    return {
        "max_offset_ms": round(max(offsets_ms), 3) if offsets_ms else None,
        "valid_pairs": len(offsets_ms),
        "orphans": len(a) + len(b) - 2 * len(offsets_ms),
    }


class CameraQualification:
    """Evaluate configured camera streams, optionally recording MP4 artifacts."""

    def __init__(
        self,
        config: CameraTestConfig,
        *,
        camera_factory: Callable[[CameraConfig], Camera] = RealSenseCamera,
        writer_factory: Callable[..., AsyncFfmpegWriter] = AsyncFfmpegWriter,
        monotonic: Callable[[], float] = time.monotonic,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.config = config
        self._camera_factory = camera_factory
        self._writer_factory = writer_factory
        self._monotonic = monotonic
        self._monotonic_ns = monotonic_ns

    def _select_camera_roles(self, camera_roles: Sequence[str] | None) -> tuple[str, ...]:
        configured = tuple(self.config.cameras)
        if camera_roles is None:
            return configured
        requested = tuple(str(role).strip() for role in camera_roles)
        if not requested or any(not role for role in requested):
            raise ValueError("at least one camera role must be selected")
        duplicates = sorted({role for role in requested if requested.count(role) > 1})
        if duplicates:
            raise ValueError(f"camera roles must not be repeated: {', '.join(duplicates)}")
        unknown = [role for role in requested if role not in self.config.cameras]
        if unknown:
            available = ", ".join(configured)
            raise ValueError(f"unknown camera role(s): {', '.join(unknown)}; available: {available}")
        requested_set = set(requested)
        return tuple(role for role in configured if role in requested_set)

    def _create_output_dirs(self, selected_roles: Sequence[str]) -> tuple[Path, Path]:
        root = self.config.settings.output_dir
        configured_roles = tuple(self.config.cameras)
        if tuple(selected_roles) != configured_roles:
            root = root / f"only-{'-'.join(selected_roles)}"
        root.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        final_dir = root / timestamp
        suffix = 1
        while final_dir.exists() or final_dir.with_name(f"{final_dir.name}.partial").exists():
            final_dir = root / f"{timestamp}-{suffix}"
            suffix += 1
        partial_dir = final_dir.with_name(f"{final_dir.name}.partial")
        partial_dir.mkdir()
        saved_config = dict(self.config.raw)
        saved_settings = dict(saved_config.get("camera_test_config", {}))
        saved_settings["validation_profile"] = self.config.settings.validation_profile
        saved_settings["selected_camera_roles"] = list(selected_roles)
        saved_config["camera_test_config"] = saved_settings
        capture_mode = getattr(self.config, "capture_mode", None)
        if capture_mode is not None:
            saved_config["selected_capture_mode"] = capture_mode
        with (partial_dir / "test_config.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(saved_config, stream, allow_unicode=True, sort_keys=False)
        return partial_dir, final_dir

    def _capture_camera(
        self,
        runtime: _CameraRuntime,
        camera: Camera,
        deadline: float,
        stop_event: threading.Event,
    ) -> None:
        while not stop_event.is_set() and self._monotonic() < deadline:
            try:
                frame = camera.read(timeout_ms=round(self.config.settings.timeout_threshold_s * 1_000))
                self._record_frame(runtime, frame)
            except Exception as exc:
                runtime.error = f"capture failed: {exc}"
                stop_event.set()
                return

    def _record_frame(self, runtime: _CameraRuntime, frame: CameraFrame) -> None:
        runtime.host_timing.add(frame.host_timestamp_ns or self._monotonic_ns())
        if frame.device_timestamp_ns is not None:
            runtime.device_timing.add(frame.device_timestamp_ns)
        runtime.capture_timing.add(frame.timestamp_ns)
        runtime.captured_frames += 1
        if runtime.writer is not None:
            accepted = runtime.writer.enqueue(frame.image_bgr, copy_frame=not frame.owns_image)
            if not accepted:
                raise CameraTestError("video queue overflow")
            runtime.accepted_frames += 1

    def _camera_report(
        self,
        runtime: _CameraRuntime,
        output_dir: Path,
        duration_s: float,
        policy: ValidationPolicy,
        record_video: bool,
    ) -> dict[str, Any]:
        camera = runtime.camera
        host = runtime.host_timing.report()
        device = runtime.device_timing.report()
        capture = runtime.capture_timing.report()
        source_name = "capture" if capture["frames"] >= 2 and capture["timestamp_regressions"] == 0 else "host"
        source = capture if source_name == "capture" else host
        writer = runtime.writer
        if record_video:
            video_path = output_dir / f"{camera.role}.mp4"
            video = _probe_video(video_path) if video_path.is_file() else {"passed": False, "error": "missing video"}
            configured_codec = self.config.settings.video_codec.strip().lower()
            expected_codec = "h264" if configured_codec in {"libx264", "h264"} else "hevc"
            expected_video: dict[str, Any] | None = {
                "codec": expected_codec,
                "width": camera.width,
                "height": camera.height,
                "fps": camera.fps,
                "frame_count": runtime.accepted_frames,
            }
            video_matches: bool | None = bool(
                video.get("passed")
                and video.get("codec") == expected_video["codec"]
                and video.get("width") == expected_video["width"]
                and video.get("height") == expected_video["height"]
                and abs(float(video.get("fps", 0.0)) - expected_video["fps"]) <= self.config.settings.fps_tolerance
                and video.get("frame_count") == expected_video["frame_count"]
            )
        else:
            video = {"status": "disabled", "reason": "--no-video"}
            expected_video = None
            video_matches = None
        max_gap_ms_limit = policy.camera_max_gap_ms
        if getattr(self.config, "capture_mode", None) is None:
            max_gap_ms_limit = min(
                max_gap_ms_limit,
                self.config.settings.max_gap_factor * 1_000 / camera.fps,
            )
        source_matches = bool(
            source["duration_s"] >= min(1.0, duration_s / 2)
            and abs(source["fps"] - camera.fps) <= self.config.settings.fps_tolerance
            and source["max_gap_ms"] <= max_gap_ms_limit
        )
        queue_overflow = bool(writer and writer.dropped_frames)
        passed = bool(runtime.error is None and source_matches and (video_matches is not False) and not queue_overflow)
        return {
            "serial": camera.serial,
            "resolution": [camera.width, camera.height],
            "target_fps": camera.fps,
            "source_clock": source_name,
            "max_gap_ms_limit": round(max_gap_ms_limit, 3),
            "host_timing": host,
            "device_timing": device,
            "capture_timing": capture,
            "source_passed": source_matches,
            "captured_frames": runtime.captured_frames,
            "accepted_frames": runtime.accepted_frames,
            "video": video,
            "video_expected": expected_video,
            "video_passed": video_matches,
            "video_writer": (
                {
                    "dropped_frames": writer.dropped_frames,
                    "queue_high_watermark": writer.max_queue_depth,
                    "queue_capacity": writer.queue_size,
                    "max_pipe_write_ms": round(writer.max_pipe_write_ms, 3),
                }
                if writer is not None
                else None
            ),
            "error": runtime.error,
            "passed": passed,
        }

    def run(
        self,
        duration_s: float | None = None,
        *,
        record_video: bool = True,
        camera_roles: Sequence[str] | None = None,
    ) -> CameraTestRun:
        """Run one concurrent camera-only test and always preserve its report."""
        selected_duration_s = duration_s if duration_s is not None else self.config.settings.duration_s
        if selected_duration_s <= 0:
            raise ValueError("duration_s must be positive")
        configured_roles = tuple(self.config.cameras)
        selected_roles = self._select_camera_roles(camera_roles)
        selected_cameras = {role: self.config.cameras[role] for role in selected_roles}
        policy = build_validation_policy(selected_cameras, self.config.settings.validation_profile)
        partial_dir, final_dir = self._create_output_dirs(selected_roles)
        runtimes = {role: _CameraRuntime(camera) for role, camera in selected_cameras.items()}
        cameras: dict[str, Camera] = {}
        stop_event = threading.Event()
        threads: list[threading.Thread] = []
        started: list[Camera] = []
        run_error: str | None = None
        try:
            for role, runtime in runtimes.items():
                camera = self._camera_factory(runtime.camera)
                cameras[role] = camera
                if record_video:
                    runtime.writer = self._writer_factory(
                        partial_dir / f"{role}.mp4",
                        runtime.camera.width,
                        runtime.camera.height,
                        runtime.camera.fps,
                        self.config.settings.video_codec,
                        self.config.settings.video_crf,
                        self.config.settings.video_preset,
                        self.config.settings.video_threads,
                        self.config.settings.video_queue_size,
                        input_pixel_format=ffmpeg_input_pixel_format(runtime.camera.stream_format),
                    )
            for role, camera in cameras.items():
                try:
                    camera.start()
                    started.append(camera)
                except Exception as exc:
                    runtimes[role].error = f"startup failed: {exc}"
                    stop_event.set()
                    break
            if not stop_event.is_set():
                deadline = self._monotonic() + selected_duration_s
                for role, camera in cameras.items():
                    thread = threading.Thread(
                        target=self._capture_camera,
                        args=(runtimes[role], camera, deadline, stop_event),
                        name=f"camera-test-{role}",
                        daemon=True,
                    )
                    threads.append(thread)
                    thread.start()
                for thread in threads:
                    thread.join()
        except KeyboardInterrupt:
            run_error = "interrupted by operator"
            stop_event.set()
        except Exception as exc:
            run_error = f"setup failed: {exc}"
            stop_event.set()
        finally:
            stop_event.set()
            for camera in reversed(started):
                try:
                    camera.stop()
                except Exception as exc:
                    role = next((key for key, value in cameras.items() if value is camera), "unknown")
                    if role in runtimes and runtimes[role].error is None:
                        runtimes[role].error = f"shutdown failed: {exc}"
            for thread in threads:
                thread.join()
            for runtime in runtimes.values():
                if runtime.writer is None:
                    continue
                try:
                    runtime.writer.close(commit=True)
                except Exception as exc:
                    if runtime.error is None:
                        runtime.error = f"video finalization failed: {exc}"
        camera_reports = {
            role: self._camera_report(runtime, partial_dir, selected_duration_s, policy, record_video)
            for role, runtime in runtimes.items()
        }
        capture_timestamps = {role: runtime.capture_timing.timestamps_ns for role, runtime in runtimes.items()}
        roles = list(selected_roles)
        synchronization = {
            f"{first}__{second}": _mutual_nearest_summary(capture_timestamps[first], capture_timestamps[second])
            for index, first in enumerate(roles)
            for second in roles[index + 1 :]
        }
        sync_applicable = len(roles) >= 2
        sync_passed = not sync_applicable or all(
            entry["max_offset_ms"] is not None and entry["max_offset_ms"] <= policy.camera_sync_ms
            for entry in synchronization.values()
        )
        if record_video:
            qualification_scope = "timing_sync_and_video" if sync_applicable else "timing_and_video"
        else:
            qualification_scope = "timing_and_sync_only" if sync_applicable else "timing_only"
        report: dict[str, Any] = {
            "passed": run_error is None and all(entry["passed"] for entry in camera_reports.values()) and sync_passed,
            "duration_s": selected_duration_s,
            "video_recording_enabled": record_video,
            "qualification_scope": qualification_scope,
            "configured_camera_roles": list(configured_roles),
            "selected_camera_roles": list(selected_roles),
            "full_three_camera_qualification": selected_roles == configured_roles and len(configured_roles) == 3,
            "policy": {
                "validation_profile": policy.profile_name,
                "fps_tolerance": self.config.settings.fps_tolerance,
                "max_gap_factor": self.config.settings.max_gap_factor,
                "camera_max_gap_ms": policy.camera_max_gap_ms,
            },
            "cameras": camera_reports,
            "camera_sync": {
                "applicable": sync_applicable,
                "threshold_ms": policy.camera_sync_ms,
                "passed": sync_passed,
                "pairwise": synchronization,
            },
            "run_error": run_error,
        }
        with (partial_dir / "report.json").open("w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        fatal_error = run_error is not None or any(runtime.error is not None for runtime in runtimes.values())
        output_dir = partial_dir
        if not fatal_error:
            partial_dir.replace(final_dir)
            output_dir = final_dir
        return CameraTestRun(output_dir=output_dir, report=report)
