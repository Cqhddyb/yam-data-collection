"""MCAP and video quality checks for Piper-compatible and YAM capture modes."""

from __future__ import annotations

import bisect
import json
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

from .config import CameraConfig
from .mcap_writer import CAMERA_TOPICS, FRAME_TIMING_DIAGNOSTICS_TOPIC, ROBOT_TOPICS
from .terminal import colorize
from .validation_profiles import (
    CUSTOM_PROFILE,
    PIPER_COMPATIBLE_PROFILE,
    get_validation_profile,
    normalize_validation_profile,
)


@dataclass(frozen=True)
class ValidationPolicy:
    """Concrete validation thresholds, optionally built from a named profile."""

    profile_name: str = CUSTOM_PROFILE
    expected_camera_fps: float = 30.0
    expected_camera_fps_by_topic: Mapping[str, float] = field(default_factory=dict)
    expected_camera_resolution_by_topic: Mapping[str, tuple[int, int]] = field(default_factory=dict)
    camera_average_tolerance: float = 0.5
    camera_min_average_fps: float | None = None
    camera_max_average_fps: float | None = None
    camera_max_gap_ms: float = 69.0
    camera_min_width: int = 640
    camera_min_height: int = 480
    camera_sync_ms: float = 34.0
    joint_camera_sync_ms: float = 34.0
    joint_min_frequency: float = 45.0
    rolling_window_s: float = 1.0
    expected_video_codecs: tuple[str, ...] = ("hevc",)
    require_exact_camera_resolution: bool = False
    auto_detect_camera_fps: bool = False
    allowed_camera_fps: tuple[float, ...] = (15.0, 30.0, 60.0)

    @classmethod
    def for_cameras(cls, cameras: Mapping[str, CameraConfig]) -> "ValidationPolicy":
        """Build the legacy configurable policy used before named profiles existed."""
        return cls(
            profile_name=CUSTOM_PROFILE,
            expected_camera_fps_by_topic={
                CAMERA_TOPICS[role]: camera.fps for role, camera in cameras.items() if role in CAMERA_TOPICS
            },
            expected_camera_resolution_by_topic={
                CAMERA_TOPICS[role]: (camera.width, camera.height)
                for role, camera in cameras.items()
                if role in CAMERA_TOPICS
            },
            require_exact_camera_resolution=True,
        )

    @classmethod
    def for_profile(cls, profile_name: str, cameras: Mapping[str, CameraConfig] | None = None) -> "ValidationPolicy":
        """Build the selected Piper-compatible or YAM mode standard."""
        normalized = normalize_validation_profile(profile_name)
        if normalized == CUSTOM_PROFILE:
            return cls.for_cameras(cameras) if cameras is not None else cls()
        profile = get_validation_profile(normalized)
        if profile.require_exact_camera_resolution and cameras is None:
            raise ValueError(f"validation profile {profile.name} requires configured cameras")
        expected_fps_by_topic = (
            {
                CAMERA_TOPICS[role]: profile.expected_camera_fps
                for role in cameras or {}
                if role in CAMERA_TOPICS and profile.expected_camera_fps is not None
            }
            if profile.expected_camera_fps is not None
            else {}
        )
        expected_resolution_by_topic = (
            {
                CAMERA_TOPICS[role]: (camera.width, camera.height)
                for role, camera in (cameras or {}).items()
                if role in CAMERA_TOPICS
            }
            if profile.require_exact_camera_resolution
            else {}
        )
        return cls(
            profile_name=profile.name,
            expected_camera_fps=profile.expected_camera_fps or 30.0,
            expected_camera_fps_by_topic=expected_fps_by_topic,
            expected_camera_resolution_by_topic=expected_resolution_by_topic,
            camera_min_average_fps=profile.camera_min_average_fps,
            camera_max_average_fps=profile.camera_max_average_fps,
            camera_max_gap_ms=profile.camera_max_gap_ms,
            camera_sync_ms=profile.camera_sync_ms,
            joint_camera_sync_ms=profile.joint_camera_sync_ms,
            require_exact_camera_resolution=profile.require_exact_camera_resolution,
            auto_detect_camera_fps=profile.auto_detect_camera_fps,
        )

    def target_fps_for(self, topic: str, observed_fps: float | None = None) -> float | None:
        """Return a configured target or Piper's closest supported historical rate."""
        configured = self.expected_camera_fps_by_topic.get(topic)
        if configured is not None:
            return float(configured)
        if not self.auto_detect_camera_fps:
            return float(self.expected_camera_fps)
        if observed_fps is None or observed_fps <= 0:
            return None
        target = min(self.allowed_camera_fps, key=lambda candidate: abs(candidate - observed_fps))
        return float(target) if abs(target - observed_fps) / target < 0.1 else None

    def target_resolution_for(self, topic: str) -> tuple[int, int] | None:
        """Return the configured profile when strict resolution validation is enabled."""
        return self.expected_camera_resolution_by_topic.get(topic)

    def camera_rate_bounds_for(self, target_fps: float) -> tuple[float, float]:
        """Return the accepted average-FPS interval for one camera target."""
        if not self.auto_detect_camera_fps and (
            self.camera_min_average_fps is not None or self.camera_max_average_fps is not None
        ):
            minimum = (
                float(self.camera_min_average_fps)
                if self.camera_min_average_fps is not None
                else target_fps - self.camera_average_tolerance
            )
            maximum = (
                float(self.camera_max_average_fps)
                if self.camera_max_average_fps is not None
                else target_fps + self.camera_average_tolerance
            )
            return minimum, maximum
        return target_fps - self.camera_average_tolerance, target_fps + self.camera_average_tolerance

    def camera_rate_passes(self, observed_fps: float, target_fps: float) -> bool:
        """Return whether an observed average is inside the profile's FPS bounds."""
        minimum, maximum = self.camera_rate_bounds_for(target_fps)
        return minimum <= observed_fps <= maximum


def build_validation_policy(
    cameras: Mapping[str, CameraConfig], profile_name: str = PIPER_COMPATIBLE_PROFILE
) -> ValidationPolicy:
    """Resolve one configuration's named profile to the thresholds used everywhere."""
    normalized = normalize_validation_profile(profile_name)
    if normalized == CUSTOM_PROFILE:
        return ValidationPolicy.for_cameras(cameras)
    return ValidationPolicy.for_profile(normalized, cameras)


@dataclass(frozen=True)
class TopicStats:
    message_count: int
    average_hz: float
    max_gap_ms: float


@dataclass(frozen=True)
class CameraTimingDiagnostics:
    """Capture-vs-delivery evidence written once per camera and episode."""

    capture_timestamps_ns: tuple[int, ...]
    host_receive_timestamps_ns: tuple[int, ...]
    device_timestamps_ns: tuple[int | None, ...]
    frame_numbers: tuple[int | None, ...]
    capture_timestamp_sources: Mapping[str, int]
    device_timestamp_domains: Mapping[str, int]
    synthetic_indices: tuple[int, ...] = ()
    repair_policy: Mapping[str, Any] = field(default_factory=dict)
    repair_events: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class McapInspection:
    log_times: Mapping[str, tuple[int, ...]]
    publish_times: Mapping[str, tuple[int, ...]]
    header_stamps: Mapping[str, tuple[int, ...]]
    resolutions: Mapping[str, tuple[tuple[int, int], ...]]
    camera_timing_diagnostics: Mapping[str, CameraTimingDiagnostics]

    @property
    def stats(self) -> Mapping[str, TopicStats]:
        return {topic: _topic_stats(timestamps) for topic, timestamps in self.log_times.items()}


def _topic_stats(timestamps: Iterable[int]) -> TopicStats:
    ordered = sorted(int(value) for value in timestamps)
    if len(ordered) < 2:
        return TopicStats(message_count=len(ordered), average_hz=0.0, max_gap_ms=0.0)
    duration_s = (ordered[-1] - ordered[0]) / 1_000_000_000
    gaps = [ordered[index + 1] - ordered[index] for index in range(len(ordered) - 1)]
    return TopicStats(
        message_count=len(ordered),
        average_hz=(len(ordered) - 1) / duration_s if duration_s > 0 else 0.0,
        max_gap_ms=max(gaps) / 1_000_000,
    )


def _header_stamp(decoded: Any) -> int | None:
    header = getattr(decoded, "header", None)
    stamp = getattr(header, "stamp", None)
    seconds = getattr(stamp, "sec", None)
    nanoseconds = getattr(stamp, "nanosec", None)
    if seconds is None or nanoseconds is None:
        return None
    return int(seconds) * 1_000_000_000 + int(nanoseconds)


def inspect_mcap(path: str | Path) -> McapInspection:
    """Decode the minimum MCAP information used by the Piper standalone checker."""
    log_times: dict[str, list[int]] = defaultdict(list)
    publish_times: dict[str, list[int]] = defaultdict(list)
    header_stamps: dict[str, list[int]] = defaultdict(list)
    resolutions: dict[str, set[tuple[int, int]]] = defaultdict(set)
    camera_timing_diagnostics: dict[str, CameraTimingDiagnostics] = {}
    with Path(path).open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for _schema, channel, message, decoded in reader.iter_decoded_messages():
            topic = channel.topic
            log_times[topic].append(int(message.log_time))
            publish_times[topic].append(int(message.publish_time))
            stamp = _header_stamp(decoded)
            if stamp is not None:
                header_stamps[topic].append(stamp)
            if topic.endswith("camera_info"):
                width = getattr(decoded, "width", None)
                height = getattr(decoded, "height", None)
                if width is not None and height is not None:
                    resolutions[topic].add((int(width), int(height)))
            if topic == FRAME_TIMING_DIAGNOSTICS_TOPIC:
                raw_data = getattr(decoded, "data", None)
                try:
                    payload = json.loads(raw_data) if isinstance(raw_data, str) else {}
                    role = str(payload["role"])
                    capture_timestamps = tuple(int(value) for value in payload["capture_timestamp_ns"])
                    host_timestamps = tuple(int(value) for value in payload["host_receive_timestamp_ns"])
                    device_timestamps = tuple(
                        None if value is None else int(value) for value in payload["device_timestamp_ns"]
                    )
                    frame_numbers = tuple(
                        None if value is None else int(value) for value in payload["device_frame_number"]
                    )
                    sources = {str(key): int(value) for key, value in payload["capture_timestamp_sources"].items()}
                    domains = {str(key): int(value) for key, value in payload["device_timestamp_domains"].items()}
                    synthetic_indices = tuple(int(value) for value in payload.get("synthetic_indices", ()))
                    repair_policy_value = payload.get("repair_policy", {})
                    repair_events_value = payload.get("repair_events", [])
                    if not isinstance(repair_policy_value, Mapping) or not isinstance(repair_events_value, list):
                        raise TypeError("invalid frame-repair diagnostics")
                    repair_policy = {str(key): value for key, value in repair_policy_value.items()}
                    repair_events = tuple(
                        {str(key): value for key, value in event.items()}
                        for event in repair_events_value
                        if isinstance(event, Mapping)
                    )
                    if len(repair_events) != len(repair_events_value):
                        raise TypeError("invalid frame-repair event")
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
                camera_timing_diagnostics[role] = CameraTimingDiagnostics(
                    capture_timestamps_ns=capture_timestamps,
                    host_receive_timestamps_ns=host_timestamps,
                    device_timestamps_ns=device_timestamps,
                    frame_numbers=frame_numbers,
                    capture_timestamp_sources=sources,
                    device_timestamp_domains=domains,
                    synthetic_indices=synthetic_indices,
                    repair_policy=repair_policy,
                    repair_events=repair_events,
                )
    return McapInspection(
        log_times={topic: tuple(values) for topic, values in log_times.items()},
        publish_times={topic: tuple(values) for topic, values in publish_times.items()},
        header_stamps={topic: tuple(values) for topic, values in header_stamps.items()},
        resolutions={topic: tuple(sorted(values)) for topic, values in resolutions.items()},
        camera_timing_diagnostics=camera_timing_diagnostics,
    )


def _frame_counter_summary(frame_numbers: Iterable[int | None]) -> dict[str, int]:
    skipped = 0
    regressions = 0
    unavailable = 0
    previous: int | None = None
    for frame_number in frame_numbers:
        if frame_number is None:
            unavailable += 1
            continue
        if previous is not None:
            delta = frame_number - previous
            if delta > 1:
                skipped += delta - 1
            elif delta <= 0:
                regressions += 1
        previous = frame_number
    return {"skipped_frames": skipped, "regressions": regressions, "unavailable": unavailable}


def _camera_timing_diagnostics(
    inspection: McapInspection,
    camera_topics: list[str],
    policy: ValidationPolicy,
) -> dict[str, dict[str, Any]]:
    """Explain host jitter and source continuity without changing Piper's six checks."""
    delivery: dict[str, Any] = {"passed": True, "per_topic": {}, "threshold_ms": policy.camera_max_gap_ms}
    continuity: dict[str, Any] = {"passed": True, "per_topic": {}}
    repairs: dict[str, Any] = {"passed": True, "per_topic": {}, "total_synthetic_frames": 0}
    role_by_topic = {topic: role for role, topic in CAMERA_TOPICS.items()}
    for topic in camera_topics:
        role = role_by_topic[topic]
        diagnostics = inspection.camera_timing_diagnostics.get(role)
        if diagnostics is None:
            delivery["per_topic"][topic] = {"passed": True, "skipped": True, "reason": "legacy MCAP"}
            continuity["per_topic"][topic] = {"passed": True, "skipped": True, "reason": "legacy MCAP"}
            repairs["per_topic"][topic] = {"passed": True, "skipped": True, "reason": "legacy MCAP"}
            continue
        synthetic_set = set(diagnostics.synthetic_indices)
        synthetic_indices_valid = (
            len(synthetic_set) == len(diagnostics.synthetic_indices)
            and all(0 <= index < len(diagnostics.capture_timestamps_ns) for index in synthetic_set)
        )
        real_indices = [
            index for index in range(len(diagnostics.capture_timestamps_ns)) if index not in synthetic_set
        ]
        output_capture_stats = _topic_stats(diagnostics.capture_timestamps_ns)
        capture_stats = _topic_stats(diagnostics.capture_timestamps_ns[index] for index in real_indices)
        host_stats = _topic_stats(diagnostics.host_receive_timestamps_ns[index] for index in real_indices)
        mcap_capture_timestamps = inspection.log_times.get(topic, ())
        lengths = {
            len(diagnostics.capture_timestamps_ns),
            len(diagnostics.host_receive_timestamps_ns),
            len(diagnostics.device_timestamps_ns),
            len(diagnostics.frame_numbers),
        }
        complete = (
            len(lengths) == 1
            and diagnostics.capture_timestamps_ns == mcap_capture_timestamps
            and synthetic_indices_valid
        )
        host_only_jitter = host_stats.max_gap_ms > policy.camera_max_gap_ms >= capture_stats.max_gap_ms
        delivery_entry_passed = complete and host_stats.max_gap_ms <= policy.camera_max_gap_ms
        delivery["per_topic"][topic] = {
            "passed": delivery_entry_passed,
            "diagnostic_complete": complete,
            "capture_max_gap_ms": round(capture_stats.max_gap_ms, 2),
            "output_capture_max_gap_ms": round(output_capture_stats.max_gap_ms, 2),
            "host_receive_max_gap_ms": round(host_stats.max_gap_ms, 2),
            "host_only_jitter": host_only_jitter,
            "synthetic_frames": len(synthetic_set),
            "capture_timestamp_sources": dict(diagnostics.capture_timestamp_sources),
            "device_timestamp_domains": dict(diagnostics.device_timestamp_domains),
        }
        delivery["passed"] = bool(delivery["passed"] and delivery_entry_passed)
        counter = _frame_counter_summary(diagnostics.frame_numbers)
        continuity_entry_passed = complete and counter["skipped_frames"] == 0 and counter["regressions"] == 0
        continuity["per_topic"][topic] = {
            "passed": continuity_entry_passed,
            "diagnostic_complete": complete,
            **counter,
            "synthetic_frames": len(synthetic_set),
        }
        continuity["passed"] = bool(continuity["passed"] and continuity_entry_passed)
        repair_count = len(synthetic_set)
        repairs["per_topic"][topic] = {
            "passed": repair_count == 0,
            "diagnostic_complete": complete,
            "synthetic_frames": repair_count,
            "repair_policy": dict(diagnostics.repair_policy),
            "repair_events": [dict(event) for event in diagnostics.repair_events],
        }
        repairs["total_synthetic_frames"] += repair_count
        repairs["passed"] = bool(repairs["passed"] and repair_count == 0)
    return {
        "camera_host_delivery_within_gap_budget": delivery,
        "camera_source_frame_continuity": continuity,
        "camera_contains_offline_repaired_frames": repairs,
    }


def _nearest(timestamp: int, ordered: list[int]) -> int | None:
    if not ordered:
        return None
    index = bisect.bisect_left(ordered, timestamp)
    candidates = []
    if index < len(ordered):
        candidates.append(ordered[index])
    if index:
        candidates.append(ordered[index - 1])
    return min(candidates, key=lambda item: abs(item - timestamp))


def _mutual_pair_offsets(first: Iterable[int], second: Iterable[int]) -> tuple[float, int, int]:
    """Use the same mutual-nearest-neighbour pairing rule as the Piper checker."""
    a = sorted(first)
    b = sorted(second)
    if not a or not b:
        return 0.0, 0, len(a) + len(b)
    nearest_b = {value: _nearest(value, b) for value in a}
    nearest_a = {value: _nearest(value, a) for value in b}
    offsets = [
        abs(value - closest) / 1_000_000
        for value, closest in nearest_b.items()
        if closest is not None and nearest_a.get(closest) == value
    ]
    pairs = len(offsets)
    return (max(offsets) if offsets else 0.0), pairs, len(a) + len(b) - 2 * pairs


def _rolling_frequency(timestamps: Iterable[int], window_s: float) -> tuple[list[float], float]:
    ordered = sorted(timestamps)
    if len(ordered) < 2:
        return [], 0.0
    window_ns = int(window_s * 1_000_000_000)
    start = 0
    frequencies: list[float] = []
    for end in range(1, len(ordered)):
        while start < end and ordered[end] - ordered[start] > window_ns:
            start += 1
        elapsed_s = (ordered[end] - ordered[start]) / 1_000_000_000
        if elapsed_s >= 0.5:
            frequencies.append((end - start) / elapsed_s)
    return frequencies, window_s


def _check_camera_rate(
    stats: Mapping[str, TopicStats], camera_topics: list[str], policy: ValidationPolicy
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "passed": True,
        "expected_fps": policy.expected_camera_fps if not policy.auto_detect_camera_fps else "auto (15/30/60)",
        "expected_fps_by_topic": {},
        "tolerance": policy.camera_average_tolerance,
        "minimum_fps": policy.camera_min_average_fps,
        "maximum_fps": policy.camera_max_average_fps,
        "per_topic": {},
    }
    for topic in camera_topics:
        average = stats.get(topic, TopicStats(0, 0.0, 0.0)).average_hz
        target_fps = policy.target_fps_for(topic, average)
        deviation = abs(average - target_fps) if target_fps is not None else float("inf")
        minimum_fps, maximum_fps = (
            policy.camera_rate_bounds_for(target_fps) if target_fps is not None else (None, None)
        )
        passed = target_fps is not None and policy.camera_rate_passes(average, target_fps)
        result["per_topic"][topic] = {
            "avg_fps": round(average, 2),
            "target_fps": target_fps,
            "deviation": round(deviation, 2) if target_fps is not None else None,
            "minimum_fps": minimum_fps,
            "maximum_fps": maximum_fps,
            "passed": passed,
        }
        result["expected_fps_by_topic"][topic] = target_fps
        result["passed"] = bool(result["passed"] and passed)
    if not camera_topics:
        result.update(passed=False, error="No required camera topics found")
    return result


def _check_camera_gaps(
    stats: Mapping[str, TopicStats], camera_topics: list[str], policy: ValidationPolicy
) -> dict[str, Any]:
    result: dict[str, Any] = {"passed": True, "threshold_ms": policy.camera_max_gap_ms, "per_topic": {}}
    for topic in camera_topics:
        topic_stats = stats.get(topic, TopicStats(0, 0.0, 0.0))
        passed = topic_stats.max_gap_ms <= policy.camera_max_gap_ms
        result["per_topic"][topic] = {
            "max_gap_ms": round(topic_stats.max_gap_ms, 2),
            "message_count": topic_stats.message_count,
            "passed": passed,
        }
        result["passed"] = bool(result["passed"] and passed)
    if not camera_topics:
        result.update(passed=False, error="No required camera topics found")
    return result


def _resolution_entry(resolutions: tuple[tuple[int, int], ...]) -> dict[str, Any]:
    """Make resolution reports readable while retaining all observed CameraInfo profiles."""
    first_width, first_height = resolutions[0] if resolutions else (0, 0)
    return {
        "width": first_width,
        "height": first_height,
        "observed_resolutions": [list(resolution) for resolution in resolutions],
    }


def _check_resolution(
    resolutions: Mapping[str, tuple[tuple[int, int], ...]], camera_topics: list[str], policy: ValidationPolicy
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "passed": True,
        "threshold": f"{policy.camera_min_width}x{policy.camera_min_height}",
        "per_topic": {},
    }
    for topic in camera_topics:
        observed = resolutions.get(topic, ())
        passed = bool(observed) and all(
            width >= policy.camera_min_width and height >= policy.camera_min_height for width, height in observed
        )
        result["per_topic"][topic] = {**_resolution_entry(observed), "passed": passed}
        result["passed"] = bool(result["passed"] and passed)
    if not camera_topics:
        result.update(passed=False, error="No required CameraInfo topics found")
    return result


def _check_configured_resolution(
    resolutions: Mapping[str, tuple[tuple[int, int], ...]], camera_topics: list[str], policy: ValidationPolicy
) -> dict[str, Any]:
    """Require every CameraInfo profile to match the explicit capture configuration.

    Piper accepts any resolution at or above 640x480. New YAM episodes also
    need the configured profile so a camera mode change cannot pass unnoticed.
    Legacy callers without configured profiles retain the Piper-only check.
    """
    configured_topics = [topic for topic in camera_topics if policy.target_resolution_for(topic) is not None]
    result: dict[str, Any] = {"passed": True, "per_topic": {}}
    if not configured_topics:
        result.update(skipped=True, reason="No exact camera profiles configured")
        return result
    for topic in camera_topics:
        expected = policy.target_resolution_for(topic)
        observed = resolutions.get(topic, ())
        if expected is None:
            entry = {**_resolution_entry(observed), "skipped": True, "passed": True}
        else:
            passed = bool(observed) and all(resolution == expected for resolution in observed)
            entry = {
                **_resolution_entry(observed),
                "expected_width": expected[0],
                "expected_height": expected[1],
                "passed": passed,
            }
            result["passed"] = bool(result["passed"] and passed)
        result["per_topic"][topic] = entry
    return result


def _check_camera_sync(
    header_stamps: Mapping[str, tuple[int, ...]], camera_topics: list[str], policy: ValidationPolicy
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "passed": True,
        "threshold_ms": policy.camera_sync_ms,
        "max_pair_offset_ms": 0.0,
        "total_valid_pairs": 0,
        "total_orphans": 0,
        "all_cameras_have_timestamps": True,
    }
    available = [topic for topic in camera_topics if header_stamps.get(topic)]
    if len(available) != len(camera_topics):
        result["all_cameras_have_timestamps"] = False
        result["passed"] = False
    for first_index, first in enumerate(available):
        for second in available[first_index + 1 :]:
            offset, pairs, orphans = _mutual_pair_offsets(header_stamps[first], header_stamps[second])
            result["max_pair_offset_ms"] = max(float(result["max_pair_offset_ms"]), offset)
            result["total_valid_pairs"] += pairs
            result["total_orphans"] += orphans
    if not available:
        result.update(passed=False, error="No required camera timestamps found")
    if float(result["max_pair_offset_ms"]) > policy.camera_sync_ms:
        result["passed"] = False
    result["max_pair_offset_ms"] = round(float(result["max_pair_offset_ms"]), 2)
    return result


def _check_joint_camera_sync(
    header_stamps: Mapping[str, tuple[int, ...]],
    camera_topics: list[str],
    joint_topics: list[str],
    policy: ValidationPolicy,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "passed": True,
        "threshold_ms": policy.joint_camera_sync_ms,
        "max_offset_ms": 0.0,
        "uncovered_camera_frames": 0,
        "total_camera_frames": 0,
        "method": "nearest_joint_per_camera",
    }
    all_joint_stamps = sorted({stamp for topic in joint_topics for stamp in header_stamps.get(topic, ())})
    if not all_joint_stamps:
        return {**result, "passed": False, "error": "No required joint timestamps found"}
    for topic in camera_topics:
        camera_stamps = header_stamps.get(topic, ())
        if not camera_stamps:
            return {**result, "passed": False, "error": f"No timestamps for {topic}"}
        for camera_stamp in camera_stamps:
            result["total_camera_frames"] += 1
            closest = _nearest(camera_stamp, all_joint_stamps)
            if closest is None:
                result["uncovered_camera_frames"] += 1
                continue
            offset = abs(camera_stamp - closest) / 1_000_000
            result["max_offset_ms"] = max(float(result["max_offset_ms"]), offset)
            if offset > policy.joint_camera_sync_ms:
                result["uncovered_camera_frames"] += 1
    if int(result["uncovered_camera_frames"]) > 0:
        result["passed"] = False
    result["max_offset_ms"] = round(float(result["max_offset_ms"]), 2)
    return result


def _check_joint_frequency(
    stats: Mapping[str, TopicStats], joint_topics: list[str], policy: ValidationPolicy
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "passed": True,
        "threshold": policy.joint_min_frequency,
        "per_topic": {},
        "min_avg_freq": 0.0,
    }
    averages: list[float] = []
    for topic in joint_topics:
        average = stats.get(topic, TopicStats(0, 0.0, 0.0)).average_hz
        passed = average > policy.joint_min_frequency
        result["per_topic"][topic] = {"avg_freq": round(average, 2), "passed": passed}
        averages.append(average)
        result["passed"] = bool(result["passed"] and passed)
    if not joint_topics:
        result.update(passed=False, error="No required joint topics found")
    result["min_avg_freq"] = round(min(averages), 2) if averages else 0.0
    return result


def _informational_rolling(
    header_stamps: Mapping[str, tuple[int, ...]],
    log_times: Mapping[str, tuple[int, ...]],
    camera_topics: list[str],
    joint_topics: list[str],
    policy: ValidationPolicy,
) -> dict[str, Any]:
    camera_result: dict[str, Any] = {"passed": True, "per_topic": {}, "tolerance": policy.camera_average_tolerance}
    for topic in camera_topics:
        timestamps = header_stamps.get(topic, ()) or log_times.get(topic, ())
        frequencies, _ = _rolling_frequency(timestamps, policy.rolling_window_s)
        target_fps = policy.target_fps_for(topic, _topic_stats(timestamps).average_hz)
        if target_fps is None:
            camera_result["per_topic"][topic] = {
                "passed": False,
                "target_fps": None,
                "violations": 0,
                "total_windows": len(frequencies),
                "max_deviation": None,
                "error": "average FPS does not match a supported Piper rate",
            }
            camera_result["passed"] = False
            continue
        deviations = [abs(frequency - target_fps) for frequency in frequencies]
        minimum_fps, maximum_fps = policy.camera_rate_bounds_for(target_fps)
        violations = sum(not policy.camera_rate_passes(frequency, target_fps) for frequency in frequencies)
        passed = violations == 0
        camera_result["per_topic"][topic] = {
            "passed": passed,
            "target_fps": target_fps,
            "violations": violations,
            "total_windows": len(frequencies),
            "max_deviation": round(max(deviations), 2) if deviations else 0.0,
            "minimum_fps": minimum_fps,
            "maximum_fps": maximum_fps,
        }
        camera_result["passed"] = bool(camera_result["passed"] and passed)
    joint_result: dict[str, Any] = {"passed": True, "per_topic": {}, "threshold": policy.joint_min_frequency}
    for topic in joint_topics:
        frequencies, _ = _rolling_frequency(
            header_stamps.get(topic, ()) or log_times.get(topic, ()), policy.rolling_window_s
        )
        violations = sum(frequency <= policy.joint_min_frequency for frequency in frequencies)
        passed = violations == 0
        joint_result["per_topic"][topic] = {
            "passed": passed,
            "violations": violations,
            "total_windows": len(frequencies),
            "min_rolling_freq": round(min(frequencies), 2) if frequencies else 0.0,
        }
        joint_result["passed"] = bool(joint_result["passed"] and passed)
    return {
        "camera_rolling_1s_within_0.5_of_target": camera_result,
        "joint_rolling_1s_above_45hz": joint_result,
    }


def _threshold_label(value: float) -> str:
    return f"{value:g}".replace(".", "_")


def validate_mcap(path: str | Path, policy: ValidationPolicy | None = None) -> dict[str, Any]:
    """Run the selected quality standard against one episode MCAP."""
    selected_policy = policy or ValidationPolicy()
    inspection = inspect_mcap(path)
    stats = inspection.stats
    camera_topics = list(CAMERA_TOPICS.values())
    joint_topics = list(ROBOT_TOPICS.values())
    missing_cameras = [topic for topic in camera_topics if topic not in stats]
    missing_joints = [topic for topic in joint_topics if topic not in stats]
    mandatory = {
        "camera_avg_fps_within_0.5_of_target": _check_camera_rate(stats, camera_topics, selected_policy),
        f"camera_max_gap_below_{_threshold_label(selected_policy.camera_max_gap_ms)}ms": _check_camera_gaps(
            stats, camera_topics, selected_policy
        ),
        "camera_resolution_min_640x480": _check_resolution(inspection.resolutions, camera_topics, selected_policy),
        f"camera_sync_below_{_threshold_label(selected_policy.camera_sync_ms)}ms": _check_camera_sync(
            inspection.header_stamps, camera_topics, selected_policy
        ),
        f"joint_camera_sync_below_{_threshold_label(selected_policy.joint_camera_sync_ms)}ms": _check_joint_camera_sync(
            inspection.header_stamps, camera_topics, joint_topics, selected_policy
        ),
        "joint_avg_freq_above_45hz": _check_joint_frequency(stats, joint_topics, selected_policy),
    }
    if selected_policy.require_exact_camera_resolution:
        mandatory["camera_resolution_matches_config"] = _check_configured_resolution(
            inspection.resolutions, camera_topics, selected_policy
        )
    failed_checks = [name for name, check in mandatory.items() if not check["passed"]]
    informational = _informational_rolling(
        inspection.header_stamps, inspection.log_times, camera_topics, joint_topics, selected_policy
    )
    informational.update(_camera_timing_diagnostics(inspection, camera_topics, selected_policy))
    informational_failures = [name for name, check in informational.items() if not check["passed"]]
    return {
        "output_mcap_passed": not failed_checks,
        "failed_checks": failed_checks,
        "informational_failures": informational_failures,
        "passed_checks": len(mandatory) - len(failed_checks),
        "total_checks": len(mandatory),
        "mandatory_checks": mandatory,
        "informational_checks": informational,
        "required_topics": {"missing_camera_topics": missing_cameras, "missing_joint_topics": missing_joints},
        "policy": asdict(selected_policy),
    }


def _parse_frame_rate(value: str) -> float:
    numerator, denominator = value.split("/", maxsplit=1)
    parsed_denominator = int(denominator)
    if parsed_denominator == 0:
        raise ValueError(f"invalid frame rate {value!r}")
    return int(numerator) / parsed_denominator


def validate_video_assets(
    episode_mcap: str | Path,
    policy: ValidationPolicy | None = None,
    *,
    video_dir: str | Path | None = None,
    expected_video_codecs: tuple[str, ...] | None = None,
    require_fps_match: bool = True,
) -> dict[str, Any]:
    """Verify external MP4 assets decode consistently with CameraInfo.

    The default preserves the public dataset contract: final HEVC files live
    below ``videos/<episode>/`` and use the codecs declared by ``policy``.
    The optional directory and codec override let the recorder verify a live
    intermediate before it is eligible for offline HEVC transcoding. Set
    ``require_fps_match`` to false only for the internal publication path:
    formal timing acceptance remains visible in ``validate_mcap``, while the
    publication path must independently guarantee that no camera frame was
    added, removed, or substituted during encoding. Camera timestamps and
    frame identity remain sourced from the MCAP in both cases.
    """
    selected_policy = policy or ValidationPolicy()
    mcap_path = Path(episode_mcap)
    if video_dir is not None:
        resolved_video_dir = Path(video_dir)
    elif mcap_path.parent.name.isdigit():
        # Current layout: <dataset>/<episode>/<episode>.mcap and videos/.
        resolved_video_dir = mcap_path.parent / "videos"
    else:
        # Legacy layout: <dataset>/<episode>.mcap and videos/<episode>/.
        resolved_video_dir = mcap_path.parent / "videos" / mcap_path.stem
    source_timeline_manifest = resolved_video_dir / "source_timeline.json"
    source_timeline_mode = source_timeline_manifest.is_file()
    effective_require_fps_match = bool(require_fps_match and not source_timeline_mode)
    expected_codecs = expected_video_codecs or selected_policy.expected_video_codecs
    if not expected_codecs:
        raise ValueError("expected_video_codecs must contain at least one codec")
    inspection = inspect_mcap(mcap_path)
    camera_message_counts = {
        topic: stats.message_count for topic, stats in inspection.stats.items() if topic in CAMERA_TOPICS.values()
    }
    result: dict[str, Any] = {
        "passed": True,
        "video_dir": str(resolved_video_dir),
        "expected_video_codecs": list(expected_codecs),
        "require_fps_match": effective_require_fps_match,
        "source_timeline": source_timeline_mode,
        "per_camera": {},
    }
    for role in CAMERA_TOPICS:
        path = resolved_video_dir / f"{role}.mp4"
        entry: dict[str, Any] = {"path": str(path), "passed": False}
        if not path.is_file():
            entry["error"] = "missing video"
            result["per_camera"][role] = entry
            result["passed"] = False
            continue
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
            entry["error"] = "ffprobe is not installed or not on PATH"
            result["per_camera"][role] = entry
            result["passed"] = False
            continue
        if completed.returncode != 0:
            entry["error"] = completed.stderr.strip() or "ffprobe failed"
            result["per_camera"][role] = entry
            result["passed"] = False
            continue
        try:
            streams = json.loads(completed.stdout).get("streams", [])
        except json.JSONDecodeError:
            streams = []
        if not streams:
            entry["error"] = "no video stream"
            result["per_camera"][role] = entry
            result["passed"] = False
            continue
        stream = streams[0]
        try:
            fps = _parse_frame_rate(stream["avg_frame_rate"])
            width, height = int(stream["width"]), int(stream["height"])
            raw_frame_count = stream.get("nb_read_frames") or stream.get("nb_frames") or 0
            frame_count = int(raw_frame_count)
        except (KeyError, TypeError, ValueError):
            entry["error"] = "ffprobe returned incomplete stream metadata"
            result["per_camera"][role] = entry
            result["passed"] = False
            continue
        expected_frame_count = camera_message_counts.get(CAMERA_TOPICS[role], 0)
        codec = stream.get("codec_name")
        codec_matches = codec in expected_codecs
        frame_count_matches = frame_count == expected_frame_count
        target_fps = selected_policy.target_fps_for(
            CAMERA_TOPICS[role], inspection.stats.get(CAMERA_TOPICS[role], TopicStats(0, 0.0, 0.0)).average_hz
        )
        expected_resolution = selected_policy.target_resolution_for(CAMERA_TOPICS[role])
        resolution_matches = (
            (width >= selected_policy.camera_min_width and height >= selected_policy.camera_min_height)
            if expected_resolution is None
            else (width, height) == expected_resolution
        )
        fps_matches = target_fps is not None and selected_policy.camera_rate_passes(fps, target_fps)
        passed = (
            (fps_matches or not effective_require_fps_match)
            and resolution_matches
            and codec_matches
            and frame_count_matches
        )
        entry.update(
            passed=passed,
            codec=codec,
            codec_matches=codec_matches,
            expected_codecs=list(expected_codecs),
            width=width,
            height=height,
            expected_width=expected_resolution[0] if expected_resolution is not None else None,
            expected_height=expected_resolution[1] if expected_resolution is not None else None,
            resolution_matches=resolution_matches,
            fps=round(fps, 3),
            target_fps=target_fps,
            fps_matches=fps_matches,
            fps_check_skipped=source_timeline_mode and require_fps_match,
            frame_count=frame_count,
            expected_camera_info_count=expected_frame_count,
            frame_count_matches=frame_count_matches,
        )
        result["per_camera"][role] = entry
        result["passed"] = bool(result["passed"] and passed)
    return result


def print_validation(result: Mapping[str, Any]) -> None:
    """Print a compact operator-facing report without hiding failed check names."""
    passed_checks = int(result["passed_checks"])
    total_checks = int(result["total_checks"])
    all_checks_passed = passed_checks == total_checks
    status = "PASS" if all_checks_passed else "FAIL"
    profile_name = result.get("policy", {}).get("profile_name", CUSTOM_PROFILE)
    print(
        colorize(
            f"MCAP validation [{profile_name}]: {status} "
            f"({passed_checks}/{total_checks} mandatory checks)",
            "green" if all_checks_passed else "red",
        )
    )
    if result["failed_checks"]:
        print("Failed: " + ", ".join(result["failed_checks"]))
    if result["informational_failures"]:
        print("Warnings: " + ", ".join(result["informational_failures"]))
