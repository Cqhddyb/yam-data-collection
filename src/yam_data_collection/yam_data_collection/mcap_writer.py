"""ROS2 CDR MCAP writer matching the Piper episode topic contract."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Mapping

from mcap_ros2.writer import Writer as McapWriter

from .config import CameraConfig
from .models import CameraTimingSample, RobotSnapshot

JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper")

ROBOT_TOPICS = {
    "left_leader": "/robot/arm_left_leader/joint_states",
    "right_leader": "/robot/arm_right_leader/joint_states",
    "left_follower": "/robot/arm_left_follow/joint_states",
    "right_follower": "/robot/arm_right_follow/joint_states",
}

CAMERA_TOPICS = {
    "cam_high": "/robot/camera/cam_high/image_raw/compressed/camera_info",
    "cam_left": "/robot/camera/cam_left/image_raw/compressed/camera_info",
    "cam_right": "/robot/camera/cam_right/image_raw/compressed/camera_info",
}

# Deliberately contains none of the Piper checker's camera-topic keywords.
# Consumers that only know the original Piper contract therefore ignore it.
FRAME_TIMING_DIAGNOSTICS_TOPIC = "/diagnostics/frame_timing"

_JOINT_STATE_SCHEMA = """std_msgs/Header header
string[] name
float64[] position
float64[] velocity
float64[] effort
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
"""

_CAMERA_INFO_SCHEMA = """std_msgs/Header header
uint32 height
uint32 width
string distortion_model
float64[] d
float64[9] k
float64[9] r
float64[12] p
uint32 binning_x
uint32 binning_y
sensor_msgs/RegionOfInterest roi
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
================================================================================
MSG: sensor_msgs/RegionOfInterest
uint32 x_offset
uint32 y_offset
uint32 height
uint32 width
bool do_rectify
"""

_STRING_SCHEMA = """string data
"""


def _stamp(timestamp_ns: int) -> dict[str, int]:
    seconds, nanoseconds = divmod(int(timestamp_ns), 1_000_000_000)
    return {"sec": seconds, "nanosec": nanoseconds}


def camera_info_message(camera: CameraConfig, timestamp_ns: int) -> dict[str, Any]:
    """Build the CameraInfo payload used by live and derived episode writers."""
    intrinsics = camera.intrinsics
    return {
        "header": {"stamp": _stamp(timestamp_ns), "frame_id": camera.frame_id},
        "height": camera.height,
        "width": camera.width,
        "distortion_model": intrinsics.distortion_model,
        "d": list(intrinsics.d),
        "k": list(intrinsics.k),
        "r": list(intrinsics.r),
        "p": list(intrinsics.p),
        "binning_x": 0,
        "binning_y": 0,
        "roi": {"x_offset": 0, "y_offset": 0, "height": 0, "width": 0, "do_rectify": False},
    }


class EpisodeMcapWriter:
    """Write one dual-arm episode without requiring a ROS runtime."""

    def __init__(self, path: Path, task_info: Mapping[str, Any]) -> None:
        self.path = path
        self._stream = path.open("xb")
        self._writer = McapWriter(self._stream)
        self._joint_schema = self._writer.register_msgdef("sensor_msgs/msg/JointState", _JOINT_STATE_SCHEMA)
        self._camera_schema = self._writer.register_msgdef("sensor_msgs/msg/CameraInfo", _CAMERA_INFO_SCHEMA)
        self._string_schema = self._writer.register_msgdef("std_msgs/msg/String", _STRING_SCHEMA)
        self._lock = threading.Lock()
        self._sequences: dict[str, int] = {}
        self._closed = False
        self.write_task_info(task_info)

    def _next_sequence(self, topic: str) -> int:
        sequence = self._sequences.get(topic, 0)
        self._sequences[topic] = sequence + 1
        return sequence

    def _write(
        self,
        topic: str,
        schema: Any,
        message: Mapping[str, Any],
        log_time_ns: int,
        publish_time_ns: int | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("MCAP writer is already closed")
        publish_time_ns = log_time_ns if publish_time_ns is None else publish_time_ns
        self._writer.write_message(
            topic=topic,
            schema=schema,
            message=message,
            log_time=int(log_time_ns),
            publish_time=int(publish_time_ns),
            sequence=self._next_sequence(topic),
        )

    def write_task_info(self, task_info: Mapping[str, Any]) -> None:
        import time

        with self._lock:
            self._write(
                "/task_info", self._string_schema, {"data": json.dumps(task_info, ensure_ascii=False)}, time.time_ns()
            )

    def write_joint_state(
        self,
        snapshot: RobotSnapshot,
        log_time_ns: int,
        *,
        header_timestamp_ns: int | None = None,
    ) -> None:
        """Write one state sample with separate recorder and source timestamps.

        ``log_time_ns`` describes the recorder's fixed-rate sampling schedule.
        ``header_timestamp_ns`` is the time when i2rt read the actual hardware
        state, allowing the Piper-compatible joint-camera check to detect stale
        telemetry instead of treating delayed RPC replies as fresh measurements.
        Legacy callers that omit it retain the original single-timestamp form.
        """
        if snapshot.role not in ROBOT_TOPICS:
            raise ValueError(f"unknown robot role {snapshot.role!r}")
        header_timestamp_ns = log_time_ns if header_timestamp_ns is None else header_timestamp_ns
        position = list(snapshot.position)
        if len(position) != len(JOINT_NAMES):
            raise ValueError(
                f"{snapshot.role} reports {len(position)} joints; expected {len(JOINT_NAMES)} ({', '.join(JOINT_NAMES)})"
            )
        velocity = list(snapshot.velocity) if snapshot.role.endswith("follower") else []
        effort = list(snapshot.effort) if snapshot.role.endswith("follower") else []
        if snapshot.role.endswith("follower") and (
            len(velocity) != len(JOINT_NAMES) or len(effort) != len(JOINT_NAMES)
        ):
            raise ValueError(f"{snapshot.role} must provide velocity and effort for all {len(JOINT_NAMES)} joints")
        message = {
            "header": {"stamp": _stamp(header_timestamp_ns), "frame_id": ""},
            "name": list(JOINT_NAMES),
            "position": position,
            # Piper leader streams have empty derivatives; preserve this compatible distinction.
            "velocity": velocity,
            "effort": effort,
        }
        with self._lock:
            self._write(
                ROBOT_TOPICS[snapshot.role],
                self._joint_schema,
                message,
                log_time_ns,
                publish_time_ns=header_timestamp_ns,
            )

    def write_camera_info(self, camera: CameraConfig, timestamp_ns: int) -> None:
        if camera.role not in CAMERA_TOPICS:
            raise ValueError(f"unknown camera role {camera.role!r}")
        message = camera_info_message(camera, timestamp_ns)
        with self._lock:
            self._write(CAMERA_TOPICS[camera.role], self._camera_schema, message, timestamp_ns)

    def write_camera_timing_diagnostics(
        self, role: str, samples: list[CameraTimingSample]
    ) -> None:
        """Persist capture-vs-delivery evidence without adding callback load."""
        if role not in CAMERA_TOPICS:
            raise ValueError(f"unknown camera role {role!r}")
        if not samples:
            return
        if any(sample.role != role for sample in samples):
            raise ValueError(f"camera timing batch for {role} contains another role")
        sources: dict[str, int] = {}
        domains: dict[str, int] = {}
        for sample in samples:
            sources[sample.capture_timestamp_source] = sources.get(sample.capture_timestamp_source, 0) + 1
            domain = sample.device_timestamp_domain or "unavailable"
            domains[domain] = domains.get(domain, 0) + 1
        payload = {
            "version": 1,
            "role": role,
            "capture_timestamp_ns": [sample.capture_timestamp_ns for sample in samples],
            "host_receive_timestamp_ns": [sample.host_receive_timestamp_ns for sample in samples],
            "device_timestamp_ns": [sample.device_timestamp_ns for sample in samples],
            "device_frame_number": [sample.device_frame_number for sample in samples],
            "capture_timestamp_sources": sources,
            "device_timestamp_domains": domains,
        }
        capture_time_ns = samples[-1].capture_timestamp_ns
        host_time_ns = samples[-1].host_receive_timestamp_ns
        with self._lock:
            self._write(
                FRAME_TIMING_DIAGNOSTICS_TOPIC,
                self._string_schema,
                {"data": json.dumps(payload, ensure_ascii=True, separators=(",", ":"))},
                host_time_ns,
                publish_time_ns=capture_time_ns,
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._writer.finish()
            finally:
                self._stream.close()

    def __enter__(self) -> "EpisodeMcapWriter":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
