"""Resolve a physical arm pair and its cameras without modifying the host config."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import yaml

from yam_data_collection.config import (
    CameraConfig,
    CameraIntrinsics,
    CameraTestConfig,
    CameraTestSettings,
    DatasetConfig,
    TelemetryEndpoint,
    _load_collection,
    load_config as load_host_config,
)
from yam_data_collection.validation_profiles import validate_profile_camera_fps


DEFAULT_CONFIG = Path(__file__).with_name("config.host.yaml")
SIDES = ("left", "right")
CAMERA_SETUPS = ("wrist", "wrist-top")
CAPTURE_MODES = ("60hz", "30hz")


def robot_roles(side: str) -> tuple[str, str]:
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}")
    return f"{side}_leader", f"{side}_follower"


def camera_roles(side: str, camera_setup: str) -> tuple[str, ...]:
    robot_roles(side)
    if camera_setup not in CAMERA_SETUPS:
        raise ValueError(f"cameras must be one of {CAMERA_SETUPS}")
    wrist = f"cam_{side}"
    return (wrist, "cam_high") if camera_setup == "wrist-top" else (wrist,)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _path(value: Any, parent: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path")
    configured = Path(value).expanduser()
    return (configured if configured.is_absolute() else parent / configured).resolve()


def _camera(role: str, raw: Mapping[str, Any]) -> CameraConfig:
    serial = str(raw.get("serial", "")).strip()
    width, height, fps = int(raw.get("width", 0)), int(raw.get("height", 0)), int(raw.get("fps", 0))
    stream_format = str(raw.get("stream_format", "bgr8"))
    if not serial or width < 640 or height < 480 or fps <= 0 or stream_format not in {"bgr8", "yuyv"}:
        raise ValueError(f"invalid serial, resolution, FPS or stream format for {role}")
    intrinsics = raw.get("intrinsics")
    return CameraConfig(
        role=role,
        serial=serial,
        width=width,
        height=height,
        fps=fps,
        frame_id=str(raw.get("frame_id", f"{role}_color_optical_frame")),
        intrinsics=CameraIntrinsics.from_mapping(_mapping(intrinsics, f"{role}.intrinsics"))
        if intrinsics is not None
        else CameraIntrinsics.uncalibrated(),
        stream_format=stream_format,
    )


def load_config(
    path: str | Path = DEFAULT_CONFIG,
    *,
    side: str,
    camera_setup: str = "wrist",
    capture_mode: str = "60hz",
) -> DatasetConfig:
    """Load the extension config or an independently usable recorded snapshot."""
    selected_robots = robot_roles(side)
    selected_cameras = camera_roles(side, camera_setup)
    if capture_mode not in CAPTURE_MODES:
        raise ValueError(f"mode must be one of {CAPTURE_MODES}")
    source_path = Path(path).expanduser().resolve()
    settings = _mapping(yaml.safe_load(source_path.read_text(encoding="utf-8")), "configuration")
    selection = {"side": side, "cameras": camera_setup, "format_version": 1}
    if "host_config" in settings:
        host_path = _path(settings["host_config"], source_path.parent, "host_config")
        host = load_host_config(host_path, capture_mode=capture_mode)
        raw = deepcopy(dict(host.raw))
        dataset_root = _path(settings.get("dataset_root"), source_path.parent, "dataset_root")
        if dataset_root.is_relative_to(host.collection.dataset_dir.parent):
            raise ValueError("single-arm dataset_root must not be inside the bimanual dataset")
        suffix = Path(side) / camera_setup / capture_mode
        raw["collection_config"]["dataset_dir"] = str(dataset_root / suffix)
        test_root = _path(settings.get("camera_test_root"), source_path.parent, "camera_test_root")
        raw["camera_test_config"]["output_dir"] = str(test_root / suffix)
        raw["task_info"] = _mapping(settings.get("task_info"), "task_info")
        raw["telemetry"] = {role: raw["telemetry"][role] for role in selected_robots}
        raw["cameras"] = {role: raw["cameras"][role] for role in selected_cameras}
        raw["single_arm"] = {**selection, "host_config": str(host_path)}
        raw["task_info"]["collection_layout"] = {
            **selection,
            "robot_roles": list(selected_robots),
            "camera_roles": list(selected_cameras),
        }
    else:
        raw = deepcopy(settings)
        recorded = _mapping(raw.get("single_arm"), "single_arm")
        if any(recorded.get(key) != value for key, value in selection.items()):
            raise ValueError("recorded single-arm selection does not match --side / --cameras")
        if raw.get("selected_capture_mode") != capture_mode:
            raise ValueError("recorded capture mode does not match --mode")
    telemetry_raw = _mapping(raw.get("telemetry"), "telemetry")
    cameras_raw = _mapping(raw.get("cameras"), "cameras")
    if set(telemetry_raw) != set(selected_robots) or set(cameras_raw) != set(selected_cameras):
        raise ValueError("single-arm configuration must contain exactly the selected robot and camera roles")
    collection = _load_collection(_mapping(raw.get("collection_config"), "collection_config"), source_path.parent)
    if collection.video_codec not in {"libx264", "h264"} or collection.live_video_codec not in {"libx264", "h264"}:
        raise ValueError("single-arm collection requires H.264 final and live video codecs")
    expected_fps = 60 if capture_mode == "60hz" else 30
    if collection.target_fps != expected_fps or collection.validation_profile != f"yam_{capture_mode}":
        raise ValueError("collection FPS and validation profile must match the selected capture mode")
    cameras = {role: _camera(role, _mapping(cameras_raw[role], role)) for role in selected_cameras}
    if len({camera.serial for camera in cameras.values()}) != len(cameras):
        raise ValueError("selected cameras must have distinct serial numbers")
    validate_profile_camera_fps(collection.validation_profile, cameras)
    telemetry: dict[str, TelemetryEndpoint] = {}
    for role in selected_robots:
        endpoint = _mapping(telemetry_raw[role], role)
        port = int(endpoint.get("port", 0))
        host = str(endpoint.get("host", "127.0.0.1")).strip()
        if not host or not 1 <= port <= 65535:
            raise ValueError(f"invalid telemetry host or port for {role}")
        telemetry[role] = TelemetryEndpoint(role=role, host=host, port=port)
    if len({(endpoint.host, endpoint.port) for endpoint in telemetry.values()}) != len(telemetry):
        raise ValueError("leader and follower must use distinct telemetry endpoints")
    raw["collection_config"]["dataset_dir"] = str(collection.dataset_dir)
    return DatasetConfig(
        source_path=source_path,
        raw=raw,
        collection=collection,
        telemetry=telemetry,
        cameras=cameras,
        task_info=_mapping(raw.get("task_info"), "task_info"),
        capture_mode=capture_mode,
    )


def camera_test_config(config: DatasetConfig) -> CameraTestConfig:
    """Use the same selected devices and profile for camera-only qualification."""
    raw = _mapping(config.raw.get("camera_test_config"), "camera_test_config")
    collection = config.collection
    duration_s = float(raw.get("duration_s", 60.0))
    queue_size = int(raw.get("video_queue_size", 240))
    if not math.isfinite(duration_s) or duration_s <= 0 or queue_size <= 0:
        raise ValueError("camera test duration and video queue size must be positive")
    settings = CameraTestSettings(
        output_dir=_path(raw.get("output_dir"), config.source_path.parent, "camera_test_config.output_dir"),
        duration_s=duration_s,
        timeout_threshold_s=collection.timeout_threshold_s,
        video_codec=collection.live_video_codec,
        video_crf=collection.live_video_crf,
        video_preset=collection.live_video_preset,
        video_threads=collection.live_video_threads,
        video_queue_size=queue_size,
        fps_tolerance=0.5,
        max_gap_factor=2.1,
        validation_profile=collection.validation_profile,
    )
    saved = deepcopy(dict(config.raw))
    saved["camera_test_config"] = {**asdict(settings), "output_dir": str(settings.output_dir)}
    return CameraTestConfig(config.source_path, saved, settings, config.cameras, config.capture_mode)
