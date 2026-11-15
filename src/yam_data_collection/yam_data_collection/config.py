"""Configuration loading and validation for portable capture deployments."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping, Optional

import yaml

from .validation_profiles import (
    PIPER_COMPATIBLE_PROFILE,
    YAM_30HZ_PROFILE,
    YAM_60HZ_PROFILE,
    normalize_validation_profile,
    validate_profile_camera_fps,
)

_REQUIRED_ROBOT_ROLES = ("left_leader", "right_leader", "left_follower", "right_follower")
_REQUIRED_CAMERA_ROLES = ("cam_high", "cam_left", "cam_right")
_CAPTURE_MODE_CONTRACTS = {
    "30hz": (30.0, YAM_30HZ_PROFILE),
    "60hz": (60.0, YAM_60HZ_PROFILE),
}
_CAPTURE_MODE_STATE_SAVE_FREQ = 180.0
_VIDEO_PRESETS = frozenset(
    {
        "ultrafast",
        "superfast",
        "veryfast",
        "faster",
        "fast",
        "medium",
        "slow",
        "slower",
        "veryslow",
    }
)
# Final dataset assets are normally HEVC.  H.264 is also supported for an
# explicit direct-publication deployment, where the verified live MP4s become
# the final dataset assets without a second encode.  The default remains HEVC
# so existing configurations retain their two-stage behavior.
_FINAL_VIDEO_CODECS = frozenset({"libx264", "h264", "libx265", "hevc"})
_LIVE_VIDEO_CODECS = frozenset({"libx264", "h264", "libx265", "hevc"})
_CPU_SET_ITEM_RE = re.compile(r"(\d+)\s*(?:-\s*(\d+))?")


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _as_float_list(value: Any, name: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must be a list with {length} values")
    return tuple(float(item) for item in value)


@dataclass(frozen=True)
class CameraIntrinsics:
    """Static ROS CameraInfo values for one installed physical camera."""

    distortion_model: str
    d: tuple[float, ...]
    k: tuple[float, ...]
    r: tuple[float, ...]
    p: tuple[float, ...]

    @classmethod
    def uncalibrated(cls) -> "CameraIntrinsics":
        """Return an explicit zero calibration until the camera is calibrated."""
        return cls(
            distortion_model="plumb_bob",
            d=(),
            k=(0.0,) * 9,
            r=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            p=(0.0,) * 12,
        )

    @property
    def is_calibrated(self) -> bool:
        """Whether the minimum focal-length values needed for CameraInfo are present."""
        return self.k[0] > 0.0 and self.k[4] > 0.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CameraIntrinsics":
        return cls(
            distortion_model=str(value.get("distortion_model", "plumb_bob")),
            d=tuple(float(item) for item in value.get("D", value.get("d", []))),
            k=_as_float_list(value.get("K", value.get("k")), "intrinsics.K", 9),
            r=_as_float_list(value.get("R", value.get("r")), "intrinsics.R", 9),
            p=_as_float_list(value.get("P", value.get("p")), "intrinsics.P", 12),
        )


@dataclass(frozen=True)
class CameraConfig:
    role: str
    serial: str
    width: int
    height: int
    fps: int
    frame_id: str
    intrinsics: CameraIntrinsics
    # Input transport format. The recorder always emits BGR frames to ffmpeg.
    stream_format: str = "bgr8"


@dataclass(frozen=True)
class TelemetryEndpoint:
    role: str
    host: str
    port: int


@dataclass(frozen=True)
class CollectionSettings:
    dataset_dir: Path
    state_save_freq: float
    target_fps: float
    timeout_threshold_s: float
    total_sampled_frames: int
    video_codec: str
    video_crf: int
    video_preset: str
    video_threads: int
    allow_uncalibrated_intrinsics: bool
    validation_profile: str
    # ``video_*`` control final dataset assets. Existing configurations use
    # HEVC through the offline pipeline; an explicit H.264-final configuration
    # promotes the verified live files directly. ``live_video_*`` control the
    # per-camera file written while teleoperation is active.
    live_video_codec: str = "libx264"
    live_video_crf: int = 23
    live_video_preset: str = "ultrafast"
    live_video_threads: int = 1
    live_video_queue_size: int = 120
    # Optional CPU affinity for the capture process itself. When paired with
    # ``live_video_cpu_set`` this reserves disjoint cores for camera/telemetry
    # callbacks and the three live FFmpeg children.
    recorder_cpu_set: str | None = None
    # Optional CPU affinity reserved for the live FFmpeg children.  The
    # collector applies ``recorder_cpu_set`` before it opens input streams.
    live_video_cpu_set: str | None = None


@dataclass(frozen=True)
class DatasetConfig:
    source_path: Path
    raw: Mapping[str, Any]
    collection: CollectionSettings
    telemetry: Mapping[str, TelemetryEndpoint]
    cameras: Mapping[str, CameraConfig]
    task_info: Mapping[str, Any]
    capture_mode: str | None = None


@dataclass(frozen=True)
class CameraTestSettings:
    """Options for a camera-only qualification run, independent of robot telemetry."""

    output_dir: Path
    duration_s: float
    timeout_threshold_s: float
    video_codec: str
    video_crf: int
    video_preset: str
    video_threads: int
    video_queue_size: int
    fps_tolerance: float
    max_gap_factor: float
    validation_profile: str = PIPER_COMPATIBLE_PROFILE


@dataclass(frozen=True)
class CameraTestConfig:
    source_path: Path
    raw: Mapping[str, Any]
    settings: CameraTestSettings
    cameras: Mapping[str, CameraConfig]
    capture_mode: str | None = None


@dataclass(frozen=True)
class _ResolvedCaptureMode:
    """The fixed capture contract selected from a portable configuration."""

    name: str
    target_fps: float
    validation_profile: str
    dataset_subdir: str
    cameras: Mapping[str, Any]
    collection_config: Mapping[str, Any]
    camera_test_config: Mapping[str, Any]


def capture_mode_cli_choices() -> tuple[str, ...]:
    """Return the two formal YAM acquisition modes accepted by the CLI."""
    return tuple(_CAPTURE_MODE_CONTRACTS)


def _normalize_capture_mode(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "")
    aliases = {"30": "30hz", "30hz": "30hz", "60": "60hz", "60hz": "60hz"}
    try:
        return aliases[normalized]
    except KeyError as exc:
        choices = ", ".join(capture_mode_cli_choices())
        raise ValueError(f"unknown capture mode {value!r}; expected one of: {choices}") from exc


def _load_video_codec(
    raw: Mapping[str, Any],
    key: str,
    default: str,
    allowed: frozenset[str],
    section_name: str,
) -> str:
    value = raw.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"{section_name}.{key} must be a string")
    codec = value.strip().lower()
    if codec not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"{section_name}.{key} must be one of: {choices}")
    return codec


def _load_video_integer(
    raw: Mapping[str, Any], key: str, default: int, section_name: str, *, minimum: int, maximum: int | None = None
) -> int:
    value = raw.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{section_name}.{key} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{section_name}.{key} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{section_name}.{key} must be an integer")
    if parsed < minimum or (maximum is not None and parsed > maximum):
        range_label = f"{minimum}..{maximum}" if maximum is not None else f"at least {minimum}"
        raise ValueError(f"{section_name}.{key} must be in {range_label}")
    return parsed


def _load_video_preset(raw: Mapping[str, Any], key: str, default: str, section_name: str) -> str:
    value = raw.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"{section_name}.{key} must be a string")
    preset = value.strip().lower()
    if preset not in _VIDEO_PRESETS:
        raise ValueError(f"{section_name}.{key} must be a supported x264/x265 preset")
    return preset


def normalize_cpu_set(value: str, name: str = "CPU set") -> str:
    """Validate a Linux CPU-list expression and return its compact canonical form.

    The configuration remains portable by checking syntax only.  Whether a
    particular CPU is available to this process is deliberately checked by
    ``taskset`` when the live encoder starts.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string such as 16-19 or 16,18-20")
    parsed_ranges: list[tuple[int, int]] = []
    for item in value.split(","):
        match = _CPU_SET_ITEM_RE.fullmatch(item.strip())
        if match is None:
            raise ValueError(f"{name} has invalid CPU-set item {item!r}; use values such as 16-19 or 16,18-20")
        first = int(match.group(1))
        last = int(match.group(2) or first)
        if last < first:
            raise ValueError(f"{name} has descending CPU range {item!r}")
        parsed_ranges.append((first, last))

    merged_ranges: list[tuple[int, int]] = []
    for start, end in sorted(parsed_ranges):
        if merged_ranges and start <= merged_ranges[-1][1] + 1:
            previous_start, previous_end = merged_ranges[-1]
            merged_ranges[-1] = (previous_start, max(previous_end, end))
        else:
            merged_ranges.append((start, end))
    return ",".join(str(start) if start == end else f"{start}-{end}" for start, end in merged_ranges)


def cpu_set_members(value: str, name: str = "CPU set") -> tuple[int, ...]:
    """Validate a Linux CPU-list expression and return its expanded members."""
    normalized = normalize_cpu_set(value, name)
    members: list[int] = []
    for item in normalized.split(","):
        first_text, separator, last_text = item.partition("-")
        first = int(first_text)
        last = int(last_text) if separator else first
        members.extend(range(first, last + 1))
    return tuple(members)


def _load_optional_cpu_set(raw: Mapping[str, Any], key: str, section_name: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    return normalize_cpu_set(value, f"{section_name}.{key}")


def _load_collection(raw: Mapping[str, Any], config_dir: Path) -> CollectionSettings:
    dataset_value = raw.get("dataset_dir")
    if not isinstance(dataset_value, str) or not dataset_value:
        raise ValueError("collection_config.dataset_dir must be a non-empty path")
    configured_path = Path(dataset_value).expanduser()
    dataset_dir = configured_path if configured_path.is_absolute() else (config_dir / configured_path).resolve()
    state_save_freq = float(raw.get("state_save_freq", 100.0))
    target_fps = float(raw.get("target_fps", raw.get("image_save_freq", 30.0)))
    timeout_threshold_s = float(raw.get("timeout_threshold_s", raw.get("timeout_threshold", 1.0)))
    total_sampled_frames = int(raw.get("total_sampled_frames", 50000))
    # ``video_*`` are intentionally final-output settings. They retain their
    # stable YAML names so older host configs continue to describe HEVC output.
    video_codec = _load_video_codec(
        raw, "video_codec", "libx265", _FINAL_VIDEO_CODECS, "collection_config"
    )
    video_crf = _load_video_integer(raw, "video_crf", 23, "collection_config", minimum=0, maximum=51)
    video_preset = _load_video_preset(raw, "video_preset", "medium", "collection_config")
    video_threads = _load_video_integer(raw, "video_threads", 0, "collection_config", minimum=0)
    live_video_codec = _load_video_codec(
        raw, "live_video_codec", "libx264", _LIVE_VIDEO_CODECS, "collection_config"
    )
    if video_codec in {"libx264", "h264"} and live_video_codec not in {"libx264", "h264"}:
        raise ValueError(
            "collection_config.live_video_codec must be H.264 when "
            "collection_config.video_codec is H.264-final"
        )
    live_video_crf = _load_video_integer(raw, "live_video_crf", video_crf, "collection_config", minimum=0, maximum=51)
    live_video_preset = _load_video_preset(
        raw, "live_video_preset", "ultrafast", "collection_config"
    )
    live_video_threads = _load_video_integer(raw, "live_video_threads", 1, "collection_config", minimum=0)
    live_video_queue_size = _load_video_integer(
        raw, "live_video_queue_size", 120, "collection_config", minimum=1
    )
    recorder_cpu_set = _load_optional_cpu_set(raw, "recorder_cpu_set", "collection_config")
    live_video_cpu_set = _load_optional_cpu_set(raw, "live_video_cpu_set", "collection_config")
    if recorder_cpu_set is not None and live_video_cpu_set is not None:
        overlap = sorted(set(cpu_set_members(recorder_cpu_set)) & set(cpu_set_members(live_video_cpu_set)))
        if overlap:
            overlap_text = ", ".join(str(cpu) for cpu in overlap)
            raise ValueError(
                "collection_config.recorder_cpu_set and collection_config.live_video_cpu_set "
                f"must not overlap (shared CPUs: {overlap_text})"
            )
    if state_save_freq <= 45.0:
        raise ValueError("collection_config.state_save_freq must be greater than 45 Hz")
    if target_fps <= 0:
        raise ValueError("collection_config.target_fps must be positive")
    if timeout_threshold_s <= 0:
        raise ValueError("collection_config.timeout_threshold_s must be positive")
    if total_sampled_frames <= 0:
        raise ValueError("collection_config.total_sampled_frames must be positive")
    allow_uncalibrated_intrinsics = raw.get("allow_uncalibrated_intrinsics", False)
    if not isinstance(allow_uncalibrated_intrinsics, bool):
        raise ValueError("collection_config.allow_uncalibrated_intrinsics must be true or false")
    validation_profile = normalize_validation_profile(raw.get("validation_profile"))
    return CollectionSettings(
        dataset_dir=dataset_dir,
        state_save_freq=state_save_freq,
        target_fps=target_fps,
        timeout_threshold_s=timeout_threshold_s,
        total_sampled_frames=total_sampled_frames,
        video_codec=video_codec,
        video_crf=video_crf,
        video_preset=video_preset,
        video_threads=video_threads,
        allow_uncalibrated_intrinsics=allow_uncalibrated_intrinsics,
        validation_profile=validation_profile,
        live_video_codec=live_video_codec,
        live_video_crf=live_video_crf,
        live_video_preset=live_video_preset,
        live_video_threads=live_video_threads,
        live_video_queue_size=live_video_queue_size,
        recorder_cpu_set=recorder_cpu_set,
        live_video_cpu_set=live_video_cpu_set,
    )


def _load_telemetry(raw: Mapping[str, Any]) -> dict[str, TelemetryEndpoint]:
    missing = [role for role in _REQUIRED_ROBOT_ROLES if role not in raw]
    if missing:
        raise ValueError(f"telemetry is missing required roles: {', '.join(missing)}")
    endpoints: dict[str, TelemetryEndpoint] = {}
    for role in _REQUIRED_ROBOT_ROLES:
        value = _require_mapping(raw[role], f"telemetry.{role}")
        port = int(value.get("port", 0))
        if not 1 <= port <= 65535:
            raise ValueError(f"telemetry.{role}.port must be in 1..65535")
        endpoints[role] = TelemetryEndpoint(role=role, host=str(value.get("host", "127.0.0.1")), port=port)
    return endpoints


def _load_cameras(raw: Mapping[str, Any], default_fps: float | None) -> dict[str, CameraConfig]:
    missing = [role for role in _REQUIRED_CAMERA_ROLES if role not in raw]
    if missing:
        raise ValueError(f"cameras is missing required roles: {', '.join(missing)}")
    cameras: dict[str, CameraConfig] = {}
    for role in _REQUIRED_CAMERA_ROLES:
        value = _require_mapping(raw[role], f"cameras.{role}")
        serial = str(value.get("serial", "")).strip()
        width = int(value.get("width", 0))
        height = int(value.get("height", 0))
        fps_value = value.get("fps", default_fps)
        stream_format = str(value.get("stream_format", "bgr8")).lower()
        if fps_value is None:
            raise ValueError(f"cameras.{role}.fps must be configured")
        fps = int(fps_value)
        if not serial:
            raise ValueError(f"cameras.{role}.serial must be configured")
        if width < 640 or height < 480:
            raise ValueError(f"cameras.{role} must be at least 640x480")
        if fps <= 0:
            raise ValueError(f"cameras.{role}.fps must be positive")
        if stream_format not in {"bgr8", "yuyv"}:
            raise ValueError(f"cameras.{role}.stream_format must be bgr8 or yuyv")
        intrinsics_value: Optional[Mapping[str, Any]]
        raw_intrinsics = value.get("intrinsics")
        intrinsics_value = _require_mapping(raw_intrinsics, f"cameras.{role}.intrinsics") if raw_intrinsics else None
        cameras[role] = CameraConfig(
            role=role,
            serial=serial,
            width=width,
            height=height,
            fps=fps,
            frame_id=str(value.get("frame_id", f"{role}_color_optical_frame")),
            intrinsics=CameraIntrinsics.from_mapping(intrinsics_value)
            if intrinsics_value is not None
            else CameraIntrinsics.uncalibrated(),
            stream_format=stream_format,
        )
    serials = [camera.serial for camera in cameras.values()]
    if len(set(serials)) != len(serials):
        raise ValueError("each camera role must use a distinct serial number")
    return cameras


def _load_yaml_mapping(path: str | Path) -> tuple[Path, Mapping[str, Any]]:
    """Load one YAML mapping once for either recorder or camera-only use."""
    source_path = Path(path).expanduser().resolve()
    with source_path.open("r", encoding="utf-8") as stream:
        raw_value = yaml.safe_load(stream)
    return source_path, _require_mapping(raw_value, "configuration root")


def _merge_mapping(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively overlay a mode profile without mutating the source YAML."""
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge_mapping(_require_mapping(merged[key], str(key)), value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_capture_mode(raw: Mapping[str, Any], capture_mode: str | None) -> _ResolvedCaptureMode | None:
    """Resolve the selected formal mode and ensure its declared contract is intact."""
    normalized = _normalize_capture_mode(capture_mode)
    modes_value = raw.get("capture_modes")
    if modes_value is None:
        persisted = _normalize_capture_mode(raw.get("selected_capture_mode"))
        if normalized is not None and persisted is not None and normalized != persisted:
            raise ValueError(
                f"saved configuration was captured in {persisted}; it cannot be loaded with --mode {normalized}"
            )
        if normalized is not None:
            if persisted == normalized:
                return None
            raise ValueError("--mode requires a capture_modes mapping in the configuration")
        return None
    modes = _require_mapping(modes_value, "capture_modes")
    if normalized is None:
        available = ", ".join(str(name) for name in modes)
        raise ValueError(f"configuration defines capture modes ({available}); choose one with --mode")
    if normalized not in modes:
        available = ", ".join(str(name) for name in modes)
        raise ValueError(f"capture mode {normalized!r} is not configured; available: {available}")
    expected_fps, expected_profile = _CAPTURE_MODE_CONTRACTS[normalized]
    mode_raw = _require_mapping(modes[normalized], f"capture_modes.{normalized}")
    declared_profile = normalize_validation_profile(mode_raw.get("validation_profile", expected_profile))
    if declared_profile != expected_profile:
        raise ValueError(
            f"capture_modes.{normalized}.validation_profile must be {expected_profile}, got {declared_profile}"
        )
    dataset_subdir = str(mode_raw.get("dataset_subdir", normalized)).strip()
    if not dataset_subdir or Path(dataset_subdir).is_absolute() or ".." in Path(dataset_subdir).parts:
        raise ValueError(f"capture_modes.{normalized}.dataset_subdir must be a relative child directory")
    cameras = _require_mapping(mode_raw.get("cameras"), f"capture_modes.{normalized}.cameras")
    collection_config = _require_mapping(
        mode_raw.get("collection_config", {}),
        f"capture_modes.{normalized}.collection_config",
    )
    camera_test_config = _require_mapping(
        mode_raw.get("camera_test_config", {}),
        f"capture_modes.{normalized}.camera_test_config",
    )
    return _ResolvedCaptureMode(
        name=normalized,
        target_fps=expected_fps,
        validation_profile=expected_profile,
        dataset_subdir=dataset_subdir,
        cameras=cameras,
        collection_config=collection_config,
        camera_test_config=camera_test_config,
    )


def _resolved_cameras_raw(raw: Mapping[str, Any], mode: _ResolvedCaptureMode | None) -> Mapping[str, Any]:
    """Combine fixed physical camera identity with the selected stream profile."""
    cameras = _require_mapping(raw.get("cameras"), "cameras")
    if mode is None:
        return cameras
    missing = [role for role in _REQUIRED_CAMERA_ROLES if role not in mode.cameras]
    if missing:
        raise ValueError(f"capture_modes.{mode.name}.cameras is missing required roles: {', '.join(missing)}")
    return {
        role: _merge_mapping(
            _require_mapping(cameras[role], f"cameras.{role}"),
            _require_mapping(mode.cameras[role], f"capture_modes.{mode.name}.cameras.{role}"),
        )
        for role in _REQUIRED_CAMERA_ROLES
    }


def _resolved_collection_raw(raw: Mapping[str, Any], mode: _ResolvedCaptureMode | None) -> Mapping[str, Any]:
    """Apply tunable mode overrides, then restore its fixed capture contract."""
    collection = _require_mapping(raw.get("collection_config"), "collection_config")
    if mode is None:
        return collection
    resolved = _merge_mapping(collection, mode.collection_config)
    # The dataset root belongs to the host-level config. A mode can tune the
    # recorder, but it cannot escape its own contract directory.
    configured_dir = str(collection.get("dataset_dir", "")).strip()
    if not configured_dir:
        raise ValueError("collection_config.dataset_dir must be a non-empty path")
    resolved["dataset_dir"] = str(Path(configured_dir) / mode.dataset_subdir)
    resolved["state_save_freq"] = _CAPTURE_MODE_STATE_SAVE_FREQ
    resolved["target_fps"] = mode.target_fps
    resolved["validation_profile"] = mode.validation_profile
    return resolved


def _resolved_camera_test_settings_raw(raw: Mapping[str, Any], mode: _ResolvedCaptureMode | None) -> Mapping[str, Any]:
    """Apply a mode's test tuning, then restore its fixed camera contract."""
    settings = _require_mapping(raw.get("camera_test_config"), "camera_test_config")
    if mode is None:
        return settings
    resolved = _merge_mapping(settings, mode.camera_test_config)
    configured_dir = str(settings.get("output_dir", "")).strip()
    if not configured_dir:
        raise ValueError("camera_test_config.output_dir must be a non-empty path")
    resolved["output_dir"] = str(Path(configured_dir) / mode.dataset_subdir)
    resolved["validation_profile"] = mode.validation_profile
    return resolved


def _resolved_raw_for_record(raw: Mapping[str, Any], mode: _ResolvedCaptureMode | None) -> Mapping[str, Any]:
    """Create the self-contained configuration saved beside each dataset/test run."""
    saved = deepcopy(dict(raw))
    if mode is None:
        return saved
    if "collection_config" in raw:
        saved["collection_config"] = _resolved_collection_raw(raw, mode)
    if "camera_test_config" in raw:
        saved["camera_test_config"] = _resolved_camera_test_settings_raw(raw, mode)
    saved["cameras"] = _resolved_cameras_raw(raw, mode)
    # The saved file is already mode-resolved. Keeping the source mode map
    # would make a later load append the mode subdirectory a second time.
    saved.pop("capture_modes", None)
    saved["selected_capture_mode"] = mode.name
    return saved


def load_config(path: str | Path, capture_mode: str | None = None) -> DatasetConfig:
    """Load a portable capture configuration, resolving paths from its location."""
    source_path, raw = _load_yaml_mapping(path)
    mode = _resolve_capture_mode(raw, capture_mode)
    persisted_mode = _normalize_capture_mode(raw.get("selected_capture_mode")) if mode is None else mode.name
    collection = _load_collection(_resolved_collection_raw(raw, mode), source_path.parent)
    telemetry = _load_telemetry(_require_mapping(raw.get("telemetry"), "telemetry"))
    # ``target_fps`` remains a backwards-compatible default for old configs
    # that omitted a per-camera rate. New deployments may use mixed profiles.
    cameras = _load_cameras(_resolved_cameras_raw(raw, mode), collection.target_fps)
    validate_profile_camera_fps(collection.validation_profile, cameras)
    task_info = _require_mapping(raw.get("task_info", {}), "task_info")
    resolved_raw = dict(_resolved_raw_for_record(raw, mode))
    resolved_collection_raw = dict(_require_mapping(resolved_raw.get("collection_config"), "collection_config"))
    resolved_collection_raw["dataset_dir"] = str(collection.dataset_dir)
    if collection.recorder_cpu_set is not None:
        resolved_collection_raw["recorder_cpu_set"] = collection.recorder_cpu_set
    if collection.live_video_cpu_set is not None:
        resolved_collection_raw["live_video_cpu_set"] = collection.live_video_cpu_set
    resolved_raw["collection_config"] = resolved_collection_raw
    return DatasetConfig(
        source_path=source_path,
        raw=resolved_raw,
        collection=collection,
        telemetry=telemetry,
        cameras=cameras,
        task_info=task_info,
        capture_mode=persisted_mode,
    )


def load_camera_test_config(path: str | Path, capture_mode: str | None = None) -> CameraTestConfig:
    """Load a portable mixed-frame-rate RealSense qualification configuration."""
    source_path, raw = _load_yaml_mapping(path)
    mode = _resolve_capture_mode(raw, capture_mode)
    persisted_mode = _normalize_capture_mode(raw.get("selected_capture_mode")) if mode is None else mode.name
    settings_raw = _resolved_camera_test_settings_raw(raw, mode)
    output_value = settings_raw.get("output_dir")
    if not isinstance(output_value, str) or not output_value:
        raise ValueError("camera_test_config.output_dir must be a non-empty path")
    configured_output = Path(output_value).expanduser()
    output_dir = (
        configured_output if configured_output.is_absolute() else (source_path.parent / configured_output).resolve()
    )
    duration_s = float(settings_raw.get("duration_s", 60.0))
    timeout_threshold_s = float(settings_raw.get("timeout_threshold_s", 1.0))
    video_codec = _load_video_codec(
        settings_raw, "video_codec", "libx265", _LIVE_VIDEO_CODECS, "camera_test_config"
    )
    video_crf = _load_video_integer(settings_raw, "video_crf", 23, "camera_test_config", minimum=0, maximum=51)
    video_preset = _load_video_preset(settings_raw, "video_preset", "veryfast", "camera_test_config")
    video_threads = _load_video_integer(settings_raw, "video_threads", 0, "camera_test_config", minimum=0)
    video_queue_size = int(settings_raw.get("video_queue_size", 120))
    fps_tolerance = float(settings_raw.get("fps_tolerance", 0.5))
    max_gap_factor = float(settings_raw.get("max_gap_factor", 2.1))
    validation_profile = normalize_validation_profile(settings_raw.get("validation_profile"))
    if duration_s <= 0:
        raise ValueError("camera_test_config.duration_s must be positive")
    if timeout_threshold_s <= 0:
        raise ValueError("camera_test_config.timeout_threshold_s must be positive")
    if video_queue_size <= 0:
        raise ValueError("camera_test_config.video_queue_size must be positive")
    if fps_tolerance < 0:
        raise ValueError("camera_test_config.fps_tolerance must be non-negative")
    if max_gap_factor < 1.0:
        raise ValueError("camera_test_config.max_gap_factor must be at least 1.0")
    cameras = _load_cameras(_resolved_cameras_raw(raw, mode), default_fps=None)
    validate_profile_camera_fps(validation_profile, cameras)
    resolved_raw = dict(_resolved_raw_for_record(raw, mode))
    resolved_test_raw = dict(_require_mapping(resolved_raw.get("camera_test_config"), "camera_test_config"))
    resolved_test_raw["output_dir"] = str(output_dir)
    resolved_raw["camera_test_config"] = resolved_test_raw
    return CameraTestConfig(
        source_path=source_path,
        raw=resolved_raw,
        settings=CameraTestSettings(
            output_dir=output_dir,
            duration_s=duration_s,
            timeout_threshold_s=timeout_threshold_s,
            video_codec=video_codec,
            video_crf=video_crf,
            video_preset=video_preset,
            video_threads=video_threads,
            video_queue_size=video_queue_size,
            fps_tolerance=fps_tolerance,
            max_gap_factor=max_gap_factor,
            validation_profile=validation_profile,
        ),
        cameras=cameras,
        capture_mode=persisted_mode,
    )
