"""Apply the existing quality thresholds to exactly one arm pair and its cameras."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from yam_data_collection.config import DatasetConfig
from yam_data_collection.mcap_writer import CAMERA_TOPICS, ROBOT_TOPICS
from yam_data_collection.quality import (
    ValidationPolicy,
    _camera_timing_diagnostics,
    _check_camera_gaps,
    _check_camera_rate,
    _check_camera_sync,
    _check_configured_resolution,
    _check_joint_camera_sync,
    _check_joint_frequency,
    _check_resolution,
    _informational_rolling,
    build_validation_policy,
    inspect_mcap,
    validate_video_assets as validate_all_video_assets,
)


def validation_policy(config: DatasetConfig) -> ValidationPolicy:
    policy = build_validation_policy(config.cameras, config.collection.validation_profile)
    return replace(
        policy,
        require_exact_camera_resolution=True,
        expected_camera_resolution_by_topic={
            CAMERA_TOPICS[role]: (camera.width, camera.height) for role, camera in config.cameras.items()
        },
    )


def validate_mcap(path: str | Path, config: DatasetConfig) -> dict[str, Any]:
    """Validate selected streams without requiring or synthesizing the other side."""
    policy = validation_policy(config)
    inspection = inspect_mcap(path)
    stats = inspection.stats
    camera_topics = [CAMERA_TOPICS[role] for role in config.cameras]
    joint_topics = [ROBOT_TOPICS[role] for role in config.telemetry]
    missing_cameras = [topic for topic in camera_topics if topic not in stats]
    missing_joints = [topic for topic in joint_topics if topic not in stats]
    unexpected = sorted(
        set(stats).intersection({*CAMERA_TOPICS.values(), *ROBOT_TOPICS.values()})
        - set(camera_topics)
        - set(joint_topics)
    )
    camera_sync = _check_camera_sync(inspection.header_stamps, camera_topics, policy)
    camera_sync["applicable"] = len(camera_topics) > 1
    if len(camera_topics) == 1:
        camera_sync["reason"] = "single camera; joint-camera synchronization is still mandatory"
    per_joint_sync = {
        topic: _check_joint_camera_sync(inspection.header_stamps, camera_topics, [topic], policy)
        for topic in joint_topics
    }
    mandatory = {
        "selected_topics_present_and_exclusive": {
            "passed": not (missing_cameras or missing_joints or unexpected),
            "unexpected_topics": unexpected,
        },
        "camera_avg_fps_within_0.5_of_target": _check_camera_rate(stats, camera_topics, policy),
        "camera_max_gap": _check_camera_gaps(stats, camera_topics, policy),
        "camera_resolution_min_640x480": _check_resolution(inspection.resolutions, camera_topics, policy),
        "camera_resolution_matches_config": _check_configured_resolution(
            inspection.resolutions, camera_topics, policy
        ),
        "camera_sync": camera_sync,
        "joint_camera_sync": {
            "passed": all(entry["passed"] for entry in per_joint_sync.values()),
            "threshold_ms": policy.joint_camera_sync_ms,
            "per_joint_topic": per_joint_sync,
        },
        "joint_avg_freq_above_45hz": _check_joint_frequency(stats, joint_topics, policy),
    }
    informational = _informational_rolling(
        inspection.header_stamps, inspection.log_times, camera_topics, joint_topics, policy
    )
    informational.update(_camera_timing_diagnostics(inspection, camera_topics, policy))
    failed = [name for name, entry in mandatory.items() if not entry["passed"]]
    return {
        "output_mcap_passed": not failed,
        "failed_checks": failed,
        "passed_checks": len(mandatory) - len(failed),
        "total_checks": len(mandatory),
        "mandatory_checks": mandatory,
        "informational_checks": informational,
        "informational_failures": [name for name, entry in informational.items() if not entry["passed"]],
        "required_topics": {
            "missing_camera_topics": missing_cameras,
            "missing_joint_topics": missing_joints,
            "unexpected_topics": unexpected,
        },
        "single_arm": dict(config.raw["single_arm"]),
        "policy": asdict(policy),
    }


def validate_video_assets(
    path: str | Path,
    config: DatasetConfig,
    *,
    video_dir: Path | None = None,
) -> dict[str, Any]:
    """Retain the existing codec/resolution/frame-count checks for selected MP4s."""
    policy = validation_policy(config)
    result = validate_all_video_assets(
        path, policy, video_dir=video_dir, expected_video_codecs=("h264",), require_fps_match=False
    )
    selected = {role: result["per_camera"][role] for role in config.cameras}
    unexpected = sorted(
        video.stem for video in Path(result["video_dir"]).glob("*.mp4") if video.stem not in config.cameras
    )
    result["per_camera"] = selected
    result["unexpected_videos"] = unexpected
    result["passed"] = not unexpected and all(
        entry["passed"] and entry.get("expected_camera_info_count", 0) > 0 for entry in selected.values()
    )
    return result
