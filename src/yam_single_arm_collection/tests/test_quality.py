from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from yam_data_collection.config import DatasetConfig
from yam_data_collection.mcap_writer import CAMERA_TOPICS, ROBOT_TOPICS, EpisodeMcapWriter
from yam_data_collection.models import RobotSnapshot
from yam_data_collection.quality import build_validation_policy, validate_mcap as validate_bimanual_mcap

from yam_single_arm_collection.quality import validate_mcap, validate_video_assets


BASE = 1_700_000_000_000_000_000


def _snapshot(role: str, timestamp_ns: int, sequence: int) -> RobotSnapshot:
    return RobotSnapshot(role, timestamp_ns, sequence, (0.0,) * 7, (0.0,) * 7, (0.0,) * 7)


def _write_mcap(
    path: Path,
    config: DatasetConfig,
    *,
    skip_joint: str | None = None,
    skip_camera: str | None = None,
    shifted_joint: str | None = None,
    extra_joint: str | None = None,
    gapped_camera: str | None = None,
    wrong_resolution: bool = False,
) -> None:
    with EpisodeMcapWriter(path, config.task_info) as writer:
        for index in range(201):
            timestamp = BASE + index * 10_000_000
            for role in config.telemetry:
                if role != skip_joint:
                    source = timestamp + (500_000_000 if role == shifted_joint else 0)
                    writer.write_joint_state(_snapshot(role, source, index), timestamp, header_timestamp_ns=source)
        if extra_joint is not None:
            writer.write_joint_state(_snapshot(extra_joint, BASE, 0), BASE)
        for role, camera in config.cameras.items():
            if role == skip_camera:
                continue
            if wrong_resolution:
                camera = replace(camera, width=640)
            for index in range(camera.fps * 2 + 1):
                if role == gapped_camera and 30 <= index <= 34:
                    continue
                timestamp = BASE + round(index * 1_000_000_000 / camera.fps)
                writer.write_camera_info(camera, timestamp)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("setup", ["wrist", "wrist-top"])
@pytest.mark.parametrize("mode", ["60hz", "30hz"])
def test_real_mcap_passes_only_the_scoped_contract(
    tmp_path: Path, config_factory: object, side: str, setup: str, mode: str
) -> None:
    config = config_factory(side, setup, mode)
    path = tmp_path / "episode.mcap"
    _write_mcap(path, config)

    result = validate_mcap(path, config)

    assert result["output_mcap_passed"], result
    assert result["required_topics"] == {
        "missing_camera_topics": [],
        "missing_joint_topics": [],
        "unexpected_topics": [],
    }
    assert result["mandatory_checks"]["camera_sync"]["applicable"] == (setup == "wrist-top")
    assert not validate_bimanual_mcap(path)["output_mcap_passed"]


@pytest.mark.parametrize("missing", ["right_leader", "right_follower"])
def test_missing_selected_joint_fails(tmp_path: Path, config_factory: object, missing: str) -> None:
    config = config_factory()
    path = tmp_path / "episode.mcap"
    _write_mcap(path, config, skip_joint=missing)
    result = validate_mcap(path, config)
    assert not result["output_mcap_passed"]
    assert ROBOT_TOPICS[missing] in result["required_topics"]["missing_joint_topics"]


def test_missing_optional_top_is_required_after_selection(tmp_path: Path, config_factory: object) -> None:
    config = config_factory("right", "wrist-top")
    path = tmp_path / "episode.mcap"
    _write_mcap(path, config, skip_camera="cam_high")
    result = validate_mcap(path, config)
    assert not result["output_mcap_passed"]
    assert result["required_topics"]["missing_camera_topics"] == [CAMERA_TOPICS["cam_high"]]


@pytest.mark.parametrize("shifted", ["right_leader", "right_follower"])
def test_each_joint_stream_must_align_with_camera(tmp_path: Path, config_factory: object, shifted: str) -> None:
    config = config_factory()
    path = tmp_path / "episode.mcap"
    _write_mcap(path, config, shifted_joint=shifted)
    result = validate_mcap(path, config)
    assert "joint_camera_sync" in result["failed_checks"]
    assert not result["mandatory_checks"]["joint_camera_sync"]["per_joint_topic"][ROBOT_TOPICS[shifted]]["passed"]


def test_wrong_side_joint_is_not_accepted(tmp_path: Path, config_factory: object) -> None:
    config = config_factory()
    path = tmp_path / "episode.mcap"
    _write_mcap(path, config, extra_joint="left_follower")
    result = validate_mcap(path, config)
    assert not result["output_mcap_passed"]
    assert result["required_topics"]["unexpected_topics"] == [ROBOT_TOPICS["left_follower"]]


def test_gap_threshold_is_not_relaxed_for_single_camera(tmp_path: Path, config_factory: object) -> None:
    config = config_factory()
    path = tmp_path / "episode.mcap"
    _write_mcap(path, config, gapped_camera="cam_right")
    result = validate_mcap(path, config)
    assert "camera_max_gap" in result["failed_checks"]
    original = build_validation_policy(config.cameras, config.collection.validation_profile)
    assert result["policy"]["camera_max_gap_ms"] == original.camera_max_gap_ms


def test_exact_resolution_is_still_required(tmp_path: Path, config_factory: object) -> None:
    config = config_factory()
    path = tmp_path / "episode.mcap"
    _write_mcap(path, config, wrong_resolution=True)
    assert "camera_resolution_matches_config" in validate_mcap(path, config)["failed_checks"]


def test_only_selected_video_errors_affect_result(
    tmp_path: Path, config_factory: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = config_factory()
    monkeypatch.setattr(
        "yam_single_arm_collection.quality.validate_all_video_assets",
        lambda *args, **kwargs: {
            "passed": False,
            "video_dir": str(tmp_path),
            "per_camera": {
                "cam_right": {"passed": True, "expected_camera_info_count": 61},
                "cam_left": {"passed": False, "error": "missing video"},
                "cam_high": {"passed": False, "error": "missing video"},
            },
        },
    )
    assert validate_video_assets(tmp_path / "episode.mcap", config)["passed"]
    (tmp_path / "cam_high.mp4").touch()
    assert not validate_video_assets(tmp_path / "episode.mcap", config)["passed"]
