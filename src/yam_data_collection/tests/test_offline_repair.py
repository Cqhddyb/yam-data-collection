from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import yam_data_collection.offline_repair as offline_repair
from yam_data_collection.config import CollectionSettings, DatasetConfig
from yam_data_collection.mcap_writer import CAMERA_TOPICS, ROBOT_TOPICS, EpisodeMcapWriter
from yam_data_collection.models import CameraTimingSample, RobotSnapshot
from yam_data_collection.offline_repair import (
    FRAME_PERIOD_NS,
    NoRepairNeededError,
    OfflineRepairError,
    RepairPlan,
    build_repair_plan,
    rewrite_mcap_with_repairs,
)
from yam_data_collection.quality import build_validation_policy, inspect_mcap, validate_mcap

from .test_mcap_contract import _camera


def _config(dataset: Path, *, capture_mode: str = "30hz") -> DatasetConfig:
    cameras = {role: _camera(role, width=1280, height=720, fps=30) for role in CAMERA_TOPICS}
    collection = CollectionSettings(
        dataset_dir=dataset,
        state_save_freq=180.0,
        target_fps=30.0,
        timeout_threshold_s=1.0,
        total_sampled_frames=50_000,
        video_codec="libx265",
        video_crf=23,
        video_preset="ultrafast",
        video_threads=1,
        allow_uncalibrated_intrinsics=True,
        validation_profile="yam_30hz",
    )
    return DatasetConfig(
        source_path=dataset / "config.yaml",
        raw={
            "collection_config": {"dataset_dir": str(dataset)},
            "selected_capture_mode": capture_mode,
        },
        collection=collection,
        telemetry={},
        cameras=cameras,
        task_info={"task_name": "repair-test"},
        capture_mode=capture_mode,
    )


def _snapshot(role: str) -> RobotSnapshot:
    follower = role.endswith("follower")
    return RobotSnapshot(
        role=role,
        source_timestamp_ns=1,
        sequence=1,
        position=(0.0,) * 7,
        velocity=(0.0,) * 7 if follower else (),
        effort=(0.0,) * 7 if follower else (),
    )


def _write_source(path: Path, config: DatasetConfig, *, missing_indices: set[int]) -> None:
    path.parent.mkdir(parents=True)
    base = 1_700_000_000_000_000_000
    indices_by_role = {
        role: [index for index in range(301) if role != "cam_left" or index not in missing_indices]
        for role in CAMERA_TOPICS
    }
    with EpisodeMcapWriter(path, {"task_name": "repair-test"}) as writer:
        for index in range(1_001):
            timestamp = base + index * 10_000_000
            for role in ROBOT_TOPICS:
                writer.write_joint_state(_snapshot(role), timestamp)
        for role, indices in indices_by_role.items():
            camera = config.cameras[role]
            for index in indices:
                writer.write_camera_info(camera, base + index * FRAME_PERIOD_NS)
        for role, indices in indices_by_role.items():
            samples = [
                CameraTimingSample(
                    role=role,
                    capture_timestamp_ns=base + index * FRAME_PERIOD_NS,
                    host_receive_timestamp_ns=base + index * FRAME_PERIOD_NS + 1_000_000,
                    device_timestamp_ns=base + index * FRAME_PERIOD_NS,
                    device_frame_number=10_000 + index,
                    device_timestamp_domain="timestamp_domain.global_time",
                    capture_timestamp_source="realsense_global_time",
                )
                for index in indices
            ]
            writer.write_camera_timing_diagnostics(role, samples)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_delta_three_gap_builds_one_previous_frame_repair_and_rewrites_mcap(tmp_path: Path) -> None:
    config = _config(tmp_path)
    source = tmp_path / ".pending" / "0" / "0.mcap"
    _write_source(source, config, missing_indices={150, 151})
    source_hash = _sha256(source)

    plan = build_repair_plan(source, config)

    assert plan.event_count == 1
    event = plan.events_by_role["cam_left"][0]
    assert event.source_frame_index == 149
    assert event.trigger_frame_index == 150
    assert event.synthetic_output_index == 150
    assert event.source_device_frame_number == 10_149
    assert event.trigger_device_frame_number == 10_152
    assert event.synthetic_timestamp_ns == event.source_capture_timestamp_ns + FRAME_PERIOD_NS

    repaired = tmp_path / "derived" / "0.mcap"
    rewrite_mcap_with_repairs(source, repaired, config, plan)

    assert _sha256(source) == source_hash
    policy = build_validation_policy(config.cameras, config.collection.validation_profile)
    result = validate_mcap(repaired, policy)
    assert result["output_mcap_passed"]
    assert result["failed_checks"] == []
    assert "camera_source_frame_continuity" in result["informational_failures"]
    assert "camera_contains_offline_repaired_frames" in result["informational_failures"]
    left_topic = CAMERA_TOPICS["cam_left"]
    continuity = result["informational_checks"]["camera_source_frame_continuity"]["per_topic"][left_topic]
    assert continuity["skipped_frames"] == 2
    assert continuity["synthetic_frames"] == 1

    inspection = inspect_mcap(repaired)
    diagnostics = inspection.camera_timing_diagnostics["cam_left"]
    assert diagnostics.synthetic_indices == (150,)
    assert diagnostics.frame_numbers[149:152] == (10_149, None, 10_152)
    assert diagnostics.capture_timestamps_ns[150] == event.synthetic_timestamp_ns
    assert inspection.stats[left_topic].max_gap_ms == pytest.approx(66.666666, abs=0.01)


def test_single_missing_frame_at_66ms_is_not_repaired(tmp_path: Path) -> None:
    config = _config(tmp_path)
    source = tmp_path / ".pending" / "0" / "0.mcap"
    _write_source(source, config, missing_indices={150})

    with pytest.raises(NoRepairNeededError, match="no eligible delta=3"):
        build_repair_plan(source, config)


def test_delta_four_gap_is_rejected_instead_of_partially_repaired(tmp_path: Path) -> None:
    config = _config(tmp_path)
    source = tmp_path / ".pending" / "0" / "0.mcap"
    _write_source(source, config, missing_indices={150, 151, 152})

    with pytest.raises(OfflineRepairError, match="no eligible delta=3"):
        build_repair_plan(source, config)


def test_repair_rejects_non_30hz_mode(tmp_path: Path) -> None:
    config = _config(tmp_path, capture_mode="60hz")
    source = tmp_path / ".pending" / "0" / "0.mcap"
    _write_source(source, config, missing_indices={150, 151})

    with pytest.raises(OfflineRepairError, match="requires --mode 30hz"):
        build_repair_plan(source, config)


def test_repair_reports_plan_before_video_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    plan = RepairPlan(
        source_mcap=tmp_path / "0.mcap",
        episode_id=0,
        events_by_role={role: () for role in CAMERA_TOPICS},
        source_validation={},
    )
    stages: list[str] = []

    monkeypatch.setattr(offline_repair, "build_repair_plan", lambda *args, **kwargs: plan)

    def stop_before_video(*args: object, **kwargs: object) -> dict[str, object]:
        assert stages == ["plan"]
        raise RuntimeError("video validation reached")

    monkeypatch.setattr(offline_repair, "validate_video_assets", stop_before_video)

    with pytest.raises(RuntimeError, match="video validation reached"):
        offline_repair.repair_30hz_episode(
            plan.source_mcap,
            config,
            on_plan_ready=lambda received: stages.append("plan") if received is plan else None,
        )

    assert stages == ["plan"]
