from __future__ import annotations

from pathlib import Path

from yam_data_collection.config import CameraConfig, CameraIntrinsics
from yam_data_collection.mcap_writer import CAMERA_TOPICS, ROBOT_TOPICS, EpisodeMcapWriter
from yam_data_collection.models import CameraTimingSample, RobotSnapshot
from yam_data_collection.quality import (
    TopicStats,
    ValidationPolicy,
    _check_camera_rate,
    _check_camera_sync,
    _check_joint_camera_sync,
    inspect_mcap,
    validate_mcap,
)
from yam_data_collection.validation_profiles import (
    PIPER_COMPATIBLE_PROFILE,
    YAM_60HZ_PROFILE,
    YAM_60HZ_STRICT_PROFILE,
)


def _camera(role: str, *, width: int = 1280, height: int = 720, fps: int = 30) -> CameraConfig:
    return CameraConfig(
        role=role,
        serial=role,
        width=width,
        height=height,
        fps=fps,
        frame_id=role,
        intrinsics=CameraIntrinsics.uncalibrated(),
    )


def _snapshot(role: str) -> RobotSnapshot:
    return RobotSnapshot(
        role=role,
        source_timestamp_ns=1,
        sequence=1,
        position=(0.0,) * 7,
        velocity=(0.0,) * 7,
        effort=(0.0,) * 7,
    )


def _write_valid_episode(path: Path) -> None:
    base = 1_700_000_000_000_000_000
    with EpisodeMcapWriter(path, {"task_name": "test"}) as writer:
        for index in range(201):
            timestamp = base + index * 10_000_000
            for role in ROBOT_TOPICS:
                writer.write_joint_state(_snapshot(role), timestamp)
        for index in range(61):
            timestamp = base + round(index * 1_000_000_000 / 30)
            for role in CAMERA_TOPICS:
                writer.write_camera_info(_camera(role), timestamp)


def _write_valid_episode_with_timing_diagnostics(
    path: Path,
    *,
    host_gap_role: str | None = None,
    capture_and_host_gap_role: str | None = None,
    skipped_frame_role: str | None = None,
) -> None:
    """Write a valid Piper episode plus one diagnostic sample per camera frame."""
    base = 1_700_000_000_000_000_000
    camera_timestamps = [base + round(index * 1_000_000_000 / 30) for index in range(61)]
    with EpisodeMcapWriter(path, {"task_name": "test"}) as writer:
        for index in range(201):
            timestamp = base + index * 10_000_000
            for role in ROBOT_TOPICS:
                writer.write_joint_state(_snapshot(role), timestamp)
        timestamps_by_role = {role: list(camera_timestamps) for role in CAMERA_TOPICS}
        if capture_and_host_gap_role is not None:
            # Shift the second half so both capture and host timing have one
            # 83.33 ms gap. The long-series average remains 30 Hz.
            timestamps_by_role[capture_and_host_gap_role] = [
                timestamp + (50_000_000 if index >= 30 else 0)
                for index, timestamp in enumerate(camera_timestamps)
            ]
        for role, timestamps in timestamps_by_role.items():
            for timestamp in timestamps:
                writer.write_camera_info(_camera(role), timestamp)
        for role in CAMERA_TOPICS:
            samples = []
            for index, capture_timestamp_ns in enumerate(timestamps_by_role[role]):
                # A persistent 50 ms host-delivery delay produces one 83.33 ms
                # host gap without changing the camera capture clock.
                host_offset_ns = 50_000_000 if role == host_gap_role and index >= 30 else 0
                # The diagnostic represents a device frame-counter discontinuity,
                # while capture timestamps remain a valid 30 Hz series.
                frame_offset = 1 if role == skipped_frame_role and index >= 30 else 0
                samples.append(
                    CameraTimingSample(
                        role=role,
                        capture_timestamp_ns=capture_timestamp_ns,
                        host_receive_timestamp_ns=capture_timestamp_ns + host_offset_ns,
                        device_timestamp_ns=capture_timestamp_ns,
                        device_frame_number=1_000 + index + frame_offset,
                        device_timestamp_domain="timestamp_domain.global_time",
                        capture_timestamp_source="realsense_global_time",
                    )
                )
            writer.write_camera_timing_diagnostics(role, samples)


def _write_mixed_rate_episode(path: Path) -> dict[str, CameraConfig]:
    base = 1_700_000_000_000_000_000
    cameras = {
        "cam_high": _camera("cam_high", width=1280, height=720, fps=30),
        "cam_left": _camera("cam_left", width=848, height=480, fps=60),
        "cam_right": _camera("cam_right", width=848, height=480, fps=60),
    }
    with EpisodeMcapWriter(path, {"task_name": "test"}) as writer:
        for index in range(201):
            timestamp = base + index * 10_000_000
            for role in ROBOT_TOPICS:
                writer.write_joint_state(_snapshot(role), timestamp)
        for role, camera in cameras.items():
            for index in range(camera.fps * 2 + 1):
                timestamp = base + round(index * 1_000_000_000 / camera.fps)
                writer.write_camera_info(camera, timestamp)
    return cameras


def _write_60hz_episode(path: Path, *, add_36ms_gap: bool = False) -> dict[str, CameraConfig]:
    base = 1_700_000_000_000_000_000
    cameras = {role: _camera(role, width=848, height=480, fps=60) for role in CAMERA_TOPICS}
    with EpisodeMcapWriter(path, {"task_name": "test"}) as writer:
        for index in range(1_003):
            timestamp = base + index * 10_000_000
            for role in ROBOT_TOPICS:
                writer.write_joint_state(_snapshot(role), timestamp)
        for role, camera in cameras.items():
            for index in range(601):
                # A permanent 20 ms shift at the midpoint produces one 36.67 ms
                # gap while keeping the ten-second average within 60 +/- 0.5 Hz.
                shift_ns = 20_000_000 if add_36ms_gap and role == "cam_right" and index >= 300 else 0
                timestamp = base + round(index * 1_000_000_000 / 60) + shift_ns
                writer.write_camera_info(camera, timestamp)
    return cameras


def test_writer_produces_piper_compatible_30hz_mcap(tmp_path: Path) -> None:
    mcap_path = tmp_path / "0.mcap"
    _write_valid_episode(mcap_path)

    result = validate_mcap(mcap_path, ValidationPolicy(expected_camera_fps=30.0))

    assert result["output_mcap_passed"]
    assert result["failed_checks"] == []
    assert result["required_topics"] == {"missing_camera_topics": [], "missing_joint_topics": []}


def test_camera_timing_diagnostics_round_trip_through_mcap_inspection(tmp_path: Path) -> None:
    mcap_path = tmp_path / "0.mcap"
    _write_valid_episode_with_timing_diagnostics(mcap_path)

    inspection = inspect_mcap(mcap_path)
    diagnostics = inspection.camera_timing_diagnostics

    assert set(diagnostics) == set(CAMERA_TOPICS)
    high = diagnostics["cam_high"]
    assert len(high.capture_timestamps_ns) == 61
    assert high.capture_timestamps_ns == high.host_receive_timestamps_ns
    assert high.device_timestamps_ns == high.capture_timestamps_ns
    assert high.frame_numbers == tuple(range(1_000, 1_061))
    assert high.capture_timestamp_sources == {"realsense_global_time": 61}
    assert high.device_timestamp_domains == {"timestamp_domain.global_time": 61}
    result = validate_mcap(mcap_path, ValidationPolicy(expected_camera_fps=30.0))
    assert result["informational_checks"]["camera_host_delivery_within_gap_budget"]["passed"]
    assert result["informational_checks"]["camera_source_frame_continuity"]["passed"]


def test_host_delivery_gap_is_informational_when_capture_timing_is_valid(tmp_path: Path) -> None:
    mcap_path = tmp_path / "0.mcap"
    _write_valid_episode_with_timing_diagnostics(mcap_path, host_gap_role="cam_left")

    result = validate_mcap(mcap_path, ValidationPolicy(expected_camera_fps=30.0))

    assert result["output_mcap_passed"]
    assert result["failed_checks"] == []
    assert "camera_host_delivery_within_gap_budget" in result["informational_failures"]
    check = result["informational_checks"]["camera_host_delivery_within_gap_budget"]
    left = check["per_topic"][CAMERA_TOPICS["cam_left"]]
    assert not left["passed"]
    assert left["diagnostic_complete"]
    assert left["capture_max_gap_ms"] <= 69.0
    assert left["host_receive_max_gap_ms"] > 69.0
    assert left["host_only_jitter"]
    assert result["informational_checks"]["camera_source_frame_continuity"]["passed"]


def test_host_delivery_check_fails_when_capture_and_host_gaps_both_exceed_budget(tmp_path: Path) -> None:
    mcap_path = tmp_path / "0.mcap"
    _write_valid_episode_with_timing_diagnostics(mcap_path, capture_and_host_gap_role="cam_left")

    result = validate_mcap(mcap_path, ValidationPolicy(expected_camera_fps=30.0))

    # The existing MCAP-rate gate detects the capture gap. The supplemental
    # delivery diagnostic must also report its own host-side gap, rather than
    # accidentally passing because it is not host-only jitter.
    assert "camera_max_gap_below_69ms" in result["failed_checks"]
    delivery = result["informational_checks"]["camera_host_delivery_within_gap_budget"]
    left = delivery["per_topic"][CAMERA_TOPICS["cam_left"]]
    assert not delivery["passed"]
    assert not left["passed"]
    assert not left["host_only_jitter"]
    assert left["capture_max_gap_ms"] > 69.0
    assert left["host_receive_max_gap_ms"] > 69.0


def test_device_frame_counter_gap_is_an_informational_source_continuity_failure(tmp_path: Path) -> None:
    mcap_path = tmp_path / "0.mcap"
    _write_valid_episode_with_timing_diagnostics(mcap_path, skipped_frame_role="cam_right")

    result = validate_mcap(mcap_path, ValidationPolicy(expected_camera_fps=30.0))

    assert result["output_mcap_passed"]
    assert result["failed_checks"] == []
    assert "camera_source_frame_continuity" in result["informational_failures"]
    check = result["informational_checks"]["camera_source_frame_continuity"]
    right = check["per_topic"][CAMERA_TOPICS["cam_right"]]
    assert not right["passed"]
    assert right["diagnostic_complete"]
    assert right["skipped_frames"] == 1
    assert right["regressions"] == 0
    assert right["unavailable"] == 0
    assert result["informational_checks"]["camera_host_delivery_within_gap_budget"]["passed"]


def test_legacy_episode_without_timing_diagnostics_is_explicitly_skipped(tmp_path: Path) -> None:
    mcap_path = tmp_path / "0.mcap"
    _write_valid_episode(mcap_path)

    result = validate_mcap(mcap_path, ValidationPolicy(expected_camera_fps=30.0))

    assert result["output_mcap_passed"]
    for name in ("camera_host_delivery_within_gap_budget", "camera_source_frame_continuity"):
        check = result["informational_checks"][name]
        assert check["passed"]
        for topic in CAMERA_TOPICS.values():
            assert check["per_topic"][topic] == {"passed": True, "skipped": True, "reason": "legacy MCAP"}


def test_explicit_fps_rejects_a_30hz_episode_when_60hz_is_required(tmp_path: Path) -> None:
    mcap_path = tmp_path / "0.mcap"
    _write_valid_episode(mcap_path)

    result = validate_mcap(mcap_path, ValidationPolicy(expected_camera_fps=60.0))

    assert not result["output_mcap_passed"]
    assert "camera_avg_fps_within_0.5_of_target" in result["failed_checks"]


def test_piper_compatible_profile_preserves_the_original_six_checks(tmp_path: Path) -> None:
    mcap_path = tmp_path / "0.mcap"
    _write_valid_episode(mcap_path)

    result = validate_mcap(mcap_path, ValidationPolicy.for_profile(PIPER_COMPATIBLE_PROFILE))

    assert result["output_mcap_passed"]
    assert result["policy"]["profile_name"] == PIPER_COMPATIBLE_PROFILE
    assert result["total_checks"] == 6
    assert "camera_resolution_matches_config" not in result["mandatory_checks"]
    assert result["mandatory_checks"]["camera_max_gap_below_69ms"]["threshold_ms"] == 69.0
    assert result["mandatory_checks"]["camera_sync_below_34ms"]["threshold_ms"] == 34.0


def test_writer_preserves_hardware_time_in_joint_header(tmp_path: Path) -> None:
    mcap_path = tmp_path / "0.mcap"
    record_time_ns = 1_700_000_000_100_000_000
    hardware_time_ns = record_time_ns - 20_000_000

    with EpisodeMcapWriter(mcap_path, {"task_name": "test"}) as writer:
        writer.write_joint_state(
            _snapshot("left_leader"),
            record_time_ns,
            header_timestamp_ns=hardware_time_ns,
        )

    inspection = inspect_mcap(mcap_path)
    topic = ROBOT_TOPICS["left_leader"]
    assert inspection.log_times[topic] == (record_time_ns,)
    assert inspection.publish_times[topic] == (hardware_time_ns,)
    assert inspection.header_stamps[topic] == (hardware_time_ns,)


def test_yam_60hz_profile_rejects_30hz_camera_data(tmp_path: Path) -> None:
    mcap_path = tmp_path / "0.mcap"
    _write_valid_episode(mcap_path)
    cameras = {role: _camera(role, width=848, height=480, fps=60) for role in CAMERA_TOPICS}

    result = validate_mcap(mcap_path, ValidationPolicy.for_profile(YAM_60HZ_STRICT_PROFILE, cameras))

    assert not result["output_mcap_passed"]
    assert result["policy"]["profile_name"] == YAM_60HZ_STRICT_PROFILE
    assert "camera_avg_fps_within_0.5_of_target" in result["failed_checks"]


def test_yam_60hz_profile_rejects_a_gap_that_piper_allows(tmp_path: Path) -> None:
    mcap_path = tmp_path / "0.mcap"
    cameras = _write_60hz_episode(mcap_path, add_36ms_gap=True)

    piper_result = validate_mcap(mcap_path, ValidationPolicy.for_profile(PIPER_COMPATIBLE_PROFILE, cameras))
    strict_result = validate_mcap(mcap_path, ValidationPolicy.for_profile(YAM_60HZ_STRICT_PROFILE, cameras))

    assert piper_result["output_mcap_passed"]
    assert not strict_result["output_mcap_passed"]
    assert "camera_max_gap_below_35ms" in strict_result["failed_checks"]
    gap = strict_result["mandatory_checks"]["camera_max_gap_below_35ms"]
    assert gap["per_topic"][CAMERA_TOPICS["cam_right"]]["max_gap_ms"] > 35.0


def test_new_yam_60hz_profile_keeps_the_piper_gap_allowance(tmp_path: Path) -> None:
    mcap_path = tmp_path / "0.mcap"
    cameras = _write_60hz_episode(mcap_path, add_36ms_gap=True)

    result = validate_mcap(mcap_path, ValidationPolicy.for_profile(YAM_60HZ_PROFILE, cameras))

    gap_check = result["mandatory_checks"]["camera_max_gap_below_69ms"]
    assert gap_check["passed"]
    assert gap_check["threshold_ms"] == 69.0


def test_yam_60hz_profile_uses_a_17ms_sync_budget() -> None:
    configured_cameras = {role: _camera(role, width=848, height=480, fps=60) for role in CAMERA_TOPICS}
    policy = ValidationPolicy.for_profile(YAM_60HZ_STRICT_PROFILE, configured_cameras)
    cameras = list(CAMERA_TOPICS.values())
    joints = list(ROBOT_TOPICS.values())
    header_stamps = {
        cameras[0]: (0, 40_000_000),
        cameras[1]: (18_000_000, 58_000_000),
        cameras[2]: (0, 40_000_000),
        **{topic: (0, 40_000_000) for topic in joints},
    }

    camera_sync = _check_camera_sync(header_stamps, cameras, policy)
    joint_sync = _check_joint_camera_sync(header_stamps, cameras, joints, policy)

    assert not camera_sync["passed"]
    assert camera_sync["threshold_ms"] == 17.0
    assert camera_sync["max_pair_offset_ms"] == 18.0
    assert not joint_sync["passed"]
    assert joint_sync["threshold_ms"] == 17.0
    assert joint_sync["max_offset_ms"] == 18.0


def test_yam_60hz_profile_requires_camera_configuration() -> None:
    try:
        ValidationPolicy.for_profile(YAM_60HZ_STRICT_PROFILE)
    except ValueError as exc:
        assert "requires configured cameras" in str(exc)
    else:  # pragma: no cover - makes the strict profile invariant explicit
        raise AssertionError("strict profile accepted missing camera configuration")


def test_configured_policy_accepts_mixed_30_and_60hz_mcap(tmp_path: Path) -> None:
    mcap_path = tmp_path / "0.mcap"
    cameras = _write_mixed_rate_episode(mcap_path)

    result = validate_mcap(mcap_path, ValidationPolicy.for_cameras(cameras))

    assert result["output_mcap_passed"]
    camera_check = result["mandatory_checks"]["camera_avg_fps_within_0.5_of_target"]
    assert camera_check["per_topic"][CAMERA_TOPICS["cam_high"]]["target_fps"] == 30
    assert camera_check["per_topic"][CAMERA_TOPICS["cam_left"]]["target_fps"] == 60
    assert camera_check["per_topic"][CAMERA_TOPICS["cam_right"]]["target_fps"] == 60


def test_yam_60hz_camera_rate_accepts_the_new_59hz_minimum() -> None:
    cameras = {role: _camera(role, width=848, height=480, fps=60) for role in CAMERA_TOPICS}
    policy = ValidationPolicy.for_profile(YAM_60HZ_PROFILE, cameras)
    stats = {
        topic: TopicStats(message_count=60, average_hz=59.0, max_gap_ms=16.7)
        for topic in CAMERA_TOPICS.values()
    }

    result = _check_camera_rate(stats, list(CAMERA_TOPICS.values()), policy)

    assert result["passed"]
    assert all(entry["passed"] for entry in result["per_topic"].values())
    assert all(entry["minimum_fps"] == 59.0 for entry in result["per_topic"].values())
    assert all(entry["maximum_fps"] == 60.5 for entry in result["per_topic"].values())


def test_yam_60hz_camera_rate_rejects_below_59hz_and_above_the_existing_upper_bound() -> None:
    cameras = {role: _camera(role, width=848, height=480, fps=60) for role in CAMERA_TOPICS}
    policy = ValidationPolicy.for_profile(YAM_60HZ_PROFILE, cameras)
    topics = list(CAMERA_TOPICS.values())

    below = _check_camera_rate(
        {topic: TopicStats(message_count=60, average_hz=58.999, max_gap_ms=16.7) for topic in topics},
        topics,
        policy,
    )
    above = _check_camera_rate(
        {topic: TopicStats(message_count=60, average_hz=60.501, max_gap_ms=16.7) for topic in topics},
        topics,
        policy,
    )

    assert not below["passed"]
    assert not above["passed"]


def test_legacy_60hz_strict_and_30hz_camera_rate_bounds_are_unchanged() -> None:
    cameras_60 = {role: _camera(role, width=848, height=480, fps=60) for role in CAMERA_TOPICS}
    strict = ValidationPolicy.for_profile(YAM_60HZ_STRICT_PROFILE, cameras_60)
    cameras_30 = {role: _camera(role, fps=30) for role in CAMERA_TOPICS}
    thirty = ValidationPolicy.for_cameras(cameras_30)
    topics = list(CAMERA_TOPICS.values())

    strict_result = _check_camera_rate(
        {topic: TopicStats(message_count=60, average_hz=59.49, max_gap_ms=16.7) for topic in topics},
        topics,
        strict,
    )
    thirty_result = _check_camera_rate(
        {topic: TopicStats(message_count=30, average_hz=29.49, max_gap_ms=33.4) for topic in topics},
        topics,
        thirty,
    )

    assert not strict_result["passed"]
    assert not thirty_result["passed"]


def test_configured_policy_rejects_a_camera_info_resolution_that_differs_from_the_profile(tmp_path: Path) -> None:
    mcap_path = tmp_path / "0.mcap"
    cameras = _write_mixed_rate_episode(mcap_path)
    expected_cameras = {
        **cameras,
        "cam_left": _camera("cam_left", width=640, height=480, fps=60),
    }

    result = validate_mcap(mcap_path, ValidationPolicy.for_cameras(expected_cameras))

    assert not result["output_mcap_passed"]
    assert "camera_resolution_matches_config" in result["failed_checks"]
    check = result["mandatory_checks"]["camera_resolution_matches_config"]
    assert check["per_topic"][CAMERA_TOPICS["cam_left"]]["observed_resolutions"] == [[848, 480]]
    assert not check["per_topic"][CAMERA_TOPICS["cam_left"]]["passed"]


def test_follower_state_requires_velocity_and_effort_for_every_joint(tmp_path: Path) -> None:
    mcap_path = tmp_path / "0.mcap"
    malformed = RobotSnapshot(
        role="left_follower",
        source_timestamp_ns=1,
        sequence=1,
        position=(0.0,) * 7,
        velocity=(),
        effort=(),
    )

    with EpisodeMcapWriter(mcap_path, {"task_name": "test"}) as writer:
        try:
            writer.write_joint_state(malformed, 1)
        except ValueError as exc:
            assert "velocity and effort" in str(exc)
        else:  # pragma: no cover - makes the expected writer invariant explicit
            raise AssertionError("malformed follower state was accepted")
