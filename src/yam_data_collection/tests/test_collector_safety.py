from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import numpy as np
import yaml

from yam_data_collection.collector import CaptureError, CaptureState, EpisodeCollector, _FrameTiming
from yam_data_collection.models import CameraFrame, RobotSnapshot
from yam_data_collection.telemetry import TelemetryError
from yam_data_collection.validation_profiles import YAM_30HZ_PROFILE, YAM_60HZ_PROFILE, YAM_60HZ_STRICT_PROFILE


def _snapshot(sequence: int, source_timestamp_ns: int) -> RobotSnapshot:
    return RobotSnapshot(
        role="left_leader",
        source_timestamp_ns=source_timestamp_ns,
        sequence=sequence,
        position=(0.0,) * 7,
        velocity=(),
        effort=(),
    )


def _collector_for_progress(timeout_s: float) -> EpisodeCollector:
    collector = object.__new__(EpisodeCollector)
    collector.config = SimpleNamespace(collection=SimpleNamespace(timeout_threshold_s=timeout_s))
    collector._telemetry_progress = {}
    return collector


def _set_healthy_camera_arrivals(
    collector: EpisodeCollector, roles: tuple[str, ...], fps_by_role: dict[str, int] | None = None
) -> None:
    now_ns = time.monotonic_ns()
    collector._camera_arrival_lock = threading.Lock()
    collector._camera_arrivals = {
        role: deque(
            now_ns - offset * round(1_000_000_000 / (fps_by_role or {}).get(role, 30))
            for offset in range(2 * (fps_by_role or {}).get(role, 30) - 1, -1, -1)
        )
        for role in roles
    }


def test_replayed_telemetry_is_rejected_after_timeout() -> None:
    collector = _collector_for_progress(timeout_s=0.01)
    snapshots = {"left_leader": _snapshot(sequence=5, source_timestamp_ns=10)}

    collector._validate_telemetry_progress(snapshots)
    time.sleep(0.02)

    with pytest.raises(TelemetryError, match="stopped advancing"):
        collector._validate_telemetry_progress(snapshots)


def test_regressing_telemetry_sequence_is_rejected() -> None:
    collector = _collector_for_progress(timeout_s=1.0)
    collector._validate_telemetry_progress({"left_leader": _snapshot(sequence=5, source_timestamp_ns=10)})

    with pytest.raises(TelemetryError, match="moved backwards"):
        collector._validate_telemetry_progress({"left_leader": _snapshot(sequence=4, source_timestamp_ns=11)})


def test_episode_telemetry_timing_records_a_rejected_source_gap_before_it_is_written() -> None:
    collector = object.__new__(EpisodeCollector)
    collector.config = SimpleNamespace(
        cameras={},
        collection=SimpleNamespace(validation_profile="yam_60hz"),
    )
    episode = SimpleNamespace(
        last_state_sample_monotonic_ns=1_000_000_000,
        max_state_sample_gap_ns=0,
        telemetry_last_source_timestamp_ns={"left_leader": 2_000_000_000},
        telemetry_last_sequence={"left_leader": 4},
        telemetry_last_progress_monotonic_ns={"left_leader": 1_000_000_000},
    )

    error = collector._episode_telemetry_timing_error(
        episode,
        {"left_leader": _snapshot(sequence=5, source_timestamp_ns=2_050_000_000)},
        sample_monotonic_ns=1_005_000_000,
    )

    assert error is not None
    assert "source gap" in str(error)
    assert episode.telemetry_max_source_gap_ns == {"left_leader": 50_000_000}


def test_episode_telemetry_timing_records_a_rejected_state_gap() -> None:
    collector = object.__new__(EpisodeCollector)
    collector.config = SimpleNamespace(
        cameras={},
        collection=SimpleNamespace(validation_profile="yam_60hz"),
    )
    episode = SimpleNamespace(
        episode_id=9,
        state_samples=123,
        camera_frames={},
        last_state_sample_monotonic_ns=1_000_000_000,
        max_state_sample_gap_ns=10_000_000,
    )

    error = collector._episode_telemetry_timing_error(
        episode,
        {},
        sample_monotonic_ns=1_034_100_000,
    )

    assert error is not None
    assert "state sampling gap" in str(error)
    assert episode.max_state_sample_gap_ns == 34_100_000
    collector._state_lock = threading.RLock()
    collector._background_error = error
    collector._last_episode_diagnostics = None
    collector._remember_episode_diagnostics(episode)
    assert collector.last_episode_diagnostics is not None
    assert collector.last_episode_diagnostics["max_state_sample_gap_ms"] == pytest.approx(34.1)


def test_recorder_affinity_applies_the_configured_cpu_set(monkeypatch: pytest.MonkeyPatch) -> None:
    collector = object.__new__(EpisodeCollector)
    collector.config = SimpleNamespace(collection=SimpleNamespace(recorder_cpu_set="16-19"))
    collector._effective_recorder_cpu_set = None
    current = {16, 17, 18, 19, 20, 21, 22, 23}

    monkeypatch.setattr("yam_data_collection.collector.os.sched_getaffinity", lambda _pid: set(current))

    def set_affinity(_pid: int, cpus: set[int]) -> None:
        current.clear()
        current.update(cpus)

    monkeypatch.setattr("yam_data_collection.collector.os.sched_setaffinity", set_affinity)

    collector._configure_recorder_affinity()

    assert current == {16, 17, 18, 19}
    assert collector._effective_recorder_cpu_set == (16, 17, 18, 19)


def test_recorder_affinity_rejects_encoder_cpus_outside_process_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = object.__new__(EpisodeCollector)
    collector.config = SimpleNamespace(
        collection=SimpleNamespace(recorder_cpu_set="16-19", live_video_cpu_set="20-23")
    )
    collector._effective_recorder_cpu_set = None
    current = {16, 17, 18, 19}

    monkeypatch.setattr("yam_data_collection.collector.os.sched_getaffinity", lambda _pid: set(current))

    with pytest.raises(CaptureError, match="live_video_cpu_set is outside"):
        collector._configure_recorder_affinity()

    assert current == {16, 17, 18, 19}


def test_state_recorder_preserves_source_telemetry_timestamp() -> None:
    class FakeMcap:
        def __init__(self) -> None:
            self.calls: list[tuple[RobotSnapshot, int, int | None]] = []

        def write_joint_state(
            self,
            snapshot: RobotSnapshot,
            log_time_ns: int,
            *,
            header_timestamp_ns: int | None = None,
        ) -> None:
            self.calls.append((snapshot, log_time_ns, header_timestamp_ns))

    source_timestamp_ns = time.time_ns() - 20_000_000
    snapshot = _snapshot(sequence=5, source_timestamp_ns=source_timestamp_ns)
    mcap = FakeMcap()
    collector = object.__new__(EpisodeCollector)
    collector._state = CaptureState.RECORDING
    collector._state_lock = threading.RLock()
    collector._background_error = None
    collector._last_telemetry_success_ns = 0
    collector._episode = SimpleNamespace(
        mcap=mcap,
        state_samples=0,
        active_operations=0,
        operations_drained=threading.Event(),
    )
    collector._episode.operations_drained.set()
    collector.config = SimpleNamespace(
        telemetry={"left_leader": object()},
        collection=SimpleNamespace(total_sampled_frames=100),
    )
    collector._read_snapshots = lambda require_fresh: {"left_leader": snapshot}

    collector._record_state_sample()

    assert collector._episode.state_samples == 1
    assert len(mcap.calls) == 1
    recorded_snapshot, log_time_ns, header_timestamp_ns = mcap.calls[0]
    assert recorded_snapshot is snapshot
    assert log_time_ns >= source_timestamp_ns
    assert header_timestamp_ns == source_timestamp_ns


def test_joint_preview_failure_does_not_change_recorded_state() -> None:
    class FakeMcap:
        def __init__(self) -> None:
            self.calls = 0

        def write_joint_state(
            self,
            snapshot: RobotSnapshot,
            log_time_ns: int,
            *,
            header_timestamp_ns: int | None = None,
        ) -> None:
            del snapshot, log_time_ns, header_timestamp_ns
            self.calls += 1

    class RaisingPreview:
        def submit_joints(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            raise RuntimeError("preview window closed")

    snapshot = _snapshot(sequence=5, source_timestamp_ns=time.time_ns() - 20_000_000)
    mcap = FakeMcap()
    collector = object.__new__(EpisodeCollector)
    collector._state = CaptureState.RECORDING
    collector._state_lock = threading.RLock()
    collector._background_error = None
    collector._preview_error_reported = False
    collector._last_telemetry_success_ns = 0
    collector._live_preview = RaisingPreview()
    collector._episode = SimpleNamespace(
        mcap=mcap,
        state_samples=0,
        active_operations=0,
        operations_drained=threading.Event(),
    )
    collector._episode.operations_drained.set()
    collector._print = lambda _message: None
    collector.config = SimpleNamespace(
        telemetry={"left_leader": object()},
        collection=SimpleNamespace(total_sampled_frames=100),
    )
    collector._read_snapshots = lambda require_fresh: {"left_leader": snapshot}

    collector._record_state_sample()

    assert mcap.calls == 1
    assert collector._episode.state_samples == 1
    assert collector._background_error is None


def test_camera_preview_failure_does_not_change_recorded_video() -> None:
    class FakeWriter:
        def __init__(self) -> None:
            self.accepted = 0

        def enqueue(self, image: np.ndarray, **kwargs: object) -> bool:
            del image, kwargs
            self.accepted += 1
            return True

    class FakeMcap:
        def write_camera_info(self, camera: object, timestamp_ns: int) -> None:
            del camera, timestamp_ns

    class RaisingPreview:
        def submit_camera(self, frame: CameraFrame) -> None:
            del frame
            raise RuntimeError("preview process stopped")

    writer = FakeWriter()
    operations_drained = threading.Event()
    operations_drained.set()
    collector = object.__new__(EpisodeCollector)
    collector._state = CaptureState.RECORDING
    collector._state_lock = threading.RLock()
    collector._background_error = None
    collector._preview_error_reported = False
    collector._live_preview = RaisingPreview()
    collector._camera_arrival_lock = threading.Lock()
    collector._camera_arrivals = {"cam_high": deque()}
    collector._print = lambda _message: None
    collector.config = SimpleNamespace(cameras={"cam_high": object()})
    collector._episode = SimpleNamespace(
        videos={"cam_high": writer},
        mcap=FakeMcap(),
        camera_frames={},
        camera_timing={"cam_high": _FrameTiming()},
        camera_host_timing={"cam_high": _FrameTiming()},
        camera_timing_samples={"cam_high": []},
        camera_start_timestamp_ns=0,
        active_operations=0,
        operations_drained=operations_drained,
    )

    collector._on_camera_frame(CameraFrame("cam_high", 1, np.zeros((1, 1, 3), dtype=np.uint8)))

    assert writer.accepted == 1
    assert collector._episode.camera_frames == {"cam_high": 1}
    assert collector._background_error is None


def test_preview_cpu_set_excludes_recorder_and_encoder_masks() -> None:
    collector = object.__new__(EpisodeCollector)
    collector._allowed_cpu_set = tuple(range(24))
    collector._effective_recorder_cpu_set = (16, 17, 18, 19)
    collector.config = SimpleNamespace(
        collection=SimpleNamespace(
            recorder_cpu_set="16-19",
            live_video_cpu_set="20-23",
        )
    )

    assert collector._available_preview_cpu_set() == tuple(range(16))


def test_preview_cpu_set_uses_allowed_mask_when_recorder_is_unpinned() -> None:
    collector = object.__new__(EpisodeCollector)
    collector._allowed_cpu_set = tuple(range(24))
    collector._effective_recorder_cpu_set = tuple(range(24))
    collector.config = SimpleNamespace(
        collection=SimpleNamespace(recorder_cpu_set=None, live_video_cpu_set=None)
    )

    assert collector._available_preview_cpu_set() == tuple(range(24))


def test_preview_can_share_recorder_mask_when_no_cpu_is_free() -> None:
    collector = object.__new__(EpisodeCollector)
    collector._allowed_cpu_set = tuple(range(16, 24))
    collector._effective_recorder_cpu_set = (16, 17, 18, 19)
    collector.config = SimpleNamespace(
        collection=SimpleNamespace(
            recorder_cpu_set="16-19",
            live_video_cpu_set="20-23",
        )
    )

    assert collector._available_preview_cpu_set() == ()


def test_idle_telemetry_is_forwarded_to_joint_preview() -> None:
    snapshot = _snapshot(sequence=1, source_timestamp_ns=2_000_000_000)
    submitted: list[tuple[dict[str, RobotSnapshot], int]] = []
    collector = object.__new__(EpisodeCollector)
    collector._state_lock = threading.RLock()
    collector._background_error = None
    collector._last_telemetry_success_ns = 0
    collector._read_snapshots = lambda require_fresh: {"left_leader": snapshot}
    collector._begin_episode_operation = lambda: None
    collector._submit_preview_joints = lambda snapshots, timestamp_ns: submitted.append(
        (snapshots, timestamp_ns)
    )

    collector._record_state_sample()

    assert len(submitted) == 1
    assert submitted[0][0] == {"left_leader": snapshot}
    assert submitted[0][1] > 0


def test_camera_frames_wait_for_the_first_joint_timestamp_watermark() -> None:
    class FakeMcap:
        def __init__(self) -> None:
            self.joint_calls: list[tuple[RobotSnapshot, int, int | None]] = []
            self.camera_timestamps: list[int] = []

        def write_joint_state(
            self,
            snapshot: RobotSnapshot,
            log_time_ns: int,
            *,
            header_timestamp_ns: int | None = None,
        ) -> None:
            self.joint_calls.append((snapshot, log_time_ns, header_timestamp_ns))

        def write_camera_info(self, camera: object, timestamp_ns: int) -> None:
            del camera
            self.camera_timestamps.append(timestamp_ns)

    class FakeWriter:
        def __init__(self) -> None:
            self.accepted = 0

        def enqueue(self, image: np.ndarray, **kwargs: object) -> bool:
            del image, kwargs
            self.accepted += 1
            return True

    mcap = FakeMcap()
    writer = FakeWriter()
    operations_drained = threading.Event()
    operations_drained.set()
    snapshots = {
        "left_leader": _snapshot(sequence=1, source_timestamp_ns=2_003),
        "right_leader": RobotSnapshot(
            role="right_leader",
            source_timestamp_ns=2_000,
            sequence=1,
            position=(0.0,) * 7,
            velocity=(),
            effort=(),
        ),
        "left_follower": RobotSnapshot(
            role="left_follower",
            source_timestamp_ns=2_001,
            sequence=1,
            position=(0.0,) * 7,
            velocity=(0.0,) * 7,
            effort=(0.0,) * 7,
        ),
        "right_follower": RobotSnapshot(
            role="right_follower",
            source_timestamp_ns=2_002,
            sequence=1,
            position=(0.0,) * 7,
            velocity=(0.0,) * 7,
            effort=(0.0,) * 7,
        ),
    }
    collector = object.__new__(EpisodeCollector)
    collector._state = CaptureState.RECORDING
    collector._state_lock = threading.RLock()
    collector._background_error = None
    collector._last_telemetry_success_ns = 0
    collector._camera_arrival_lock = threading.Lock()
    collector._camera_arrivals = {"cam_high": deque()}
    collector.config = SimpleNamespace(
        cameras={"cam_high": object()},
        telemetry={role: object() for role in snapshots},
        collection=SimpleNamespace(total_sampled_frames=100),
    )
    collector._episode = SimpleNamespace(
        mcap=mcap,
        videos={"cam_high": writer},
        state_samples=0,
        camera_frames={},
        camera_timing={"cam_high": _FrameTiming()},
        camera_host_timing={"cam_high": _FrameTiming()},
        camera_timing_samples={"cam_high": []},
        camera_start_timestamp_ns=None,
        active_operations=0,
        operations_drained=operations_drained,
    )
    collector._read_snapshots = lambda require_fresh: snapshots

    # The first camera callback races ahead of the first 180 Hz state sample.
    collector._on_camera_frame(CameraFrame("cam_high", 1_999, np.zeros((1, 1, 3), dtype=np.uint8)))
    assert writer.accepted == 0
    assert mcap.camera_timestamps == []

    collector._record_state_sample()
    assert collector._episode.camera_start_timestamp_ns == 2_000
    assert len(mcap.joint_calls) == 4

    # The boundary frame is pairable with the initial joint snapshot and is
    # retained. A stale native-queue frame remains excluded.
    collector._on_camera_frame(CameraFrame("cam_high", 1_999, np.zeros((1, 1, 3), dtype=np.uint8)))
    collector._on_camera_frame(CameraFrame("cam_high", 2_000, np.zeros((1, 1, 3), dtype=np.uint8)))

    assert writer.accepted == 1
    assert mcap.camera_timestamps == [2_000]
    assert collector._episode.camera_frames == {"cam_high": 1}


def test_camera_frames_remain_blocked_when_initial_joint_write_fails() -> None:
    class FailingMcap:
        def __init__(self) -> None:
            self.joint_write_count = 0

        def write_joint_state(
            self,
            snapshot: RobotSnapshot,
            log_time_ns: int,
            *,
            header_timestamp_ns: int | None = None,
        ) -> None:
            del snapshot, log_time_ns, header_timestamp_ns
            self.joint_write_count += 1
            if self.joint_write_count == 2:
                raise RuntimeError("synthetic joint write failure")

    class ProbeWriter:
        def __init__(self) -> None:
            self.enqueue_count = 0

        def enqueue(self, image: np.ndarray, **kwargs: object) -> bool:
            del image, kwargs
            self.enqueue_count += 1
            return True

    snapshots = {
        "left_leader": _snapshot(sequence=1, source_timestamp_ns=2_000),
        "right_leader": RobotSnapshot(
            role="right_leader",
            source_timestamp_ns=2_001,
            sequence=1,
            position=(0.0,) * 7,
            velocity=(),
            effort=(),
        ),
    }
    mcap = FailingMcap()
    writer = ProbeWriter()
    operations_drained = threading.Event()
    operations_drained.set()
    collector = object.__new__(EpisodeCollector)
    collector._state = CaptureState.RECORDING
    collector._state_lock = threading.RLock()
    collector._background_error = None
    collector._last_telemetry_success_ns = 0
    collector._camera_arrival_lock = threading.Lock()
    collector._camera_arrivals = {"cam_high": deque()}
    collector.config = SimpleNamespace(
        cameras={"cam_high": object()},
        telemetry={role: object() for role in snapshots},
        collection=SimpleNamespace(total_sampled_frames=100),
    )
    collector._episode = SimpleNamespace(
        mcap=mcap,
        videos={"cam_high": writer},
        state_samples=0,
        camera_frames={},
        camera_timing={"cam_high": _FrameTiming()},
        camera_host_timing={"cam_high": _FrameTiming()},
        camera_timing_samples={"cam_high": []},
        camera_start_timestamp_ns=None,
        active_operations=0,
        operations_drained=operations_drained,
    )
    collector._read_snapshots = lambda require_fresh: snapshots
    collector._discard_for_background_error = lambda error: None

    collector._record_state_sample()
    collector._on_camera_frame(CameraFrame("cam_high", 2_001, np.zeros((1, 1, 3), dtype=np.uint8)))

    assert collector._episode.camera_start_timestamp_ns is None
    assert collector._episode.state_samples == 0
    assert writer.enqueue_count == 0


def _arming_collector(*, max_queue_depth: int = 1) -> EpisodeCollector:
    class FakeWriter:
        def diagnostic_snapshot(self) -> dict[str, object]:
            return {
                "queue_depth": 0,
                "queue_capacity": 120,
                "max_queue_depth": max_queue_depth,
                "max_pipe_write_ms": 8.0,
                "dropped_frames": 0,
                "process_returncode": None,
                "writer_error": None,
            }

    collector = object.__new__(EpisodeCollector)
    collector._state = CaptureState.ARMING
    collector._state_lock = threading.RLock()
    collector._background_error = None
    collector._finalization_done = threading.Event()
    collector._finalization_done.set()
    collector._print = lambda _message: None
    collector._ARMING_DURATION_S = 0.0
    collector.config = SimpleNamespace(
        cameras={role: object() for role in ("cam_high", "cam_left", "cam_right")},
        collection=SimpleNamespace(validation_profile="yam_60hz"),
    )
    collector._episode = SimpleNamespace(
        episode_id=9,
        state_samples=20,
        recording_state_samples=0,
        max_state_sample_gap_ns=10_000_000,
        telemetry_max_source_gap_ns={role: 10_000_000 for role in ("left_leader", "right_leader")},
        camera_frames={role: 20 for role in ("cam_high", "cam_left", "cam_right")},
        videos={role: FakeWriter() for role in ("cam_high", "cam_left", "cam_right")},
    )
    return collector


def test_arming_enters_recording_only_after_a_healthy_full_load_window() -> None:
    collector = _arming_collector()

    collector._wait_for_episode_arming(9)

    assert collector.state is CaptureState.RECORDING
    assert collector._episode is not None
    assert collector._episode.recording_started_at_ns > 0


def test_recording_duration_excludes_arming_and_is_frozen_before_finalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = object.__new__(EpisodeCollector)
    collector._state = CaptureState.RECORDING
    collector._state_lock = threading.RLock()
    collector._finalization_done = threading.Event()
    episode = SimpleNamespace(
        started_at_ns=1_000_000_000,
        recording_started_at_ns=4_000_000_000,
        recording_stopped_at_ns=None,
    )
    collector._episode = episode
    now_ns = 14_000_000_000
    monkeypatch.setattr("yam_data_collection.collector.time.time_ns", lambda: now_ns)

    detached = collector._detach_active_episode()

    assert detached is episode
    assert episode.recording_stopped_at_ns == now_ns
    assert collector._recording_duration_s(episode) == 10.0
    # Simulate a long encoder/finalization phase. The displayed duration must
    # remain pinned to the detach boundary rather than continuing to increase.
    assert collector._recording_duration_s(episode, now_ns=114_000_000_000) == 10.0


def test_recording_duration_never_falls_back_to_episode_creation_time() -> None:
    episode = SimpleNamespace(
        started_at_ns=1_000_000_000,
        recording_started_at_ns=None,
        recording_stopped_at_ns=None,
    )

    assert EpisodeCollector._recording_duration_s(episode, now_ns=101_000_000_000) == 0.0


def test_arming_does_not_reject_a_short_window_camera_rate() -> None:
    collector = _arming_collector()
    collector._episode.camera_timing = {
        role: SimpleNamespace(duration_s=2.0, average_fps=59.39, max_gap_ms=16.8)
        for role in ("cam_high", "cam_left", "cam_right")
    }

    collector._wait_for_episode_arming(9)

    assert collector.state is CaptureState.RECORDING


def test_arming_discards_when_the_live_encoder_queue_grows() -> None:
    collector = _arming_collector(max_queue_depth=3)
    discarded: list[bool] = []
    collector.discard_episode = lambda: discarded.append(True) or True

    with pytest.raises(CaptureError, match="arming health check failed"):
        collector._wait_for_episode_arming(9)

    assert discarded == [True]


def test_next_episode_id_skips_interrupted_episode_artifacts(tmp_path: Path) -> None:
    (tmp_path / "0.mcap").touch()
    (tmp_path / "1.partial.mcap").touch()
    (tmp_path / "videos" / "2").mkdir(parents=True)
    collector = object.__new__(EpisodeCollector)
    collector.config = SimpleNamespace(collection=SimpleNamespace(dataset_dir=tmp_path))

    assert collector._next_episode_id() == 3


def test_record_config_persists_the_resolved_validation_profile(tmp_path: Path) -> None:
    collector = object.__new__(EpisodeCollector)
    collector.config = SimpleNamespace(
        raw={"collection_config": {"dataset_dir": "data/example"}},
        collection=SimpleNamespace(dataset_dir=tmp_path, validation_profile=YAM_60HZ_STRICT_PROFILE),
        cameras={},
    )
    collector._print = lambda message: None

    collector._prepare_dataset_dir()

    saved_config = yaml.safe_load((tmp_path / "record_config.yaml").read_text(encoding="utf-8"))
    assert saved_config["collection_config"]["validation_profile"] == YAM_60HZ_STRICT_PROFILE


def test_start_episode_rejects_uncalibrated_capture_config() -> None:
    collector = object.__new__(EpisodeCollector)
    collector._state = CaptureState.IDLE
    collector._state_lock = threading.RLock()
    collector._background_error = None
    collector.config = SimpleNamespace(
        cameras={"cam_high": SimpleNamespace(intrinsics=SimpleNamespace(is_calibrated=False))},
        collection=SimpleNamespace(allow_uncalibrated_intrinsics=False),
    )

    with pytest.raises(RuntimeError, match="cannot record without calibrated intrinsics"):
        collector.start_episode()


def test_background_error_property_exposes_worker_failure_safely() -> None:
    collector = object.__new__(EpisodeCollector)
    collector._state_lock = threading.RLock()
    collector._background_error = CaptureError("video queue overflow on cam_right")

    assert str(collector.background_error) == "video queue overflow on cam_right"


def test_idle_snapshot_keeps_final_failure_diagnostics_for_runtime_monitoring() -> None:
    collector = object.__new__(EpisodeCollector)
    collector._state = CaptureState.IDLE
    collector._state_lock = threading.RLock()
    collector._background_error = CaptureError("state sampling gap was 34.1 ms")
    collector._episode = None
    collector._last_episode_diagnostics = {
        "max_state_sample_gap_ms": 34.1,
        "telemetry_max_source_gap_ms": {},
    }

    snapshot = collector.active_episode_snapshot

    assert snapshot["background_error"] == "state sampling gap was 34.1 ms"
    assert snapshot["last_episode_diagnostics"] == {
        "max_state_sample_gap_ms": 34.1,
        "telemetry_max_source_gap_ms": {},
    }
    snapshot["last_episode_diagnostics"]["max_state_sample_gap_ms"] = 0.0  # type: ignore[index]
    assert collector._last_episode_diagnostics["max_state_sample_gap_ms"] == 34.1


def test_active_episode_snapshot_copies_live_counters_without_mutating_the_episode() -> None:
    class FakeWriter:
        def diagnostic_snapshot(self) -> dict[str, object]:
            return {"queue_depth": 3, "queue_capacity": 120, "max_pipe_write_ms": 9.5}

    writer = FakeWriter()
    collector = object.__new__(EpisodeCollector)
    collector._state = CaptureState.RECORDING
    collector._state_lock = threading.RLock()
    collector._background_error = None
    collector._episode = SimpleNamespace(
        episode_id=12,
        started_at_ns=1_700_000_000_000_000_000,
        state_samples=456,
        camera_frames={"cam_high": 75, "cam_left": 74},
        videos={"cam_high": writer},
        active_operations=1,
    )

    snapshot = collector.active_episode_snapshot

    assert snapshot["state"] == "recording"
    assert snapshot["episode_id"] == 12
    assert snapshot["state_samples"] == 456
    assert snapshot["camera_frames"] == {"cam_high": 75, "cam_left": 74}
    assert snapshot["video_writers"]["cam_high"]["queue_depth"] == 3
    snapshot["camera_frames"]["cam_high"] = 0
    assert collector._episode.camera_frames["cam_high"] == 75


def test_detected_camera_failure_cannot_be_saved_before_async_cleanup() -> None:
    collector = object.__new__(EpisodeCollector)
    collector._state = CaptureState.RECORDING
    collector._state_lock = threading.RLock()
    collector._background_error = None
    collector.config = SimpleNamespace(cameras={"cam_high": object()})
    collector._camera_arrival_lock = threading.Lock()
    collector._camera_arrivals = {"cam_high": deque()}
    operations_drained = threading.Event()
    operations_drained.set()
    collector._episode = SimpleNamespace(
        videos={"cam_high": SimpleNamespace(enqueue=lambda image, **kwargs: False)},
        mcap=object(),
        camera_frames={},
        camera_timing={"cam_high": _FrameTiming()},
        camera_start_timestamp_ns=0,
        active_operations=0,
        operations_drained=operations_drained,
    )
    # Hold the episode open long enough to inspect the marker that precedes
    # asynchronous discard/finalization.
    collector._discard_for_background_error = lambda error: None

    collector._on_camera_frame(CameraFrame("cam_high", 1, np.zeros((1, 1, 3), dtype=np.uint8)))

    assert collector._background_error is not None
    commits: list[bool] = []
    collector._detach_active_episode = lambda: object()
    collector._finish_episode = lambda episode, commit: commits.append(commit) or commit
    assert not collector.save_episode()
    assert commits == [False]


def test_saved_episode_callback_can_request_input_shutdown(tmp_path: Path) -> None:
    collector = object.__new__(EpisodeCollector)
    collector._state = CaptureState.RECORDING
    collector._state_lock = threading.RLock()
    collector._background_error = None
    collector._stop_after_saved_episode = threading.Event()
    source = tmp_path / ".pending" / "12" / "12.mcap"
    episode = SimpleNamespace(episode_id=12, final_mcap_path=source)
    collector._episode = episode
    received: list[Path] = []
    collector._on_episode_saved = lambda path: received.append(path) or True
    collector._detach_active_episode = lambda: episode
    collector._finish_episode = lambda candidate, commit: candidate is episode and commit

    assert collector.save_episode()

    assert received == [source]
    assert collector._stop_after_saved_episode.is_set()


def test_saved_episode_callback_failure_preserves_successful_save(tmp_path: Path) -> None:
    collector = object.__new__(EpisodeCollector)
    collector._state = CaptureState.RECORDING
    collector._state_lock = threading.RLock()
    collector._background_error = None
    collector._stop_after_saved_episode = threading.Event()
    collector._print = lambda message: printed.append(message)
    episode = SimpleNamespace(
        episode_id=13,
        final_mcap_path=tmp_path / ".pending" / "13" / "13.mcap",
    )
    collector._episode = episode
    collector._detach_active_episode = lambda: episode
    collector._finish_episode = lambda candidate, commit: candidate is episode and commit
    collector._on_episode_saved = lambda _path: (_ for _ in ()).throw(RuntimeError("assessment failed"))
    printed: list[str] = []

    assert collector.save_episode()

    assert not collector._stop_after_saved_episode.is_set()
    assert "raw episode remains saved: assessment failed" in printed[0]


def test_failed_post_validation_is_deleted_automatically(tmp_path: Path) -> None:
    pending = tmp_path / ".pending" / "4" / "4.mcap"
    pending.parent.mkdir(parents=True)
    pending.write_bytes(b"mcap")
    live = tmp_path / "videos_live" / "4"
    live.mkdir(parents=True)
    (live / "cam_high.mp4").write_bytes(b"video")

    collector = object.__new__(EpisodeCollector)
    collector._state = CaptureState.RECORDING
    collector._state_lock = threading.RLock()
    collector.config = SimpleNamespace(collection=SimpleNamespace(dataset_dir=tmp_path))
    collector._background_error = None
    collector._auto_delete_failed_validation = True
    collector._last_post_validation = {
        "passed": False,
        "failed_checks": ["camera_max_gap_below_69ms"],
    }
    collector._last_saved_episode_path = None
    collector._last_saved_episode_id = None
    collector._last_saved_episode_duration_s = None
    printed: list[str] = []
    collector._print = lambda message, **kwargs: printed.append(message)
    episode = SimpleNamespace(
        episode_id=4,
        final_mcap_path=pending,
        committed_mcap_path=None,
        recording_started_at_ns=1_000_000_000,
        recording_stopped_at_ns=13_500_000_000,
    )
    collector._episode = episode
    collector._detach_active_episode = lambda: episode

    def finish(candidate: object, *, commit: bool) -> bool:
        assert candidate is episode
        assert commit
        collector._state = CaptureState.IDLE
        return True

    collector._finish_episode = finish

    assert not collector.save_episode()
    assert not pending.parent.exists()
    assert not live.exists()
    assert any("deleting it automatically" in message for message in printed)
    duration_log = (tmp_path / "deleted_episode_durations.txt").read_text(encoding="utf-8")
    assert "\t4\tvalidation_failed\t12.500\t12.500\t12.500\n" in duration_log


@pytest.mark.parametrize("key", ["q", "\n"])
def test_post_save_review_q_deletes_and_enter_keeps_episode(tmp_path: Path, key: str) -> None:
    episode_root = tmp_path / "7"
    mcap = episode_root / "7.mcap"
    mcap.parent.mkdir(parents=True)
    mcap.write_bytes(b"mcap")
    (episode_root / "videos").mkdir()

    class Keys:
        def poll(self, timeout_s: float) -> str:
            del timeout_s
            return key

    collector = object.__new__(EpisodeCollector)
    collector._state = CaptureState.IDLE
    collector._state_lock = threading.RLock()
    collector.config = SimpleNamespace(collection=SimpleNamespace(dataset_dir=tmp_path))
    collector._last_post_validation = {"passed": True}
    collector._last_saved_episode_path = mcap
    collector._last_saved_episode_id = 7
    collector._last_saved_episode_duration_s = 23.75
    collector._print = lambda message, **kwargs: None

    collector._review_saved_episode(Keys())

    assert episode_root.exists() is (key == "\n")
    duration_log = tmp_path / "deleted_episode_durations.txt"
    if key == "q":
        assert "\t7\tuser_deleted\t23.750\t23.750\t23.750\n" in duration_log.read_text(encoding="utf-8")
    else:
        assert not duration_log.exists()


def test_active_recording_error_duration_is_recorded_after_discard(tmp_path: Path) -> None:
    collector = object.__new__(EpisodeCollector)
    collector.config = SimpleNamespace(collection=SimpleNamespace(dataset_dir=tmp_path))
    collector._state = CaptureState.RECORDING
    collector._state_lock = threading.RLock()
    collector._finalization_done = threading.Event()
    collector._episode = SimpleNamespace(
        episode_id=8,
        recording_started_at_ns=2_000_000_000,
        recording_stopped_at_ns=None,
    )
    collector._print = lambda message, **kwargs: None

    def finish(candidate: object, *, commit: bool) -> bool:
        assert candidate.episode_id == 8
        assert not commit
        collector._state = CaptureState.IDLE
        return False

    collector._finish_episode = finish
    with patch("yam_data_collection.collector.time.time_ns", return_value=12_250_000_000):
        assert collector.discard_episode(deletion_reason="recording_error")

    duration_log = (tmp_path / "deleted_episode_durations.txt").read_text(encoding="utf-8")
    assert "\t8\trecording_error\t10.250\t10.250\t10.250\n" in duration_log


def test_deleted_duration_log_accumulates_reason_and_overall_totals(tmp_path: Path) -> None:
    collector = object.__new__(EpisodeCollector)
    collector.config = SimpleNamespace(collection=SimpleNamespace(dataset_dir=tmp_path))
    collector._print = lambda message, **kwargs: None

    collector._record_deleted_episode_duration(1, 10.0, "validation_failed")
    collector._record_deleted_episode_duration(2, 2.5, "user_deleted")
    collector._record_deleted_episode_duration(3, 4.0, "validation_failed")

    lines = (tmp_path / "deleted_episode_durations.txt").read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("deleted_at\tepisode_id\treason\tduration_seconds")
    assert lines[-1].endswith("\t3\tvalidation_failed\t4.000\t14.000\t16.500")


def test_last_episode_diagnostics_preserve_writer_stats_after_discard(tmp_path: Path) -> None:
    class FakeMcap:
        def close(self) -> None:
            return None

    class FakeWriter:
        def __init__(self) -> None:
            self.closed = False

        def diagnostic_snapshot(self) -> dict[str, object]:
            return {
                "queue_capacity": 120,
                "max_queue_depth": 120,
                "accepted_frames": 42,
                "dropped_frames": 1,
                "closed": self.closed,
            }

        def close(self, commit: bool) -> None:
            assert not commit
            self.closed = True

    writer = FakeWriter()
    collector = object.__new__(EpisodeCollector)
    collector._state = CaptureState.FINALIZING
    collector._state_lock = threading.RLock()
    collector._background_error = CaptureError("video queue overflow on cam_left")
    collector._last_episode_diagnostics = None
    collector._finalization_done = threading.Event()
    collector.config = SimpleNamespace(cameras={"cam_left": object()})
    pending_dir = tmp_path / "pending"
    episode = SimpleNamespace(
        episode_id=7,
        state_samples=123,
        camera_frames={"cam_left": 42},
        videos={"cam_left": writer},
        operations_drained=threading.Event(),
        mcap=FakeMcap(),
        partial_mcap_path=pending_dir / "7.partial.mcap",
        final_mcap_path=pending_dir / "7.mcap",
        pending_dir=pending_dir,
        video_dir=tmp_path / "video.partial",
    )
    episode.operations_drained.set()

    assert not collector._finish_episode(episode, commit=False)

    diagnostics = collector.last_episode_diagnostics
    assert diagnostics is not None
    assert diagnostics["episode_id"] == 7
    before = diagnostics["video_writers_before_close"]["cam_left"]
    after = diagnostics["video_writers_after_close"]["cam_left"]
    assert before["dropped_frames"] == 1
    assert before["closed"] is False
    assert after["closed"] is True
    # A failed episode must not poison the next Space-triggered episode after
    # its asynchronous cleanup has completed.
    assert collector.background_error is None


def test_close_waits_for_an_existing_finalizer() -> None:
    collector = object.__new__(EpisodeCollector)
    collector._state = CaptureState.FINALIZING
    collector._state_lock = threading.RLock()
    collector._finalization_done = threading.Event()
    collector._stop_event = threading.Event()
    collector._cameras = {}
    collector._camera_workers = []
    collector._state_thread = None
    collector._executor = SimpleNamespace(shutdown=lambda **kwargs: None)
    collector._telemetry_clients = {}
    collector.discard_episode = lambda: False
    collector._print = lambda message: None
    collector._FINALIZATION_WAIT_S = 1.0
    threading.Timer(0.02, collector._finalization_done.set).start()

    started = time.monotonic()
    collector.close()

    assert time.monotonic() - started >= 0.01
    assert collector.state is CaptureState.CLOSED


def test_start_episode_closes_mcap_when_video_setup_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakeMcap:
        instance: "FakeMcap | None" = None

        def __init__(self, path: Path, task_info: object) -> None:
            del task_info
            self.path = path
            self.closed = False
            path.touch()
            FakeMcap.instance = self

        def close(self) -> None:
            self.closed = True

    class FakeWriter:
        instances: list["FakeWriter"] = []
        fps_values: list[object] = []
        input_pixel_formats: list[object] = []
        encoder_cpu_sets: list[object] = []

        def __init__(self, *args: object, **kwargs: object) -> None:
            FakeWriter.fps_values.append(args[3])
            FakeWriter.input_pixel_formats.append(kwargs["input_pixel_format"])
            FakeWriter.encoder_cpu_sets.append(kwargs["encoder_cpu_set"])
            if len(FakeWriter.instances) == 1:
                raise RuntimeError("synthetic video initialization failure")
            FakeWriter.instances.append(self)

        def close(self, commit: bool) -> None:
            del commit

    high_camera = SimpleNamespace(
        width=1280, height=720, fps=30, stream_format="yuyv", intrinsics=SimpleNamespace(is_calibrated=True)
    )
    left_camera = SimpleNamespace(
        width=848, height=480, fps=60, stream_format="bgr8", intrinsics=SimpleNamespace(is_calibrated=True)
    )
    collector = object.__new__(EpisodeCollector)
    collector._state = CaptureState.IDLE
    collector._state_lock = threading.RLock()
    collector._background_error = None
    _set_healthy_camera_arrivals(collector, ("cam_high", "cam_left"), {"cam_high": 30, "cam_left": 60})
    collector.config = SimpleNamespace(
        cameras={"cam_high": high_camera, "cam_left": left_camera},
        collection=SimpleNamespace(
            allow_uncalibrated_intrinsics=False,
            dataset_dir=tmp_path,
            target_fps=30,
            video_codec="libx265",
            video_crf=23,
            video_preset="veryfast",
            video_threads=4,
            live_video_cpu_set="20-23",
        ),
        task_info={},
    )
    monkeypatch.setattr("yam_data_collection.collector.EpisodeMcapWriter", FakeMcap)
    monkeypatch.setattr("yam_data_collection.collector.AsyncFfmpegWriter", FakeWriter)

    with pytest.raises(RuntimeError, match="synthetic video initialization failure"):
        collector.start_episode()

    assert FakeMcap.instance is not None and FakeMcap.instance.closed
    assert FakeWriter.fps_values == [30, 60]
    assert FakeWriter.input_pixel_formats == ["yuyv422", "bgr24"]
    assert FakeWriter.encoder_cpu_sets == ["20-23", "20-23"]
    assert not (tmp_path / "0.partial.mcap").exists()
    assert not (tmp_path / "videos" / "0").exists()


def test_camera_preflight_rejects_a_stream_below_target_rate() -> None:
    collector = object.__new__(EpisodeCollector)
    collector.config = SimpleNamespace(
        collection=SimpleNamespace(target_fps=30.0),
        cameras={"cam_high": SimpleNamespace(fps=30)},
    )
    collector._camera_arrival_lock = threading.Lock()
    now_ns = time.monotonic_ns()
    collector._camera_arrivals = {
        "cam_high": deque(now_ns - offset * 40_000_000 for offset in range(49, -1, -1)),
    }

    with pytest.raises(RuntimeError, match="cam_high=25.00Hz"):
        collector._validate_camera_preflight()


def test_camera_preflight_accepts_a_boundary_rate_with_runtime_epsilon() -> None:
    collector = object.__new__(EpisodeCollector)
    collector.config = SimpleNamespace(
        capture_mode="60hz",
        collection=SimpleNamespace(validation_profile=YAM_60HZ_STRICT_PROFILE),
        cameras={"cam_high": SimpleNamespace(fps=60, width=960, height=540)},
    )
    collector._camera_preflight_timings = lambda: {
        "cam_high": SimpleNamespace(duration_s=2.0, frame_count=120, average_fps=59.495, max_gap_ms=16.8)
    }

    collector._validate_camera_preflight()


def test_camera_preflight_still_rejects_a_rate_beyond_runtime_epsilon() -> None:
    collector = object.__new__(EpisodeCollector)
    collector.config = SimpleNamespace(
        capture_mode="60hz",
        collection=SimpleNamespace(validation_profile=YAM_60HZ_STRICT_PROFILE),
        cameras={"cam_high": SimpleNamespace(fps=60, width=960, height=540)},
    )
    collector._camera_preflight_timings = lambda: {
        "cam_high": SimpleNamespace(duration_s=2.0, frame_count=120, average_fps=59.48, max_gap_ms=16.8)
    }

    with pytest.raises(CaptureError, match="cam_high=59.48Hz"):
        collector._validate_camera_preflight()


def test_formal_60hz_camera_preflight_accepts_the_new_59hz_minimum() -> None:
    collector = object.__new__(EpisodeCollector)
    collector.config = SimpleNamespace(
        capture_mode="60hz",
        collection=SimpleNamespace(validation_profile=YAM_60HZ_PROFILE),
        cameras={"cam_high": SimpleNamespace(fps=60, width=960, height=540)},
    )
    collector._camera_preflight_timings = lambda: {
        "cam_high": SimpleNamespace(duration_s=2.0, frame_count=119, average_fps=59.0, max_gap_ms=16.95)
    }

    collector._validate_camera_preflight()


def test_formal_60hz_camera_preflight_reports_warming_up_without_unbound_bounds() -> None:
    collector = object.__new__(EpisodeCollector)
    collector.config = SimpleNamespace(
        capture_mode="60hz",
        collection=SimpleNamespace(validation_profile=YAM_60HZ_PROFILE),
        cameras={"cam_high": SimpleNamespace(fps=60, width=960, height=540)},
    )
    collector._camera_preflight_timings = lambda: {
        "cam_high": SimpleNamespace(duration_s=0.5, frame_count=30, average_fps=60.0, max_gap_ms=16.7)
    }

    with pytest.raises(CaptureError, match=r"cam_high=warming up \(30 frames\).*59-60\.5 Hz"):
        collector._validate_camera_preflight()


def test_formal_60hz_camera_preflight_rejects_below_the_new_minimum() -> None:
    collector = object.__new__(EpisodeCollector)
    collector.config = SimpleNamespace(
        capture_mode="60hz",
        collection=SimpleNamespace(validation_profile=YAM_60HZ_PROFILE),
        cameras={"cam_high": SimpleNamespace(fps=60, width=960, height=540)},
    )
    collector._camera_preflight_timings = lambda: {
        "cam_high": SimpleNamespace(duration_s=2.0, frame_count=118, average_fps=58.98, max_gap_ms=17.0)
    }

    with pytest.raises(CaptureError, match=r"cam_high=58\.98Hz.*59-60\.5 Hz"):
        collector._validate_camera_preflight()


def test_formal_30hz_camera_preflight_keeps_its_existing_minimum() -> None:
    collector = object.__new__(EpisodeCollector)
    collector.config = SimpleNamespace(
        capture_mode="30hz",
        collection=SimpleNamespace(validation_profile=YAM_30HZ_PROFILE),
        cameras={"cam_high": SimpleNamespace(fps=30, width=1280, height=720)},
    )
    collector._camera_preflight_timings = lambda: {
        "cam_high": SimpleNamespace(duration_s=2.0, frame_count=59, average_fps=29.48, max_gap_ms=34.0)
    }

    with pytest.raises(CaptureError, match=r"cam_high=29\.48Hz.*29\.5-30\.5 Hz"):
        collector._validate_camera_preflight()


def test_camera_preflight_rejects_a_60hz_camera_running_at_30hz() -> None:
    collector = object.__new__(EpisodeCollector)
    collector.config = SimpleNamespace(
        collection=SimpleNamespace(target_fps=30.0),
        cameras={"cam_high": SimpleNamespace(fps=30), "cam_left": SimpleNamespace(fps=60)},
    )
    collector._camera_arrival_lock = threading.Lock()
    now_ns = time.monotonic_ns()
    collector._camera_arrivals = {
        "cam_high": deque(now_ns - offset * 33_333_333 for offset in range(59, -1, -1)),
        "cam_left": deque(now_ns - offset * 33_333_333 for offset in range(59, -1, -1)),
    }

    with pytest.raises(RuntimeError, match="cam_left=30.00Hz"):
        collector._validate_camera_preflight()


def test_camera_preflight_uses_the_60hz_profile_gap_limit() -> None:
    collector = object.__new__(EpisodeCollector)
    collector.config = SimpleNamespace(
        collection=SimpleNamespace(target_fps=60.0, validation_profile=YAM_60HZ_STRICT_PROFILE),
        cameras={
            role: SimpleNamespace(fps=60, width=848, height=480) for role in ("cam_high", "cam_left", "cam_right")
        },
    )
    collector._camera_arrival_lock = threading.Lock()
    now_ns = time.monotonic_ns()
    period_ns = round(1_000_000_000 / 60)
    base_ns = now_ns - 119 * period_ns - 20_000_000
    healthy = deque(base_ns + index * period_ns for index in range(120))
    gapped = deque(base_ns + index * period_ns + (20_000_000 if index >= 60 else 0) for index in range(120))
    collector._camera_arrivals = {
        "cam_high": gapped,
        "cam_left": healthy,
        "cam_right": healthy.copy(),
    }

    with pytest.raises(CaptureError, match=r"cam_high=.*max_gap=.*?/35.0ms"):
        collector._validate_camera_preflight()


def test_formal_60hz_camera_preflight_uses_the_69ms_gap_contract() -> None:
    collector = object.__new__(EpisodeCollector)
    collector.config = SimpleNamespace(
        capture_mode="60hz",
        collection=SimpleNamespace(target_fps=60.0, validation_profile="yam_60hz"),
        cameras={
            role: SimpleNamespace(fps=60, width=848, height=480) for role in ("cam_high", "cam_left", "cam_right")
        },
    )
    collector._camera_arrival_lock = threading.Lock()
    now_ns = time.monotonic_ns()
    period_ns = round(1_000_000_000 / 60)
    base_ns = now_ns - 119 * period_ns - 20_000_000
    healthy = deque(base_ns + index * period_ns for index in range(120))
    gapped = deque(base_ns + index * period_ns + (20_000_000 if index == 60 else 0) for index in range(120))
    collector._camera_arrivals = {
        "cam_high": gapped,
        "cam_left": healthy,
        "cam_right": healthy.copy(),
    }

    collector._validate_camera_preflight()


def test_camera_callback_does_not_hold_global_state_lock_while_enqueuing() -> None:
    class ProbeWriter:
        def __init__(self) -> None:
            self.called = False

        def enqueue(self, image: np.ndarray, **kwargs: object) -> bool:
            del image, kwargs
            assert collector._state_lock.acquire(blocking=False)
            collector._state_lock.release()
            self.called = True
            return True

    class ProbeMcap:
        def write_camera_info(self, camera: object, timestamp_ns: int) -> None:
            del camera, timestamp_ns

    writer = ProbeWriter()
    operations_drained = threading.Event()
    operations_drained.set()
    collector = object.__new__(EpisodeCollector)
    collector._state = CaptureState.RECORDING
    collector._state_lock = threading.Lock()
    collector._background_error = None
    collector.config = SimpleNamespace(cameras={"cam_high": object()})
    collector._camera_arrival_lock = threading.Lock()
    collector._camera_arrivals = {"cam_high": deque()}
    collector._episode = SimpleNamespace(
        videos={"cam_high": writer},
        mcap=ProbeMcap(),
        camera_frames={},
        camera_timing={"cam_high": _FrameTiming()},
        camera_host_timing={"cam_high": _FrameTiming()},
        camera_timing_samples={"cam_high": []},
        camera_start_timestamp_ns=0,
        active_operations=0,
        operations_drained=operations_drained,
    )

    collector._on_camera_frame(
        CameraFrame(
            "cam_high",
            100,
            np.zeros((1, 1, 3), dtype=np.uint8),
            device_timestamp_ns=100,
            device_frame_number=42,
            device_timestamp_domain="timestamp_domain.global_time",
            host_timestamp_ns=120,
            capture_timestamp_source="realsense_global_time",
        )
    )

    assert writer.called
    assert collector._episode.camera_frames == {"cam_high": 1}
    assert collector._episode.camera_timing["cam_high"].last_timestamp_ns == 100
    assert collector._episode.camera_host_timing["cam_high"].last_timestamp_ns == 120
    assert collector._episode.camera_timing["cam_high"]._last_device_frame_number == 42
    assert collector._episode.camera_timing_samples["cam_high"][0].capture_timestamp_source == "realsense_global_time"
    assert collector._episode.active_operations == 0
    assert collector._episode.operations_drained.is_set()


def test_camera_preflight_uses_capture_time_not_host_delivery_time() -> None:
    collector = object.__new__(EpisodeCollector)
    collector.config = SimpleNamespace(
        capture_mode="30hz",
        collection=SimpleNamespace(target_fps=30.0, validation_profile="yam_30hz"),
        cameras={"cam_high": SimpleNamespace(fps=30, width=1280, height=720)},
    )
    collector._camera_arrival_lock = threading.Lock()
    collector._camera_arrivals = {"cam_high": deque()}
    base_ns = 1_700_000_000_000_000_000

    for index in range(61):
        collector._observe_camera_arrival(
            CameraFrame(
                "cam_high",
                base_ns + round(index * 1_000_000_000 / 30),
                np.zeros((1, 1, 3), dtype=np.uint8),
                host_timestamp_ns=base_ns + index * 70_000_000,
                capture_timestamp_source="realsense_global_time",
            )
        )

    collector._validate_camera_preflight()


def test_camera_timing_counts_missing_realsense_frame_numbers() -> None:
    timing = _FrameTiming()

    for frame_number in (100, 101, 104, 105, 103):
        timing.add_device_frame_number(frame_number)

    assert timing.device_frame_number_gaps == 2
    assert timing.device_frame_number_regressions == 1


def test_camera_startup_retries_until_every_camera_delivers_a_frame() -> None:
    class FlakyCamera:
        role = "cam_high"

        def __init__(self) -> None:
            self.start_count = 0
            self.stop_count = 0

        def start(self) -> None:
            self.start_count += 1

        def read(self, timeout_ms: int = 1000) -> CameraFrame:
            del timeout_ms
            if self.start_count == 1:
                raise RuntimeError("first pipeline startup timed out")
            return CameraFrame("cam_high", 1, np.zeros((1, 1, 3), dtype=np.uint8))

        def stop(self) -> None:
            self.stop_count += 1

    camera = FlakyCamera()
    collector = object.__new__(EpisodeCollector)
    collector._cameras = {camera.role: camera}
    collector._stop_event = threading.Event()
    collector.config = SimpleNamespace(collection=SimpleNamespace(timeout_threshold_s=1.0))
    collector._CAMERA_START_MAX_ATTEMPTS = 2
    collector._CAMERA_START_RETRY_DELAY_S = 0.0

    started = collector._start_cameras_with_readiness()

    assert started == [camera]
    assert camera.start_count == 2
    assert camera.stop_count == 1


@pytest.mark.parametrize("entry_point", ["check", "start"])
@pytest.mark.parametrize("read_error", [RuntimeError("Frame did not arrive in time!"), TimeoutError()])
def test_camera_startup_preserves_backend_errors_and_stops_cameras(
    entry_point: str, read_error: Exception
) -> None:
    class SilentCamera:
        role = "cam_right"

        def __init__(self) -> None:
            self.start_count = 0
            self.stop_count = 0

        def start(self) -> None:
            self.start_count += 1

        def read(self, timeout_ms: int = 1000) -> CameraFrame:
            # A native read may report its error after the Future's deadline.
            # The original implementation replaced that error with TimeoutError().
            time.sleep(0.03)
            raise read_error

        def stop(self) -> None:
            self.stop_count += 1

    camera = SilentCamera()
    collector = object.__new__(EpisodeCollector)
    collector._cameras = {camera.role: camera}
    collector._stop_event = threading.Event()
    collector.config = SimpleNamespace(collection=SimpleNamespace(timeout_threshold_s=0.01))
    collector._CAMERA_START_MAX_ATTEMPTS = 2
    collector._CAMERA_START_RETRY_DELAY_S = 0.0
    collector._configure_recorder_affinity = lambda: None
    collector._read_snapshots = lambda: {}
    messages: list[str] = []
    collector._print = messages.append

    with pytest.raises(CaptureError) as caught:
        if entry_point == "check":
            collector.check(start_cameras=True)
        else:
            collector._start_cameras_with_readiness()

    detail = str(read_error) or type(read_error).__name__
    assert f"camera cam_right startup read failed: {detail}" in str(caught.value)
    assert camera.start_count == camera.stop_count == (1 if entry_point == "check" else 2)
    if entry_point == "start":
        assert len(messages) == 2
        assert all(detail in message and "cam_right" in message for message in messages)
        assert caught.value.__cause__.__cause__ is read_error
    else:
        assert caught.value.__cause__ is read_error


def test_camera_startup_waits_for_the_normal_preflight_gate() -> None:
    collector = object.__new__(EpisodeCollector)
    collector._state_lock = threading.RLock()
    collector._background_error = None
    collector._stop_event = threading.Event()
    collector._CAMERA_STARTUP_PREFLIGHT_TIMEOUT_S = 1.0
    collector._CAMERA_STARTUP_PREFLIGHT_POLL_S = 0.0
    attempts = 0

    def preflight() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise CaptureError("warming up")

    collector._validate_camera_preflight = preflight

    collector._wait_for_camera_preflight()

    assert attempts == 2
