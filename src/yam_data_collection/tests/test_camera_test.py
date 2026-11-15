from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from yam_data_collection.camera_test import CameraQualification, _mutual_nearest_summary
from yam_data_collection.config import (
    CameraConfig,
    CameraIntrinsics,
    CameraTestConfig,
    CameraTestSettings,
    load_camera_test_config,
)
from yam_data_collection.models import CameraFrame
from yam_data_collection.validation_profiles import YAM_60HZ_STRICT_PROFILE


def test_camera_test_config_allows_mixed_30_and_60_hz(tmp_path: Path) -> None:
    config_path = tmp_path / "camera_test.yaml"
    config_path.write_text(
        """
camera_test_config:
  output_dir: artifacts
  duration_s: 60
cameras:
  cam_high: {serial: high, width: 1280, height: 720, fps: 30}
  cam_left: {serial: left, width: 848, height: 480, fps: 60}
  cam_right: {serial: right, width: 848, height: 480, fps: 60}
""",
        encoding="utf-8",
    )

    config = load_camera_test_config(config_path)

    assert config.settings.output_dir == tmp_path / "artifacts"
    assert [camera.fps for camera in config.cameras.values()] == [30, 60, 60]
    assert config.cameras["cam_left"].intrinsics == CameraIntrinsics.uncalibrated()


def test_camera_test_config_rejects_non_positive_camera_fps(tmp_path: Path) -> None:
    config_path = tmp_path / "camera_test.yaml"
    config_path.write_text(
        """
camera_test_config:
  output_dir: artifacts
cameras:
  cam_high: {serial: high, width: 1280, height: 720, fps: 30}
  cam_left: {serial: left, width: 848, height: 480, fps: 0}
  cam_right: {serial: right, width: 848, height: 480, fps: 60}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cam_left.fps must be positive"):
        load_camera_test_config(config_path)


class _FakeCamera:
    def __init__(self, config: CameraConfig, *, source_period_ns: int | None = None) -> None:
        self.role = config.role
        self._config = config
        self._source_period_ns = source_period_ns or round(1_000_000_000 / config.fps)
        self._source_timestamp_ns = 1_000_000_000
        self.started = False

    def start(self) -> None:
        self.started = True

    def read(self, timeout_ms: int = 1000) -> CameraFrame:
        del timeout_ms
        time.sleep(1 / self._config.fps)
        self._source_timestamp_ns += self._source_period_ns
        return CameraFrame(
            role=self.role,
            timestamp_ns=self._source_timestamp_ns,
            image_bgr=np.zeros((self._config.height, self._config.width, 3), dtype=np.uint8),
            owns_image=True,
            device_timestamp_ns=self._source_timestamp_ns,
            host_timestamp_ns=time.time_ns(),
            capture_timestamp_source="realsense_global_time",
        )

    def stop(self) -> None:
        self.started = False


class _FakeWriter:
    instances: dict[Path, "_FakeWriter"] = {}

    def __init__(self, path: Path, width: int, height: int, fps: int, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self.path = path
        self.width = width
        self.height = height
        self.fps = fps
        self.accepted = 0
        self.dropped_frames = 0
        self.max_queue_depth = 1
        self.queue_size = 240
        self.max_pipe_write_ms = 0.1
        _FakeWriter.instances[path] = self

    def enqueue(self, image_bgr: np.ndarray, *, copy_frame: bool = True) -> bool:
        del image_bgr, copy_frame
        self.accepted += 1
        return True

    def close(self, commit: bool) -> None:
        assert commit
        self.path.touch()


def _camera_test_config(tmp_path: Path) -> CameraTestConfig:
    cameras = {
        "cam_high": CameraConfig("cam_high", "high", 640, 480, 30, "high", CameraIntrinsics.uncalibrated()),
        "cam_left": CameraConfig("cam_left", "left", 640, 480, 60, "left", CameraIntrinsics.uncalibrated()),
        "cam_right": CameraConfig("cam_right", "right", 640, 480, 60, "right", CameraIntrinsics.uncalibrated()),
    }
    settings = CameraTestSettings(
        output_dir=tmp_path / "camera_tests",
        duration_s=0.25,
        timeout_threshold_s=1.0,
        video_codec="libx265",
        video_crf=23,
        video_preset="ultrafast",
        video_threads=1,
        video_queue_size=240,
        fps_tolerance=0.5,
        max_gap_factor=2.1,
    )
    return CameraTestConfig(source_path=tmp_path / "camera_test.yaml", raw={}, settings=settings, cameras=cameras)


def _fake_probe(path: Path) -> dict[str, object]:
    writer = _FakeWriter.instances[path]
    return {
        "passed": True,
        "codec": "hevc",
        "width": writer.width,
        "height": writer.height,
        "fps": writer.fps,
        "frame_count": writer.accepted,
    }


def test_camera_qualification_records_mixed_rate_artifacts(monkeypatch, tmp_path: Path) -> None:
    _FakeWriter.instances = {}
    config = _camera_test_config(tmp_path)
    monkeypatch.setattr("yam_data_collection.camera_test._probe_video", _fake_probe)

    run = CameraQualification(config, camera_factory=_FakeCamera, writer_factory=_FakeWriter).run()

    assert run.passed
    assert run.output_dir.is_dir()
    assert not run.output_dir.name.endswith(".partial")
    assert (run.output_dir / "report.json").is_file()
    assert all((run.output_dir / f"{role}.mp4").is_file() for role in config.cameras)
    assert run.report["cameras"]["cam_high"]["target_fps"] == 30
    assert run.report["cameras"]["cam_left"]["target_fps"] == 60
    assert "camera_sync" in run.report
    assert run.report["camera_sync"]["threshold_ms"] == 34.0
    saved_config = yaml.safe_load((run.output_dir / "test_config.yaml").read_text(encoding="utf-8"))
    assert saved_config["camera_test_config"]["validation_profile"] == "piper_compatible"


def test_yam_60hz_camera_test_fails_on_a_sync_offset(monkeypatch, tmp_path: Path) -> None:
    _FakeWriter.instances = {}
    config = _camera_test_config(tmp_path)
    cameras = {
        role: CameraConfig(role, role, 640, 480, 60, role, CameraIntrinsics.uncalibrated()) for role in config.cameras
    }
    strict_config = CameraTestConfig(
        source_path=config.source_path,
        raw=config.raw,
        settings=CameraTestSettings(
            output_dir=config.settings.output_dir,
            duration_s=config.settings.duration_s,
            timeout_threshold_s=config.settings.timeout_threshold_s,
            video_codec=config.settings.video_codec,
            video_crf=config.settings.video_crf,
            video_preset=config.settings.video_preset,
            video_threads=config.settings.video_threads,
            video_queue_size=config.settings.video_queue_size,
            fps_tolerance=config.settings.fps_tolerance,
            max_gap_factor=config.settings.max_gap_factor,
            validation_profile=YAM_60HZ_STRICT_PROFILE,
        ),
        cameras=cameras,
    )
    monkeypatch.setattr("yam_data_collection.camera_test._probe_video", _fake_probe)
    monkeypatch.setattr(
        "yam_data_collection.camera_test._mutual_nearest_summary",
        lambda first, second: {"max_offset_ms": 18.0, "valid_pairs": 2, "orphans": 0},
    )

    run = CameraQualification(strict_config, camera_factory=_FakeCamera, writer_factory=_FakeWriter).run()

    assert not run.passed
    assert not run.output_dir.name.endswith(".partial")
    assert run.report["camera_sync"]["threshold_ms"] == 17.0
    assert not run.report["camera_sync"]["passed"]


def test_camera_qualification_marks_source_rate_failure_but_keeps_artifacts(monkeypatch, tmp_path: Path) -> None:
    _FakeWriter.instances = {}
    config = _camera_test_config(tmp_path)
    monkeypatch.setattr("yam_data_collection.camera_test._probe_video", _fake_probe)

    def slow_left_camera(camera: CameraConfig) -> _FakeCamera:
        source_period_ns = 40_000_000 if camera.role == "cam_left" else None
        return _FakeCamera(camera, source_period_ns=source_period_ns)

    run = CameraQualification(config, camera_factory=slow_left_camera, writer_factory=_FakeWriter).run()

    assert not run.passed
    assert not run.output_dir.name.endswith(".partial")
    assert not run.report["cameras"]["cam_left"]["source_passed"]
    assert (run.output_dir / "cam_left.mp4").is_file()
    assert (run.output_dir / "report.json").is_file()


def test_camera_qualification_can_skip_video_encoding_and_writes(tmp_path: Path) -> None:
    config = _camera_test_config(tmp_path)

    def writer_must_not_be_created(*args: object, **kwargs: object) -> _FakeWriter:
        del args, kwargs
        raise AssertionError("writer must not be created when video recording is disabled")

    run = CameraQualification(config, camera_factory=_FakeCamera, writer_factory=writer_must_not_be_created).run(
        record_video=False
    )

    assert run.passed
    assert not (run.output_dir / "cam_high.mp4").exists()
    assert run.report["video_recording_enabled"] is False
    assert run.report["qualification_scope"] == "timing_and_sync_only"
    for entry in run.report["cameras"].values():
        assert entry["captured_frames"] > 0
        assert entry["accepted_frames"] == 0
        assert entry["video"] == {"status": "disabled", "reason": "--no-video"}
        assert entry["video_passed"] is None
        assert entry["video_writer"] is None


def test_camera_qualification_without_video_still_fails_a_slow_source(tmp_path: Path) -> None:
    config = _camera_test_config(tmp_path)

    def slow_left_camera(camera: CameraConfig) -> _FakeCamera:
        source_period_ns = 40_000_000 if camera.role == "cam_left" else None
        return _FakeCamera(camera, source_period_ns=source_period_ns)

    run = CameraQualification(config, camera_factory=slow_left_camera).run(record_video=False)

    assert not run.passed
    assert not run.report["cameras"]["cam_left"]["source_passed"]
    assert run.report["cameras"]["cam_left"]["accepted_frames"] == 0


def test_camera_qualification_sync_uses_capture_time_not_host_dequeue_time(monkeypatch, tmp_path: Path) -> None:
    class CaptureSynchronizedCamera:
        def __init__(self, config: CameraConfig) -> None:
            self.role = config.role
            self._config = config
            self._capture_timestamp_ns = 1_000_000_000
            self._host_delay_s = {"cam_high": 0.0, "cam_left": 0.02, "cam_right": 0.04}[config.role]

        def start(self) -> None:
            pass

        def read(self, timeout_ms: int = 1000) -> CameraFrame:
            del timeout_ms
            time.sleep(1 / self._config.fps + self._host_delay_s)
            self._capture_timestamp_ns += round(1_000_000_000 / self._config.fps)
            return CameraFrame(
                role=self.role,
                timestamp_ns=self._capture_timestamp_ns,
                image_bgr=np.zeros((self._config.height, self._config.width, 3), dtype=np.uint8),
                owns_image=True,
                device_timestamp_ns=self._capture_timestamp_ns,
                host_timestamp_ns=time.time_ns(),
                capture_timestamp_source="realsense_global_time",
            )

        def stop(self) -> None:
            pass

    base_config = _camera_test_config(tmp_path)
    cameras = {
        role: CameraConfig(role, role, 640, 480, 60, role, CameraIntrinsics.uncalibrated())
        for role in base_config.cameras
    }
    config = CameraTestConfig(
        source_path=base_config.source_path,
        raw=base_config.raw,
        settings=CameraTestSettings(
            output_dir=base_config.settings.output_dir,
            duration_s=base_config.settings.duration_s,
            timeout_threshold_s=base_config.settings.timeout_threshold_s,
            video_codec=base_config.settings.video_codec,
            video_crf=base_config.settings.video_crf,
            video_preset=base_config.settings.video_preset,
            video_threads=base_config.settings.video_threads,
            video_queue_size=base_config.settings.video_queue_size,
            fps_tolerance=base_config.settings.fps_tolerance,
            max_gap_factor=base_config.settings.max_gap_factor,
            validation_profile=YAM_60HZ_STRICT_PROFILE,
        ),
        cameras=cameras,
    )

    run = CameraQualification(config, camera_factory=CaptureSynchronizedCamera).run(record_video=False)

    # The camera test must use the same capture clock as the recorder and
    # offline MCAP checker. Host dequeuing is intentionally staggered here.
    assert run.report["camera_sync"]["passed"]
    for pair in run.report["camera_sync"]["pairwise"].values():
        assert pair["max_offset_ms"] == 0.0
    assert run.report["cameras"]["cam_left"]["source_clock"] == "capture"


def test_camera_qualification_can_select_one_camera(tmp_path: Path) -> None:
    config = _camera_test_config(tmp_path)
    started_roles: list[str] = []

    def tracking_camera(camera: CameraConfig) -> _FakeCamera:
        started_roles.append(camera.role)
        return _FakeCamera(camera)

    run = CameraQualification(config, camera_factory=tracking_camera).run(
        record_video=False,
        camera_roles=["cam_high"],
    )

    assert run.passed
    assert started_roles == ["cam_high"]
    assert list(run.report["cameras"]) == ["cam_high"]
    assert run.report["selected_camera_roles"] == ["cam_high"]
    assert run.report["configured_camera_roles"] == ["cam_high", "cam_left", "cam_right"]
    assert run.report["full_three_camera_qualification"] is False
    assert run.report["qualification_scope"] == "timing_only"
    assert run.report["camera_sync"] == {
        "applicable": False,
        "threshold_ms": 34.0,
        "passed": True,
        "pairwise": {},
    }
    assert run.output_dir.parent.name == "only-cam_high"
    saved_config = yaml.safe_load((run.output_dir / "test_config.yaml").read_text(encoding="utf-8"))
    assert saved_config["camera_test_config"]["selected_camera_roles"] == ["cam_high"]


def test_camera_qualification_selects_two_cameras_for_one_sync_pair(tmp_path: Path) -> None:
    config = _camera_test_config(tmp_path)

    run = CameraQualification(config, camera_factory=_FakeCamera).run(
        record_video=False,
        camera_roles=["cam_high", "cam_right"],
    )

    assert run.passed
    assert list(run.report["cameras"]) == ["cam_high", "cam_right"]
    assert list(run.report["camera_sync"]["pairwise"]) == ["cam_high__cam_right"]
    assert run.report["camera_sync"]["applicable"] is True


@pytest.mark.parametrize(
    ("camera_roles", "message"),
    [
        (["cam_high", "cam_high"], "must not be repeated"),
        (["missing"], "unknown camera role"),
        ([], "at least one camera role"),
    ],
)
def test_camera_qualification_rejects_invalid_camera_selection(
    tmp_path: Path, camera_roles: list[str], message: str
) -> None:
    config = _camera_test_config(tmp_path)

    with pytest.raises(ValueError, match=message):
        CameraQualification(config, camera_factory=_FakeCamera).run(record_video=False, camera_roles=camera_roles)


def test_mutual_nearest_summary_matches_piper_pairing() -> None:
    summary = _mutual_nearest_summary([0, 16_000_000, 32_000_000], [3_000_000, 19_000_000, 35_000_000])

    assert summary == {"max_offset_ms": 3.0, "valid_pairs": 3, "orphans": 0}
