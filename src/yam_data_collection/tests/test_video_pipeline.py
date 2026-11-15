"""Safety boundaries for the two-stage live-video to HEVC pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from yam_data_collection import video_pipeline
from yam_data_collection.config import CameraConfig, CameraIntrinsics, CollectionSettings, DatasetConfig
from yam_data_collection.dataset_lock import DatasetBusyError, DatasetOperationLock
from yam_data_collection.mcap_writer import CAMERA_TOPICS
from yam_data_collection.video import VideoMetadata, VideoTranscodeError, VideoTranscodeResult


def _config(
    dataset_dir: Path,
    *,
    video_codec: str = "libx265",
    live_video_codec: str = "libx264",
) -> DatasetConfig:
    cameras = {
        role: CameraConfig(
            role=role,
            serial=role,
            width=640,
            height=480,
            fps=30,
            frame_id=role,
            intrinsics=CameraIntrinsics.uncalibrated(),
        )
        for role in CAMERA_TOPICS
    }
    return DatasetConfig(
        source_path=dataset_dir / "config.yaml",
        raw={},
        collection=CollectionSettings(
            dataset_dir=dataset_dir,
            state_save_freq=180.0,
            target_fps=30.0,
            timeout_threshold_s=1.0,
            total_sampled_frames=1,
            video_codec=video_codec,
            video_crf=23,
            video_preset="ultrafast",
            video_threads=1,
            allow_uncalibrated_intrinsics=True,
            validation_profile="piper_compatible",
            live_video_codec=live_video_codec,
            live_video_crf=23,
            live_video_preset="ultrafast",
            live_video_threads=1,
        ),
        telemetry={},
        cameras=cameras,
        task_info={"task_name": "pipeline-test"},
    )


def _seed_pending_episode(dataset_dir: Path, episode_id: int = 0) -> tuple[Path, Path]:
    pending_mcap = video_pipeline.pending_mcap_path(dataset_dir, episode_id)
    pending_mcap.parent.mkdir(parents=True)
    pending_mcap.write_bytes(b"pending MCAP fixture")
    live_dir = video_pipeline.live_video_dir(dataset_dir, episode_id)
    live_dir.mkdir(parents=True)
    for role in CAMERA_TOPICS:
        (live_dir / f"{role}.mp4").write_bytes(f"live-{role}".encode("ascii"))
    return pending_mcap, live_dir


def test_remove_episode_assets_cleans_public_episode_without_touching_siblings(tmp_path: Path) -> None:
    first = video_pipeline.final_mcap_path(tmp_path, 1)
    first.parent.mkdir(parents=True)
    first.write_bytes(b"episode-1")
    (first.parent / "videos").mkdir()
    (first.parent / "videos" / "cam_high.mp4").write_bytes(b"video")
    sibling = video_pipeline.final_mcap_path(tmp_path, 2)
    sibling.parent.mkdir(parents=True)
    sibling.write_bytes(b"episode-2")

    video_pipeline.remove_episode_assets(first)

    assert not first.parent.exists()
    assert sibling.is_file()


def test_remove_episode_assets_cleans_pending_mcap_and_live_video(tmp_path: Path) -> None:
    pending, live_dir = _seed_pending_episode(tmp_path, 3)

    video_pipeline.remove_episode_assets(pending)

    assert not pending.parent.exists()
    assert not live_dir.exists()


def _valid_video_validation() -> dict[str, object]:
    return {
        "passed": True,
        "per_camera": {
            role: {
                "passed": True,
                "frame_count_matches": True,
                "expected_camera_info_count": 61,
                "frame_count": 61,
            }
            for role in CAMERA_TOPICS
        },
    }


def _frame_count_mismatch(role: str = "cam_left") -> dict[str, object]:
    per_camera = {
        camera_role: {
            "passed": camera_role != role,
            "frame_count_matches": camera_role != role,
            "expected_camera_info_count": 61,
            "frame_count": 60 if camera_role == role else 61,
        }
        for camera_role in CAMERA_TOPICS
    }
    per_camera[role]["error"] = "decoded video frames=60, CameraInfo messages=61"
    return {"passed": False, "per_camera": per_camera}


def _transcode_result(source: Path, destination: Path) -> VideoTranscodeResult:
    destination.write_bytes(b"verified HEVC fixture")
    return VideoTranscodeResult(
        source=VideoMetadata(source, "h264", 640, 480, 30.0, 61, "yuv420p"),
        output=VideoMetadata(destination, "hevc", 640, 480, 30.0, 61, "yuv420p"),
    )


def _successful_transcode(source: Path, destination: Path, **_: object) -> VideoTranscodeResult:
    return _transcode_result(Path(source), Path(destination))


def test_live_frame_count_mismatch_never_starts_transcode_or_publishes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A live MP4 mismatch must leave the entire episode in its retryable state."""
    config = _config(tmp_path)
    pending_mcap, live_dir = _seed_pending_episode(tmp_path)
    calls: list[tuple[Path, tuple[str, ...] | None]] = []
    attempted_transcodes: list[Path] = []

    monkeypatch.setattr(
        video_pipeline,
        "validate_mcap",
        lambda *_: {"output_mcap_passed": True, "failed_checks": []},
    )

    def mismatch_assets(
        _mcap_path: Path,
        _policy: object,
        *,
        video_dir: Path | None = None,
        expected_video_codecs: tuple[str, ...] | None = None,
        require_fps_match: bool = True,
    ) -> dict[str, object]:
        assert video_dir == live_dir
        assert not require_fps_match
        calls.append((Path(video_dir), expected_video_codecs))
        return _frame_count_mismatch()

    def unexpected_transcode(source: Path, destination: Path, **_: object) -> VideoTranscodeResult:
        del destination
        attempted_transcodes.append(Path(source))
        raise AssertionError("a mismatched live video must never be transcoded")

    monkeypatch.setattr(video_pipeline, "validate_video_assets", mismatch_assets)
    monkeypatch.setattr(video_pipeline, "transcode_to_hevc", unexpected_transcode)

    with pytest.raises(video_pipeline.VideoPipelineError, match="live video validation failed"):
        video_pipeline.transcode_episode(pending_mcap, config)

    assert calls == [(live_dir, ("h264",))]
    assert attempted_transcodes == []
    assert pending_mcap.is_file()
    assert live_dir.is_dir()
    assert not (tmp_path / "0" / "0.mcap").exists()
    assert not video_pipeline.final_video_dir(tmp_path, 0).exists()


def test_live_identity_gate_preserves_a_formal_quality_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Timing acceptance must not be confused with frame-identity safety."""
    config = _config(tmp_path)
    pending_mcap, live_dir = _seed_pending_episode(tmp_path)

    monkeypatch.setattr(
        video_pipeline,
        "validate_mcap",
        lambda *_: {"output_mcap_passed": False, "failed_checks": ["camera_max_gap_below_69ms"]},
    )

    def identity_matches(
        _mcap_path: Path,
        _policy: object,
        *,
        video_dir: Path | None = None,
        expected_video_codecs: tuple[str, ...] | None = None,
        require_fps_match: bool = True,
    ) -> dict[str, object]:
        assert video_dir == live_dir
        assert expected_video_codecs == ("h264",)
        assert not require_fps_match
        return _valid_video_validation()

    monkeypatch.setattr(video_pipeline, "validate_video_assets", identity_matches)

    result = video_pipeline.validate_live_episode(pending_mcap, config)

    assert result["passed"]
    assert not result["formal_mcap_passed"]
    assert result["formal_failed_checks"] == ["camera_max_gap_below_69ms"]


def test_final_frame_count_mismatch_never_publishes_root_mcap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Even completed x265 outputs stay staged when final count validation fails."""
    config = _config(tmp_path)
    pending_mcap, live_dir = _seed_pending_episode(tmp_path)
    transcoded_roles: list[str] = []
    final_validation_dirs: list[Path] = []

    monkeypatch.setattr(video_pipeline, "validate_live_episode", lambda *_: _valid_video_validation())

    def transcode(source: Path, destination: Path, **_: object) -> VideoTranscodeResult:
        transcoded_roles.append(Path(source).stem)
        return _transcode_result(Path(source), Path(destination))

    def final_mismatch(
        _mcap_path: Path,
        _policy: object,
        *,
        video_dir: Path | None = None,
        require_fps_match: bool = True,
        **_: object,
    ) -> dict[str, object]:
        assert video_dir is not None
        assert not require_fps_match
        final_validation_dirs.append(Path(video_dir))
        return _frame_count_mismatch()

    monkeypatch.setattr(video_pipeline, "transcode_to_hevc", transcode)
    monkeypatch.setattr(video_pipeline, "validate_video_assets", final_mismatch)

    with pytest.raises(video_pipeline.VideoPipelineError, match="final HEVC validation failed"):
        video_pipeline.transcode_episode(pending_mcap, config)

    assert transcoded_roles == list(CAMERA_TOPICS)
    assert len(final_validation_dirs) == 1
    assert final_validation_dirs[0].parent == tmp_path / "0"
    assert pending_mcap.is_file()
    assert live_dir.is_dir()
    assert not (tmp_path / "0" / "0.mcap").exists()
    assert not video_pipeline.final_video_dir(tmp_path, 0).exists()


def test_success_publishes_final_video_directory_before_root_mcap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The root numeric MCAP appears only after all final HEVC assets are visible."""
    config = _config(tmp_path)
    pending_mcap, live_dir = _seed_pending_episode(tmp_path)
    final_dir = video_pipeline.final_video_dir(tmp_path, 0)
    final_mcap = tmp_path / "0" / "0.mcap"
    publication_events: list[tuple[str, bool]] = []
    original_publish_directory = video_pipeline._publish_directory
    original_publish_mcap = video_pipeline._publish_mcap

    monkeypatch.setattr(video_pipeline, "validate_live_episode", lambda *_: _valid_video_validation())
    monkeypatch.setattr(video_pipeline, "transcode_to_hevc", _successful_transcode)
    monkeypatch.setattr(video_pipeline, "validate_video_assets", lambda *_args, **_kwargs: _valid_video_validation())

    def publish_directory(staged: Path, destination: Path, *, overwrite: bool) -> None:
        publication_events.append(("videos", final_mcap.exists()))
        original_publish_directory(staged, destination, overwrite=overwrite)

    def publish_mcap(source: Path, destination: Path, *, overwrite: bool) -> None:
        publication_events.append(("mcap", final_dir.is_dir()))
        original_publish_mcap(source, destination, overwrite=overwrite)

    monkeypatch.setattr(video_pipeline, "_publish_directory", publish_directory)
    monkeypatch.setattr(video_pipeline, "_publish_mcap", publish_mcap)

    result = video_pipeline.transcode_episode(pending_mcap, config)

    assert publication_events == [("videos", False), ("mcap", True)]
    assert result.final_mcap_path == final_mcap
    assert final_mcap.is_file()
    assert not pending_mcap.exists()
    assert live_dir.is_dir()
    assert sorted(path.name for path in final_dir.glob("*.mp4")) == [
        "cam_high.mp4",
        "cam_left.mp4",
        "cam_right.mp4",
    ]
    assert not (live_dir / "pipeline.json").exists()


def test_h264_final_publication_moves_verified_live_assets_without_transcoding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A direct H.264 episode is atomically published without invoking x265."""
    config = _config(tmp_path, video_codec="libx264")
    pending_mcap, live_dir = _seed_pending_episode(tmp_path)
    final_dir = video_pipeline.final_video_dir(tmp_path, 0)
    final_mcap = tmp_path / "0" / "0.mcap"
    publication_events: list[tuple[str, bool]] = []
    original_publish_directory = video_pipeline._publish_directory
    original_publish_mcap = video_pipeline._publish_mcap

    monkeypatch.setattr(video_pipeline, "validate_live_episode", lambda *_: _valid_video_validation())

    def unexpected_transcode(*_args: object, **_kwargs: object) -> VideoTranscodeResult:
        raise AssertionError("direct H.264 publication must not invoke the offline HEVC encoder")

    def publish_directory(staged: Path, destination: Path, *, overwrite: bool) -> None:
        publication_events.append(("videos", final_mcap.exists()))
        original_publish_directory(staged, destination, overwrite=overwrite)

    def publish_mcap(source: Path, destination: Path, *, overwrite: bool) -> None:
        publication_events.append(("mcap", final_dir.is_dir()))
        original_publish_mcap(source, destination, overwrite=overwrite)

    monkeypatch.setattr(video_pipeline, "transcode_to_hevc", unexpected_transcode)
    monkeypatch.setattr(video_pipeline, "_publish_directory", publish_directory)
    monkeypatch.setattr(video_pipeline, "_publish_mcap", publish_mcap)

    result = video_pipeline.publish_live_episode(pending_mcap, config)

    assert publication_events == [("videos", False), ("mcap", True)]
    assert result.final_mcap_path == final_mcap
    assert final_mcap.is_file()
    assert not pending_mcap.exists()
    assert not live_dir.exists()
    assert sorted(path.name for path in final_dir.glob("*.mp4")) == [
        "cam_high.mp4",
        "cam_left.mp4",
        "cam_right.mp4",
    ]
    assert (final_dir / "cam_left.mp4").read_bytes() == b"live-cam_left"
    assert not (final_dir / "pipeline.json").exists()


def test_timestamp_named_mcap_publishes_and_resolves_by_episode_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _config(tmp_path, video_codec="h264")
    filename = "2026-10-18_14-37.mcap"
    pending_mcap = video_pipeline.pending_mcap_path(tmp_path, 6, filename=filename)
    pending_mcap.parent.mkdir(parents=True)
    pending_mcap.write_bytes(b"pending MCAP fixture")
    live_dir = video_pipeline.live_video_dir(tmp_path, 6)
    live_dir.mkdir(parents=True)
    for role in CAMERA_TOPICS:
        (live_dir / f"{role}.mp4").write_bytes(f"live-{role}".encode("ascii"))
    monkeypatch.setattr(video_pipeline, "validate_live_episode", lambda *_: _valid_video_validation())

    result = video_pipeline.publish_live_episode(pending_mcap, config)

    expected = tmp_path / "6" / filename
    assert result.episode_id == 6
    assert result.final_mcap_path == expected
    assert video_pipeline.final_mcap_path(tmp_path, 6) == expected
    assert video_pipeline._episode_id(expected) == 6
    assert expected.is_file()
    assert not pending_mcap.exists()


def test_timestamp_named_mcap_deletion_removes_episode_assets(tmp_path: Path) -> None:
    episode_dir = tmp_path / "8"
    episode_dir.mkdir()
    mcap = episode_dir / "2026-10-18_14-37.mcap"
    mcap.write_bytes(b"episode")
    (episode_dir / "videos").mkdir()

    video_pipeline.remove_episode_assets(mcap)

    assert not episode_dir.exists()


def test_h264_final_frame_mismatch_never_publishes_or_transcodes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A direct-publish configuration keeps identity-mismatched assets retryable."""
    config = _config(tmp_path, video_codec="h264")
    pending_mcap, live_dir = _seed_pending_episode(tmp_path)
    validation_calls: list[tuple[Path, tuple[str, ...] | None]] = []

    monkeypatch.setattr(
        video_pipeline,
        "validate_mcap",
        lambda *_: {"output_mcap_passed": True, "failed_checks": []},
    )

    def mismatch_assets(
        _mcap_path: Path,
        _policy: object,
        *,
        video_dir: Path | None = None,
        expected_video_codecs: tuple[str, ...] | None = None,
        require_fps_match: bool = True,
    ) -> dict[str, object]:
        assert video_dir == live_dir
        assert not require_fps_match
        validation_calls.append((Path(video_dir), expected_video_codecs))
        return _frame_count_mismatch()

    def unexpected_publish(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a frame mismatch must never publish final assets")

    monkeypatch.setattr(video_pipeline, "validate_video_assets", mismatch_assets)
    monkeypatch.setattr(video_pipeline, "_publish_verified_episode", unexpected_publish)

    with pytest.raises(video_pipeline.VideoPipelineError, match="live video validation failed"):
        video_pipeline.publish_live_episode(pending_mcap, config)

    assert validation_calls == [(live_dir, ("h264",))]
    assert pending_mcap.is_file()
    assert live_dir.is_dir()
    assert not (tmp_path / "0" / "0.mcap").exists()
    assert not video_pipeline.final_video_dir(tmp_path, 0).exists()


def test_h264_final_publish_failure_restores_pending_live_assets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed final-MCAP move returns direct H.264 assets to their retryable paths."""
    config = _config(tmp_path, video_codec="libx264")
    pending_mcap, live_dir = _seed_pending_episode(tmp_path)

    monkeypatch.setattr(video_pipeline, "validate_live_episode", lambda *_: _valid_video_validation())

    def fail_mcap_publish(*_args: object, **_kwargs: object) -> None:
        raise OSError("synthetic final MCAP write failure")

    monkeypatch.setattr(video_pipeline, "_publish_mcap", fail_mcap_publish)

    with pytest.raises(OSError, match="synthetic final MCAP write failure"):
        video_pipeline.publish_live_episode(pending_mcap, config)

    assert pending_mcap.is_file()
    assert live_dir.is_dir()
    assert all((live_dir / f"{role}.mp4").is_file() for role in CAMERA_TOPICS)
    assert not (tmp_path / "0" / "0.mcap").exists()
    assert not video_pipeline.final_video_dir(tmp_path, 0).exists()
    assert not (live_dir / "pipeline.json").exists()


def test_h264_final_config_is_never_sent_to_the_offline_transcoder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The legacy transcode entry point cannot turn a direct H.264 dataset into HEVC."""
    config = _config(tmp_path, video_codec="libx264")
    pending_mcap, live_dir = _seed_pending_episode(tmp_path)

    def unexpected_transcode(*_args: object, **_kwargs: object) -> VideoTranscodeResult:
        raise AssertionError("H.264-final config must be rejected before x265 starts")

    monkeypatch.setattr(video_pipeline, "transcode_to_hevc", unexpected_transcode)

    with pytest.raises(video_pipeline.VideoPipelineError, match="published during capture"):
        video_pipeline.transcode_episode(pending_mcap, config)

    assert pending_mcap.is_file()
    assert live_dir.is_dir()


def test_h264_pending_publication_waits_for_the_dataset_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A recovery publish cannot race a live capture or offline operation."""
    config = _config(tmp_path, video_codec="libx264")
    pending_mcap, _live_dir = _seed_pending_episode(tmp_path)
    monkeypatch.setattr(video_pipeline, "validate_live_episode", lambda *_: _valid_video_validation())

    with DatasetOperationLock(tmp_path, "capture"):
        with pytest.raises(DatasetBusyError, match="dataset is busy"):
            video_pipeline.publish_pending_episode(pending_mcap, config)

    result = video_pipeline.publish_pending_episode(pending_mcap, config)

    assert result.final_mcap_path.is_file()
    assert result.final_video_dir.is_dir()


def test_failed_transcode_preserves_pending_live_assets_for_a_later_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A temporary offline encoder failure must not destroy the verified live episode."""
    config = _config(tmp_path)
    pending_mcap, live_dir = _seed_pending_episode(tmp_path)
    final_mcap = tmp_path / "0" / "0.mcap"

    monkeypatch.setattr(video_pipeline, "validate_live_episode", lambda *_: _valid_video_validation())

    def fail_transcode(*_args: object, **_kwargs: object) -> VideoTranscodeResult:
        raise VideoTranscodeError("synthetic x265 failure")

    monkeypatch.setattr(video_pipeline, "transcode_to_hevc", fail_transcode)

    with pytest.raises(VideoTranscodeError, match="synthetic x265 failure"):
        video_pipeline.transcode_episode(pending_mcap, config)

    assert pending_mcap.is_file()
    assert live_dir.is_dir()
    assert all((live_dir / f"{role}.mp4").is_file() for role in CAMERA_TOPICS)
    assert not final_mcap.exists()
    assert not video_pipeline.final_video_dir(tmp_path, 0).exists()

    monkeypatch.setattr(video_pipeline, "transcode_to_hevc", _successful_transcode)
    monkeypatch.setattr(video_pipeline, "validate_video_assets", lambda *_args, **_kwargs: _valid_video_validation())

    result = video_pipeline.transcode_episode(pending_mcap, config)

    assert result.final_mcap_path == final_mcap
    assert final_mcap.is_file()
    assert video_pipeline.final_video_dir(tmp_path, 0).is_dir()
    assert live_dir.is_dir()


def test_overwrite_rolls_back_old_public_assets_when_mcap_publish_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An interrupted replacement cannot leave new videos beside an old MCAP."""
    config = _config(tmp_path)
    pending_mcap, _live_dir = _seed_pending_episode(tmp_path)
    final_dir = video_pipeline.final_video_dir(tmp_path, 0)
    final_dir.mkdir(parents=True)
    (final_dir / "old-marker.txt").write_text("old videos", encoding="ascii")
    final_mcap = tmp_path / "0" / "0.mcap"
    final_mcap.write_bytes(b"old MCAP")

    monkeypatch.setattr(video_pipeline, "validate_live_episode", lambda *_: _valid_video_validation())
    monkeypatch.setattr(video_pipeline, "transcode_to_hevc", _successful_transcode)
    monkeypatch.setattr(video_pipeline, "validate_video_assets", lambda *_args, **_kwargs: _valid_video_validation())

    def fail_mcap_publish(*_args: object, **_kwargs: object) -> None:
        raise OSError("synthetic final MCAP write failure")

    monkeypatch.setattr(video_pipeline, "_publish_mcap", fail_mcap_publish)

    with pytest.raises(OSError, match="synthetic final MCAP write failure"):
        video_pipeline.transcode_episode(pending_mcap, config, overwrite=True)

    assert pending_mcap.is_file()
    assert final_mcap.read_bytes() == b"old MCAP"
    assert (final_dir / "old-marker.txt").read_text(encoding="ascii") == "old videos"
    assert _live_dir.is_dir()


def test_transcode_is_excluded_while_capture_owns_the_dataset_lock(tmp_path: Path) -> None:
    """The public transcode entry point observes the same lock held by capture."""
    config = _config(tmp_path)
    pending_mcap, _live_dir = _seed_pending_episode(tmp_path)

    with DatasetOperationLock(tmp_path, "capture"):
        with pytest.raises(DatasetBusyError, match="dataset is busy"):
            video_pipeline.transcode_episode(pending_mcap, config)

    # The failed acquisition did not leak the flock, so a later operation can
    # own it. This also keeps the test independent of in-process flock quirks.
    with DatasetOperationLock(tmp_path, "transcode"):
        pass
