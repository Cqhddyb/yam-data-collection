from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
import yaml

from yam_data_collection.collector import CaptureError, CaptureState, _Episode, _FrameTiming
from yam_data_collection.mcap_writer import EpisodeMcapWriter
from yam_data_collection.models import CameraTimingSample, RobotSnapshot
from yam_data_collection.quality import inspect_mcap
from yam_data_collection.queued_collector import _StateWrite
from yam_data_collection.video import TimestampedVideoWriter, close_video_writers
from yam_data_collection.video_pipeline import (
    VideoPipelineError,
    live_video_dir,
    pending_episode_dir,
    pending_mcap_path,
)

from yam_single_arm_collection.collector import SingleArmCollector
from yam_single_arm_collection.config import load_config
from yam_single_arm_collection.quality import validate_mcap, validate_video_assets


BASE = 1_700_000_000_000_000_000


def _stage_episode(collector: SingleArmCollector, *, state_offset_ns: int = 0) -> _Episode:
    dataset = collector.config.collection.dataset_dir
    pending_dir = pending_episode_dir(dataset, 0)
    pending_dir.mkdir(parents=True)
    live_dir = live_video_dir(dataset, 0)
    video_dir = live_dir.with_name("0.partial")
    video_dir.mkdir(parents=True)
    mcap_path = pending_dir / "synthetic.partial.mcap"
    mcap = EpisodeMcapWriter(mcap_path, collector.config.task_info)
    episode = _Episode(
        episode_id=0,
        partial_mcap_path=mcap_path,
        final_mcap_path=pending_mcap_path(dataset, 0, filename="synthetic.mcap"),
        video_dir=video_dir,
        live_video_dir=live_dir,
        pending_dir=pending_dir,
        mcap=mcap,
        videos={},
        started_at_ns=BASE,
        recording_started_at_ns=BASE,
        recording_stopped_at_ns=BASE + 1_000_000_000,
        camera_start_timestamp_ns=BASE,
    )
    collector._episode = episode
    collector._state = CaptureState.RECORDING
    try:
        state_writer = collector._state_writer_for(episode)
        for index in range(181):
            timestamp = BASE + round(index * 1_000_000_000 / 180)
            snapshots = {
                role: RobotSnapshot(role, timestamp + state_offset_ns, index, (0.0,) * 7, (0.0,) * 7, (0.0,) * 7)
                for role in collector.config.telemetry
            }
            state_writer.submit(_StateWrite(snapshots, timestamp))
            episode.state_samples += 1
        for role, camera in collector.config.cameras.items():
            video = TimestampedVideoWriter(
                video_dir / f"{role}.mp4",
                camera.width,
                camera.height,
                camera.fps,
                "libx264",
                23,
                "ultrafast",
                1,
                queue_size=240,
            )
            episode.videos[role] = video
            image = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
            timings: list[CameraTimingSample] = []
            source_timing, host_timing = _FrameTiming(), _FrameTiming()
            for index in range(camera.fps + 1):
                timestamp = BASE + round(index * 1_000_000_000 / camera.fps)
                assert video.enqueue(image, copy_frame=False, timestamp_ns=timestamp)
                mcap.write_camera_info(camera, timestamp)
                timings.append(
                    CameraTimingSample(
                        role,
                        timestamp,
                        timestamp,
                        timestamp,
                        index,
                        "timestamp_domain.global_time",
                        "realsense_global_time",
                    )
                )
                source_timing.add(timestamp)
                host_timing.add(timestamp)
            episode.camera_frames[role] = len(timings)
            episode.camera_timing[role] = source_timing
            episode.camera_host_timing[role] = host_timing
            episode.camera_timing_samples[role] = timings
        return episode
    except BaseException:
        writer = collector._state_mcap_writers.pop(id(episode), None)
        if writer is not None:
            writer.close()
        mcap.close()
        close_video_writers(episode.videos.values(), commit=False)
        collector._episode = None
        collector._state = CaptureState.IDLE
        raise


@pytest.mark.parametrize(
    "side,setup", [("right", "wrist"), ("right", "wrist-top"), ("left", "wrist"), ("left", "wrist-top")]
)
def test_full_queued_save_publishes_only_selected_real_mp4s(
    config_factory: object,
    collector_factory: object,
    side: str,
    setup: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = config_factory(side, setup)
    collector = collector_factory(config)
    assert set(collector._telemetry_clients) == set(config.telemetry)
    assert set(collector._cameras) == set(config.cameras)
    collector._prepare_dataset_dir()
    episode = _stage_episode(collector)
    monkeypatch.setattr("yam_data_collection.collector.time.time_ns", lambda: BASE + 1_000_000_000)

    assert collector.save_episode()
    assert "本条采集时长（episode 0）：1.00 秒（不含预热和保存耗时）。" in capsys.readouterr().out

    final_mcap = episode.committed_mcap_path
    assert final_mcap is not None and final_mcap.is_file()
    assert final_mcap.parent == config.collection.dataset_dir / "0"
    assert collector.last_post_validation["passed"]
    assert validate_mcap(final_mcap, config)["output_mcap_passed"]
    assert validate_video_assets(final_mcap, config)["passed"]
    videos = final_mcap.parent / "videos"
    assert {path.stem for path in videos.glob("*.mp4")} == set(config.cameras)
    manifest = json.loads((videos / "source_timeline.json").read_text(encoding="utf-8"))
    assert set(manifest["cameras"]) == set(config.cameras)
    assert all(entry["frame_count"] == 61 for entry in manifest["cameras"].values())
    assert not episode.pending_dir.exists()
    assert not episode.video_dir.exists()
    assert not episode.live_video_dir.exists()
    assert not collector._state_mcap_writers
    assert collector.state is CaptureState.IDLE
    assert inspect_mcap(final_mcap).camera_timing_diagnostics.keys() == config.cameras.keys()
    snapshot = config.collection.dataset_dir / "record_config.yaml"
    restored = load_config(snapshot, side=side, camera_setup=setup)
    assert set(restored.telemetry) == set(config.telemetry)
    assert set(restored.cameras) == set(config.cameras)


def test_formal_timing_failure_is_deleted_after_save(
    config_factory: object, collector_factory: object, capsys: pytest.CaptureFixture[str]
) -> None:
    config = config_factory()
    collector = collector_factory(config)
    _stage_episode(collector, state_offset_ns=500_000_000)

    assert not collector.save_episode()
    assert "本条采集时长" not in capsys.readouterr().out

    assert not (config.collection.dataset_dir / "0").exists()
    assert collector.last_post_validation is not None
    assert not collector.last_post_validation["passed"]
    assert "joint_camera_sync" in collector.last_post_validation["failed_checks"]


def test_publication_failure_retains_pending_and_rolls_video_back(
    config_factory: object, collector_factory: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = config_factory()
    collector = collector_factory(config)
    episode = _stage_episode(collector)

    def fail_mcap_publication(*args: object, **kwargs: object) -> None:
        raise OSError("synthetic MCAP publication failure")

    monkeypatch.setattr("yam_data_collection.video_pipeline._publish_mcap", fail_mcap_publication)

    assert not collector.save_episode()

    assert episode.final_mcap_path.is_file()
    assert (episode.live_video_dir / "cam_right.mp4").is_file()
    assert not (config.collection.dataset_dir / "0" / "videos").exists()
    assert episode.committed_mcap_path is None
    assert collector.state is CaptureState.IDLE


def test_video_identity_failure_is_not_published(
    config_factory: object, collector_factory: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = config_factory()
    collector = collector_factory(config)
    episode = _stage_episode(collector)

    monkeypatch.setattr(
        "yam_single_arm_collection.video_pipeline.validate_video_assets",
        lambda *args, **kwargs: {
            "passed": False,
            "per_camera": {"cam_right": {"passed": False, "frame_count_matches": False}},
        },
    )

    assert not collector.save_episode()
    assert episode.final_mcap_path.is_file()
    assert episode.live_video_dir.is_dir()
    assert episode.committed_mcap_path is None


def test_discard_closes_queued_writes_and_removes_partials(config_factory: object, collector_factory: object) -> None:
    config = config_factory()
    collector = collector_factory(config)
    episode = _stage_episode(collector)

    assert collector.discard_episode()

    assert not episode.pending_dir.exists()
    assert not episode.video_dir.exists()
    assert not collector._state_mcap_writers
    assert collector.state is CaptureState.IDLE


def test_background_failure_discards_instead_of_saving(config_factory: object, collector_factory: object) -> None:
    config = config_factory()
    collector = collector_factory(config)
    episode = _stage_episode(collector)
    collector._background_error = CaptureError("synthetic telemetry failure")

    assert not collector.save_episode()
    assert not episode.pending_dir.exists()
    assert not episode.video_dir.exists()
    assert not (config.collection.dataset_dir / "0").exists()
    assert collector.background_error is None


def test_cannot_construct_collector_with_an_extra_robot(config_factory: object) -> None:
    config = config_factory()
    endpoints = dict(config.telemetry)
    endpoints["left_follower"] = endpoints["right_follower"]
    with pytest.raises(ValueError, match="exactly the selected"):
        SingleArmCollector(replace(config, telemetry=endpoints))


def test_recorded_config_excludes_unused_devices(config_factory: object, collector_factory: object) -> None:
    config = config_factory("left")
    collector = collector_factory(config)
    collector._prepare_dataset_dir()
    raw = yaml.safe_load((config.collection.dataset_dir / "record_config.yaml").read_text(encoding="utf-8"))
    assert set(raw["cameras"]) == {"cam_left"}
    assert set(raw["telemetry"]) == {"left_leader", "left_follower"}


def test_refuses_existing_public_assets(config_factory: object, monkeypatch: pytest.MonkeyPatch) -> None:
    from yam_single_arm_collection.video_pipeline import publish_live_episode

    config = config_factory()
    path = pending_mcap_path(config.collection.dataset_dir, 0, filename="episode.mcap")
    path.parent.mkdir(parents=True)
    path.touch()
    final = config.collection.dataset_dir / "0"
    final.mkdir()
    protected = final / "episode.mcap"
    protected.write_bytes(b"existing episode")
    with pytest.raises(VideoPipelineError, match="refusing to overwrite"):
        publish_live_episode(path, config)
    assert protected.read_bytes() == b"existing episode"
