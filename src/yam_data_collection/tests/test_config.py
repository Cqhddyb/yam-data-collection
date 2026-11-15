from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from yam_data_collection.config import load_camera_test_config, load_config


def test_config_resolves_dataset_relative_to_configuration_file(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
collection_config:
  dataset_dir: data/example
  state_save_freq: 100
  target_fps: 30
telemetry:
  left_leader: {host: 127.0.0.1, port: 1245}
  right_leader: {host: 127.0.0.1, port: 1244}
  left_follower: {host: 127.0.0.1, port: 1235}
  right_follower: {host: 127.0.0.1, port: 1234}
cameras:
  cam_high: {serial: high, width: 1280, height: 720, fps: 30}
  cam_left: {serial: left, width: 1280, height: 720, fps: 30}
  cam_right: {serial: right, width: 1280, height: 720, fps: 30}
task_info: {task_name: test}
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.collection.dataset_dir == tmp_path / "data" / "example"
    assert config.collection.video_preset == "medium"
    assert config.collection.video_threads == 0
    assert config.collection.live_video_codec == "libx264"
    assert config.collection.live_video_crf == 23
    assert config.collection.live_video_preset == "ultrafast"
    assert config.collection.live_video_threads == 1
    assert config.collection.live_video_queue_size == 120
    assert config.collection.recorder_cpu_set is None
    assert config.collection.live_video_cpu_set is None
    assert config.collection.validation_profile == "piper_compatible"
    assert config.cameras["cam_high"].intrinsics.k == (0.0,) * 9


def test_config_allows_mixed_camera_profiles(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
collection_config:
  dataset_dir: data/example
  state_save_freq: 100
  target_fps: 30
telemetry:
  left_leader: {host: 127.0.0.1, port: 1245}
  right_leader: {host: 127.0.0.1, port: 1244}
  left_follower: {host: 127.0.0.1, port: 1235}
  right_follower: {host: 127.0.0.1, port: 1234}
cameras:
  cam_high: {serial: high, width: 1280, height: 720, fps: 30}
  cam_left: {serial: left, width: 848, height: 480, fps: 60}
  cam_right: {serial: right, width: 848, height: 480, fps: 60}
task_info: {task_name: test}
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert [(camera.width, camera.height, camera.fps) for camera in config.cameras.values()] == [
        (1280, 720, 30),
        (848, 480, 60),
        (848, 480, 60),
    ]


@pytest.mark.parametrize(
    (
        "capture_mode",
        "collection_dataset_subdir",
        "camera_test_dataset_subdir",
        "validation_profile",
        "collection_camera_profiles",
        "camera_test_camera_profiles",
        "collection_video_threads",
        "camera_test_video_threads",
    ),
    (
        (
            "30hz",
            "30hz_720p_yuyv_threads1",
            "30hz_720p_yuyv_threads1",
            "yam_30hz",
            ((1280, 720, 30), (1280, 720, 30), (1280, 720, 30)),
            ((1280, 720, 30), (1280, 720, 30), (1280, 720, 30)),
            1,
            1,
        ),
        (
            "60hz",
            "60hz",
            "60hz",
            "yam_60hz",
            ((960, 540, 60), (848, 480, 60), (848, 480, 60)),
            ((960, 540, 60), (848, 480, 60), (848, 480, 60)),
            4,
            1,
        ),
    ),
)
def test_deployment_examples_resolve_each_capture_contract(
    capture_mode: str,
    collection_dataset_subdir: str,
    camera_test_dataset_subdir: str,
    validation_profile: str,
    collection_camera_profiles: tuple[tuple[int, int, int], ...],
    camera_test_camera_profiles: tuple[tuple[int, int, int], ...],
    collection_video_threads: int,
    camera_test_video_threads: int,
) -> None:
    project_dir = Path(__file__).parents[1]
    collection_config = load_config(project_dir / "config.example.yaml", capture_mode=capture_mode)
    camera_test_config = load_camera_test_config(project_dir / "camera_test.example.yaml", capture_mode=capture_mode)

    assert collection_config.capture_mode == capture_mode
    assert (
        collection_config.collection.dataset_dir
        == project_dir / "data" / "yam_pick_and_place" / collection_dataset_subdir
    )
    assert collection_config.collection.state_save_freq == 180.0
    assert collection_config.collection.validation_profile == validation_profile
    assert (
        tuple((camera.width, camera.height, camera.fps) for camera in collection_config.cameras.values())
        == collection_camera_profiles
    )
    assert [camera.stream_format for camera in collection_config.cameras.values()] == ["yuyv", "yuyv", "yuyv"]
    assert collection_config.collection.video_preset == "ultrafast"
    assert collection_config.collection.video_threads == collection_video_threads
    assert collection_config.collection.live_video_codec == "libx264"
    assert collection_config.collection.live_video_crf == 23
    assert collection_config.collection.live_video_preset == "ultrafast"
    assert collection_config.collection.live_video_threads == 1
    assert collection_config.collection.live_video_queue_size == 120
    assert collection_config.collection.live_video_cpu_set is None
    assert collection_config.raw["collection_config"]["video_threads"] == collection_video_threads
    assert collection_config.raw["selected_capture_mode"] == capture_mode
    assert "capture_modes" not in collection_config.raw
    assert camera_test_config.capture_mode == capture_mode
    assert camera_test_config.settings.output_dir == project_dir / "camera_tests" / camera_test_dataset_subdir
    assert camera_test_config.settings.validation_profile == validation_profile
    # Camera qualification exercises the live H.264 writer, not the offline
    # HEVC encoder. Its worker count therefore stays independent of the final
    # x265 setting, including in the 60 Hz profile.
    assert camera_test_config.settings.video_threads == camera_test_video_threads
    assert (
        tuple((camera.width, camera.height, camera.fps) for camera in camera_test_config.cameras.values())
        == camera_test_camera_profiles
    )
    assert [camera.stream_format for camera in camera_test_config.cameras.values()] == ["yuyv", "yuyv", "yuyv"]
    assert camera_test_config.raw["selected_capture_mode"] == capture_mode
    assert "capture_modes" not in camera_test_config.raw
    assert all(camera.intrinsics.is_calibrated for camera in collection_config.cameras.values())


def test_d405_640x480_60hz_validation_config_reserves_encoder_cpus() -> None:
    project_dir = Path(__file__).parents[1]

    config = load_config(project_dir / "config.d405_640x480.validation.yaml", capture_mode="60hz")

    assert config.collection.recorder_cpu_set == "16-19"
    assert config.collection.live_video_cpu_set == "20-23"
    assert config.raw["collection_config"]["recorder_cpu_set"] == "16-19"
    assert config.raw["collection_config"]["live_video_cpu_set"] == "20-23"


def test_d405_848x480_validation_config_is_isolated_from_640x480() -> None:
    project_dir = Path(__file__).parents[1]

    candidate_path = project_dir / "config.d405_848x480.validation.yaml"
    candidate = load_config(candidate_path, capture_mode="60hz")
    candidate_camera_test = load_camera_test_config(candidate_path, capture_mode="60hz")
    baseline = load_config(project_dir / "config.d405_640x480.validation.yaml", capture_mode="60hz")

    assert [(camera.width, camera.height, camera.fps) for camera in candidate.cameras.values()] == [
        (960, 540, 60),
        (848, 480, 60),
        (848, 480, 60),
    ]
    assert [(camera.width, camera.height, camera.fps) for camera in baseline.cameras.values()] == [
        (960, 540, 60),
        (640, 480, 60),
        (640, 480, 60),
    ]
    assert candidate.collection.dataset_dir == project_dir / "data" / "yam_pick_and_place_d405_848x480_validation" / "60hz"
    assert (
        candidate_camera_test.settings.output_dir
        == project_dir / "camera_tests" / "d405_848x480_validation" / "60hz"
    )
    assert candidate.collection.dataset_dir != baseline.collection.dataset_dir
    assert candidate.collection.recorder_cpu_set == "16-19"
    assert candidate.collection.live_video_cpu_set == "20-23"
    assert candidate.cameras["cam_left"].intrinsics.k[0] == pytest.approx(434.689628)
    assert candidate.cameras["cam_right"].intrinsics.k[0] == pytest.approx(438.548052)


def test_resolved_capture_config_is_reusable_without_a_mode(tmp_path: Path) -> None:
    project_dir = Path(__file__).parents[1]
    selected = load_config(project_dir / "config.example.yaml", capture_mode="60hz")
    record_path = tmp_path / "record_config.yaml"

    record_path.write_text(yaml.safe_dump(selected.raw, sort_keys=False), encoding="utf-8")

    reloaded = load_config(record_path)

    assert reloaded.capture_mode == "60hz"
    assert reloaded.collection.dataset_dir == selected.collection.dataset_dir
    assert reloaded.cameras["cam_high"] == selected.cameras["cam_high"]
    assert load_config(record_path, capture_mode="60hz").collection.dataset_dir == selected.collection.dataset_dir

    with pytest.raises(ValueError, match="was captured in 60hz"):
        load_config(record_path, capture_mode="30hz")


def test_multi_mode_configuration_requires_explicit_mode() -> None:
    project_dir = Path(__file__).parents[1]

    with pytest.raises(ValueError, match="choose one with --mode"):
        load_config(project_dir / "config.example.yaml")
    with pytest.raises(ValueError, match="choose one with --mode"):
        load_camera_test_config(project_dir / "camera_test.example.yaml")


def test_capture_mode_collection_overrides_tunables_but_not_the_fixed_contract(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
collection_config:
  dataset_dir: data/example
  state_save_freq: 100
  target_fps: 60
  validation_profile: yam_60hz
  video_crf: 23
  video_preset: medium
  video_threads: 4
  live_video_cpu_set: "12-15"
telemetry:
  left_leader: {host: 127.0.0.1, port: 1245}
  right_leader: {host: 127.0.0.1, port: 1244}
  left_follower: {host: 127.0.0.1, port: 1235}
  right_follower: {host: 127.0.0.1, port: 1234}
cameras:
  cam_high: {serial: high}
  cam_left: {serial: left}
  cam_right: {serial: right}
capture_modes:
  30hz:
    dataset_subdir: thirty
    validation_profile: yam_30hz
    collection_config:
      dataset_dir: ignored-mode-root
      state_save_freq: 1000
      target_fps: 60
      validation_profile: yam_60hz
      video_crf: 28
      video_preset: ultrafast
      video_threads: 1
      live_video_cpu_set: "20-23"
    cameras:
      cam_high: {width: 1280, height: 720, fps: 30}
      cam_left: {width: 1280, height: 720, fps: 30}
      cam_right: {width: 1280, height: 720, fps: 30}
  60hz:
    dataset_subdir: sixty
    validation_profile: yam_60hz
    cameras:
      cam_high: {width: 960, height: 540, fps: 60}
      cam_left: {width: 848, height: 480, fps: 60}
      cam_right: {width: 848, height: 480, fps: 60}
task_info: {task_name: test}
""",
        encoding="utf-8",
    )

    config = load_config(config_path, capture_mode="30hz")

    assert config.collection.dataset_dir == tmp_path / "data" / "example" / "thirty"
    assert config.collection.state_save_freq == 180.0
    assert config.collection.target_fps == 30.0
    assert config.collection.validation_profile == "yam_30hz"
    assert config.collection.video_crf == 28
    assert config.collection.video_preset == "ultrafast"
    assert config.collection.video_threads == 1
    assert config.collection.live_video_cpu_set == "20-23"
    assert config.raw["collection_config"] == {
        "dataset_dir": str(tmp_path / "data" / "example" / "thirty"),
        "state_save_freq": 180.0,
        "target_fps": 30.0,
        "validation_profile": "yam_30hz",
        "video_crf": 28,
        "video_preset": "ultrafast",
        "video_threads": 1,
        "live_video_cpu_set": "20-23",
    }


def test_collection_config_keeps_final_hevc_and_live_settings_independent(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
collection_config:
  dataset_dir: data/example
  state_save_freq: 100
  target_fps: 30
  video_codec: libx265
  video_crf: 28
  video_preset: slow
  video_threads: 3
  live_video_codec: h264
  live_video_crf: 31
  live_video_preset: ultrafast
  live_video_threads: 1
  live_video_queue_size: 75
  live_video_cpu_set: "16, 18-20, 19-21"
telemetry:
  left_leader: {host: 127.0.0.1, port: 1245}
  right_leader: {host: 127.0.0.1, port: 1244}
  left_follower: {host: 127.0.0.1, port: 1235}
  right_follower: {host: 127.0.0.1, port: 1234}
cameras:
  cam_high: {serial: high, width: 1280, height: 720, fps: 30}
  cam_left: {serial: left, width: 1280, height: 720, fps: 30}
  cam_right: {serial: right, width: 1280, height: 720, fps: 30}
task_info: {task_name: test}
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert (
        config.collection.video_codec,
        config.collection.video_crf,
        config.collection.video_preset,
        config.collection.video_threads,
    ) == ("libx265", 28, "slow", 3)
    assert (
        config.collection.live_video_codec,
        config.collection.live_video_crf,
        config.collection.live_video_preset,
        config.collection.live_video_threads,
        config.collection.live_video_queue_size,
        config.collection.live_video_cpu_set,
    ) == ("h264", 31, "ultrafast", 1, 75, "16,18-21")


@pytest.mark.parametrize("value", ["", "16-", "20-16", "16,,18", "-1", 16, [16, 17]])
def test_collection_config_rejects_invalid_live_video_cpu_set(tmp_path: Path, value: object) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "collection_config": {
                    "dataset_dir": "data/example",
                    "state_save_freq": 100,
                    "target_fps": 30,
                    "live_video_cpu_set": value,
                },
                "telemetry": {
                    "left_leader": {"host": "127.0.0.1", "port": 1245},
                    "right_leader": {"host": "127.0.0.1", "port": 1244},
                    "left_follower": {"host": "127.0.0.1", "port": 1235},
                    "right_follower": {"host": "127.0.0.1", "port": 1234},
                },
                "cameras": {
                    "cam_high": {"serial": "high", "width": 1280, "height": 720, "fps": 30},
                    "cam_left": {"serial": "left", "width": 1280, "height": 720, "fps": 30},
                    "cam_right": {"serial": "right", "width": 1280, "height": 720, "fps": 30},
                },
                "task_info": {"task_name": "test"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="collection_config.live_video_cpu_set"):
        load_config(config_path)


def test_collection_config_rejects_overlapping_recorder_and_live_encoder_cpu_sets(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
collection_config:
  dataset_dir: data/example
  state_save_freq: 100
  target_fps: 30
  recorder_cpu_set: "16-19"
  live_video_cpu_set: "18-21"
telemetry:
  left_leader: {host: 127.0.0.1, port: 1245}
  right_leader: {host: 127.0.0.1, port: 1244}
  left_follower: {host: 127.0.0.1, port: 1235}
  right_follower: {host: 127.0.0.1, port: 1234}
cameras:
  cam_high: {serial: high, width: 1280, height: 720, fps: 30}
  cam_left: {serial: left, width: 1280, height: 720, fps: 30}
  cam_right: {serial: right, width: 1280, height: 720, fps: 30}
task_info: {task_name: test}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must not overlap"):
        load_config(config_path)


def test_collection_config_accepts_h264_as_a_direct_final_codec(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
collection_config:
  dataset_dir: data/example
  state_save_freq: 100
  target_fps: 30
  video_codec: libx264
telemetry:
  left_leader: {host: 127.0.0.1, port: 1245}
  right_leader: {host: 127.0.0.1, port: 1244}
  left_follower: {host: 127.0.0.1, port: 1235}
  right_follower: {host: 127.0.0.1, port: 1234}
cameras:
  cam_high: {serial: high, width: 1280, height: 720, fps: 30}
  cam_left: {serial: left, width: 1280, height: 720, fps: 30}
  cam_right: {serial: right, width: 1280, height: 720, fps: 30}
task_info: {task_name: test}
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.collection.video_codec == "libx264"
    assert config.collection.live_video_codec == "libx264"


def test_collection_config_rejects_non_h264_live_codec_for_h264_final(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
collection_config:
  dataset_dir: data/example
  state_save_freq: 100
  target_fps: 30
  video_codec: h264
  live_video_codec: libx265
telemetry:
  left_leader: {host: 127.0.0.1, port: 1245}
  right_leader: {host: 127.0.0.1, port: 1244}
  left_follower: {host: 127.0.0.1, port: 1235}
  right_follower: {host: 127.0.0.1, port: 1234}
cameras:
  cam_high: {serial: high, width: 1280, height: 720, fps: 30}
  cam_left: {serial: left, width: 1280, height: 720, fps: 30}
  cam_right: {serial: right, width: 1280, height: 720, fps: 30}
task_info: {task_name: test}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="live_video_codec must be H.264"):
        load_config(config_path)


def test_collection_config_rejects_unknown_final_codec(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
collection_config:
  dataset_dir: data/example
  state_save_freq: 100
  target_fps: 30
  video_codec: vp9
telemetry:
  left_leader: {host: 127.0.0.1, port: 1245}
  right_leader: {host: 127.0.0.1, port: 1244}
  left_follower: {host: 127.0.0.1, port: 1235}
  right_follower: {host: 127.0.0.1, port: 1234}
cameras:
  cam_high: {serial: high, width: 1280, height: 720, fps: 30}
  cam_left: {serial: left, width: 1280, height: 720, fps: 30}
  cam_right: {serial: right, width: 1280, height: 720, fps: 30}
task_info: {task_name: test}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="collection_config.video_codec must be one of"):
        load_config(config_path)


def test_h264_final_config_preserves_30hz_and_60hz_camera_contracts() -> None:
    project_dir = Path(__file__).parents[1]
    h264_config_path = project_dir / "config.d405_848x480.h264_final.yaml"
    h264_30hz = load_config(h264_config_path, capture_mode="30hz")
    h264_60hz = load_config(h264_config_path, capture_mode="60hz")
    hevc_30hz = load_config(project_dir / "config.example.yaml", capture_mode="30hz")
    hevc_60hz = load_config(project_dir / "config.d405_848x480.validation.yaml", capture_mode="60hz")

    for config in (h264_30hz, h264_60hz):
        assert config.collection.video_codec == "libx264"
        assert config.collection.live_video_codec == "libx264"
        assert [camera.stream_format for camera in config.cameras.values()] == ["yuyv", "yuyv", "yuyv"]

    assert [(camera.width, camera.height, camera.fps) for camera in h264_30hz.cameras.values()] == [
        (1280, 720, 30),
        (1280, 720, 30),
        (1280, 720, 30),
    ]
    assert [(camera.width, camera.height, camera.fps) for camera in h264_60hz.cameras.values()] == [
        (960, 540, 60),
        (848, 480, 60),
        (848, 480, 60),
    ]
    assert h264_30hz.collection.validation_profile == "yam_30hz"
    assert h264_60hz.collection.validation_profile == "yam_60hz"
    assert h264_60hz.collection.recorder_cpu_set == "16-19"
    assert h264_60hz.collection.live_video_cpu_set == "20-23"
    assert h264_30hz.collection.dataset_dir != hevc_30hz.collection.dataset_dir
    assert h264_60hz.collection.dataset_dir != hevc_60hz.collection.dataset_dir
    assert h264_30hz.collection.dataset_dir != h264_60hz.collection.dataset_dir
    assert hevc_30hz.collection.video_codec == "libx265"
    assert hevc_60hz.collection.video_codec == "libx265"


def test_camera_test_mode_overrides_tunables_but_not_the_fixed_contract(tmp_path: Path) -> None:
    config_path = tmp_path / "camera_test.yaml"
    config_path.write_text(
        """
camera_test_config:
  output_dir: camera_tests
  video_threads: 4
cameras:
  cam_high: {serial: high}
  cam_left: {serial: left}
  cam_right: {serial: right}
capture_modes:
  30hz:
    dataset_subdir: thirty
    validation_profile: yam_30hz
    camera_test_config:
      output_dir: ignored-mode-root
      validation_profile: yam_60hz
      video_threads: 1
    cameras:
      cam_high: {width: 1280, height: 720, fps: 30}
      cam_left: {width: 1280, height: 720, fps: 30}
      cam_right: {width: 1280, height: 720, fps: 30}
  60hz:
    dataset_subdir: sixty
    validation_profile: yam_60hz
    cameras:
      cam_high: {width: 960, height: 540, fps: 60}
      cam_left: {width: 848, height: 480, fps: 60}
      cam_right: {width: 848, height: 480, fps: 60}
""",
        encoding="utf-8",
    )

    config = load_camera_test_config(config_path, capture_mode="30hz")

    assert config.settings.output_dir == tmp_path / "camera_tests" / "thirty"
    assert config.settings.validation_profile == "yam_30hz"
    assert config.settings.video_threads == 1
    assert config.raw["camera_test_config"]["output_dir"] == str(tmp_path / "camera_tests" / "thirty")
    assert config.raw["camera_test_config"]["validation_profile"] == "yam_30hz"
    assert config.raw["camera_test_config"]["video_threads"] == 1


def test_yam_60hz_profile_rejects_a_non_60hz_camera(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
collection_config:
  dataset_dir: data/example
  state_save_freq: 100
  validation_profile: yam-60hz
telemetry:
  left_leader: {host: 127.0.0.1, port: 1245}
  right_leader: {host: 127.0.0.1, port: 1244}
  left_follower: {host: 127.0.0.1, port: 1235}
  right_follower: {host: 127.0.0.1, port: 1234}
cameras:
  cam_high: {serial: high, width: 848, height: 480, fps: 60}
  cam_left: {serial: left, width: 848, height: 480, fps: 30}
  cam_right: {serial: right, width: 848, height: 480, fps: 60}
task_info: {task_name: test}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires every camera at 60 Hz"):
        load_config(config_path)
