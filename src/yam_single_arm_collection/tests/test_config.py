from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from yam_data_collection.config import load_config as load_bimanual_config

from yam_single_arm_collection.config import DEFAULT_CONFIG, camera_roles, camera_test_config, load_config


ROOT = DEFAULT_CONFIG.parent.parent


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("setup", ["wrist", "wrist-top"])
@pytest.mark.parametrize("mode", ["60hz", "30hz"])
def test_selection_scopes_devices_and_keeps_profiles(side: str, setup: str, mode: str) -> None:
    config = load_config(side=side, camera_setup=setup, capture_mode=mode)
    original = load_bimanual_config(ROOT / "yam_data_collection/config.host.yaml", capture_mode=mode)

    assert set(config.telemetry) == {f"{side}_leader", f"{side}_follower"}
    assert tuple(config.cameras) == camera_roles(side, setup)
    assert set(config.raw["cameras"]) == set(config.cameras)
    assert set(config.raw["telemetry"]) == set(config.telemetry)
    for role, camera in config.cameras.items():
        assert camera == original.cameras[role]
    assert config.collection.dataset_dir == DEFAULT_CONFIG.parent / "data" / side / setup / mode
    assert not config.collection.dataset_dir.is_relative_to(original.collection.dataset_dir)
    assert config.collection.state_save_freq == 180
    assert config.collection.validation_profile == f"yam_{mode}"
    assert config.task_info["task_id"] == "REPLACE_TASK_ID"
    assert config.task_info["collection_layout"]["camera_roles"] == list(config.cameras)
    assert config.collection.recorder_cpu_set == original.collection.recorder_cpu_set
    assert camera_test_config(config).cameras == config.cameras


def test_left_wrist_keeps_actual_serial_and_distortion_model() -> None:
    camera = load_config(side="left").cameras["cam_left"]
    assert camera.serial == "REPLACE_CAM_LEFT_SERIAL"
    assert camera.intrinsics.distortion_model == "inverse_brown_conrady"


def test_default_is_wrist_only_at_60hz() -> None:
    config = load_config(side="right")
    assert list(config.cameras) == ["cam_right"]
    assert config.collection.target_fps == 60


def test_different_selections_never_share_output_directory() -> None:
    paths = {
        load_config(side=side, camera_setup=setup, capture_mode=mode).collection.dataset_dir
        for side in ("left", "right")
        for setup in ("wrist", "wrist-top")
        for mode in ("60hz", "30hz")
    }
    assert len(paths) == 8


def test_snapshot_loads_without_the_original_host_file(tmp_path: Path, config_factory: object) -> None:
    config = config_factory("left", "wrist-top")
    saved = deepcopy(config.raw)
    saved["single_arm"]["host_config"] = "/missing/original/host.yaml"
    path = tmp_path / "record_config.yaml"
    path.write_text(yaml.safe_dump(saved), encoding="utf-8")

    restored = load_config(path, side="left", camera_setup="wrist-top")

    assert restored.cameras == config.cameras
    assert restored.telemetry == config.telemetry
    assert restored.collection == config.collection
    assert restored.task_info == config.task_info


@pytest.mark.parametrize(
    "side,setup,mode", [("right", "wrist", "60hz"), ("left", "wrist", "60hz"), ("left", "wrist-top", "30hz")]
)
def test_snapshot_rejects_wrong_selection_or_mode(
    tmp_path: Path, config_factory: object, side: str, setup: str, mode: str
) -> None:
    config = config_factory("left", "wrist-top")
    path = tmp_path / "record_config.yaml"
    path.write_text(yaml.safe_dump(dict(config.raw)), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        load_config(path, side=side, camera_setup=setup, capture_mode=mode)


@pytest.mark.parametrize(
    "field,value",
    [("telemetry", {"right_follower": {"port": 9999}}), ("cameras", {"cam_high": {"serial": "unexpected"}})],
)
def test_snapshot_rejects_additional_device_roles(
    tmp_path: Path, config_factory: object, field: str, value: dict
) -> None:
    raw = deepcopy(config_factory("left").raw)
    raw[field].update(value)
    path = tmp_path / "record_config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly the selected"):
        load_config(path, side="left")


@pytest.mark.parametrize(
    "field,value",
    [
        ("video_codec", "libx265"),
        ("live_video_codec", "libx265"),
        ("target_fps", 30),
        ("validation_profile", "yam_30hz"),
    ],
)
def test_snapshot_rejects_inconsistent_video_or_timing_contract(
    tmp_path: Path, config_factory: object, field: str, value: object
) -> None:
    raw = deepcopy(config_factory().raw)
    raw["collection_config"][field] = value
    path = tmp_path / "record_config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path, side="right")


def test_no_source_configuration_is_modified(tmp_path: Path) -> None:
    host = ROOT / "yam_data_collection/config.host.yaml"
    before_host, before_extension = host.read_bytes(), DEFAULT_CONFIG.read_bytes()
    for side in ("left", "right"):
        for setup in ("wrist", "wrist-top"):
            load_config(side=side, camera_setup=setup)
    assert host.read_bytes() == before_host
    assert DEFAULT_CONFIG.read_bytes() == before_extension
