from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("transfer_configure_cpu", ROOT / "single_arm_transfer/configure_cpu.py")
configure_cpu = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(configure_cpu)


@pytest.mark.parametrize("count", [6, 8, 16, 20])
def test_default_cpu_partition_uses_valid_disjoint_groups(count: int) -> None:
    available = {index * 2 for index in range(count)}
    groups = configure_cpu.choose_cpus(available, [None, None, None])
    assert set.union(*groups.values()) == available
    assert sum(map(len, groups.values())) == len(available)
    assert len(groups["control"]) >= 2


@pytest.mark.parametrize(
    "requested",
    [
        ["0-2", "2-4", "5-7"],
        ["0-2", "3-4", "8-9"],
        ["0-2", None, None],
        ["3-1", "4-5", "6-7"],
    ],
)
def test_invalid_cpu_partitions_are_rejected(requested: list[str | None]) -> None:
    with pytest.raises(ValueError):
        configure_cpu.choose_cpus(set(range(8)), requested)


def test_configure_refuses_the_original_deployment(tmp_path: Path) -> None:
    groups = configure_cpu.choose_cpus(set(range(8)), [None, None, None])
    with pytest.raises(ValueError, match="original deployment"):
        configure_cpu.configure(tmp_path, groups, apply=True)


def test_configure_is_previewable_repeatable_and_preserves_hardware_identity(tmp_path: Path) -> None:
    (tmp_path / ".yam-single-arm-transfer").touch()
    copied = ("yam.sh", "yam_single_arm_collection/leader_motion.sh", "yam_data_collection/config.host.yaml")
    for relative in copied:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    groups = configure_cpu.choose_cpus(set(range(8)), [None, None, None])
    config_path = tmp_path / "yam_data_collection/config.host.yaml"
    original = config_path.read_bytes()
    original_config = yaml.safe_load(original)
    configure_cpu.configure(tmp_path, groups, apply=False)
    assert config_path.read_bytes() == original
    configure_cpu.configure(tmp_path, groups, apply=True)
    config = yaml.safe_load(config_path.read_text())
    assert config["cameras"] == original_config["cameras"]
    assert config["telemetry"] == original_config["telemetry"]
    for mode in ("30hz", "60hz"):
        assert config["capture_modes"][mode]["cameras"] == original_config["capture_modes"][mode]["cameras"]
        assert config["capture_modes"][mode]["collection_config"]["recorder_cpu_set"] == "4,5"
        assert config["capture_modes"][mode]["collection_config"]["live_video_cpu_set"] == "6,7"
    assert "taskset -c 0,1,2,3 .venv/bin/python" in (tmp_path / "yam.sh").read_text()
    assert (
        "taskset -c 0,1,2,3 .venv/bin/python" in (tmp_path / "yam_single_arm_collection/leader_motion.sh").read_text()
    )
    assert "taskset -c 4,5,6,7 .venv/bin/yam-data-collect camera-test" in (tmp_path / "yam.sh").read_text()
    assert len(list((tmp_path / "single_arm_transfer/backups").glob("*/yam.sh"))) == 1
    configure_cpu.configure(tmp_path, groups, apply=True)
    assert len(list((tmp_path / "single_arm_transfer/backups").glob("*/yam.sh"))) == 2
