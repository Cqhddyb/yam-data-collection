from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "side,expected,absent",
    [
        ("left", ("REPLACE_LEADER_L_CAN_SERIAL", "REPLACE_FOLLOWER_L_CAN_SERIAL"), ("REPLACE_LEADER_R_CAN_SERIAL", "REPLACE_FOLLOWER_R_CAN_SERIAL")),
        ("right", ("REPLACE_LEADER_R_CAN_SERIAL", "REPLACE_FOLLOWER_R_CAN_SERIAL"), ("REPLACE_LEADER_L_CAN_SERIAL", "REPLACE_FOLLOWER_L_CAN_SERIAL")),
    ],
)
def test_can_preview_selects_only_one_pair_without_system_commands(
    tmp_path: Path, side: str, expected: tuple[str, ...], absent: tuple[str, ...]
) -> None:
    (tmp_path / ".yam-single-arm-transfer").touch()
    for relative in (
        "single_arm_transfer/install_can_single.sh",
        "deploy/install_can.sh",
        "deploy/90-yam-can.rules",
        "deploy/yam-can@.service",
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    completed = subprocess.run(
        ["bash", str(tmp_path / "single_arm_transfer/install_can_single.sh"), side, "--preview"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    assert all(serial in completed.stdout for serial in expected)
    assert all(serial not in completed.stdout for serial in absent)
    assert completed.stdout.count('NAME="can_') == 2
    assert "restart-ms 0" in completed.stdout
    assert "Preview only; no system or hardware state was changed." in completed.stdout
