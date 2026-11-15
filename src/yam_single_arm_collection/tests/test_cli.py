from __future__ import annotations

import json
from pathlib import Path

import pytest

from yam_single_arm_collection import cli


@pytest.mark.parametrize("command", ["show-config", "check", "capture", "camera-test", "follower", "leader"])
def test_side_is_never_chosen_implicitly(command: str) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main([command])
    assert error.value.code == 2


@pytest.mark.parametrize("command", ["follower", "leader"])
def test_motor_start_requires_explicit_motion_confirmation(command: str) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main([command, "--side", "right"])
    assert error.value.code == 2


@pytest.mark.parametrize("side,suffix", [("left", "l"), ("right", "r")])
@pytest.mark.parametrize("command", ["follower", "leader"])
def test_controller_start_only_delegates_selected_role(
    side: str, suffix: str, command: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple] = []
    monkeypatch.setattr(cli.os, "execv", lambda *args: calls.append(args))
    assert cli.main([command, "--side", side, "--confirm-motion"]) == 0
    assert calls == [("/bin/bash", ["bash", str(cli.ROOT / "yam.sh"), f"{command}-{suffix}", "--confirm-motion"])]


def test_show_config_has_no_hardware_connections(monkeypatch: pytest.MonkeyPatch, capsys: object) -> None:
    def fail_on_hardware(*args: object, **kwargs: object) -> None:
        raise AssertionError("show-config must not create a hardware collector")

    monkeypatch.setattr(cli, "SingleArmCollector", fail_on_hardware)
    assert cli.main(["show-config", "--side", "left", "--cameras", "wrist-top"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert set(result["telemetry"]) == {"left_leader", "left_follower"}
    assert set(result["cameras"]) == {"cam_left", "cam_high"}


def test_capture_requires_terminal_before_any_hardware_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(cli, "SingleArmCollector", lambda config: pytest.fail("hardware opened"))
    with pytest.raises(SystemExit) as error:
        cli.main(["capture", "--side", "right"])
    assert error.value.code == 2


def test_check_does_not_require_other_side_or_start_cameras(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list = []

    class FakeCollector:
        def __init__(self, config: object) -> None:
            seen.append(set(config.telemetry))

        def check(self, *, start_cameras: bool) -> None:
            seen.append(start_cameras)

        def close(self) -> None:
            seen.append("closed")

    monkeypatch.setattr(cli, "SingleArmCollector", FakeCollector)
    assert cli.main(["check", "--side", "right"]) == 0
    assert seen == [{"right_leader", "right_follower"}, False, "closed"]


def test_bad_camera_test_duration_fails_without_starting_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.os, "sched_getaffinity", lambda pid: set(range(20)))
    monkeypatch.setattr(cli.os, "sched_setaffinity", lambda pid, cpus: None)

    class FakeQualification:
        def __init__(self, config: object) -> None:
            self.config = config

        def run(self, **kwargs: object) -> None:
            raise ValueError("duration must be positive")

    monkeypatch.setattr(cli, "CameraQualification", FakeQualification)
    with pytest.raises(SystemExit) as error:
        cli.main(["camera-test", "--side", "right", "--duration", "-1"])
    assert error.value.code == 1


def test_shell_wrapper_works_from_an_unrelated_directory(tmp_path: Path) -> None:
    import subprocess

    completed = subprocess.run(
        ["bash", str(cli.ROOT / "yam_single_arm.sh"), "show-config", "--side", "right"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert set(result["cameras"]) == {"cam_right"}
