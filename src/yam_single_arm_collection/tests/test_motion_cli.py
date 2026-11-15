from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from yam_single_arm_collection import cli


def test_move_cli_requires_confirmed_execute_pair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple] = []
    plan = tmp_path / "move.json"
    plan.write_text(
        json.dumps(
            {
                "side": "right",
                "start": [0.0] * 6 + [0.5],
                "target": [0.1] * 6 + [0.5],
                "duration_s": 2,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli,
        "move_follower",
        lambda plan, *, execute: calls.append((execute, plan.side)) or {},
    )

    assert cli.main(["move", "--side", "right", "--plan", str(plan)]) == 0
    with pytest.raises(SystemExit) as missing_confirm:
        cli.main(["move", "--side", "right", "--plan", str(plan), "--execute"])
    with pytest.raises(SystemExit) as confirm_without_execute:
        cli.main(["move", "--side", "right", "--plan", str(plan), "--confirm", "MOVE"])
    with pytest.raises(SystemExit) as confirm_during_preview:
        cli.main(["move", "--side", "right", "--plan", str(plan), "--confirm", "TYPO"])
    assert missing_confirm.value.code == 2
    assert confirm_without_execute.value.code == 2
    assert confirm_during_preview.value.code == 2
    assert calls == [(False, "right")]


def test_pair_pose_read_is_a_hardware_read_only_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []
    monkeypatch.setattr(
        cli,
        "read_pair_poses",
        lambda config: seen.append(config.raw["single_arm"]["side"]),
    )

    assert cli.main(["read-poses", "--side", "left"]) == 0

    assert seen == ["left"]


@pytest.mark.parametrize("side", ["left", "right"])
def test_motion_leader_launch_uses_opt_in_launcher(side: str, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple] = []
    monkeypatch.setattr(cli.os, "execv", lambda *args: calls.append(args))
    assert cli.main(["leader-motion", "--side", side, "--confirm-motion"]) == 0
    assert calls == [
        ("/bin/bash", ["bash", str(cli.ROOT / "yam_single_arm_collection/leader_motion.sh"), side, "--confirm-motion"])
    ]


def test_paired_preview_does_not_request_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []
    monkeypatch.setattr(
        cli, "preposition_pair", lambda config, path, *, execute, from_current: calls.append(execute) or {}
    )
    assert cli.main(["preview-preposition", "--side", "right", "--plan", str(tmp_path / "pair.json")]) == 0
    assert calls == [False]


def test_preposition_failure_does_not_construct_collector(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)

    def fail_move(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated preposition failure")

    monkeypatch.setattr(cli, "preposition_pair", fail_move)
    monkeypatch.setattr(cli, "SingleArmCollector", lambda config: pytest.fail("collection must not start"))
    with pytest.raises(SystemExit) as error:
        cli.main(["capture", "--side", "right", "--preposition", "plan.json", "--confirm", "PREPOSITION"])
    assert error.value.code == 1


def test_capture_requires_preposition_confirmation_before_opening_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        cli,
        "preposition_pair",
        lambda *args: pytest.fail("move requested without confirmation"),
    )
    with pytest.raises(SystemExit) as error:
        cli.main(["capture", "--side", "right", "--preposition", "not-a-plan.json"])
    assert error.value.code == 2


def test_capture_finishes_pair_preposition_before_starting_recording(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = tmp_path / "pair.json"
    plan.write_text("{}", encoding="utf-8")
    calls: list[str] = []
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)

    def preposition(config: object, plan_path: Path, *, from_current: bool) -> dict[str, Any]:
        calls.append("preposition")
        assert from_current
        assert plan_path == plan
        return {
            "side": "right",
            "plan_sha256": "abc",
            "duration_s": 2.0,
            "target": [0.1] * 6 + [0.5],
            "leader_synchronized": False,
        }

    class FakeCollector:
        def __init__(self, config: object) -> None:
            self.config = config
            calls.append("collector")
            assert config.task_info["collection_layout"]["preprogrammed_preposition"]["side"] == "right"

        def run_interactive(self) -> None:
            calls.append("recording")

    monkeypatch.setattr(cli, "preposition_pair", preposition)
    monkeypatch.setattr(cli, "SingleArmCollector", FakeCollector)

    assert (
        cli.main(
            [
                "capture",
                "--side",
                "right",
                "--preposition",
                str(plan),
                "--confirm",
                "PREPOSITION",
            ]
        )
        == 0
    )

    assert calls == ["preposition", "collector", "recording"]


def test_preposition_report_is_recorded_in_saved_dataset_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = tmp_path / "pair.json"
    plan.write_text("{}", encoding="utf-8")
    captured: list[dict] = []
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli, "preposition_pair", lambda *args, **kwargs: {"plan_sha256": "verified"})

    class FakeCollector:
        def __init__(self, config: object) -> None:
            captured.append(config.raw)

        def run_interactive(self) -> None:
            pass

    monkeypatch.setattr(cli, "SingleArmCollector", FakeCollector)
    assert (
        cli.main(
            [
                "capture",
                "--side",
                "right",
                "--preposition",
                str(plan),
                "--confirm",
                "PREPOSITION",
            ]
        )
        == 0
    )
    assert captured[0]["task_info"]["collection_layout"]["preprogrammed_preposition"]["plan_sha256"] == "verified"
    assert captured[0]["single_arm"]["preprogrammed_preposition"]["plan_sha256"] == "verified"


@pytest.mark.parametrize("execute", [False, True])
@pytest.mark.parametrize("start_args", [[], ["--from-current"]])
def test_move_defaults_to_current_start_with_target_only_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, execute: bool, start_args: list[str]
) -> None:
    path = tmp_path / "target.json"
    path.write_text(json.dumps({"side": "left", "target": [0.1] * 6 + [0.5], "duration_s": 2}), encoding="utf-8")
    seen: list[tuple[bool, bool]] = []
    monkeypatch.setattr(cli, "move_follower", lambda plan, *, execute: seen.append((plan.from_current, execute)) or {})
    args = ["move", "--side", "left", "--plan", str(path), *start_args]
    if execute:
        args += ["--execute", "--confirm", "MOVE"]
    assert cli.main(args) == 0
    assert seen == [(True, execute)]


@pytest.mark.parametrize("start_option", ["--from-current", "--from-recorded"])
def test_explicit_capture_start_mode_still_requires_a_preposition_plan(
    monkeypatch: pytest.MonkeyPatch, start_option: str
) -> None:
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli, "SingleArmCollector", lambda config: pytest.fail("devices must not open"))
    with pytest.raises(SystemExit) as error:
        cli.main(["capture", "--side", "left", start_option])
    assert error.value.code == 2


@pytest.mark.parametrize("start_args", [[], ["--from-current"]])
def test_capture_defaults_to_current_start_and_records_mode_before_collection(
    monkeypatch: pytest.MonkeyPatch, start_args: list[str]
) -> None:
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    calls: list[str] = []

    def preposition(config: object, path: Path, *, from_current: bool) -> dict[str, Any]:
        assert from_current
        calls.append("preposition")
        return {"start_mode": "current", "leader_before": [0.2] * 6 + [0.5], "follower_before": [0.1] * 6 + [0.5]}

    class FakeCollector:
        def __init__(self, config: object) -> None:
            report = config.task_info["collection_layout"]["preprogrammed_preposition"]
            assert report["start_mode"] == "current"
            assert report["leader_before"] != report["follower_before"]
            calls.append("collector")

        def run_interactive(self) -> None:
            calls.append("recording")

    monkeypatch.setattr(cli, "preposition_pair", preposition)
    monkeypatch.setattr(cli, "SingleArmCollector", FakeCollector)
    assert (
        cli.main(
            ["capture", "--side", "left", "--preposition", "target.json", *start_args, "--confirm", "PREPOSITION"]
        )
        == 0
    )
    assert calls == ["preposition", "collector", "recording"]


@pytest.mark.parametrize("start_args,expected", [([], True), (["--from-current"], True), (["--from-recorded"], False)])
def test_pair_preview_forwards_selected_start_mode_without_execution(
    monkeypatch: pytest.MonkeyPatch, start_args: list[str], expected: bool
) -> None:
    seen: list[tuple[bool, bool]] = []
    monkeypatch.setattr(
        cli,
        "preposition_pair",
        lambda config, path, *, from_current, execute: seen.append((from_current, execute)) or {},
    )
    assert cli.main(["preview-preposition", "--side", "left", "--plan", "target.json", *start_args]) == 0
    assert seen == [(expected, False)]


def test_recorded_start_mode_is_available_explicitly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "recorded.json"
    path.write_text(
        json.dumps({"side": "left", "start": [0.0] * 6 + [0.5], "target": [0.1] * 6 + [0.5], "duration_s": 2}),
        encoding="utf-8",
    )
    seen: list[Any] = []
    monkeypatch.setattr(cli, "move_follower", lambda plan, *, execute: seen.append(plan) or {})
    assert cli.main(["move", "--side", "left", "--plan", str(path), "--from-recorded"]) == 0
    assert not seen[0].from_current
    assert seen[0].start == (0.0,) * 6 + (0.5,)


@pytest.mark.parametrize("command", ["move", "preview-preposition", "capture"])
def test_current_and_recorded_start_flags_are_mutually_exclusive(command: str) -> None:
    plan_args = ["--preposition", "target.json"] if command == "capture" else ["--plan", "target.json"]
    with pytest.raises(SystemExit) as error:
        cli.main([command, "--side", "left", *plan_args, "--from-current", "--from-recorded"])
    assert error.value.code == 2


def test_capture_without_preposition_still_starts_without_motion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli, "preposition_pair", lambda *args, **kwargs: pytest.fail("unexpected preposition"))
    calls: list[str] = []

    class CaptureCollector:
        def __init__(self, config: object) -> None:
            calls.append("collector")

        def run_interactive(self) -> None:
            calls.append("capture")

    monkeypatch.setattr(cli, "SingleArmCollector", CaptureCollector)
    assert cli.main(["capture", "--side", "left"]) == 0
    assert calls == ["collector", "capture"]
