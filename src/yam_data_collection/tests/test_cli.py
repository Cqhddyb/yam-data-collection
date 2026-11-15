from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from yam_data_collection import cli
from yam_data_collection.cli import main
from yam_data_collection.offline_repair import NoRepairNeededError, OfflineRepairError


def test_transcode_command_exposes_episode_and_overwrite_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["transcode", "--help"])

    assert raised.value.code == 0
    help_text = capsys.readouterr().out
    assert "Path to a pending .pending/<id>/<id>.mcap" in help_text
    assert "--config CONFIG" in help_text
    assert "--mode {30hz,60hz}" in help_text
    assert "--overwrite" in help_text


def test_publish_pending_command_exposes_h264_recovery_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["publish-pending", "--help"])

    assert raised.value.code == 0
    help_text = capsys.readouterr().out
    assert "H.264-final" in help_text
    assert "Path to a pending .pending/<id>/<id>.mcap" in help_text
    assert "--config CONFIG" in help_text
    assert "--mode {30hz,60hz}" in help_text
    assert "--overwrite" in help_text


def test_publish_pending_command_forwards_the_h264_final_configuration(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    config = object()
    result = SimpleNamespace(episode_id=7, final_mcap_path=Path("dataset/7.mcap"))
    received: list[tuple[Path, object, bool]] = []

    monkeypatch.setattr("yam_data_collection.cli.load_config", lambda *args, **kwargs: config)

    def publish(path: Path, received_config: object, *, overwrite: bool) -> SimpleNamespace:
        received.append((path, received_config, overwrite))
        return result

    monkeypatch.setattr("yam_data_collection.cli.publish_pending_episode", publish)

    assert main(
        [
            "publish-pending",
            ".pending/7/7.mcap",
            "--config",
            "config.yaml",
            "--mode",
            "60hz",
            "--overwrite",
        ]
    ) == 0

    assert received == [(Path(".pending/7/7.mcap"), config, True)]
    assert "Published H.264 final episode 7: dataset/7.mcap" in capsys.readouterr().out


def test_repair_30hz_command_exposes_derived_dataset_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["repair-30hz", "--help"])

    assert raised.value.code == 0
    help_text = capsys.readouterr().out
    assert ".pending/<id>/<id>.mcap" in help_text
    assert "--config CONFIG" in help_text
    assert "--mode {30hz,60hz}" in help_text
    assert "--output-dataset OUTPUT_DATASET" in help_text
    assert "--overwrite" in help_text


def test_capture_auto_repair_command_exposes_30hz_and_output_arguments(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["capture-auto-repair", "--help"])

    assert raised.value.code == 0
    help_text = capsys.readouterr().out
    assert "--config CONFIG" in help_text
    assert "--mode {30hz,60hz}" in help_text
    assert "--output-dataset OUTPUT_DATASET" in help_text
    assert "--preview" in help_text


def test_capture_command_enables_terminal_episode_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = object()
    ran: list[bool] = []

    class FakeCollector:
        def __init__(self, received_config: object, **kwargs: object) -> None:
            assert received_config is config
            assert kwargs == {
                "enable_preview": False,
                "auto_delete_failed_validation": True,
                "enable_post_save_review": True,
            }

        def run_interactive(self) -> None:
            ran.append(True)

    monkeypatch.setattr("yam_data_collection.cli.load_config", lambda *args, **kwargs: config)
    monkeypatch.setattr("yam_data_collection.cli.EpisodeCollector", FakeCollector)

    assert main(["capture", "--config", "config.yaml", "--mode", "30hz"]) == 0
    assert ran == [True]


def test_capture_help_exposes_optional_preview(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["capture", "--help"])

    assert raised.value.code == 0
    assert "--preview" in capsys.readouterr().out


def test_capture_source_video_timeline_is_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    config = object()
    received: list[dict[str, object]] = []

    class FakeCollector:
        def __init__(self, received_config: object, **kwargs: object) -> None:
            assert received_config is config
            received.append(kwargs)

        def run_interactive(self) -> None:
            return None

    monkeypatch.setattr("yam_data_collection.cli.load_config", lambda *args, **kwargs: config)
    monkeypatch.setattr("yam_data_collection.cli.EpisodeCollector", FakeCollector)

    assert main(
        [
            "capture",
            "--config",
            "config.yaml",
            "--mode",
            "60hz",
            "--video-timeline",
            "source",
        ]
    ) == 0
    assert received == [
        {
            "enable_preview": False,
            "auto_delete_failed_validation": True,
            "enable_post_save_review": True,
            "video_timeline": "source",
        }
    ]


@pytest.mark.parametrize(
    ("final_codec", "expected_codecs"),
    (("libx264", ("h264",)), ("libx265", ("hevc",))),
)
def test_validate_uses_the_configured_final_video_codec(
    final_codec: str,
    expected_codecs: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Final-asset validation must stay strict for both H.264 and legacy HEVC datasets."""
    config = SimpleNamespace(
        collection=SimpleNamespace(validation_profile="yam_30hz", video_codec=final_codec),
        cameras={"cam_high": object(), "cam_left": object(), "cam_right": object()},
    )
    observed_codecs: list[tuple[str, ...] | None] = []

    monkeypatch.setattr("yam_data_collection.cli.load_config", lambda *args, **kwargs: config)
    monkeypatch.setattr("yam_data_collection.cli.build_validation_policy", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        "yam_data_collection.cli.validate_mcap",
        lambda *args, **kwargs: {"output_mcap_passed": True},
    )
    monkeypatch.setattr("yam_data_collection.cli.print_validation", lambda *args, **kwargs: None)

    def validate_assets(*_args: object, **kwargs: object) -> dict[str, object]:
        observed_codecs.append(kwargs.get("expected_video_codecs"))
        return {"passed": True, "per_camera": {}}

    monkeypatch.setattr("yam_data_collection.cli.validate_video_assets", validate_assets)

    assert cli._validate(Path("0.mcap"), Path("config.yaml"), True, "config", "30hz") == 0
    assert observed_codecs == [expected_codecs]


@pytest.mark.parametrize("mode", ["30hz", "60hz"])
def test_capture_preview_flag_is_forwarded_for_both_modes(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = object()
    received: list[dict[str, object]] = []

    class FakeCollector:
        def __init__(self, received_config: object, **kwargs: object) -> None:
            assert received_config is config
            received.append(kwargs)

        def run_interactive(self) -> None:
            return None

    monkeypatch.setattr("yam_data_collection.cli.load_config", lambda *args, **kwargs: config)
    monkeypatch.setattr("yam_data_collection.cli.EpisodeCollector", FakeCollector)

    assert main(["capture", "--config", "config.yaml", "--mode", mode, "--preview"]) == 0
    assert received == [
        {
            "enable_preview": True,
            "auto_delete_failed_validation": True,
            "enable_post_save_review": True,
        }
    ]


def test_capture_auto_repair_preview_flag_is_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(capture_mode="30hz")
    received: list[dict[str, object]] = []

    class FakeCollector:
        def __init__(
            self,
            received_config: object,
            *,
            on_episode_saved: object,
            **kwargs: object,
        ) -> None:
            assert received_config is config
            assert callable(on_episode_saved)
            received.append(kwargs)

        def run_interactive(self) -> None:
            return None

    monkeypatch.setattr("yam_data_collection.cli.load_config", lambda *args, **kwargs: config)
    monkeypatch.setattr("yam_data_collection.cli.EpisodeCollector", FakeCollector)

    assert main(
        ["capture-auto-repair", "--config", "config.yaml", "--mode", "30hz", "--preview"]
    ) == 0
    assert received == [{"enable_preview": True}]


def test_repair_30hz_prints_each_planned_event_and_insert_totals(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    events = {
        "cam_high": (),
        "cam_left": (
            SimpleNamespace(
                source_device_frame_number=100,
                trigger_device_frame_number=103,
                original_gap_ns=100_000_000,
                synthetic_timestamp_ns=1_733_333_333,
            ),
            SimpleNamespace(
                source_device_frame_number=400,
                trigger_device_frame_number=403,
                original_gap_ns=99_900_000,
                synthetic_timestamp_ns=9_033_333_333,
            ),
        ),
        "cam_right": (),
    }
    plan = SimpleNamespace(episode_id=48, event_count=2, events_by_role=events)

    def fake_repair(*args: object, **kwargs: object) -> SimpleNamespace:
        callback = kwargs["on_plan_ready"]
        assert callable(callback)
        callback(plan)
        return SimpleNamespace(
            episode_id=48,
            output_mcap=Path("derived/48.mcap"),
            output_video_dir=Path("derived/videos/48"),
            events_by_role=events,
            manifest_path=Path("derived/repair_manifests/48.json"),
        )

    monkeypatch.setattr("yam_data_collection.cli.load_config", lambda *args, **kwargs: object())
    monkeypatch.setattr("yam_data_collection.cli.repair_30hz_episode", fake_repair)

    assert main(["repair-30hz", "48.mcap", "--config", "config.yaml", "--mode", "30hz"]) == 0

    output = capsys.readouterr().out
    assert "Repair plan for episode 48: 2 eligible delta=3 gap(s)." in output
    assert "cam_left event 1: device frames 100 -> 103, gap=100.0ms" in output
    assert "duplicate previous frame at 1733333333ns" in output
    assert "cam_left event 2: device frames 400 -> 403, gap=99.9ms" in output
    assert "Planned inserted frames: 2" in output
    assert "cam_high: 0 previous-frame duplicate(s)" in output
    assert "cam_left: 2 previous-frame duplicate(s)" in output
    assert "cam_right: 0 previous-frame duplicate(s)" in output
    assert "Total inserted frames: 2" in output


def test_capture_auto_repair_closes_repairs_and_recreates_collector(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SimpleNamespace(capture_mode="30hz")
    event = SimpleNamespace(
        source_device_frame_number=100,
        trigger_device_frame_number=103,
        original_gap_ns=100_000_000,
        synthetic_timestamp_ns=1_733_333_333,
    )
    events = {"cam_high": (), "cam_left": (event,), "cam_right": ()}
    plan = SimpleNamespace(episode_id=7, event_count=1, events_by_role=events)
    collectors: list[object] = []
    repaired_paths: list[Path] = []

    class FakeCollector:
        def __init__(self, received_config: object, *, on_episode_saved: object) -> None:
            assert received_config is config
            self.callback = on_episode_saved
            self.index = len(collectors)
            self.returned = False
            collectors.append(self)

        def run_interactive(self) -> None:
            if self.index == 0:
                assert callable(self.callback)
                assert self.callback(Path("raw/.pending/7/7.mcap")) is True
            self.returned = True

    def fake_repair(path: Path, received_config: object, **kwargs: object) -> SimpleNamespace:
        assert received_config is config
        assert collectors[0].returned
        repaired_paths.append(path)
        assert kwargs["output_dataset"] == Path("derived")
        callback = kwargs["on_plan_ready"]
        assert callable(callback)
        callback(plan)
        return SimpleNamespace(
            episode_id=7,
            output_mcap=Path("derived/7.mcap"),
            output_video_dir=Path("derived/videos/7"),
            events_by_role=events,
            manifest_path=Path("derived/repair_manifests/7.json"),
        )

    monkeypatch.setattr("yam_data_collection.cli.load_config", lambda *args, **kwargs: config)
    monkeypatch.setattr("yam_data_collection.cli.EpisodeCollector", FakeCollector)
    monkeypatch.setattr("yam_data_collection.cli.build_repair_plan", lambda *args, **kwargs: plan)
    monkeypatch.setattr("yam_data_collection.cli.repair_30hz_episode", fake_repair)

    assert main(
        [
            "capture-auto-repair",
            "--config",
            "config.yaml",
            "--mode",
            "30hz",
            "--output-dataset",
            "derived",
        ]
    ) == 0

    assert len(collectors) == 2
    assert repaired_paths == [Path("raw/.pending/7/7.mcap")]
    output = capsys.readouterr().out
    assert "Episode 7: eligible for 1 offline frame repair(s)" in output
    assert "Capture inputs are closed. Starting verified repair and HEVC encoding" in output
    assert "Repair plan for episode 7: 1 eligible delta=3 gap(s)." in output
    assert "cam_left event 1: device frames 100 -> 103, gap=100.0ms" in output
    assert "Total inserted frames: 1" in output
    assert "Restarting interactive capture." in output


def test_capture_auto_repair_classifies_saved_episodes_and_recovers_from_repair_error(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SimpleNamespace(capture_mode="30hz")
    plan = SimpleNamespace(episode_id=3, event_count=1, events_by_role={})
    collectors: list[object] = []

    class FakeCollector:
        def __init__(self, received_config: object, *, on_episode_saved: object) -> None:
            assert received_config is config
            self.callback = on_episode_saved
            self.index = len(collectors)
            collectors.append(self)

        def run_interactive(self) -> None:
            if self.index == 0:
                assert callable(self.callback)
                assert self.callback(Path("raw/.pending/1/1.mcap")) is False
                assert self.callback(Path("raw/.pending/2/2.mcap")) is False
                assert self.callback(Path("raw/.pending/3/3.mcap")) is True

    def fake_plan(path: Path, config: object) -> object:
        if path.stem == "1":
            raise NoRepairNeededError("no eligible event")
        if path.stem == "2":
            raise OfflineRepairError("delta=4 gap remains")
        return plan

    def failed_repair(*args: object, **kwargs: object) -> object:
        raise OfflineRepairError("video identity check failed")

    monkeypatch.setattr("yam_data_collection.cli.load_config", lambda *args, **kwargs: config)
    monkeypatch.setattr("yam_data_collection.cli.EpisodeCollector", FakeCollector)
    monkeypatch.setattr("yam_data_collection.cli.build_repair_plan", fake_plan)
    monkeypatch.setattr("yam_data_collection.cli.repair_30hz_episode", failed_repair)

    assert main(["capture-auto-repair", "--config", "config.yaml", "--mode", "30hz"]) == 0

    assert len(collectors) == 2
    output = capsys.readouterr().out
    assert "Episode 1: no eligible approximately 100 ms/delta=3 gap" in output
    assert "Episode 2: not automatically repairable; raw episode retained: delta=4 gap remains" in output
    assert "Automatic repair failed for episode 3; raw episode retained: video identity check failed" in output
    assert "Restarting interactive capture." in output


def test_capture_auto_repair_rejects_resolved_60hz_mode(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "yam_data_collection.cli.load_config",
        lambda *args, **kwargs: SimpleNamespace(capture_mode="60hz"),
    )

    with pytest.raises(SystemExit) as raised:
        main(["capture-auto-repair", "--config", "config.yaml", "--mode", "60hz"])

    assert raised.value.code == 2
    assert "capture-auto-repair requires the resolved capture mode to be 30hz" in capsys.readouterr().err
