from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from yam_data_collection.mcap_writer import EpisodeMcapWriter
from yam_data_collection.models import RobotSnapshot


SCRIPT = Path(__file__).resolve().parents[2] / "summarize_dataset_duration.sh"
BASE = 1_700_000_000_000_000_000


def _write_episode(path: Path, config: Any, seconds: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with EpisodeMcapWriter(path, config.task_info) as writer:
        for timestamp in (BASE, BASE + round(seconds * 1_000_000_000)):
            for role in config.telemetry:
                writer.write_joint_state(
                    RobotSnapshot(role, timestamp, 1, (0.0,) * 7, (0.0,) * 7, (0.0,) * 7), timestamp
                )
            for camera in config.cameras.values():
                writer.write_camera_info(camera, timestamp)


def _run(folder: str, cwd: Path, *extra_args: str, as_json: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), folder, *(["--json"] if as_json else []), *extra_args],
        cwd=cwd,
        env={**os.environ, "PYTHONPATH": "/invalid-python-path", "PYTHONHOME": "/invalid-python-home"},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


@pytest.mark.parametrize("setup", ["wrist", "wrist-top"])
def test_recursive_total_counts_each_episode_once(tmp_path: Path, config_factory: Any, setup: str) -> None:
    config = config_factory("left", setup)
    folder = tmp_path / "采集 data"
    _write_episode(folder / "left" / "0" / "record.mcap", config, 12.5)
    _write_episode(folder / "left" / "1" / "record.mcap", config, 7.25)
    for excluded in (".pending/2/record.mcap", "derived/record.mcap", "3/record.partial.mcap"):
        _write_episode(folder / excluded, config, 100)

    result = _run(folder.name, tmp_path)

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["episode_count"] == 2
    assert report["total_mcap_duration_s"] == 19.75
    assert report["average_mcap_duration_s"] == 9.875
    assert report["longest_10_percent_count"] == 1
    assert report["longest_10_percent"] == [
        {"path": str(folder / "left/0/record.mcap"), "relative_path": "left/0/record.mcap", "duration_s": 12.5}
    ]
    assert not report["failures"]
    assert sorted(episode["duration_s"] for episode in report["episodes"]) == [7.25, 12.5]


def test_pending_and_derived_are_only_included_explicitly(tmp_path: Path, config_factory: Any) -> None:
    config = config_factory()
    _write_episode(tmp_path / ".pending" / "record.mcap", config, 2)
    _write_episode(tmp_path / "derived" / "record.mcap", config, 3)
    result = _run(".", tmp_path, "--include-pending", "--include-derived")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["episode_count"] == 2
    assert report["total_mcap_duration_s"] == 5


def test_empty_folder_and_missing_folder_are_distinguished(tmp_path: Path) -> None:
    result = _run(".", tmp_path)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["episode_count"] == 0
    assert report["total_mcap_duration_s"] == 0
    assert report["average_mcap_duration_s"] == 0
    assert report["longest_10_percent"] == []
    missing = _run("does-not-exist", tmp_path)
    assert missing.returncode != 0
    assert "does not exist" in missing.stderr


@pytest.mark.parametrize("count,top_count", [(0, 0), (1, 1), (9, 1), (10, 1), (11, 2), (20, 2), (23, 3)])
def test_summary_separates_all_episode_durations_from_longest_ten_percent(
    tmp_path: Path, config_factory: Any, count: int, top_count: int
) -> None:
    config = config_factory("left", "wrist-top")
    for index in range(1, count + 1):
        _write_episode(tmp_path / f"{index:02d}" / "same_name.mcap", config, index)
    _write_episode(tmp_path / ".pending" / "hidden.mcap", config, 1000)
    _write_episode(tmp_path / "derived" / "hidden.mcap", config, 1000)

    result = _run(".", tmp_path, as_json=False)

    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    total = count * (count + 1) / 2
    average = (count + 1) / 2 if count else 0
    assert lines[0].startswith("总时长：") and f"({total:.2f}s)" in lines[0]
    assert lines[1] == f"总段数：{count}"
    assert lines[2].startswith("平均时长：") and f"({average:.2f}s)" in lines[2]
    assert lines[3] == "每段时长："
    separator_index = lines.index("-" * 60)
    assert lines[4:separator_index] == (
        [f"  {index:02d}/same_name.mcap  {index:.2f} 秒" for index in range(1, count + 1)] if count else ["  无数据。"]
    )
    assert lines[separator_index + 1] == f"时长最长的 10%（前 {top_count} 段，段数向上取整）："
    assert lines[separator_index + 2 :] == (
        [f"  {index:02d}/same_name.mcap  {index:.2f} 秒" for index in range(count, count - top_count, -1)]
        if count
        else ["  无数据。"]
    )


def test_equal_duration_is_ordered_by_path_without_expanding_top_ten_percent(
    tmp_path: Path, config_factory: Any
) -> None:
    config = config_factory()
    for index in range(20, -1, -1):
        _write_episode(tmp_path / f"{index:02d}" / "record.mcap", config, 5)

    result = _run(".", tmp_path, as_json=False)

    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    separator_index = lines.index("-" * 60)
    assert lines[separator_index + 2 :] == [f"  {index:02d}/record.mcap  5.00 秒" for index in range(3)]


def test_summary_does_not_decode_unused_task_metadata(tmp_path: Path) -> None:
    from mcap.writer import Writer

    with (tmp_path / "record.mcap").open("wb") as stream:
        writer = Writer(stream)
        writer.start()
        joint = writer.register_channel("/robot/arm_left_follow/joint_states", "json", 0)
        task = writer.register_channel("/task_info", "unsupported-unused-encoding", 0)
        writer.add_message(joint, BASE + 3_000_000_000, b"{}", BASE + 3_000_000_000)
        writer.add_message(task, BASE + 90_000_000_000, b"invalid metadata", BASE + 90_000_000_000)
        writer.add_message(joint, BASE, b"{}", BASE)
        writer.finish()

    result = _run(".", tmp_path, as_json=False)

    assert result.returncode == 0, result.stderr
    assert "总段数：1" in result.stdout
    assert "总时长：00:00:03 (3.00s)" in result.stdout


@pytest.mark.parametrize("as_json", [False, True])
def test_invalid_mcap_is_reported_and_excluded_from_average(
    tmp_path: Path, config_factory: Any, as_json: bool
) -> None:
    _write_episode(tmp_path / "valid" / "record.mcap", config_factory(), 7.5)
    (tmp_path / "broken.mcap").write_bytes(b"not a valid mcap file")

    result = _run(".", tmp_path, as_json=as_json)

    assert result.returncode == 1
    if as_json:
        report = json.loads(result.stdout)
        assert report["episode_count"] == 1
        assert report["average_mcap_duration_s"] == 7.5
        assert len(report["failures"]) == 1
    else:
        assert "总段数：1" in result.stdout
        assert "平均时长：00:00:08 (7.50s)" in result.stdout
        assert "broken.mcap" in result.stderr
        assert "未计入统计的文件：1 个" in result.stderr
