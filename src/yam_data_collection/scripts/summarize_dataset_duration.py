#!/usr/bin/env python3
"""Report total, average, per-episode and longest 10% durations for a YAM dataset."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from mcap.exceptions import McapError
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory


TIMESTAMP_TOPICS = {
    "/robot/arm_left_leader/joint_states",
    "/robot/arm_right_leader/joint_states",
    "/robot/arm_left_follow/joint_states",
    "/robot/arm_right_follow/joint_states",
    "/robot/camera/cam_high/image_raw/compressed/camera_info",
    "/robot/camera/cam_left/image_raw/compressed/camera_info",
    "/robot/camera/cam_right/image_raw/compressed/camera_info",
}
CAMERA_ROLES = ("cam_high", "cam_left", "cam_right")
TASK_INFO_TOPIC = "/task_info"
UNKNOWN_TASK_ID = "<missing>"
TASK_INFO_FIELDS = ("task_name", "task_description", "task_id")


@dataclass(frozen=True)
class EpisodeDuration:
    path: str
    episode_id: str
    duration_s: float
    first_timestamp_ns: int
    last_timestamp_ns: int
    video_duration_s: float | None = None
    folder_created_at: str | None = None
    folder_created_at_source: str | None = None
    task_name: str | None = None
    task_description: str | None = None
    task_id: str | None = None


def _folder_creation_time(path: Path) -> tuple[str | None, str | None]:
    """Return the folder creation time, falling back to mtime when needed."""
    try:
        result = subprocess.run(
            ["stat", "-c", "%W", str(path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        birth_epoch = int(result.stdout.strip())
        if birth_epoch > 0:
            return datetime.fromtimestamp(birth_epoch).strftime("%Y-%m-%d %H:%M:%S"), "birthtime"
    except (FileNotFoundError, OSError, subprocess.SubprocessError, ValueError, OverflowError):
        pass
    try:
        modified_epoch = path.stat().st_mtime
        return datetime.fromtimestamp(modified_epoch).strftime("%Y-%m-%d %H:%M:%S"), "mtime_fallback"
    except (OSError, ValueError, OverflowError):
        return None, None


def _published_mcaps(root: Path, include_pending: bool, include_derived: bool) -> list[Path]:
    result: list[Path] = []
    for path in sorted(root.rglob("*.mcap")):
        parts = path.relative_to(root).parts
        if not include_pending and any(part in {".pending", "videos_live"} for part in parts):
            continue
        if not include_derived and "derived" in parts:
            continue
        if path.name.endswith(".partial.mcap"):
            continue
        result.append(path)
    return result


def mcap_duration(path: Path) -> tuple[int, int]:
    first: int | None = None
    last: int | None = None
    with path.open("rb") as stream:
        reader = make_reader(stream)
        for _schema, _channel, message in reader.iter_messages(topics=sorted(TIMESTAMP_TOPICS), log_time_order=False):
            timestamp = int(message.log_time)
            first = timestamp if first is None else min(first, timestamp)
            last = timestamp if last is None else max(last, timestamp)
    if first is None or last is None:
        raise ValueError("no joint-state or CameraInfo timestamps found")
    return first, last


def mcap_task_info(path: Path) -> dict[str, str | None]:
    """Return task metadata from the first JSON message on ``/task_info``."""
    with path.open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for _schema, _channel, _message, decoded in reader.iter_decoded_messages(topics=[TASK_INFO_TOPIC]):
            payload = getattr(decoded, "data", None)
            if not isinstance(payload, str):
                continue
            try:
                task_info = json.loads(payload)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on {TASK_INFO_TOPIC}: {error}") from error
            if not isinstance(task_info, dict):
                raise ValueError(f"{TASK_INFO_TOPIC} JSON must be an object")
            return {
                name: str(task_info[name]) if task_info.get(name) is not None else None for name in TASK_INFO_FIELDS
            }
    return {name: None for name in TASK_INFO_FIELDS}


def _video_duration(path: Path) -> float | None:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return float(result.stdout.strip())
    except (FileNotFoundError, OSError, subprocess.SubprocessError, ValueError):
        return None


def _episode_video_duration(mcap_path: Path) -> float | None:
    video_dir = mcap_path.parent / "videos"
    if not video_dir.is_dir():
        video_dir = mcap_path.parent / "videos" / mcap_path.stem
    if not video_dir.is_dir():
        video_dir = mcap_path.parent.parent / "videos" / mcap_path.stem
    durations = [_video_duration(video_dir / f"{role}.mp4") for role in CAMERA_ROLES]
    available = [duration for duration in durations if duration is not None]
    return max(available) if available else None


def _format_duration(seconds: float) -> str:
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d} ({seconds:.2f}s)"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path, help="Dataset directory containing published episode MCAP files")
    parser.add_argument("--include-pending", action="store_true")
    parser.add_argument("--include-derived", action="store_true")
    parser.add_argument(
        "--json", action="store_true", dest="as_json", help="Print full JSON including episode, task and video details"
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.dataset_dir.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {root}")
    episodes: list[EpisodeDuration] = []
    task_infos: dict[str, dict[str, str | None]] = {}
    failures: list[dict[str, str]] = []
    for path in _published_mcaps(root, args.include_pending, args.include_derived):
        try:
            first, last = mcap_duration(path)
            task_info = mcap_task_info(path) if args.as_json else dict.fromkeys(TASK_INFO_FIELDS)
            folder_created_at, folder_created_at_source = (
                _folder_creation_time(path.parent) if args.as_json else (None, None)
            )
            episodes.append(
                EpisodeDuration(
                    str(path),
                    path.stem,
                    (last - first) / 1_000_000_000,
                    first,
                    last,
                    _episode_video_duration(path) if args.as_json else None,
                    folder_created_at,
                    folder_created_at_source,
                    task_name=task_info["task_name"],
                    task_description=task_info["task_description"],
                    task_id=task_info["task_id"],
                )
            )
            task_infos[str(path)] = task_info
        except (OSError, ValueError, RuntimeError, McapError) as exc:
            failures.append({"path": str(path), "error": str(exc)})
    total_mcap = sum(item.duration_s for item in episodes)
    average_duration = total_mcap / len(episodes) if episodes else 0.0
    longest_count = (len(episodes) + 9) // 10
    longest = sorted(episodes, key=lambda item: (-item.duration_s, item.path))[:longest_count]
    if not args.as_json:
        print(f"总时长：{_format_duration(total_mcap)}")
        print(f"总段数：{len(episodes)}")
        print(f"平均时长：{_format_duration(average_duration)}")
        print("每段时长：")
        for item in episodes:
            print(f"  {Path(item.path).relative_to(root)}  {item.duration_s:.2f} 秒")
        if not episodes:
            print("  无数据。")
        print("-" * 60)
        print(f"时长最长的 10%（前 {longest_count} 段，段数向上取整）：")
        for item in longest:
            print(f"  {Path(item.path).relative_to(root)}  {item.duration_s:.2f} 秒")
        if not longest:
            print("  无数据。")
        if failures:
            print(f"读取失败、未计入统计的文件：{len(failures)} 个", file=sys.stderr)
            for failure in failures[:3]:
                print(f"  {failure['path']}: {failure['error']}", file=sys.stderr)
            if len(failures) > 3:
                print("  更多失败详情可使用 --json 查看。", file=sys.stderr)
        return 1 if failures else 0
    video_values = [item.video_duration_s for item in episodes if item.video_duration_s is not None]
    total_video = sum(video_values) if video_values else None
    task_durations: dict[str, float] = {}
    task_episode_counts: dict[str, int] = {}
    task_groups_by_key: dict[tuple[str | None, ...], dict[str, Any]] = {}
    episode_task_groups: dict[str, str] = {}
    for item in episodes:
        task_id = item.task_id or UNKNOWN_TASK_ID
        task_durations[task_id] = task_durations.get(task_id, 0.0) + item.duration_s
        task_episode_counts[task_id] = task_episode_counts.get(task_id, 0) + 1
        info = task_infos[item.path]
        key = tuple(info[name] for name in TASK_INFO_FIELDS)
        if key not in task_groups_by_key:
            task_groups_by_key[key] = {
                "task_group": f"T{len(task_groups_by_key) + 1}",
                **info,
                "episode_count": 0,
                "duration_s": 0.0,
            }
        group = task_groups_by_key[key]
        group["episode_count"] += 1
        group["duration_s"] += item.duration_s
        episode_task_groups[item.path] = group["task_group"]
    task_groups_by_key = dict(
        sorted(
            task_groups_by_key.items(),
            key=lambda entry: (
                entry[0][2] or "",
                entry[0][0] or "",
                entry[0][1] or "",
            ),
        )
    )
    for index, group in enumerate(task_groups_by_key.values(), start=1):
        group["task_group"] = f"T{index}"
    for item in episodes:
        info = task_infos[item.path]
        key = tuple(info[name] for name in TASK_INFO_FIELDS)
        episode_task_groups[item.path] = task_groups_by_key[key]["task_group"]
    report: dict[str, Any] = {
        "dataset_dir": str(root),
        "episode_count": len(episodes),
        "total_mcap_duration_s": total_mcap,
        "average_mcap_duration_s": average_duration,
        "longest_10_percent_count": longest_count,
        "longest_10_percent": [
            {"path": item.path, "relative_path": str(Path(item.path).relative_to(root)), "duration_s": item.duration_s}
            for item in longest
        ],
        "total_video_coverage_duration_s": total_video,
        "task_id_duration_s": task_durations,
        "task_id_episode_counts": task_episode_counts,
        "tasks": list(task_groups_by_key.values()),
        "episodes": [{**asdict(item), "task_group": episode_task_groups[item.path]} for item in episodes],
        "failures": failures,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
