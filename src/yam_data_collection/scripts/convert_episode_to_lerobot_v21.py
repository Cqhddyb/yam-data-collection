#!/usr/bin/env python3
"""Convert one YAM MCAP episode and its videos to a LeRobot v2.1-style dataset.

This is intentionally a one-episode, offline converter. It does not access robot
hardware or modify source MCAP/video files. Actions are leader joint positions;
observations are follower joint positions.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory


FPS = 30
CAMERAS = ("cam_high", "cam_left", "cam_right")
TOPICS = {
    "left_leader": "/robot/arm_left_leader/joint_states",
    "right_leader": "/robot/arm_right_leader/joint_states",
    "left_follower": "/robot/arm_left_follow/joint_states",
    "right_follower": "/robot/arm_right_follow/joint_states",
    "task": "/task_info",
}
CAMERA_TOPIC_PREFIX = "/robot/camera/"
JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper")


def _read_episode(path: Path) -> tuple[dict[str, list[tuple[int, np.ndarray]]], dict[str, Any]]:
    values: dict[str, list[tuple[int, np.ndarray]]] = {key: [] for key in TOPICS if key != "task"}
    task: dict[str, Any] = {}
    with path.open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for _schema, channel, message, decoded in reader.iter_decoded_messages(log_time_order=True):
            if channel.topic == TOPICS["task"]:
                try:
                    payload = json.loads(decoded.data)
                    if isinstance(payload, dict):
                        task = payload
                except (AttributeError, json.JSONDecodeError):
                    continue
                continue
            role = next((name for name, topic in TOPICS.items() if topic == channel.topic), None)
            if role is None:
                continue
            names = list(decoded.name)
            positions = np.asarray(decoded.position, dtype=np.float64)
            if len(names) != len(JOINT_NAMES) or set(names) != set(JOINT_NAMES) or positions.size != len(names):
                raise ValueError(f"{path}: unexpected joint layout on {channel.topic}: {names}")
            ordered = np.asarray([positions[names.index(name)] for name in JOINT_NAMES], dtype=np.float64)
            values[role].append((int(message.publish_time), ordered))
    missing = [role for role, samples in values.items() if not samples]
    if missing:
        raise ValueError(f"missing robot state topics: {', '.join(missing)}")
    if not task.get("task_name"):
        raise ValueError("MCAP has no task_name in /task_info")
    return values, task


def _read_timeline(path: Path) -> dict[str, Any]:
    timeline_path = path.parent / "videos" / "source_timeline.json"
    if not timeline_path.is_file():
        raise ValueError(f"source-timestamp map is missing: {timeline_path}")
    value = json.loads(timeline_path.read_text())
    if value.get("mode") != "source" or not isinstance(value.get("cameras"), dict):
        raise ValueError("expected source-timestamp video metadata")
    return value


def _ffprobe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,avg_frame_rate,nb_frames,codec_name", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise ValueError(f"no video stream in {path}")
    return streams[0]


def _nearest(samples: list[tuple[int, np.ndarray]], targets: np.ndarray) -> np.ndarray:
    stamps = np.fromiter((item[0] for item in samples), dtype=np.int64)
    values = np.stack([item[1] for item in samples])
    indexes = np.searchsorted(stamps, targets, side="left").clip(0, len(stamps) - 1)
    previous = np.maximum(indexes - 1, 0)
    choose_previous = np.abs(stamps[previous] - targets) <= np.abs(stamps[indexes] - targets)
    indexes = np.where(choose_previous, previous, indexes)
    return values[indexes]


def convert_episode(mcap: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    source_video_dir = mcap.parent / "videos"
    inputs = {role: source_video_dir / f"{role}.mp4" for role in CAMERAS}
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing source videos: " + ", ".join(missing))

    states, task = _read_episode(mcap)
    timeline = _read_timeline(mcap)
    camera_meta = timeline["cameras"]
    origin_ns = int(timeline["origin_timestamp_ns"])
    common_first_ns = max(int(camera_meta[role]["first_timestamp_ns"]) for role in CAMERAS)
    common_last_ns = min(int(camera_meta[role]["last_timestamp_ns"]) for role in CAMERAS)
    sample_count = int((common_last_ns - common_first_ns) // (1_000_000_000 // FPS)) + 1
    if sample_count < 1:
        raise ValueError("camera streams have no common time interval")
    targets_ns = common_first_ns + np.arange(sample_count, dtype=np.int64) * (1_000_000_000 // FPS)
    relative_start_s = (common_first_ns - origin_ns) / 1e9
    relative_end_s = (targets_ns[-1] - origin_ns) / 1e9
    temp = output.with_name(output.name + ".partial")
    if temp.exists():
        raise FileExistsError(f"temporary output already exists: {temp}")
    temp.mkdir(parents=True)
    try:
        role_samples = {}
        for role in ("left_follower", "right_follower", "left_leader", "right_leader"):
            # Keep timestamps sorted and remove duplicate source timestamps, retaining the newest sample.
            dedup = {stamp: qpos for stamp, qpos in states[role]}
            role_samples[role] = sorted(dedup.items())
        follower_l = _nearest(role_samples["left_follower"], targets_ns)
        follower_r = _nearest(role_samples["right_follower"], targets_ns)
        leader_l = _nearest(role_samples["left_leader"], targets_ns)
        leader_r = _nearest(role_samples["right_leader"], targets_ns)
        observation = np.concatenate((follower_r, follower_l), axis=1).astype(np.float32)
        action = np.concatenate((leader_r, leader_l), axis=1).astype(np.float32)
        data_dir = temp / "data" / "chunk-000"
        data_dir.mkdir(parents=True)
        columns = {
            "observation.state": pa.array(observation.tolist(), type=pa.list_(pa.float32(), 14)),
            "action": pa.array(action.tolist(), type=pa.list_(pa.float32(), 14)),
            "timestamp": pa.array((np.arange(sample_count) / FPS).astype(np.float32)),
            "frame_index": pa.array(np.arange(sample_count, dtype=np.int64)),
            "episode_index": pa.array(np.zeros(sample_count, dtype=np.int64)),
            "index": pa.array(np.arange(sample_count, dtype=np.int64)),
            "task_index": pa.array(np.zeros(sample_count, dtype=np.int64)),
        }
        pq.write_table(pa.table(columns), data_dir / "episode_000000.parquet", compression="zstd")

        video_root = temp / "videos" / "chunk-000"
        video_root.mkdir(parents=True)
        video_info = {}
        duration_s = relative_end_s - relative_start_s + 1 / FPS
        for role, source in inputs.items():
            role_dir = video_root / f"observation.images.{role}"
            role_dir.mkdir()
            dest = role_dir / "episode_000000.mp4"
            camera_first_s = (int(camera_meta[role]["first_timestamp_ns"]) - origin_ns) / 1e9
            trim_start = max(0.0, relative_start_s - camera_first_s)
            filter_expr = (
                f"trim=start={trim_start:.6f}:duration={duration_s:.6f},setpts=PTS-STARTPTS,"
                f"fps={FPS},tpad=stop_mode=clone:stop_duration=1,trim=end_frame={sample_count},"
                f"setpts=N/({FPS}*TB)"
            )
            subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-i", str(source), "-an", "-vf", filter_expr,
                 "-frames:v", str(sample_count), "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                 "-threads", "1", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dest)],
                check=True,
            )
            probe = _ffprobe(dest)
            if probe.get("codec_name") != "h264" or int(probe.get("nb_frames", -1)) != sample_count:
                raise ValueError(f"{role} output video mismatch: {probe}")
            video_info[f"observation.images.{role}"] = {
                "dtype": "video",
                "shape": [int(probe["height"]), int(probe["width"]), 3],
                "names": ["height", "width", "channels"],
                "info": {"video.fps": FPS, "video.codec": "h264"},
            }

        info = {
            "codebase_version": "v2.1", "robot_type": "yam", "total_episodes": 1,
            "total_frames": sample_count, "total_tasks": 1, "total_videos": len(CAMERAS),
            "total_chunks": 1, "chunks_size": 1000, "fps": FPS, "splits": {"train": "0:1"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "action": {"dtype": "float32", "shape": [14], "names": None},
                "observation.state": {"dtype": "float32", "shape": [14], "names": None},
                **video_info,
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
            },
        }
        meta = temp / "meta"
        meta.mkdir()
        (meta / "info.json").write_text(json.dumps(info, indent=2) + "\n")
        (meta / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": task["task_name"]}, ensure_ascii=False) + "\n")
        (meta / "episodes.jsonl").write_text(
            json.dumps({"episode_index": 0, "tasks": [task["task_name"]], "length": sample_count}, ensure_ascii=False)
            + "\n"
        )
        # Preserve extra task metadata for provenance without changing the LeRobot task table.
        (meta / "yam_task_info.json").write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n")
        temp.rename(output)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    print(f"Converted one episode: {sample_count} frames at {FPS} Hz ({sample_count / FPS:.2f}s)")
    print("State: right_follower[7], left_follower[7]; action: right_leader[7], left_leader[7].")
    print("Videos: H.264, source frames resampled to 30 fps; source episode was not modified.")
    print(f"Output: {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mcap", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        convert_episode(args.mcap.resolve(), args.output.resolve())
    except Exception as error:  # CLI boundary
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
