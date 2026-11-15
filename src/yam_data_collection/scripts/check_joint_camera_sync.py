#!/usr/bin/env python3
"""Check joint-camera timestamp coverage in one or more YAM MCAP episodes.

The production quality check compares every CameraInfo timestamp with the
nearest timestamp from the union of the four JointState topics. This script
exposes that one check so a failed episode can be inspected independently.
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys
from pathlib import Path
from typing import Any, Iterable

# Make direct ``.venv/bin/python scripts/check_joint_camera_sync.py`` execution
# use the checkout's package even when the project has not been installed.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yam_data_collection.mcap_writer import CAMERA_TOPICS, ROBOT_TOPICS  # noqa: E402
from yam_data_collection.quality import inspect_mcap  # noqa: E402
from yam_data_collection.terminal import colorize  # noqa: E402


MODE_THRESHOLDS_MS = {
    "30hz": 34.0,
    "60hz": 17.0,
    "piper-compatible": 34.0,
}
TIMESTAMP_FIELDS = {
    "header": "header_stamps",
    "log": "log_times",
    "publish": "publish_times",
}


def _nearest(timestamp: int, ordered: list[int]) -> int | None:
    """Return the nearest value using the same tie behavior as quality.py."""
    if not ordered:
        return None
    index = bisect.bisect_left(ordered, timestamp)
    candidates: list[int] = []
    if index < len(ordered):
        candidates.append(ordered[index])
    if index:
        candidates.append(ordered[index - 1])
    return min(candidates, key=lambda value: abs(value - timestamp))


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def _expand_inputs(inputs: Iterable[Path]) -> list[Path]:
    """Accept either MCAP files or directories containing MCAP files."""
    expanded: list[Path] = []
    seen: set[Path] = set()
    for raw_path in inputs:
        path = raw_path.expanduser().resolve()
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = sorted(candidate for candidate in path.rglob("*.mcap") if candidate.is_file())
            if not candidates:
                raise ValueError(f"no .mcap files found below {path}")
        else:
            raise ValueError(f"MCAP path does not exist: {path}")
        for candidate in candidates:
            if candidate not in seen:
                expanded.append(candidate)
                seen.add(candidate)
    return expanded


def _timestamp_mapping(inspection: Any, field: str) -> dict[str, tuple[int, ...]]:
    return getattr(inspection, TIMESTAMP_FIELDS[field])


def _analyze(
    inspection: Any,
    *,
    threshold_ms: float,
    timestamp_field: str,
    max_offenders: int,
) -> dict[str, Any]:
    """Return aggregate and per-camera details for the production rule."""
    timestamps = _timestamp_mapping(inspection, timestamp_field)
    joint_topics = list(ROBOT_TOPICS.values())
    all_joint_stamps = sorted({stamp for topic in joint_topics for stamp in timestamps.get(topic, ())})
    joint_counts = {role: len(timestamps.get(topic, ())) for role, topic in ROBOT_TOPICS.items()}
    result: dict[str, Any] = {
        "passed": True,
        "threshold_ms": threshold_ms,
        "timestamp_field": timestamp_field,
        "method": "nearest_joint_per_camera",
        "joint_timestamp_count": len(all_joint_stamps),
        "joint_topic_counts": joint_counts,
        "total_camera_frames": 0,
        "uncovered_camera_frames": 0,
        "max_offset_ms": 0.0,
        "per_camera": {},
        "offenders": [],
    }
    if not all_joint_stamps:
        result.update(passed=False, error="No required joint timestamps found")
        return result

    all_offenders: list[dict[str, Any]] = []
    for role, topic in CAMERA_TOPICS.items():
        camera_stamps = timestamps.get(topic, ())
        camera_result: dict[str, Any] = {
            "passed": True,
            "total_frames": len(camera_stamps),
            "uncovered_frames": 0,
            "max_offset_ms": 0.0,
        }
        if not camera_stamps:
            camera_result.update(passed=False, error=f"No timestamps for {topic}")
            result["passed"] = False
            result["per_camera"][role] = camera_result
            continue

        for frame_index, camera_stamp in enumerate(camera_stamps):
            result["total_camera_frames"] += 1
            closest = _nearest(camera_stamp, all_joint_stamps)
            if closest is None:
                camera_result["uncovered_frames"] += 1
                result["uncovered_camera_frames"] += 1
                continue
            offset_ms = abs(camera_stamp - closest) / 1_000_000
            camera_result["max_offset_ms"] = max(float(camera_result["max_offset_ms"]), offset_ms)
            result["max_offset_ms"] = max(float(result["max_offset_ms"]), offset_ms)
            if offset_ms > threshold_ms:
                camera_result["uncovered_frames"] += 1
                result["uncovered_camera_frames"] += 1
                all_offenders.append(
                    {
                        "camera_role": role,
                        "camera_frame_index": frame_index,
                        "camera_stamp_ns": int(camera_stamp),
                        "nearest_joint_stamp_ns": int(closest),
                        "offset_ms": round(offset_ms, 3),
                    }
                )
        camera_result["max_offset_ms"] = round(float(camera_result["max_offset_ms"]), 3)
        camera_result["passed"] = int(camera_result["uncovered_frames"]) == 0
        result["passed"] = bool(result["passed"] and camera_result["passed"])
        result["per_camera"][role] = camera_result

    all_offenders.sort(key=lambda item: (-float(item["offset_ms"]), item["camera_role"], item["camera_frame_index"]))
    result["offenders"] = all_offenders[:max_offenders]
    result["offender_count"] = len(all_offenders)
    result["max_offset_ms"] = round(float(result["max_offset_ms"]), 3)
    if int(result["uncovered_camera_frames"]) > 0:
        result["passed"] = False
    return result


def _format_status(passed: bool) -> str:
    label = "PASS" if passed else "FAIL"
    return colorize(label, "green" if passed else "red")


def _print_human(path: Path, result: dict[str, Any]) -> None:
    status = _format_status(bool(result["passed"]))
    print(f"[{status}] {path}")
    print(
        f"  field={result['timestamp_field']}  threshold={float(result['threshold_ms']):.3f} ms  "
        f"max_offset={float(result['max_offset_ms']):.3f} ms"
    )
    print(
        f"  camera_frames={int(result['total_camera_frames'])}  "
        f"over_threshold={int(result['uncovered_camera_frames'])}  "
        f"unique_joint_timestamps={int(result['joint_timestamp_count'])}"
    )
    if result.get("error"):
        print(f"  error: {result['error']}")
    for role, camera in result["per_camera"].items():
        camera_status = _format_status(bool(camera["passed"]))
        detail = (
            f"frames={int(camera['total_frames'])}  "
            f"over_threshold={int(camera['uncovered_frames'])}  "
            f"max_offset={float(camera['max_offset_ms']):.3f} ms"
        )
        if camera.get("error"):
            detail += f"  error={camera['error']}"
        print(f"  {role}: [{camera_status}] {detail}")
    offenders = result.get("offenders", [])
    if offenders:
        print(f"  largest offending frames (showing {len(offenders)} of {result['offender_count']}):")
        for offender in offenders:
            print(
                f"    {offender['camera_role']} frame={offender['camera_frame_index']}  "
                f"offset={float(offender['offset_ms']):.3f} ms  "
                f"camera_stamp={offender['camera_stamp_ns']}  "
                f"nearest_joint_stamp={offender['nearest_joint_stamp_ns']}"
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mcap",
        nargs="+",
        type=Path,
        metavar="MCAP_OR_DIR",
        help="One or more episode MCAP files, or directories to scan for MCAP files",
    )
    parser.add_argument(
        "--mode",
        choices=tuple(MODE_THRESHOLDS_MS),
        default="piper-compatible",
        help="Threshold preset: 30hz=34 ms, 60hz=17 ms (default: piper-compatible=34 ms)",
    )
    parser.add_argument(
        "--threshold-ms",
        type=_non_negative_float,
        help="Override the mode threshold, for example 34 to reproduce a legacy or client-side check",
    )
    parser.add_argument(
        "--timestamp-field",
        choices=tuple(TIMESTAMP_FIELDS),
        default="header",
        help="Timestamp source: header.stamp (production), MCAP log_time, or publish_time",
    )
    parser.add_argument(
        "--max-offenders",
        type=_positive_int,
        default=10,
        help="Maximum offending frames printed/serialized per MCAP (default: 10)",
    )
    parser.add_argument("--json", action="store_true", dest="as_json", help="Print machine-readable JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        paths = _expand_inputs(args.mcap)
        threshold_ms = float(args.threshold_ms) if args.threshold_ms is not None else MODE_THRESHOLDS_MS[args.mode]
        results: list[dict[str, Any]] = []
        read_failures: list[dict[str, str]] = []
        for path in paths:
            try:
                result = _analyze(
                    inspect_mcap(path),
                    threshold_ms=threshold_ms,
                    timestamp_field=args.timestamp_field,
                    max_offenders=args.max_offenders,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                read_failures.append({"path": str(path), "error": str(exc)})
                continue
            result["path"] = str(path)
            result["mode"] = args.mode
            results.append(result)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.as_json:
        print(
            json.dumps(
                {
                    "threshold_ms": threshold_ms,
                    "timestamp_field": args.timestamp_field,
                    "results": results,
                    "read_failures": read_failures,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        for result in results:
            _print_human(Path(result["path"]), result)
        for failure in read_failures:
            print(f"[ERROR] {failure['path']}: {failure['error']}", file=sys.stderr)

    return 0 if results and not read_failures and all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
