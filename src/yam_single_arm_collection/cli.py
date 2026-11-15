"""Explicit-side CLI for the independent single-arm collection entry point."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from yam_data_collection.camera_test import CameraQualification
from yam_data_collection.config import DatasetConfig, cpu_set_members
from yam_data_collection.quality import print_validation

from .collector import SingleArmCollector, parse_preview_fps
from .config import (
    CAMERA_SETUPS,
    CAPTURE_MODES,
    DEFAULT_CONFIG,
    SIDES,
    camera_test_config,
    load_config,
)
from .motion import load_motion_plan, move_follower, preposition_pair, read_pair_poses, validate_motion_endpoints
from .quality import validate_mcap, validate_video_assets


ROOT = Path(__file__).resolve().parent.parent


def _add_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Extension config or recorded snapshot",
    )
    parser.add_argument(
        "--side",
        choices=SIDES,
        required=True,
        help="Physical leader/follower pair to read",
    )
    parser.add_argument(
        "--cameras",
        choices=CAMERA_SETUPS,
        default="wrist",
        help="Corresponding wrist, optionally top",
    )
    parser.add_argument(
        "--mode",
        choices=CAPTURE_MODES,
        default="60hz",
        help="Capture/qualification contract",
    )


def _show_config(config: DatasetConfig) -> None:
    print(
        json.dumps(
            {
                "single_arm": config.raw["single_arm"],
                "mode": config.capture_mode,
                "dataset_dir": str(config.collection.dataset_dir),
                "camera_test_dir": str(camera_test_config(config).settings.output_dir),
                "telemetry": {role: asdict(endpoint) for role, endpoint in config.telemetry.items()},
                "cameras": {
                    role: {
                        "serial": camera.serial,
                        "width": camera.width,
                        "height": camera.height,
                        "fps": camera.fps,
                        "stream_format": camera.stream_format,
                        "distortion_model": camera.intrinsics.distortion_model,
                    }
                    for role, camera in config.cameras.items()
                },
                "state_save_freq": config.collection.state_save_freq,
                "validation_profile": config.collection.validation_profile,
                "task_info": config.task_info,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _qualify_cameras(config: DatasetConfig, duration_s: float | None, *, record_video: bool) -> int:
    requested: set[int] = set()
    for value in (
        config.collection.recorder_cpu_set,
        config.collection.live_video_cpu_set,
    ):
        if value is not None:
            requested.update(cpu_set_members(value))
    if requested:
        available = set(os.sched_getaffinity(0))
        if not requested.issubset(available):
            raise ValueError("camera-test CPU set is outside this process's allowed CPUs")
        os.sched_setaffinity(0, requested)
    run = CameraQualification(camera_test_config(config)).run(duration_s=duration_s, record_video=record_video)
    print(f"Camera {'test' if record_video else 'timing test'}: {'PASS' if run.passed else 'FAIL'}")
    print(f"Artifacts: {run.output_dir}")
    for role, entry in run.report["cameras"].items():
        timing = entry[f"{entry['source_clock']}_timing"]
        print(
            f"  {role}: {'PASS' if entry['passed'] else 'FAIL'} - "
            f"source={timing['fps']:.2f}Hz, gap={timing['max_gap_ms']:.1f}ms, "
            f"captured_frames={entry['captured_frames']}"
        )
        if entry["error"]:
            print(f"    error: {entry['error']}")
    if run.report["run_error"]:
        print(f"  run error: {run.report['run_error']}")
    sync = run.report["camera_sync"]
    if sync["applicable"]:
        print(f"  Camera sync: {'PASS' if sync['passed'] else 'FAIL'} (threshold={sync['threshold_ms']:g} ms)")
    else:
        print("  Camera sync: N/A (single-camera timing test)")
    return 0 if run.passed else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="yam-single-arm",
        description="Single leader + follower, with wrist or wrist + top cameras",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("follower", "leader", "leader-motion"):
        controller = commands.add_parser(name, help=f"Start only the selected {name}; can enable motors")
        controller.add_argument("--side", choices=SIDES, required=True)
        controller.add_argument(
            "--confirm-motion",
            action="store_true",
            required=True,
            help="Operator safety confirmation",
        )
    show = commands.add_parser("show-config", help="Display selection; no hardware connection")
    _add_selection(show)
    check = commands.add_parser("check", help="Read only the selected pair's telemetry")
    _add_selection(check)
    check.add_argument(
        "--start-cameras",
        action="store_true",
        help="Also open only selected cameras once",
    )
    commands_pose = commands.add_parser("read-poses", help="Read selected leader/follower poses for a measured plan")
    _add_selection(commands_pose)
    preposition_preview = commands.add_parser(
        "preview-preposition", help="Validate the paired preposition plan without sending motor commands"
    )
    _add_selection(preposition_preview)
    preposition_preview.add_argument("--plan", type=Path, required=True)
    camera_test = commands.add_parser("camera-test", help="Qualify only the selected camera combination")
    _add_selection(camera_test)
    camera_test.add_argument("--duration", type=float, help="Qualification duration in seconds")
    camera_test.add_argument("--no-video", action="store_true", help="Timing only; do not write MP4")
    capture = commands.add_parser("capture", help="Interactive queued H.264 collection using source timestamps")
    _add_selection(capture)
    capture.add_argument(
        "--preview",
        action="store_true",
        help="Show only the selected side's wrist camera at full camera resolution in a desktop window",
    )
    capture.add_argument(
        "--preview-fps",
        type=parse_preview_fps,
        default=10.0,
        help="Camera display FPS when --preview is enabled (default: 10; greater than 0, up to 10)",
    )
    capture.add_argument(
        "--preposition",
        type=Path,
        help="Move the leader/follower pair to a measured shared pose before capture",
    )
    capture.add_argument(
        "--confirm",
        default="",
        help="Must be exactly PREPOSITION when --preposition is supplied",
    )
    move = commands.add_parser("move", help="Preview or move one follower from its current pose to the planned target")
    _add_selection(move)
    move.add_argument("--plan", type=Path, required=True, help="Target joint pose and duration in JSON")
    move.add_argument("--execute", action="store_true", help="Actually send position setpoints")
    move.add_argument("--confirm", default="", help="Must be exactly MOVE together with --execute")
    for motion_command in (move, preposition_preview, capture):
        start_mode = motion_command.add_mutually_exclusive_group()
        start_mode.add_argument(
            "--from-current",
            action="store_const",
            const=True,
            default=None,
            help="Use current joint poses as the start (default); recorded start fields are ignored",
        )
        start_mode.add_argument(
            "--from-recorded",
            dest="from_current",
            action="store_const",
            const=False,
            help="Require the recorded start poses instead of reading a new start; requires a motion plan",
        )
    validate = commands.add_parser("validate", help="Offline scoped MCAP and video validation")
    _add_selection(validate)
    validate.add_argument("mcap", type=Path)
    validate.add_argument(
        "--json",
        action="store_true",
        help="Print full machine-readable validation report",
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "leader-motion":
            os.execv(
                "/bin/bash",
                ["bash", str(ROOT / "yam_single_arm_collection/leader_motion.sh"), args.side, "--confirm-motion"],
            )
            return 0
        if args.command in {"follower", "leader"}:
            controller_name = f"{args.command}-{'l' if args.side == 'left' else 'r'}"
            os.execv(
                "/bin/bash",
                ["bash", str(ROOT / "yam.sh"), controller_name, "--confirm-motion"],
            )
            return 0
        config = load_config(
            args.config,
            side=args.side,
            camera_setup=args.cameras,
            capture_mode=args.mode,
        )
        if args.command == "move":
            validate_motion_endpoints(config)
            plan = load_motion_plan(args.plan, side=args.side, pair=False, from_current=args.from_current is not False)
            if args.execute and args.confirm != "MOVE":
                parser.error("execute motion with --execute --confirm MOVE")
            if not args.execute and args.confirm:
                parser.error("preview motion without --confirm")
            result = move_follower(plan, execute=args.execute)
            if not args.execute:
                print("Plan is valid; no motion command was sent.")
            else:
                print(f"Follower reached the target: {result.get('follower_final')}")
            return 0
        if args.command == "read-poses":
            read_pair_poses(config)
            return 0
        if args.command == "preview-preposition":
            result = preposition_pair(config, args.plan, execute=False, from_current=args.from_current is not False)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print("Paired preposition preview passed; no motor command was sent.")
            return 0
        if args.command == "show-config":
            _show_config(config)
            return 0
        if args.command == "validate":
            mcap_result = validate_mcap(args.mcap, config)
            video_result = validate_video_assets(args.mcap, config)
            if args.json:
                print(
                    json.dumps(
                        {"mcap": mcap_result, "video": video_result},
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            else:
                print_validation(mcap_result)
                print(f"Video asset validation: {'PASS' if video_result['passed'] else 'FAIL'}")
                for role, entry in video_result["per_camera"].items():
                    if not entry["passed"]:
                        print(f"  {role}: {entry}")
                if video_result["unexpected_videos"]:
                    print(f"  Unexpected videos: {video_result['unexpected_videos']}")
            return 0 if mcap_result["output_mcap_passed"] and video_result["passed"] else 1
        if args.command == "camera-test":
            return _qualify_cameras(config, args.duration, record_video=not args.no_video)
        if args.command == "capture" and not sys.stdin.isatty():
            parser.error("interactive capture requires a terminal; no devices were opened")
        if args.command == "capture" and args.preposition is not None and args.confirm != "PREPOSITION":
            parser.error("paired prepositioning requires --preposition PLAN --confirm PREPOSITION")
        if args.command == "capture" and args.preposition is None and args.confirm:
            parser.error("--confirm PREPOSITION requires --preposition PLAN")
        if args.command == "capture" and args.from_current is not None and args.preposition is None:
            parser.error("--from-current / --from-recorded requires --preposition PLAN for capture")
        print(f"Single-arm selection: {args.side}, {args.cameras}, {args.mode}")
        if args.command == "capture":
            print(f"Dataset: {config.collection.dataset_dir}")
            if config.task_info.get("task_id") == "SINGLE_ARM_SMOKE_TEST":
                print("Smoke-test task metadata active; update the NEW config before formal collection.")
            if args.preposition is not None:
                result = preposition_pair(config, args.preposition, from_current=args.from_current is not False)
                config.raw["single_arm"]["preprogrammed_preposition"] = result
                task_info = dict(config.task_info)
                layout = dict(task_info.get("collection_layout", {}))
                layout["preprogrammed_preposition"] = result
                task_info["collection_layout"] = layout
                config.raw["task_info"] = task_info
                config.task_info.clear()
                config.task_info.update(task_info)
                print("Leader and follower are at the verified shared pose; starting camera collection.")
        if args.command == "capture" and args.preview:
            collector = SingleArmCollector(config, enable_preview=True, preview_fps=args.preview_fps)
        else:
            collector = SingleArmCollector(config)
        if args.command == "check":
            try:
                collector.check(start_cameras=args.start_cameras)
            finally:
                collector.close()
        else:
            collector.run_interactive()
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.exit(1, f"yam-single-arm: error: {exc}\n")
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
