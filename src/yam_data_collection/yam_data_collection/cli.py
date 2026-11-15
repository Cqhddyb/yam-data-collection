"""Command-line entry points for deployment checks and interactive capture."""

from __future__ import annotations

import argparse
import signal
from pathlib import Path
from typing import Sequence

from .camera_test import CameraQualification
from .cameras import CameraError, discover_realsense
from .collector import EpisodeCollector
from .config import capture_mode_cli_choices, load_camera_test_config, load_config
from .experiment import ExperimentError, load_experiment_plan, run_experiment, write_session_comparison
from .offline_repair import (
    NoRepairNeededError,
    OfflineRepairError,
    RepairPlan,
    RepairedEpisode,
    build_repair_plan,
    repair_30hz_episode,
)
from .quality import build_validation_policy, print_validation, validate_mcap, validate_video_assets
from .validation_profiles import validation_profile_cli_choices
from .video_pipeline import (
    VideoPipelineError,
    final_codec_family,
    publish_pending_episode,
    transcode_dataset,
    transcode_episode,
    uses_direct_h264_publication,
)


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path, help="Path to the host-specific YAML configuration")


def _add_capture_mode_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mode",
        choices=capture_mode_cli_choices(),
        help="Select the formal camera/state capture contract declared in the configuration",
    )


def _discover_cameras() -> int:
    devices = discover_realsense()
    if not devices:
        print("No RealSense cameras found.")
        return 1
    for device in devices:
        print(f"serial={device['serial']}  product_line={device['product_line']}  name={device['name']}")
    return 0


def _check(config_path: Path, start_cameras: bool, capture_mode: str | None) -> int:
    collector = EpisodeCollector(load_config(config_path, capture_mode=capture_mode))
    try:
        collector.check(start_cameras=start_cameras)
    finally:
        collector.close()
    return 0


def _capture(
    config_path: Path,
    capture_mode: str | None,
    preview: bool = False,
    video_timeline: str = "fixed",
) -> int:
    config = load_config(config_path, capture_mode=capture_mode)
    collector_kwargs = {
        "enable_preview": preview,
        "auto_delete_failed_validation": True,
        "enable_post_save_review": True,
    }
    if video_timeline != "fixed":
        collector_kwargs["video_timeline"] = video_timeline
    collector = EpisodeCollector(config, **collector_kwargs)
    collector.run_interactive()
    return 0


def _web_console(
    config_path: Path | None,
    capture_mode: str | None,
    host: str,
    port: int,
    arm_commands_path: Path | None,
    i2rt_dir: Path | None = None,
    arm_cpu_set: str = "0-15",
    arm_control: bool = True,
) -> int:
    """Serve the browser console without changing the terminal entry point."""
    from .web_console import ArmProcessSupervisor, PreviewBuffer, WebConsoleApplication, WebConsoleState, create_server, load_arm_commands
    from .web_runtime import WebCaptureRuntime

    if i2rt_dir is None:
        i2rt_dir = Path(__file__).resolve().parents[2] / "i2rt"

    runtime: WebCaptureRuntime | None = None
    supervisor: ArmProcessSupervisor | None = None
    if config_path is not None:
        arm_commands = load_arm_commands(arm_commands_path) if arm_commands_path is not None else None
        runtime = WebCaptureRuntime(
            config_path,
            capture_mode,
            i2rt_dir=i2rt_dir,
            arm_cpu_set=arm_cpu_set,
            arm_control=arm_control,
            arm_commands=arm_commands,
        )
        app = runtime.application
    else:
        preview = PreviewBuffer()
        state = WebConsoleState(preview=preview)
        state.message("info", "未提供 --config，仅显示网页界面；采集按钮不可用")
        if arm_commands_path is not None:
            supervisor = ArmProcessSupervisor(load_arm_commands(arm_commands_path))
        else:
            state.message("info", "未提供采集配置，机械臂进程控制已禁用")
        app = WebConsoleApplication(state, arm_supervisor=supervisor)
    server = None
    previous_handlers: dict[int, object] = {}

    def request_shutdown(signum: int, frame: object) -> None:
        del signum, frame
        raise KeyboardInterrupt

    try:
        server = create_server(host, port, application=app)
        print(f"YAM web console: http://{host}:{server.server_address[1]}", flush=True)
        for signal_name in ("SIGTERM", "SIGHUP"):
            signal_number = getattr(signal, signal_name, None)
            if signal_number is not None:
                previous_handlers[signal_number] = signal.getsignal(signal_number)
                signal.signal(signal_number, request_shutdown)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for signal_number, handler in previous_handlers.items():
            signal.signal(signal_number, handler)
        if runtime is not None:
            runtime.close()
        elif supervisor is not None:
            supervisor.stop_all()
        if server is not None:
            server.server_close()
    return 0


def _print_repair_plan(plan: RepairPlan) -> None:
    print(
        f"Repair plan for episode {plan.episode_id}: "
        f"{plan.event_count} eligible delta=3 gap(s)."
    )
    for role, events in plan.events_by_role.items():
        for event_number, event in enumerate(events, start=1):
            print(
                f"  {role} event {event_number}: "
                f"device frames {event.source_device_frame_number} -> "
                f"{event.trigger_device_frame_number}, "
                f"gap={event.original_gap_ns / 1_000_000:.1f}ms; "
                f"duplicate previous frame at {event.synthetic_timestamp_ns}ns"
            )
    print(f"Planned inserted frames: {plan.event_count}")


def _print_repaired_episode(result: RepairedEpisode) -> None:
    print(f"Published repaired episode {result.episode_id}: {result.output_mcap}")
    print(f"Repaired video directory: {result.output_video_dir}")
    print("Inserted frame summary:")
    total_inserted = 0
    for role, events in result.events_by_role.items():
        inserted = len(events)
        total_inserted += inserted
        print(f"  {role}: {inserted} previous-frame duplicate(s)")
    print(f"Total inserted frames: {total_inserted}")
    print(f"Repair manifest: {result.manifest_path}")
    print("Raw MCAP and source video assets were left unchanged.")


def _capture_auto_repair(
    config_path: Path,
    capture_mode: str | None,
    output_dataset: Path | None,
    preview: bool = False,
) -> int:
    config = load_config(config_path, capture_mode=capture_mode)
    if config.capture_mode != "30hz":
        raise ValueError("capture-auto-repair requires the resolved capture mode to be 30hz")
    print(
        "Automatic 30 Hz post-save repair is enabled. Raw episodes remain unchanged; "
        "eligible repairs are written to the derived dataset."
    )

    while True:
        repair_candidate: Path | None = None

        def assess_saved_episode(mcap_path: Path) -> bool:
            nonlocal repair_candidate
            episode = mcap_path.stem
            try:
                plan = build_repair_plan(mcap_path, config)
            except NoRepairNeededError:
                print(
                    f"Episode {episode}: no eligible approximately 100 ms/delta=3 gap; "
                    "no frames inserted. Continuing capture."
                )
                return False
            except OfflineRepairError as exc:
                print(
                    f"Episode {episode}: not automatically repairable; raw episode retained: {exc}. "
                    "Continuing capture."
                )
                return False
            repair_candidate = mcap_path
            print(
                f"Episode {episode}: eligible for {plan.event_count} offline frame repair(s); "
                "closing capture devices before repair."
            )
            return True

        collector_kwargs = {"enable_preview": True} if preview else {}
        collector = EpisodeCollector(config, on_episode_saved=assess_saved_episode, **collector_kwargs)
        collector.run_interactive()
        if repair_candidate is None:
            return 0

        collection = getattr(config, "collection", None)
        codec_label = (
            "H.264"
            if collection is not None and uses_direct_h264_publication(config)
            else "HEVC"
        )
        print(
            f"Capture inputs are closed. Starting verified repair and {codec_label} encoding; "
            "this may take several minutes."
        )
        try:
            repaired = repair_30hz_episode(
                repair_candidate,
                config,
                output_dataset=output_dataset,
                on_plan_ready=_print_repair_plan,
            )
        except (OfflineRepairError, VideoPipelineError, OSError, RuntimeError, ValueError) as exc:
            print(
                f"Automatic repair failed for episode {repair_candidate.stem}; "
                f"raw episode retained: {exc}"
            )
        else:
            _print_repaired_episode(repaired)
        print("Restarting interactive capture.")


def _transcode(mcap_path: Path, config_path: Path, capture_mode: str | None, overwrite: bool) -> int:
    """Run one serial, offline HEVC promotion outside the live recorder."""
    result = transcode_episode(
        mcap_path,
        load_config(config_path, capture_mode=capture_mode),
        overwrite=overwrite,
    )
    print(f"Published episode {result.episode_id}: {result.final_mcap_path}")
    print(f"Final HEVC video directory: {result.final_video_dir}")
    for role, video in result.videos.items():
        print(
            f"  {role}: {video.source.frame_count} frames, "
            f"{video.source.codec} -> {video.output.codec}"
        )
    return 0


def _transcode_dataset(config_path: Path, capture_mode: str | None, overwrite: bool) -> int:
    """Promote every pending episode serially, never beside a live capture."""
    results = transcode_dataset(load_config(config_path, capture_mode=capture_mode), overwrite=overwrite)
    print(f"Published {len(results)} episode(s).")
    for result in results:
        print(f"  episode {result.episode_id}: {result.final_mcap_path}")
    return 0


def _publish_pending(mcap_path: Path, config_path: Path, capture_mode: str | None, overwrite: bool) -> int:
    """Publish an H.264-final pending episode without re-encoding it."""
    result = publish_pending_episode(
        mcap_path,
        load_config(config_path, capture_mode=capture_mode),
        overwrite=overwrite,
    )
    print(f"Published H.264 final episode {result.episode_id}: {result.final_mcap_path}")
    return 0


def _repair_30hz(
    mcap_path: Path,
    config_path: Path,
    capture_mode: str | None,
    output_dataset: Path | None,
    overwrite: bool,
) -> int:
    result = repair_30hz_episode(
        mcap_path,
        load_config(config_path, capture_mode=capture_mode),
        output_dataset=output_dataset,
        overwrite=overwrite,
        on_plan_ready=_print_repair_plan,
    )
    _print_repaired_episode(result)
    return 0


def _camera_test(
    config_path: Path,
    duration_s: float | None,
    capture_mode: str | None,
    camera_roles: Sequence[str] | None,
    *,
    record_video: bool,
) -> int:
    run = CameraQualification(load_camera_test_config(config_path, capture_mode=capture_mode)).run(
        duration_s=duration_s, record_video=record_video, camera_roles=camera_roles
    )
    test_name = "Camera timing test" if not record_video else "Camera test"
    print(f"{test_name}: {'PASS' if run.passed else 'FAIL'}")
    print(f"Artifacts: {run.output_dir}")
    for role, entry in run.report["cameras"].items():
        source = entry[f"{entry['source_clock']}_timing"]
        video_detail = (
            f"captured_frames={entry['captured_frames']}, video=disabled"
            if entry["video"].get("status") == "disabled"
            else f"video_frames={entry['video'].get('frame_count', 0)}/{entry['captured_frames']}"
        )
        print(
            f"  {role}: {'PASS' if entry['passed'] else 'FAIL'} - "
            f"source={source['fps']:.2f}Hz, gap={source['max_gap_ms']:.1f}ms, {video_detail}"
        )
        if entry["error"]:
            print(f"    error: {entry['error']}")
    if run.report["run_error"]:
        print(f"  run error: {run.report['run_error']}")
    sync = run.report["camera_sync"]
    profile = run.report["policy"]["validation_profile"]
    if not sync["applicable"]:
        print(f"  Camera sync [{profile}]: N/A (single-camera timing test)")
    else:
        print(
            f"  Camera sync [{profile}]: "
            f"{'PASS' if sync['passed'] else 'FAIL'} (threshold={sync['threshold_ms']:.0f} ms)"
        )
    return 0 if run.passed else 1


def _experiment(
    config_path: Path,
    plan_path: Path,
    case: str,
    session: str,
    capture_mode: str | None,
    duration_s: float | None,
    repeats: int | None,
) -> int:
    run = run_experiment(
        load_config(config_path, capture_mode=capture_mode),
        load_experiment_plan(plan_path),
        case,
        session,
        duration_s=duration_s,
        repeats=repeats,
    )
    print(f"Experiment artifacts: {run.case_dir}")
    print(f"Case summary: {run.summary_path}")
    print(f"A/B/C comparison: {run.comparison_path}")
    return 0


def _experiment_report(session_dir: Path) -> int:
    path = write_session_comparison(session_dir)
    print(f"A/B/C comparison: {path}")
    return 0


def _validate(
    mcap_path: Path,
    config_path: Path,
    check_video_assets: bool,
    profile_override: str,
    capture_mode: str | None,
) -> int:
    config = load_config(config_path, capture_mode=capture_mode)
    profile_name = config.collection.validation_profile if profile_override == "config" else profile_override
    policy = build_validation_policy(config.cameras, profile_name)
    result = validate_mcap(mcap_path, policy)
    print_validation(result)
    assets_ok = True
    if check_video_assets:
        video_result = validate_video_assets(
            mcap_path,
            policy,
            expected_video_codecs=final_codec_family(config),
        )
        assets_ok = bool(video_result["passed"])
        print(f"Video asset validation: {'PASS' if assets_ok else 'FAIL'}")
        for role, entry in video_result["per_camera"].items():
            detail = entry.get("error") or f"{entry.get('width')}x{entry.get('height')} @ {entry.get('fps')} fps"
            print(f"  {role}: {'PASS' if entry['passed'] else 'FAIL'} - {detail}")
    return 0 if result["output_mcap_passed"] and assets_ok else 1


def _validate_dataset(
    dataset_path: Path,
    config_path: Path,
    check_video_assets: bool,
    profile_override: str,
    capture_mode: str | None,
) -> int:
    candidates = [path for path in dataset_path.glob("*.mcap") if path.stem.isdigit()]
    candidates.extend(
        path
        for path in dataset_path.glob("*/*.mcap")
        if path.parent.name.isdigit()
        and not path.name.endswith(".partial.mcap")
        and path.parent.parent == dataset_path
    )
    mcap_files = sorted(set(candidates), key=lambda path: (int(path.parent.name) if path.parent.name.isdigit() else int(path.stem), path.name))
    if not mcap_files:
        raise ValueError(f"no episode MCAP files found in {dataset_path}")
    failures = 0
    for mcap_path in mcap_files:
        print(f"\n[{mcap_path.name}]")
        failures += int(_validate(mcap_path, config_path, check_video_assets, profile_override, capture_mode) != 0)
    print(f"\nValidated {len(mcap_files)} episodes; failures={failures}")
    return 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Portable bimanual YAM data collection")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("discover-cameras", help="List RealSense cameras by physical serial number")
    check_parser = subcommands.add_parser("check", help="Check read-only telemetry and optionally camera streams")
    _add_config_argument(check_parser)
    _add_capture_mode_argument(check_parser)
    check_parser.add_argument("--start-cameras", action="store_true", help="Open each camera and read one frame")
    capture_parser = subcommands.add_parser("capture", help="Run interactive episode capture")
    _add_config_argument(capture_parser)
    _add_capture_mode_argument(capture_parser)
    capture_parser.add_argument(
        "--preview",
        action="store_true",
        help="Show low-rate camera previews and four-arm joint trends in a separate best-effort window",
    )
    capture_parser.add_argument(
        "--video-timeline",
        choices=("fixed", "source"),
        default="fixed",
        help="Video timing: fixed uses the existing configured FPS; source uses MCAP capture timestamps",
    )
    web_parser = subcommands.add_parser(
        "web",
        help="Serve the local browser console; hardware and capture start only after a button is pressed",
    )
    web_parser.add_argument(
        "--config",
        type=Path,
        help="Host capture configuration used by the Start/End capture buttons",
    )
    _add_capture_mode_argument(web_parser)
    web_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Loopback bind address only (default: 127.0.0.1; remote control is disabled)",
    )
    web_parser.add_argument("--port", type=int, default=8765, help="HTTP port (default: 8765)")
    web_parser.add_argument(
        "--arm-commands",
        type=Path,
        help="Optional JSON override containing explicit follower/leader argv and Portal readiness probes",
    )
    web_parser.add_argument(
        "--i2rt-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "i2rt",
        help="i2rt checkout used to build the default four-arm commands",
    )
    web_parser.add_argument(
        "--arm-cpu-set",
        default="0-15",
        help="CPU set passed to each default arm process (default: 0-15)",
    )
    web_parser.add_argument(
        "--no-arm-control",
        action="store_true",
        help="Disable arm process buttons; keep the browser capture console available",
    )
    auto_repair_parser = subcommands.add_parser(
        "capture-auto-repair",
        help="Run 30 Hz capture and serially repair eligible saved episodes with capture devices closed",
    )
    _add_config_argument(auto_repair_parser)
    _add_capture_mode_argument(auto_repair_parser)
    auto_repair_parser.add_argument(
        "--output-dataset",
        type=Path,
        help="Derived dataset root (default: <source-dataset>/derived/duplicate_previous_delta3_v1)",
    )
    auto_repair_parser.add_argument(
        "--preview",
        action="store_true",
        help="Show low-rate camera previews and four-arm joint trends in a separate best-effort window",
    )
    transcode_parser = subcommands.add_parser(
        "transcode",
        help="Convert one saved episode's temporary live MP4 files into final HEVC MP4 assets",
    )
    transcode_parser.add_argument(
        "mcap",
        type=Path,
        help="Path to a pending .pending/<id>/<id>.mcap saved by capture",
    )
    _add_config_argument(transcode_parser)
    _add_capture_mode_argument(transcode_parser)
    transcode_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing final MCAP/video pair only after replacement frame-count verification",
    )
    transcode_dataset_parser = subcommands.add_parser(
        "transcode-dataset",
        help="Serially convert all verified pending live episodes into final HEVC assets",
    )
    _add_config_argument(transcode_dataset_parser)
    _add_capture_mode_argument(transcode_dataset_parser)
    transcode_dataset_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing final assets only after a replacement passes frame-count verification",
    )
    publish_pending_parser = subcommands.add_parser(
        "publish-pending",
        help="Publish one verified pending H.264-final episode without re-encoding",
    )
    publish_pending_parser.add_argument(
        "mcap",
        type=Path,
        help="Path to a pending .pending/<id>/<id>.mcap saved by an H.264-final capture",
    )
    _add_config_argument(publish_pending_parser)
    _add_capture_mode_argument(publish_pending_parser)
    publish_pending_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing final MCAP/video pair only after frame-count verification",
    )
    repair_parser = subcommands.add_parser(
        "repair-30hz",
        help="Create a separate 30 Hz derivative for eligible delta=3 camera dropouts",
    )
    repair_parser.add_argument(
        "mcap",
        type=Path,
        help="Raw numeric MCAP, normally .pending/<id>/<id>.mcap",
    )
    _add_config_argument(repair_parser)
    _add_capture_mode_argument(repair_parser)
    repair_parser.add_argument(
        "--output-dataset",
        type=Path,
        help="Derived dataset root (default: <source-dataset>/derived/duplicate_previous_delta3_v1)",
    )
    repair_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace an existing repaired MCAP/video pair after full verification",
    )
    camera_test_parser = subcommands.add_parser(
        "camera-test", help="Record and qualify three RealSense streams without robot, Portal, or MCAP inputs"
    )
    _add_config_argument(camera_test_parser)
    _add_capture_mode_argument(camera_test_parser)
    camera_test_parser.add_argument(
        "--duration", type=float, help="Override camera_test_config.duration_s for this one run"
    )
    camera_test_parser.add_argument(
        "--no-video",
        action="store_true",
        help="Timing/sync diagnostic only: skip ffmpeg, MP4 writes, and video-asset validation",
    )
    camera_test_parser.add_argument(
        "--camera",
        action="append",
        dest="camera_roles",
        metavar="ROLE",
        help="Test only this configured role; repeat for multiple cameras (default: all cameras)",
    )
    experiment_parser = subcommands.add_parser(
        "experiment",
        help="Run one duration-bounded full-load A/B/C diagnostic condition in an isolated dataset",
    )
    _add_config_argument(experiment_parser)
    _add_capture_mode_argument(experiment_parser)
    experiment_parser.add_argument("--plan", required=True, type=Path, help="Host-specific A/B/C experiment-plan YAML")
    experiment_parser.add_argument("--case", required=True, help="Case name from the experiment plan, for example A, B, or C")
    experiment_parser.add_argument("--session", required=True, help="New session name shared by the A/B/C case folders")
    experiment_parser.add_argument(
        "--duration", type=float, help="Override the plan's per-trial duration in seconds for this condition"
    )
    experiment_parser.add_argument("--repeats", type=int, help="Override the plan's number of sequential trials")
    experiment_report_parser = subcommands.add_parser(
        "experiment-report", help="Regenerate the cross-condition A/B/C comparison from a session directory"
    )
    experiment_report_parser.add_argument("session", type=Path, help="Experiment session directory containing case summaries")
    validate_parser = subcommands.add_parser("validate", help="Validate one MCAP and its external video assets")
    validate_parser.add_argument("mcap", type=Path, help="Path to an episode MCAP")
    _add_config_argument(validate_parser)
    _add_capture_mode_argument(validate_parser)
    validate_parser.add_argument(
        "--skip-video-assets", action="store_true", help="Run MCAP checks without probing external MP4 files"
    )
    validate_parser.add_argument(
        "--profile",
        choices=("config", *validation_profile_cli_choices()),
        default="config",
        help="Use the configured profile or override it for this validation run",
    )
    dataset_parser = subcommands.add_parser("validate-dataset", help="Validate every numeric MCAP in a dataset root")
    dataset_parser.add_argument("dataset", type=Path, help="Dataset directory containing numeric MCAP files")
    _add_config_argument(dataset_parser)
    _add_capture_mode_argument(dataset_parser)
    dataset_parser.add_argument(
        "--skip-video-assets", action="store_true", help="Run MCAP checks without probing external MP4 files"
    )
    dataset_parser.add_argument(
        "--profile",
        choices=("config", *validation_profile_cli_choices()),
        default="config",
        help="Use the configured profile or override it for this validation run",
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "discover-cameras":
            return _discover_cameras()
        if args.command == "check":
            return _check(args.config, args.start_cameras, args.mode)
        if args.command == "capture":
            return _capture(args.config, args.mode, args.preview, args.video_timeline)
        if args.command == "web":
            return _web_console(
                args.config,
                args.mode,
                args.host,
                args.port,
                args.arm_commands,
                args.i2rt_dir,
                args.arm_cpu_set,
                not args.no_arm_control,
            )
        if args.command == "capture-auto-repair":
            return _capture_auto_repair(args.config, args.mode, args.output_dataset, args.preview)
        if args.command == "transcode":
            return _transcode(args.mcap, args.config, args.mode, args.overwrite)
        if args.command == "transcode-dataset":
            return _transcode_dataset(args.config, args.mode, args.overwrite)
        if args.command == "publish-pending":
            return _publish_pending(args.mcap, args.config, args.mode, args.overwrite)
        if args.command == "repair-30hz":
            return _repair_30hz(args.mcap, args.config, args.mode, args.output_dataset, args.overwrite)
        if args.command == "camera-test":
            return _camera_test(
                args.config,
                args.duration,
                args.mode,
                args.camera_roles,
                record_video=not args.no_video,
            )
        if args.command == "experiment":
            return _experiment(
                args.config,
                args.plan,
                args.case,
                args.session,
                args.mode,
                args.duration,
                args.repeats,
            )
        if args.command == "experiment-report":
            return _experiment_report(args.session)
        if args.command == "validate":
            return _validate(args.mcap, args.config, not args.skip_video_assets, args.profile, args.mode)
        if args.command == "validate-dataset":
            return _validate_dataset(args.dataset, args.config, not args.skip_video_assets, args.profile, args.mode)
    except (
        CameraError,
        ExperimentError,
        OfflineRepairError,
        VideoPipelineError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
    raise AssertionError(f"unexpected command {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
