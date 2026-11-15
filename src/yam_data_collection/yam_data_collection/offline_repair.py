"""Create provenance-preserving 30 Hz derivatives for one narrow camera dropout."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml
from mcap.reader import make_reader
from mcap.writer import Writer
from mcap_ros2._dynamic import serialize_dynamic

from .config import DatasetConfig
from .dataset_lock import DatasetOperationLock
from .mcap_writer import CAMERA_TOPICS, FRAME_TIMING_DIAGNOSTICS_TOPIC, camera_info_message
from .quality import CameraTimingDiagnostics, build_validation_policy, inspect_mcap, validate_mcap, validate_video_assets
from .validation_profiles import YAM_30HZ_PROFILE
from .video import (
    VideoFrameDuplicateResult,
    VideoTranscodeResult,
    transcode_to_h264_with_frame_duplicates,
    transcode_to_hevc,
    transcode_to_hevc_with_frame_duplicates,
)
from .video_pipeline import (
    _publish_verified_episode,
    final_codec_family,
    final_mcap_path,
    final_video_dir,
    live_codec_family,
    live_video_dir,
    uses_direct_h264_publication,
)


REPAIR_STRATEGY = "duplicate_previous_delta3_v1"
FRAME_PERIOD_NS = round(1_000_000_000 / 30)
MAX_OUTPUT_GAP_NS = 69_000_000
TRIPLE_PERIOD_TOLERANCE_NS = 2_000_000


class OfflineRepairError(RuntimeError):
    """A source episode is ineligible or a verified derivative cannot be published."""


class NoRepairNeededError(OfflineRepairError):
    """The source already meets the cadence contract without an eligible repair."""


@dataclass(frozen=True)
class FrameRepairEvent:
    role: str
    source_frame_index: int
    trigger_frame_index: int
    synthetic_output_index: int
    source_capture_timestamp_ns: int
    source_device_frame_number: int
    trigger_capture_timestamp_ns: int
    trigger_device_frame_number: int
    host_detection_timestamp_ns: int
    original_gap_ns: int
    synthetic_timestamp_ns: int
    frame_number_delta: int = 3
    strategy: str = REPAIR_STRATEGY

    def manifest_value(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RepairPlan:
    source_mcap: Path
    episode_id: int
    events_by_role: Mapping[str, tuple[FrameRepairEvent, ...]]
    source_validation: Mapping[str, Any]

    @property
    def event_count(self) -> int:
        return sum(len(events) for events in self.events_by_role.values())


@dataclass(frozen=True)
class RepairedEpisode:
    episode_id: int
    source_mcap: Path
    source_video_dir: Path
    output_mcap: Path
    output_video_dir: Path
    manifest_path: Path
    events_by_role: Mapping[str, tuple[FrameRepairEvent, ...]]
    validation: Mapping[str, Any]


def _episode_id(path: Path) -> int:
    if path.suffix != ".mcap":
        raise OfflineRepairError(f"source path must be an .mcap file, got {path}")
    if path.parent.name.isdigit():
        return int(path.parent.name)
    if path.stem.isdigit():
        return int(path.stem)
    raise OfflineRepairError(f"cannot identify numeric episode directory for source MCAP: {path}")


def _require_30hz_contract(config: DatasetConfig) -> None:
    if config.capture_mode != "30hz":
        raise OfflineRepairError("repair-30hz requires --mode 30hz")
    if config.collection.validation_profile != YAM_30HZ_PROFILE:
        raise OfflineRepairError(
            f"repair-30hz requires validation profile {YAM_30HZ_PROFILE}, "
            f"got {config.collection.validation_profile}"
        )
    wrong_rates = [role for role, camera in config.cameras.items() if camera.fps != 30]
    if wrong_rates:
        raise OfflineRepairError("repair-30hz requires every camera at 30 Hz: " + ", ".join(wrong_rates))


def _require_source_path(source: Path, config: DatasetConfig, episode_id: int) -> None:
    dataset = config.collection.dataset_dir
    resolved = source.resolve(strict=False)
    allowed_final_dir = (dataset / str(episode_id)).resolve(strict=False)
    allowed_pending_dir = (dataset / ".pending" / str(episode_id)).resolve(strict=False)
    allowed_legacy = (dataset / f"{episode_id}.mcap").resolve(strict=False)
    if not (
        resolved == allowed_legacy
        or source.parent.resolve(strict=False) in {allowed_final_dir, allowed_pending_dir}
    ):
        raise OfflineRepairError(
            "source MCAP must be either the published episode or its preserved pending episode under "
            f"{allowed_final_dir} or {allowed_pending_dir}"
        )
    if not source.is_file():
        raise OfflineRepairError(f"source MCAP does not exist or is not a regular file: {source}")


def _diagnostics_are_complete(
    role: str,
    diagnostics: CameraTimingDiagnostics | None,
    camera_timestamps: tuple[int, ...],
) -> CameraTimingDiagnostics:
    if diagnostics is None:
        raise OfflineRepairError(f"{role} has no /diagnostics/frame_timing evidence")
    lengths = {
        len(diagnostics.capture_timestamps_ns),
        len(diagnostics.host_receive_timestamps_ns),
        len(diagnostics.device_timestamps_ns),
        len(diagnostics.frame_numbers),
    }
    if len(lengths) != 1 or diagnostics.capture_timestamps_ns != camera_timestamps:
        raise OfflineRepairError(f"{role} frame diagnostics do not align with CameraInfo messages")
    if diagnostics.synthetic_indices or diagnostics.repair_events:
        raise OfflineRepairError(f"{role} already contains synthetic frame-repair provenance")
    expected_count = len(diagnostics.capture_timestamps_ns)
    if diagnostics.capture_timestamp_sources != {"realsense_global_time": expected_count}:
        raise OfflineRepairError(f"{role} is not entirely timestamped by RealSense Global Time")
    return diagnostics


def _role_events(role: str, diagnostics: CameraTimingDiagnostics) -> tuple[FrameRepairEvent, ...]:
    events: list[FrameRepairEvent] = []
    captures = diagnostics.capture_timestamps_ns
    hosts = diagnostics.host_receive_timestamps_ns
    frame_numbers = diagnostics.frame_numbers
    for index in range(1, len(captures)):
        previous_frame_number = frame_numbers[index - 1]
        frame_number = frame_numbers[index]
        if previous_frame_number is None or frame_number is None:
            continue
        frame_delta = frame_number - previous_frame_number
        gap_ns = captures[index] - captures[index - 1]
        if frame_delta != 3 or abs(gap_ns - 3 * FRAME_PERIOD_NS) > TRIPLE_PERIOD_TOLERANCE_NS:
            continue
        synthetic_timestamp_ns = captures[index - 1] + FRAME_PERIOD_NS
        if gap_ns <= MAX_OUTPUT_GAP_NS:
            continue
        if synthetic_timestamp_ns <= captures[index - 1] or synthetic_timestamp_ns >= captures[index]:
            continue
        if captures[index] - synthetic_timestamp_ns > MAX_OUTPUT_GAP_NS:
            continue
        events.append(
            FrameRepairEvent(
                role=role,
                source_frame_index=index - 1,
                trigger_frame_index=index,
                synthetic_output_index=index + len(events),
                source_capture_timestamp_ns=captures[index - 1],
                source_device_frame_number=previous_frame_number,
                trigger_capture_timestamp_ns=captures[index],
                trigger_device_frame_number=frame_number,
                host_detection_timestamp_ns=hosts[index],
                original_gap_ns=gap_ns,
                synthetic_timestamp_ns=synthetic_timestamp_ns,
            )
        )
    return tuple(events)


def _repaired_timestamps(
    timestamps: tuple[int, ...], events: tuple[FrameRepairEvent, ...]
) -> tuple[int, ...]:
    repaired = list(timestamps)
    for event in events:
        repaired.insert(event.synthetic_output_index, event.synthetic_timestamp_ns)
    return tuple(repaired)


def _max_gap_ns(timestamps: tuple[int, ...]) -> int:
    return max((current - previous for previous, current in zip(timestamps, timestamps[1:])), default=0)


def build_repair_plan(source_mcap: str | Path, config: DatasetConfig) -> RepairPlan:
    """Identify only delta-three 30 Hz dropouts that one prior-frame hold can cover."""
    _require_30hz_contract(config)
    source = Path(source_mcap).expanduser().resolve()
    episode_id = _episode_id(source)
    _require_source_path(source, config, episode_id)
    policy = build_validation_policy(config.cameras, config.collection.validation_profile)
    source_validation = validate_mcap(source, policy)
    unsupported_failures = set(source_validation["failed_checks"]) - {"camera_max_gap_below_69ms"}
    if unsupported_failures:
        raise OfflineRepairError(
            "source has mandatory failures that frame repair cannot change: "
            + ", ".join(sorted(unsupported_failures))
        )

    inspection = inspect_mcap(source)
    events_by_role: dict[str, tuple[FrameRepairEvent, ...]] = {}
    residual_gaps: list[str] = []
    for role, topic in CAMERA_TOPICS.items():
        diagnostics = _diagnostics_are_complete(
            role,
            inspection.camera_timing_diagnostics.get(role),
            inspection.log_times.get(topic, ()),
        )
        events = _role_events(role, diagnostics)
        events_by_role[role] = events
        repaired_timestamps = _repaired_timestamps(diagnostics.capture_timestamps_ns, events)
        max_gap_ns = _max_gap_ns(repaired_timestamps)
        if max_gap_ns > MAX_OUTPUT_GAP_NS:
            residual_gaps.append(f"{role}={max_gap_ns / 1_000_000:.1f}ms")
    event_count = sum(len(events) for events in events_by_role.values())
    if residual_gaps:
        prefix = (
            "source contains no eligible delta=3 repair and still has camera gaps above 69 ms: "
            if event_count == 0
            else "source still has camera gaps above 69 ms after the allowed repair: "
        )
        raise OfflineRepairError(
            prefix + ", ".join(residual_gaps)
        )
    if event_count == 0:
        raise NoRepairNeededError(
            "source contains no eligible delta=3, approximately 100 ms camera gap"
        )
    return RepairPlan(
        source_mcap=source,
        episode_id=episode_id,
        events_by_role=events_by_role,
        source_validation=source_validation,
    )


def _derived_diagnostics_payload(
    role: str,
    diagnostics: CameraTimingDiagnostics,
    events: tuple[FrameRepairEvent, ...],
) -> dict[str, Any]:
    captures: list[int] = list(diagnostics.capture_timestamps_ns)
    hosts: list[int] = list(diagnostics.host_receive_timestamps_ns)
    devices: list[int | None] = list(diagnostics.device_timestamps_ns)
    frame_numbers: list[int | None] = list(diagnostics.frame_numbers)
    for event in events:
        captures.insert(event.synthetic_output_index, event.synthetic_timestamp_ns)
        # This is the host time at which the following real frame exposed the gap,
        # not a fabricated camera delivery time. Raw diagnostics filter this index.
        hosts.insert(event.synthetic_output_index, event.host_detection_timestamp_ns)
        devices.insert(event.synthetic_output_index, None)
        frame_numbers.insert(event.synthetic_output_index, None)
    sources = dict(diagnostics.capture_timestamp_sources)
    domains = dict(diagnostics.device_timestamp_domains)
    if events:
        sources["offline_duplicate_previous"] = len(events)
        domains["unavailable"] = domains.get("unavailable", 0) + len(events)
    return {
        "version": 2,
        "role": role,
        "capture_timestamp_ns": captures,
        "host_receive_timestamp_ns": hosts,
        "device_timestamp_ns": devices,
        "device_frame_number": frame_numbers,
        "capture_timestamp_sources": sources,
        "device_timestamp_domains": domains,
        "synthetic_indices": [event.synthetic_output_index for event in events],
        "repair_policy": {
            "strategy": REPAIR_STRATEGY,
            "target_fps": 30,
            "trigger_frame_number_delta": 3,
            "period_ns": FRAME_PERIOD_NS,
            "max_output_gap_ns": MAX_OUTPUT_GAP_NS,
            "triple_period_tolerance_ns": TRIPLE_PERIOD_TOLERANCE_NS,
        },
        "repair_events": [event.manifest_value() for event in events],
    }


def _encoder_for_schema(schema: Any, cache: dict[int, Any]) -> Any:
    encoder = cache.get(schema.id)
    if encoder is None:
        definitions = serialize_dynamic(schema.name, schema.data.decode())
        encoder = definitions[schema.name]
        cache[schema.id] = encoder
    return encoder


def rewrite_mcap_with_repairs(
    source_mcap: str | Path,
    destination_mcap: str | Path,
    config: DatasetConfig,
    plan: RepairPlan,
) -> Path:
    """Copy an MCAP while inserting only planned CameraInfo and v2 diagnostics."""
    source = Path(source_mcap)
    destination = Path(destination_mcap)
    if destination.exists():
        raise OfflineRepairError(f"refusing to overwrite staged repaired MCAP: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    role_by_topic = {topic: role for role, topic in CAMERA_TOPICS.items()}
    event_by_trigger = {
        role: {event.trigger_frame_index: event for event in events}
        for role, events in plan.events_by_role.items()
    }
    inspection = inspect_mcap(source)

    with source.open("rb") as input_stream, destination.open("xb") as output_stream:
        reader = make_reader(input_stream)
        header = reader.get_header()
        writer = Writer(output_stream)
        writer.start(profile=header.profile, library="yam-data-collect offline-frame-repair")
        schema_ids: dict[int, int] = {}
        channel_ids: dict[int, int] = {}
        channel_sequences: dict[int, int] = {}
        camera_indices = {role: 0 for role in CAMERA_TOPICS}
        encoders: dict[int, Any] = {}
        diagnostic_channel: tuple[Any, int, int] | None = None

        for schema, channel, message in reader.iter_messages():
            new_schema_id = 0
            if schema is not None:
                if schema.id not in schema_ids:
                    schema_ids[schema.id] = writer.register_schema(schema.name, schema.encoding, schema.data)
                new_schema_id = schema_ids[schema.id]
            if channel.id not in channel_ids:
                channel_ids[channel.id] = writer.register_channel(
                    channel.topic,
                    channel.message_encoding,
                    new_schema_id,
                    dict(channel.metadata),
                )
            new_channel_id = channel_ids[channel.id]
            if channel.topic == FRAME_TIMING_DIAGNOSTICS_TOPIC:
                if schema is None:
                    raise OfflineRepairError("frame timing diagnostics channel has no schema")
                diagnostic_channel = (schema, new_channel_id, channel.id)
                continue

            role = role_by_topic.get(channel.topic)
            if role is not None:
                source_index = camera_indices[role]
                event = event_by_trigger[role].get(source_index)
                if event is not None:
                    if schema is None:
                        raise OfflineRepairError(f"{role} CameraInfo channel has no schema")
                    encoder = _encoder_for_schema(schema, encoders)
                    synthetic_data = encoder(camera_info_message(config.cameras[role], event.synthetic_timestamp_ns))
                    sequence = channel_sequences.get(channel.id, 0)
                    writer.add_message(
                        new_channel_id,
                        event.synthetic_timestamp_ns,
                        synthetic_data,
                        event.synthetic_timestamp_ns,
                        sequence,
                    )
                    channel_sequences[channel.id] = sequence + 1

            sequence = channel_sequences.get(channel.id, 0)
            writer.add_message(
                new_channel_id,
                message.log_time,
                message.data,
                message.publish_time,
                sequence,
            )
            channel_sequences[channel.id] = sequence + 1
            if role is not None:
                camera_indices[role] += 1

        if diagnostic_channel is None:
            raise OfflineRepairError("source MCAP has no /diagnostics/frame_timing channel")
        diagnostic_schema, new_diagnostic_channel_id, original_diagnostic_channel_id = diagnostic_channel
        diagnostic_encoder = _encoder_for_schema(diagnostic_schema, encoders)
        for role in CAMERA_TOPICS:
            diagnostics = inspection.camera_timing_diagnostics[role]
            payload = _derived_diagnostics_payload(role, diagnostics, plan.events_by_role[role])
            capture_time_ns = int(payload["capture_timestamp_ns"][-1])
            host_time_ns = int(payload["host_receive_timestamp_ns"][-1])
            sequence = channel_sequences.get(original_diagnostic_channel_id, 0)
            writer.add_message(
                new_diagnostic_channel_id,
                host_time_ns,
                diagnostic_encoder(
                    {"data": json.dumps(payload, ensure_ascii=True, separators=(",", ":"))}
                ),
                capture_time_ns,
                sequence,
            )
            channel_sequences[original_diagnostic_channel_id] = sequence + 1

        for metadata in reader.iter_metadata():
            writer.add_metadata(metadata.name, dict(metadata.metadata))
        for attachment in reader.iter_attachments():
            writer.add_attachment(
                attachment.create_time,
                attachment.log_time,
                attachment.name,
                attachment.media_type,
                attachment.data,
            )
        writer.finish()
    return destination


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(path)


def _write_record_config(output_dataset: Path, config: DatasetConfig, source_mcap: Path) -> None:
    saved = copy.deepcopy(dict(config.raw))
    collection = dict(saved.get("collection_config", {}))
    collection["dataset_dir"] = str(output_dataset)
    saved["collection_config"] = collection
    saved["offline_frame_repair"] = {
        "strategy": REPAIR_STRATEGY,
        "source_mcap": str(source_mcap),
        "created_at_unix_ns": time.time_ns(),
    }
    path = output_dataset / "record_config.yaml"
    partial = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    with partial.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(saved, stream, allow_unicode=True, sort_keys=False)
    partial.replace(path)


def default_repaired_dataset(config: DatasetConfig) -> Path:
    return config.collection.dataset_dir / "derived" / REPAIR_STRATEGY


def repair_30hz_episode(
    source_mcap: str | Path,
    config: DatasetConfig,
    *,
    output_dataset: str | Path | None = None,
    overwrite: bool = False,
    on_plan_ready: Callable[[RepairPlan], None] | None = None,
) -> RepairedEpisode:
    """Create and atomically publish one independently verifiable derivative."""
    plan = build_repair_plan(source_mcap, config)
    if on_plan_ready is not None:
        on_plan_ready(plan)
    source_dataset = config.collection.dataset_dir
    output_root = (
        default_repaired_dataset(config)
        if output_dataset is None
        else Path(output_dataset).expanduser().resolve()
    )
    if output_root.resolve(strict=False) == source_dataset.resolve(strict=False):
        raise OfflineRepairError("repaired output dataset must differ from the raw source dataset")
    direct_h264 = uses_direct_h264_publication(config)
    source_video_root = (
        final_video_dir(source_dataset, plan.episode_id)
        if direct_h264
        else live_video_dir(source_dataset, plan.episode_id)
    )
    source_video_codecs = final_codec_family(config) if direct_h264 else live_codec_family(config.collection.live_video_codec)
    policy = build_validation_policy(config.cameras, config.collection.validation_profile)

    with DatasetOperationLock(source_dataset, "repair-30hz"):
        source_video_validation = validate_video_assets(
            plan.source_mcap,
            policy,
            video_dir=source_video_root,
            expected_video_codecs=source_video_codecs,
            require_fps_match=False,
        )
        if not source_video_validation["passed"]:
            failures = [
                f"{role}: {entry.get('error', 'frame identity mismatch')}"
                for role, entry in source_video_validation["per_camera"].items()
                if not entry.get("passed")
            ]
            raise OfflineRepairError("raw live video validation failed: " + "; ".join(failures))

        source_paths = {"mcap": plan.source_mcap}
        source_paths.update(
            {role: source_video_root / f"{role}.mp4" for role in CAMERA_TOPICS}
        )
        source_hashes_before = {name: _sha256(path) for name, path in source_paths.items()}
        output_root.mkdir(parents=True, exist_ok=True)
        stage_token = uuid.uuid4().hex
        output_episode_dir = output_root / str(plan.episode_id)
        output_episode_dir.mkdir(parents=True, exist_ok=True)
        staged_mcap = output_episode_dir / f".{plan.episode_id}.repairing-{stage_token}.mcap"
        staged_video_dir = output_episode_dir / f".videos-repairing-{stage_token}"
        staged_video_dir.mkdir()
        final_mcap = final_mcap_path(output_root, plan.episode_id)
        final_video_root = final_video_dir(output_root, plan.episode_id)
        if (final_mcap.exists() or final_video_root.exists()) and not overwrite:
            shutil.rmtree(staged_video_dir, ignore_errors=True)
            raise OfflineRepairError(
                f"repaired episode {plan.episode_id} already exists; use --overwrite to replace it atomically"
            )

        transcodes: dict[str, VideoTranscodeResult | VideoFrameDuplicateResult] = {}
        try:
            rewrite_mcap_with_repairs(plan.source_mcap, staged_mcap, config, plan)
            repaired_mcap_validation = validate_mcap(staged_mcap, policy)
            if not repaired_mcap_validation["output_mcap_passed"]:
                raise OfflineRepairError(
                    "repaired MCAP still fails mandatory checks: "
                    + ", ".join(repaired_mcap_validation["failed_checks"])
                )

            for role in CAMERA_TOPICS:
                source_video = source_video_root / f"{role}.mp4"
                destination_video = staged_video_dir / f"{role}.mp4"
                duplicate_after = [
                    event.source_frame_index for event in plan.events_by_role[role]
                ]
                if duplicate_after:
                    duplicate_transcoder = (
                        transcode_to_h264_with_frame_duplicates
                        if direct_h264
                        else transcode_to_hevc_with_frame_duplicates
                    )
                    transcodes[role] = duplicate_transcoder(
                            source_video,
                            destination_video,
                            duplicate_after_indices=duplicate_after,
                            crf=config.collection.video_crf,
                            preset=config.collection.video_preset,
                            threads=config.collection.video_threads,
                        )
                elif direct_h264:
                    # Keep unaffected H.264 assets byte-for-byte identical in
                    # the derivative; only cameras with an eligible dropout
                    # need a re-encode to insert a frame.
                    shutil.copy2(source_video, destination_video)
                else:
                    transcodes[role] = transcode_to_hevc(
                        source_video,
                        destination_video,
                        crf=config.collection.video_crf,
                        preset=config.collection.video_preset,
                        threads=config.collection.video_threads,
                    )

            repaired_video_validation = validate_video_assets(
                staged_mcap,
                policy,
                video_dir=staged_video_dir,
                expected_video_codecs=final_codec_family(config) if direct_h264 else ("hevc",),
            )
            if not repaired_video_validation["passed"]:
                failures = [
                    f"{role}: {entry.get('error', 'asset validation failed')}"
                    for role, entry in repaired_video_validation["per_camera"].items()
                    if not entry.get("passed")
                ]
                raise OfflineRepairError("repaired video validation failed: " + "; ".join(failures))

            source_hashes_after = {name: _sha256(path) for name, path in source_paths.items()}
            if source_hashes_after != source_hashes_before:
                raise OfflineRepairError("raw source assets changed while creating the derivative")
            output_hashes = {"mcap": _sha256(staged_mcap)}
            output_hashes.update(
                {role: _sha256(staged_video_dir / f"{role}.mp4") for role in CAMERA_TOPICS}
            )
            manifest = {
                "schema_version": 1,
                "strategy": REPAIR_STRATEGY,
                "created_at_unix_ns": time.time_ns(),
                "episode_id": plan.episode_id,
                "source_mcap": str(plan.source_mcap),
                "source_video_dir": str(source_video_root),
                "source_sha256": source_hashes_before,
                "output_sha256": output_hashes,
                "events_by_role": {
                    role: [event.manifest_value() for event in events]
                    for role, events in plan.events_by_role.items()
                },
                "source_validation": dict(plan.source_validation),
                "repaired_mcap_validation": repaired_mcap_validation,
                "repaired_video_validation": repaired_video_validation,
            }
            (staged_video_dir / "repair_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _publish_verified_episode(
                staged_video_dir,
                final_video_root,
                staged_mcap,
                final_mcap,
                overwrite=overwrite,
            )
        except BaseException:
            staged_mcap.unlink(missing_ok=True)
            shutil.rmtree(staged_video_dir, ignore_errors=True)
            raise

        _write_record_config(output_root, config, plan.source_mcap)
        manifest_path = output_root / "repair_manifests" / f"{plan.episode_id}.json"
        _atomic_json(manifest_path, manifest)
        return RepairedEpisode(
            episode_id=plan.episode_id,
            source_mcap=plan.source_mcap,
            source_video_dir=source_video_root,
            output_mcap=final_mcap,
            output_video_dir=final_video_root,
            manifest_path=manifest_path,
            events_by_role=plan.events_by_role,
            validation=repaired_mcap_validation,
        )
