"""Read and rewrite the JSON payload carried by an MCAP ``/task_info`` topic."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Iterator

from mcap.reader import make_reader
from mcap.writer import Writer
from mcap_ros2._dynamic import serialize_dynamic
from mcap_ros2.decoder import DecoderFactory

TASK_INFO_TOPIC = "/task_info"


class TaskInfoError(RuntimeError):
    """Raised when an MCAP does not meet the expected task-info contract."""


def mcap_files(path: Path, *, include_pending: bool) -> Iterator[Path]:
    """Yield one input file, or all MCAPs below a directory in stable order."""
    files = [path] if path.is_file() else sorted(path.rglob("*.mcap"))
    for mcap_path in files:
        if not include_pending and ".pending" in mcap_path.parts:
            continue
        if mcap_path.name.endswith(".partial.mcap"):
            continue
        yield mcap_path


def read_task_info(path: Path) -> list[dict[str, Any]]:
    """Return every JSON object published on ``/task_info`` in an MCAP."""
    values: list[dict[str, Any]] = []
    with path.open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for _schema, _channel, _message, decoded in reader.iter_decoded_messages(topics=[TASK_INFO_TOPIC]):
            data = getattr(decoded, "data", None)
            if not isinstance(data, str):
                raise TaskInfoError(f"{path}: {TASK_INFO_TOPIC} message is not std_msgs/msg/String")
            try:
                value = json.loads(data)
            except json.JSONDecodeError as error:
                raise TaskInfoError(f"{path}: invalid JSON in {TASK_INFO_TOPIC}: {error}") from error
            if not isinstance(value, dict):
                raise TaskInfoError(f"{path}: {TASK_INFO_TOPIC} JSON must be an object")
            values.append(value)
    return values


def rewrite_task_info(
    source: Path,
    destination: Path,
    task_description: str | None = None,
    task_id: str | None = None,
) -> int:
    """Copy ``source`` to ``destination``, changing only ``/task_info`` messages."""
    if destination.exists():
        raise TaskInfoError(f"refusing to overwrite existing destination: {destination}")

    changed = 0
    schema_ids: dict[int, int] = {}
    channel_ids: dict[int, int] = {}
    encoders: dict[int, Any] = {}
    with source.open("rb") as input_stream, destination.open("xb") as output_stream:
        reader = make_reader(input_stream)
        header = reader.get_header()
        writer = Writer(output_stream)
        writer.start(profile=header.profile, library="yam-data-collect task-info editor")
        try:
            for schema, channel, message in reader.iter_messages():
                new_schema_id = 0
                if schema is not None:
                    new_schema_id = schema_ids.get(schema.id, 0)
                    if not new_schema_id:
                        new_schema_id = writer.register_schema(schema.name, schema.encoding, schema.data)
                        schema_ids[schema.id] = new_schema_id

                new_channel_id = channel_ids.get(channel.id)
                if new_channel_id is None:
                    new_channel_id = writer.register_channel(
                        channel.topic, channel.message_encoding, new_schema_id, dict(channel.metadata)
                    )
                    channel_ids[channel.id] = new_channel_id

                data = message.data
                if channel.topic == TASK_INFO_TOPIC:
                    if schema is None:
                        raise TaskInfoError(f"{source}: {TASK_INFO_TOPIC} channel has no schema")
                    encoder = encoders.get(schema.id)
                    if encoder is None:
                        try:
                            encoder = serialize_dynamic(schema.name, schema.data.decode())[schema.name]
                        except (KeyError, UnicodeDecodeError, ValueError) as error:
                            raise TaskInfoError(f"{source}: cannot encode {TASK_INFO_TOPIC}: {error}") from error
                        encoders[schema.id] = encoder
                    values = read_task_info_message(source, schema, message.data)
                    if task_description is not None:
                        values["task_description"] = task_description
                    if task_id is not None:
                        values["task_id"] = task_id
                    data = encoder({"data": json.dumps(values, ensure_ascii=False)})
                    changed += 1

                writer.add_message(new_channel_id, message.log_time, data, message.publish_time, message.sequence)

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
        except Exception:
            output_stream.close()
            destination.unlink(missing_ok=True)
            raise

    if not changed:
        destination.unlink()
        raise TaskInfoError(f"{source}: no {TASK_INFO_TOPIC} messages found")
    return changed


def read_task_info_message(source: Path, schema: Any, data: bytes) -> dict[str, Any]:
    """Decode one raw String message using its embedded ROS2 schema."""
    decoder = DecoderFactory().decoder_for("cdr", schema)
    if decoder is None:
        raise TaskInfoError(f"{source}: cannot decode {TASK_INFO_TOPIC} as ROS2 CDR")
    decoded = decoder(data)
    payload = getattr(decoded, "data", None)
    if not isinstance(payload, str):
        raise TaskInfoError(f"{source}: {TASK_INFO_TOPIC} message is not std_msgs/msg/String")
    try:
        values = json.loads(payload)
    except json.JSONDecodeError as error:
        raise TaskInfoError(f"{source}: invalid JSON in {TASK_INFO_TOPIC}: {error}") from error
    if not isinstance(values, dict):
        raise TaskInfoError(f"{source}: {TASK_INFO_TOPIC} JSON must be an object")
    return values


def _common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("path", type=Path, help="one .mcap file or a directory to scan recursively")
    parser.add_argument("--include-pending", action="store_true", help="also process files below .pending")
    return parser


def show_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show /task_info JSON from one MCAP file.")
    parser.add_argument("path", type=Path, help="the .mcap file to inspect")
    args = parser.parse_args(argv)
    if not args.path.exists():
        parser.error(f"path does not exist: {args.path}")
    if not args.path.is_file() or args.path.suffix != ".mcap":
        parser.error("path must be one .mcap file, not a directory")
    try:
        values = read_task_info(args.path)
        if not values:
            raise TaskInfoError(f"{args.path}: no {TASK_INFO_TOPIC} messages found")
        print(f"{args.path}:")
        for index, value in enumerate(values):
            print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
            if index + 1 != len(values):
                print()
    except (OSError, TaskInfoError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


def range_mcap_files(directory: Path, range_text: str) -> list[Path]:
    """Resolve an inclusive ``START-END`` range to numbered episode files."""
    try:
        start_text, end_text = range_text.split("-", maxsplit=1)
        start, end = int(start_text), int(end_text)
    except ValueError as error:
        raise TaskInfoError("--range must use START-END, for example 5-10") from error
    if start < 0 or end < start:
        raise TaskInfoError("--range must have non-negative START and END >= START")
    files = []
    for index in range(start, end + 1):
        episode_dir = directory / str(index)
        candidates = sorted(
            path for path in episode_dir.glob("*.mcap") if not path.name.endswith(".partial.mcap")
        )
        files.append(candidates[0] if len(candidates) == 1 else episode_dir / f"{index}.mcap")
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise TaskInfoError("requested range has missing MCAP files:\n" + "\n".join(missing))
    return files


def update_main(argv: list[str] | None = None) -> int:
    parser = _common_parser("Update task_description and task_id in MCAP /task_info messages.")
    parser.add_argument("--task-description", help="new task_description value (omit to keep existing)")
    parser.add_argument("--task-id", help="new task_id value (omit to keep existing)")
    parser.add_argument("--match-task-id", help="only select MCAPs with this existing task_id")
    parser.add_argument(
        "--range",
        dest="episode_range",
        metavar="START-END",
        help="inclusive numbered episodes below PATH, for example 5-10",
    )
    parser.add_argument("--apply", action="store_true", help="replace each input after a verified rewrite")
    args = parser.parse_args(argv)
    if args.task_description is None and args.task_id is None:
        parser.error("provide --task-description and/or --task-id")
    if not args.path.exists():
        parser.error(f"path does not exist: {args.path}")
    try:
        if args.episode_range:
            if not args.path.is_dir():
                parser.error("PATH must be a directory when --range is used")
            files = range_mcap_files(args.path, args.episode_range)
        else:
            files = list(mcap_files(args.path, include_pending=args.include_pending))
    except TaskInfoError as error:
        parser.error(str(error))
    if not files:
        parser.error("no .mcap files found")

    updates = {}
    if args.task_description is not None:
        updates["task_description"] = args.task_description
    if args.task_id is not None:
        updates["task_id"] = args.task_id
    # Read and validate all selected metadata before replacing any files.
    selected: list[tuple[Path, list[dict[str, Any]]]] = []
    try:
        for path in files:
            original_values = read_task_info(path)
            if args.match_task_id is not None:
                matches = [value.get("task_id") == args.match_task_id for value in original_values]
                if not any(matches):
                    continue
                if not all(matches):
                    raise TaskInfoError(f"{path}: mixed task_ids; refusing to rewrite the entire MCAP")
            if not original_values:
                raise TaskInfoError(f"{path}: no {TASK_INFO_TOPIC} messages found")
            selected.append((path, original_values))
    except (OSError, TaskInfoError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    if not selected:
        parser.error(f"no MCAP files match task_id {args.match_task_id!r}")

    updated_count = unchanged_count = planned_count = 0
    for path, original_values in selected:
        expected = [dict(value, **updates) for value in original_values]
        if expected == original_values:
            unchanged_count += 1
            print(f"UNCHANGED: {path}")
            continue
        if not args.apply:
            planned_count += 1
            print(f"DRY RUN: would update {path}: {json.dumps(updates, ensure_ascii=False)}")
            continue
        backup = path.with_suffix(path.suffix + ".bak")
        if backup.exists():
            backup = path.with_suffix(path.suffix + f".bak.{uuid.uuid4().hex}")
        temporary = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
        try:
            changed = rewrite_task_info(path, temporary, args.task_description, args.task_id)
            updated = read_task_info(temporary)
            if updated != expected:
                raise TaskInfoError(f"{path}: verification of rewritten task_info failed")
            if read_task_info(path) != original_values:
                raise TaskInfoError(f"{path}: task_info changed since selection; refusing to replace")
            with path.open("rb") as original, backup.open("xb") as backup_stream:
                shutil.copyfileobj(original, backup_stream)
            shutil.copystat(path, backup)
            os.replace(temporary, path)
            updated_count += 1
            print(f"UPDATED: {path} ({changed} {TASK_INFO_TOPIC} message(s)); backup: {backup}")
        except (OSError, TaskInfoError) as error:
            temporary.unlink(missing_ok=True)
            print(f"ERROR: {error}", file=sys.stderr)
            return 1
    print(
        f"Summary: matched {len(selected)} MCAPs; updated {updated_count}; "
        f"unchanged {unchanged_count}; would update {planned_count}"
    )
    return 0
