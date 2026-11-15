"""Adapt only an extracted transfer copy to the target host's CPU indices."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent


def parse_cpus(value: str) -> set[int]:
    result: set[int] = set()
    for part in value.split(","):
        if re.fullmatch(r"\d+", part):
            result.add(int(part))
        elif re.fullmatch(r"\d+-\d+", part):
            first, last = map(int, part.split("-"))
            if last < first:
                raise ValueError(f"invalid CPU range: {part}")
            result.update(range(first, last + 1))
        else:
            raise ValueError(f"invalid CPU list: {value}")
    return result


def cpu_text(cpus: set[int]) -> str:
    return ",".join(map(str, sorted(cpus)))


def choose_cpus(available: set[int], requested: list[str | None]) -> dict[str, set[int]]:
    if len(available) < 6:
        raise ValueError("at least 6 allowed logical CPUs are required for this suggested partition")
    if any(requested) and not all(requested):
        raise ValueError("specify control, recorder and encoder CPU lists together")
    if all(requested):
        groups = [parse_cpus(value) for value in requested]
    else:
        ordered = sorted(available)
        worker_count = min(4, max(1, len(ordered) // 4))
        groups = [
            set(ordered[: -2 * worker_count]),
            set(ordered[-2 * worker_count : -worker_count]),
            set(ordered[-worker_count:]),
        ]
    combined: set[int] = set()
    for group in groups:
        if not group or not group <= available:
            raise ValueError(f"CPU group must be nonempty and inside available CPUs: {sorted(available)}")
        if combined & group:
            raise ValueError("control, recorder and encoder CPU groups must not overlap")
        combined.update(group)
    return dict(zip(("control", "recorder", "encoder"), groups))


def configure(root: Path, groups: dict[str, set[int]], *, apply: bool) -> dict[str, str]:
    if not (root / ".yam-single-arm-transfer").is_file():
        raise ValueError("refusing to change the original deployment; use the extracted transfer copy")
    partition = {role: cpu_text(cpus) for role, cpus in groups.items()}
    config_path = root / "yam_data_collection/config.host.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    for collection in [config["collection_config"]] + [
        mode.setdefault("collection_config", {}) for mode in config["capture_modes"].values()
    ]:
        collection["recorder_cpu_set"] = partition["recorder"]
        collection["live_video_cpu_set"] = partition["encoder"]
    replacements = {config_path: yaml.safe_dump(config, sort_keys=False, allow_unicode=True)}
    for relative in ("yam.sh", "yam_single_arm_collection/leader_motion.sh"):
        path = root / relative
        content, count = re.subn(
            r"exec taskset -c [0-9,-]+ (?=\.venv/bin/python examples/minimum_gello/minimum_gello\.py)",
            f"exec taskset -c {partition['control']} ",
            path.read_text(encoding="utf-8"),
        )
        if count != 1:
            raise ValueError(f"expected one controller CPU binding in {relative}; found {count}")
        if relative == "yam.sh":
            content, count = re.subn(
                r"exec taskset -c [0-9,-]+ (?=\.venv/bin/yam-data-collect camera-test)",
                f"exec taskset -c {cpu_text(groups['recorder'] | groups['encoder'])} ",
                content,
            )
            if count != 1:
                raise ValueError("expected one camera-test CPU binding in yam.sh")
        replacements[path] = content
    if apply:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = root / "single_arm_transfer/backups" / timestamp
        for path, content in replacements.items():
            copy_path = backup / path.relative_to(root)
            copy_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, copy_path)
            path.write_text(content, encoding="utf-8")
        (root / "single_arm_transfer/target_cpu.json").write_text(
            json.dumps(partition, indent=2) + "\n", encoding="utf-8"
        )
    return partition


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--control-cpus")
    parser.add_argument("--recorder-cpus")
    parser.add_argument("--encoder-cpus")
    parser.add_argument(
        "--apply", action="store_true", help="Write target-copy files; default only previews the partition"
    )
    args = parser.parse_args()
    try:
        groups = choose_cpus(set(os.sched_getaffinity(0)), [args.control_cpus, args.recorder_cpus, args.encoder_cpus])
        partition = configure(args.root.resolve(), groups, apply=args.apply)
    except (OSError, TypeError, ValueError, KeyError) as error:
        parser.exit(1, f"configure-cpu: {error}\n")
    print(json.dumps(partition, indent=2))
    print(
        "Target-copy CPU settings written; no hardware commands sent."
        if args.apply
        else "Preview only; use --apply to write."
    )


if __name__ == "__main__":
    main()
