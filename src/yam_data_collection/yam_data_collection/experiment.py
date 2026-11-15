"""Repeatable full-load A/B/C experiments for RealSense frame-drop diagnosis.

The normal ``capture`` command remains operator-driven and writes to the
production dataset.  This module wraps the same :class:`EpisodeCollector` in
an isolated, duration-bounded experiment run.  It deliberately keeps the
capture profile, validation policy, MCAP timing diagnostics, and video writer
unchanged; it only records the host conditions around those recordings.
"""

from __future__ import annotations

import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from .collector import EpisodeCollector
from .config import CameraConfig, DatasetConfig
from .mcap_writer import CAMERA_TOPICS
from .quality import build_validation_policy, inspect_mcap, validate_mcap, validate_video_assets
from .runtime_diagnostics import RuntimeDiagnosticsMonitor
from .terminal import print_status
from .video_pipeline import (
    final_codec_family,
    final_mcap_path,
    live_codec_family,
    live_video_dir,
    pending_mcap_path,
    uses_direct_h264_publication,
)


class ExperimentError(RuntimeError):
    """An A/B/C experiment cannot be started or completed safely."""


_BDF_RE = re.compile(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]")
_SAFE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_CPU_RANGE_RE = re.compile(r"(\d+)\s*(?:-\s*(\d+))?")
_USB_PORT_RE = re.compile(r"\d+-\d+(?:\.\d+)*")


def parse_cpu_set(value: str) -> tuple[int, ...]:
    """Parse an affinity expression such as ``12-19`` or ``0,2-4``."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("CPU set must be a non-empty string such as 12-19")
    cpus: set[int] = set()
    for item in value.split(","):
        match = _CPU_RANGE_RE.fullmatch(item.strip())
        if match is None:
            raise ValueError(f"invalid CPU-set item {item!r}; use values such as 12-19 or 0,2,4")
        first = int(match.group(1))
        last = int(match.group(2) or first)
        if last < first:
            raise ValueError(f"invalid descending CPU range {item!r}")
        cpus.update(range(first, last + 1))
    return tuple(sorted(cpus))


def _as_string_tuple(value: Any, name: str) -> tuple[str, ...]:
    values = (value,) if isinstance(value, str) else value
    if not isinstance(values, list | tuple) or not values:
        raise ValueError(f"{name} must be a non-empty string or list of strings")
    normalized = tuple(str(item).strip().lower() for item in values)
    if any(not item for item in normalized):
        raise ValueError(f"{name} must not contain an empty value")
    return normalized


def _validate_controller(value: str, name: str) -> str:
    normalized = value.strip().lower()
    if _BDF_RE.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be a PCI BDF such as 0000:80:14.0, got {value!r}")
    return normalized


@dataclass(frozen=True)
class ControllerRule:
    """Expected controller relationship for one camera role in a case."""

    equals: str | None = None
    one_of: tuple[str, ...] = ()
    not_equals: tuple[str, ...] = ()
    different_from: tuple[str, ...] = ()

    @classmethod
    def from_value(cls, value: Any, name: str) -> "ControllerRule":
        if isinstance(value, str):
            return cls(equals=_validate_controller(value, name))
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be a PCI BDF string or mapping")
        allowed = {"equals", "one_of", "not_equals", "different_from"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"{name} has unsupported keys: {', '.join(unknown)}")
        equals_value = value.get("equals")
        equals = _validate_controller(str(equals_value), f"{name}.equals") if equals_value is not None else None
        one_of = tuple(
            _validate_controller(item, f"{name}.one_of") for item in _as_string_tuple(value["one_of"], f"{name}.one_of")
        ) if "one_of" in value else ()
        not_equals = tuple(
            _validate_controller(item, f"{name}.not_equals")
            for item in _as_string_tuple(value["not_equals"], f"{name}.not_equals")
        ) if "not_equals" in value else ()
        different_from = tuple(str(item).strip() for item in _as_string_tuple(value["different_from"], f"{name}.different_from")) if "different_from" in value else ()
        if not any((equals, one_of, not_equals, different_from)):
            raise ValueError(f"{name} must declare at least one controller constraint")
        return cls(equals=equals, one_of=one_of, not_equals=not_equals, different_from=different_from)


@dataclass(frozen=True)
class ExperimentCase:
    """One controlled A/B/C condition from a host-specific plan."""

    name: str
    cpu_set: tuple[int, ...]
    camera_controller_rules: Mapping[str, ControllerRule]
    description: str = ""


@dataclass(frozen=True)
class ExperimentPlan:
    """A portable experiment definition, with only host topology in YAML."""

    source_path: Path
    raw: Mapping[str, Any]
    output_dir: Path
    duration_s: float
    repeats: int
    inter_trial_delay_s: float
    cases: Mapping[str, ExperimentCase]


@dataclass(frozen=True)
class ExperimentRun:
    """Locations produced by one experiment condition."""

    case_dir: Path
    summary_path: Path
    comparison_path: Path


def load_experiment_plan(path: str | Path) -> ExperimentPlan:
    """Load the A/B/C plan without touching capture or acceptance settings."""
    source_path = Path(path).expanduser().resolve()
    with source_path.open("r", encoding="utf-8") as stream:
        raw_value = yaml.safe_load(stream)
    if not isinstance(raw_value, Mapping):
        raise ValueError("experiment plan root must be a mapping")
    root = raw_value.get("experiment", raw_value)
    if not isinstance(root, Mapping):
        raise ValueError("experiment must be a mapping")
    output_value = root.get("output_dir")
    if not isinstance(output_value, str) or not output_value.strip():
        raise ValueError("experiment.output_dir must be a non-empty path")
    output_path = Path(output_value).expanduser()
    output_dir = output_path if output_path.is_absolute() else (source_path.parent / output_path).resolve()
    duration_s = float(root.get("duration_s", 300.0))
    repeats = int(root.get("repeats", 3))
    inter_trial_delay_s = float(root.get("inter_trial_delay_s", 15.0))
    if duration_s <= 0:
        raise ValueError("experiment.duration_s must be positive")
    if repeats <= 0:
        raise ValueError("experiment.repeats must be positive")
    if inter_trial_delay_s < 0:
        raise ValueError("experiment.inter_trial_delay_s must be non-negative")
    raw_cases = root.get("cases")
    if not isinstance(raw_cases, Mapping) or not raw_cases:
        raise ValueError("experiment.cases must be a non-empty mapping")
    cases: dict[str, ExperimentCase] = {}
    for raw_name, raw_case in raw_cases.items():
        name = str(raw_name).strip()
        if _SAFE_NAME_RE.fullmatch(name) is None:
            raise ValueError(f"experiment case name {name!r} may contain only letters, digits, ., _, and -")
        if not isinstance(raw_case, Mapping):
            raise ValueError(f"experiment.cases.{name} must be a mapping")
        cpu_value = raw_case.get("cpu_set")
        if not isinstance(cpu_value, str):
            raise ValueError(f"experiment.cases.{name}.cpu_set must be a string")
        raw_rules = raw_case.get("camera_controller_rules")
        if not isinstance(raw_rules, Mapping) or not raw_rules:
            raise ValueError(f"experiment.cases.{name}.camera_controller_rules must be a non-empty mapping")
        rules = {
            str(role).strip(): ControllerRule.from_value(rule, f"experiment.cases.{name}.camera_controller_rules.{role}")
            for role, rule in raw_rules.items()
        }
        if any(not role for role in rules):
            raise ValueError(f"experiment.cases.{name}.camera_controller_rules contains an empty role")
        cases[name] = ExperimentCase(
            name=name,
            cpu_set=parse_cpu_set(cpu_value),
            camera_controller_rules=rules,
            description=str(raw_case.get("description", "")).strip(),
        )
    return ExperimentPlan(
        source_path=source_path,
        raw=dict(raw_value),
        output_dir=output_dir,
        duration_s=duration_s,
        repeats=repeats,
        inter_trial_delay_s=inter_trial_delay_s,
        cases=cases,
    )


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def _controller_for_path(path: Path) -> str | None:
    matches = _BDF_RE.findall(str(path.resolve()))
    return matches[-1].lower() if matches else None


def _realsense_physical_ports() -> tuple[dict[str, str], str | None]:
    """Resolve RealSense device serials to physical Linux paths.

    RealSense's camera serial is not necessarily the USB descriptor serial in
    ``/sys/bus/usb/devices``.  Librealsense is the authoritative bridge
    between the serial in the capture YAML and the kernel's physical port.
    """
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        return {}, f"pyrealsense2 is unavailable: {exc}"
    try:
        context = rs.context()
        ports: dict[str, str] = {}
        for device in context.query_devices():
            if not device.supports(rs.camera_info.serial_number):
                continue
            serial = str(device.get_info(rs.camera_info.serial_number)).strip()
            if not serial:
                continue
            if not device.supports(rs.camera_info.physical_port):
                continue
            physical_port = str(device.get_info(rs.camera_info.physical_port)).strip()
            if physical_port:
                ports[serial] = physical_port
        return ports, None
    except Exception as exc:
        return {}, f"could not query RealSense physical ports: {exc}"


def _usb_port_from_physical_path(path: Path) -> str | None:
    for component in reversed(path.parts):
        if _USB_PORT_RE.fullmatch(component):
            return component
    return None


def _usb_entry_from_device_path(
    *,
    serial: str,
    physical_port: str,
    usb_devices_root: Path,
) -> dict[str, Any]:
    physical_path = Path(physical_port).resolve()
    usb_port = _usb_port_from_physical_path(physical_path)
    usb_device = usb_devices_root / usb_port if usb_port is not None else None
    entry: dict[str, Any] = {
        "serial": serial,
        "physical_port": physical_port,
        "resolved_physical_port": str(physical_path),
        "controller": _controller_for_path(physical_path),
        "usb_port": usb_port,
    }
    if usb_device is None or not usb_device.is_dir():
        entry["error"] = "could not resolve a USB device directory from RealSense physical_port"
        return entry
    entry.update(
        usb_sysfs_device=str(usb_device),
        busnum=_read_text(usb_device / "busnum"),
        devnum=_read_text(usb_device / "devnum"),
        speed_mbps=_read_text(usb_device / "speed"),
        manufacturer=_read_text(usb_device / "manufacturer"),
        product=_read_text(usb_device / "product"),
        id_vendor=_read_text(usb_device / "idVendor"),
        id_product=_read_text(usb_device / "idProduct"),
    )
    if entry["controller"] is None:
        entry["error"] = "could not resolve a PCI controller from RealSense physical_port"
    return entry


def discover_camera_topology(
    cameras: Mapping[str, CameraConfig],
    usb_devices_root: Path = Path("/sys/bus/usb/devices"),
    *,
    physical_ports: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Map configured RealSense serials to their current USB device and PCI controller."""
    if physical_ports is None:
        physical_ports, discovery_error = _realsense_physical_ports()
    else:
        discovery_error = None
    from_realsense: dict[str, dict[str, Any]] = {}
    for role, camera in cameras.items():
        physical_port = physical_ports.get(camera.serial)
        if physical_port:
            from_realsense[role] = _usb_entry_from_device_path(
                serial=camera.serial,
                physical_port=physical_port,
                usb_devices_root=usb_devices_root,
            )
    if len(from_realsense) == len(cameras):
        return from_realsense

    # Fallback only for devices whose capture serial happens to equal the USB
    # descriptor serial. This is useful on some hardware revisions, but never
    # replaces the RealSense physical-port mapping when it is available.
    devices: list[tuple[Path, str]] = []
    try:
        candidates = sorted(usb_devices_root.iterdir())
    except OSError as exc:
        return {
            role: {"serial": camera.serial, "error": f"cannot read {usb_devices_root}: {exc}"}
            for role, camera in cameras.items()
        }
    for device in candidates:
        # USB interface nodes (for example ``4-3:1.0``) do not own the device
        # serial. Root-hub entries have no serial and are ignored naturally.
        serial = _read_text(device / "serial")
        if serial:
            devices.append((device, serial))
    result: dict[str, dict[str, Any]] = {}
    for role, camera in cameras.items():
        if role in from_realsense:
            result[role] = from_realsense[role]
            continue
        matched = [device for device, serial in devices if serial == camera.serial]
        if not matched:
            detail = "configured RealSense serial is not present in Librealsense physical-port discovery or USB sysfs"
            if discovery_error is not None:
                detail += f" ({discovery_error})"
            result[role] = {"serial": camera.serial, "error": detail}
            continue
        if len(matched) > 1:
            result[role] = {
                "serial": camera.serial,
                "error": "configured serial matched multiple USB devices",
                "matches": [str(device) for device in matched],
            }
            continue
        device = matched[0]
        resolved = device.resolve()
        result[role] = {
            "serial": camera.serial,
            "usb_sysfs_device": str(device),
            "resolved_device_path": str(resolved),
            "controller": _controller_for_path(device),
            "busnum": _read_text(device / "busnum"),
            "devnum": _read_text(device / "devnum"),
            "speed_mbps": _read_text(device / "speed"),
            "manufacturer": _read_text(device / "manufacturer"),
            "product": _read_text(device / "product"),
            "id_vendor": _read_text(device / "idVendor"),
            "id_product": _read_text(device / "idProduct"),
        }
    return result


def topology_violations(
    cameras: Mapping[str, CameraConfig],
    case: ExperimentCase,
    usb_devices_root: Path = Path("/sys/bus/usb/devices"),
    *,
    physical_ports: Mapping[str, str] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Return the observed topology and hard failures for a plan condition."""
    topology = discover_camera_topology(cameras, usb_devices_root, physical_ports=physical_ports)
    violations: list[str] = []
    for role, rule in case.camera_controller_rules.items():
        if role not in cameras:
            violations.append(f"plan references unknown camera role {role!r}")
            continue
        entry = topology.get(role, {})
        controller = entry.get("controller")
        if not isinstance(controller, str):
            violations.append(f"{role}: {entry.get('error', 'PCI controller could not be resolved')}")
            continue
        if rule.equals is not None and controller != rule.equals:
            violations.append(f"{role}: expected controller {rule.equals}, found {controller}")
        if rule.one_of and controller not in rule.one_of:
            violations.append(f"{role}: expected one of {', '.join(rule.one_of)}, found {controller}")
        if controller in rule.not_equals:
            violations.append(f"{role}: controller {controller} is explicitly disallowed")
        for other_role in rule.different_from:
            other = topology.get(other_role, {})
            other_controller = other.get("controller")
            if not isinstance(other_controller, str):
                violations.append(f"{role}: cannot compare against {other_role} ({other.get('error', 'not resolved')})")
            elif controller == other_controller:
                violations.append(f"{role}: must use a controller different from {other_role}, both are {controller}")
    return topology, violations


def _command_snapshot(command: Sequence[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=10)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        return {"command": list(command), "error": str(exc), "stdout": "", "stderr": ""}
    return {
        "command": list(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _xhci_interrupts() -> str:
    try:
        lines = Path("/proc/interrupts").read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"could not read /proc/interrupts: {exc}\n"
    selected = [line for line in lines if "xhci" in line.lower()]
    return "\n".join(selected) + ("\n" if selected else "")


def _i2rt_processes() -> list[dict[str, Any]]:
    """Record other local teleoperation processes without modifying them."""
    processes: list[dict[str, Any]] = []
    proc_root = Path("/proc")
    try:
        candidates = list(proc_root.iterdir())
    except OSError:
        return processes
    for path in candidates:
        if not path.name.isdigit():
            continue
        try:
            command = (path / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").strip()
        except OSError:
            continue
        if "minimum_gello.py" not in command and "bimanual_lead_follower" not in command:
            continue
        pid = int(path.name)
        try:
            affinity = sorted(os.sched_getaffinity(pid))
        except (AttributeError, OSError):
            affinity = []
        processes.append({"pid": pid, "command": command, "affinity_cpus": affinity})
    return sorted(processes, key=lambda entry: int(entry["pid"]))


def _environment_snapshot(cameras: Mapping[str, CameraConfig]) -> dict[str, Any]:
    lsusb = _command_snapshot(("lsusb", "-t"))
    return {
        "captured_at_unix_ns": time.time_ns(),
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
            "pid": os.getpid(),
            "recorder_affinity_cpus": sorted(os.sched_getaffinity(0)),
        },
        "camera_topology": discover_camera_topology(cameras),
        "lsusb": lsusb,
        "xhci_interrupts": _xhci_interrupts(),
        "i2rt_processes": _i2rt_processes(),
    }


def _write_json(path: Path, value: Mapping[str, Any] | list[Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _write_environment_snapshot(case_dir: Path, phase: str, cameras: Mapping[str, CameraConfig]) -> dict[str, Any]:
    snapshot = _environment_snapshot(cameras)
    _write_json(case_dir / f"environment-{phase}.json", snapshot)
    _write_json(case_dir / f"topology-{phase}.json", snapshot["camera_topology"])
    lsusb = snapshot["lsusb"]
    lsusb_text = str(lsusb.get("stdout", ""))
    if lsusb.get("stderr"):
        lsusb_text += "\n[stderr]\n" + str(lsusb["stderr"])
    if lsusb.get("error"):
        lsusb_text += "\n[error]\n" + str(lsusb["error"])
    (case_dir / f"lsusb-{phase}.txt").write_text(lsusb_text, encoding="utf-8")
    (case_dir / f"interrupts-{phase}.txt").write_text(str(snapshot["xhci_interrupts"]), encoding="utf-8")
    return snapshot


def _redirect_dataset(config: DatasetConfig, dataset_dir: Path, total_sampled_frames: int) -> DatasetConfig:
    """Create an experiment-local recorder configuration without mutating YAML."""
    raw = dict(config.raw)
    collection_raw = dict(raw.get("collection_config", {}))
    collection_raw["dataset_dir"] = str(dataset_dir)
    collection_raw["total_sampled_frames"] = total_sampled_frames
    raw["collection_config"] = collection_raw
    return replace(
        config,
        raw=raw,
        collection=replace(config.collection, dataset_dir=dataset_dir, total_sampled_frames=total_sampled_frames),
    )


def _source_discontinuity_events(inspection: Any) -> dict[str, list[dict[str, Any]]]:
    """Extract frame-counter jumps while preserving each recorded timing clock."""
    events: dict[str, list[dict[str, Any]]] = {role: [] for role in CAMERA_TOPICS}
    for role in CAMERA_TOPICS:
        diagnostics = inspection.camera_timing_diagnostics.get(role)
        if diagnostics is None:
            continue
        captures = diagnostics.capture_timestamps_ns
        hosts = diagnostics.host_receive_timestamps_ns
        frame_numbers = diagnostics.frame_numbers
        if not captures or len(captures) != len(hosts) or len(captures) != len(frame_numbers):
            continue
        synthetic_indices = set(getattr(diagnostics, "synthetic_indices", ()))
        real_indices = [
            index
            for index, frame_number in enumerate(frame_numbers)
            if index not in synthetic_indices and frame_number is not None
        ]
        if not real_indices:
            continue
        first_capture = captures[real_indices[0]]
        for previous_index, index in zip(real_indices, real_indices[1:]):
            previous = frame_numbers[previous_index]
            current = frame_numbers[index]
            if previous is None or current is None:
                continue
            delta = current - previous
            if delta == 1:
                continue
            event = {
                "time_since_first_capture_s": round((captures[index] - first_capture) / 1_000_000_000, 6),
                "capture_gap_ms": round((captures[index] - captures[previous_index]) / 1_000_000, 3),
                "host_receive_gap_ms": round((hosts[index] - hosts[previous_index]) / 1_000_000, 3),
                "previous_frame_number": previous,
                "frame_number": current,
                "frame_number_delta": delta,
                "skipped_frames": delta - 1 if delta > 1 else 0,
                "kind": "skipped" if delta > 1 else "regression",
                "offline_repaired": any(
                    previous_index < synthetic_index < index for synthetic_index in synthetic_indices
                ),
            }
            events[role].append(event)
    return events


def _trial_summary(
    mcap_path: Path,
    config: DatasetConfig,
    trial_number: int,
    requested_duration_s: float,
    *,
    live_assets_dir: Path | None = None,
) -> dict[str, Any]:
    policy = build_validation_policy(config.cameras, config.collection.validation_profile)
    validation = validate_mcap(mcap_path, policy)
    if live_assets_dir is None:
        assets = validate_video_assets(
            mcap_path,
            policy,
            expected_video_codecs=final_codec_family(config),
        )
        asset_stage = "final"
    else:
        live_codec = getattr(config.collection, "live_video_codec", config.collection.video_codec)
        assets = validate_video_assets(
            mcap_path,
            policy,
            video_dir=live_assets_dir,
            expected_video_codecs=live_codec_family(live_codec),
        )
        asset_stage = "live"
    inspection = inspect_mcap(mcap_path)
    delivery = validation["informational_checks"]["camera_host_delivery_within_gap_budget"]["per_topic"]
    continuity = validation["informational_checks"]["camera_source_frame_continuity"]["per_topic"]
    events = _source_discontinuity_events(inspection)
    cameras: dict[str, Any] = {}
    for role, topic in CAMERA_TOPICS.items():
        stats = inspection.stats.get(topic)
        cameras[role] = {
            "average_fps": round(stats.average_hz, 3) if stats is not None else 0.0,
            "mcap_max_gap_ms": round(stats.max_gap_ms, 3) if stats is not None else 0.0,
            "message_count": stats.message_count if stats is not None else 0,
            "delivery": delivery.get(topic, {}),
            "source_continuity": continuity.get(topic, {}),
            "frame_counter_events": events[role],
        }
    return {
        "trial_number": trial_number,
        "mcap": str(mcap_path),
        "requested_duration_s": requested_duration_s,
        "formal_passed": bool(validation["output_mcap_passed"]),
        "formal_failed_checks": list(validation["failed_checks"]),
        "informational_failures": list(validation["informational_failures"]),
        "validation": validation,
        "video_assets": assets,
        "video_asset_stage": asset_stage,
        "cameras": cameras,
    }


def _aggregate_case(
    *,
    case: ExperimentCase,
    session: str,
    config: DatasetConfig,
    requested_duration_s: float,
    requested_repeats: int,
    topology: Mapping[str, Any],
    trials: Sequence[Mapping[str, Any]],
    run_error: str | None,
) -> dict[str, Any]:
    completed = [trial for trial in trials if "validation" in trial]
    aborted = [trial for trial in trials if "validation" not in trial]
    failed_checks: Counter[str] = Counter()
    source: dict[str, dict[str, Any]] = {
        role: {
            "total_skipped_frames": 0,
            "regressions": 0,
            "frame_counter_events": 0,
            "max_mcap_gap_ms": 0.0,
            "max_capture_gap_ms": 0.0,
            "max_host_receive_gap_ms": 0.0,
            "minimum_average_fps": None,
        }
        for role in CAMERA_TOPICS
    }
    for trial in completed:
        failed_checks.update(str(name) for name in trial.get("formal_failed_checks", []))
        for role, entry in trial.get("cameras", {}).items():
            if role not in source:
                continue
            aggregate = source[role]
            aggregate["max_mcap_gap_ms"] = max(aggregate["max_mcap_gap_ms"], float(entry.get("mcap_max_gap_ms", 0.0)))
            fps = float(entry.get("average_fps", 0.0))
            minimum = aggregate["minimum_average_fps"]
            aggregate["minimum_average_fps"] = fps if minimum is None else min(float(minimum), fps)
            delivery = entry.get("delivery", {})
            aggregate["max_capture_gap_ms"] = max(
                aggregate["max_capture_gap_ms"], float(delivery.get("capture_max_gap_ms", 0.0))
            )
            aggregate["max_host_receive_gap_ms"] = max(
                aggregate["max_host_receive_gap_ms"], float(delivery.get("host_receive_max_gap_ms", 0.0))
            )
            continuity = entry.get("source_continuity", {})
            aggregate["total_skipped_frames"] += int(continuity.get("skipped_frames", 0))
            aggregate["regressions"] += int(continuity.get("regressions", 0))
            aggregate["frame_counter_events"] += len(entry.get("frame_counter_events", []))
    for entry in source.values():
        for key in ("max_mcap_gap_ms", "max_capture_gap_ms", "max_host_receive_gap_ms"):
            entry[key] = round(float(entry[key]), 3)
        if entry["minimum_average_fps"] is not None:
            entry["minimum_average_fps"] = round(float(entry["minimum_average_fps"]), 3)
    formal_passes = sum(bool(trial.get("formal_passed")) for trial in completed)
    return {
        "session": session,
        "case": case.name,
        "description": case.description,
        "requested_duration_s": requested_duration_s,
        "requested_repeats": requested_repeats,
        "completed_trials": len(completed),
        "aborted_trials": len(aborted),
        "formal_passes": formal_passes,
        "formal_failures": len(completed) - formal_passes,
        "formal_failed_check_counts": dict(sorted(failed_checks.items())),
        "capture_contract": {
            "mode": config.capture_mode,
            "validation_profile": config.collection.validation_profile,
            "state_save_freq_hz": config.collection.state_save_freq,
            "cameras": {
                role: {
                    "serial": camera.serial,
                    "width": camera.width,
                    "height": camera.height,
                    "fps": camera.fps,
                    "stream_format": camera.stream_format,
                }
                for role, camera in config.cameras.items()
            },
        },
        "requested_cpu_set": list(case.cpu_set),
        "topology_before": topology,
        "source_continuity": source,
        "trials": list(trials),
        "run_error": run_error,
    }


def _case_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        f"# RealSense A/B/C Experiment: {summary['case']}",
        "",
        f"Session: `{summary['session']}`",
        f"Requested recorder CPUs: `{','.join(str(cpu) for cpu in summary['requested_cpu_set'])}`",
        f"Capture contract: `{summary['capture_contract']['mode']}` / "
        f"`{summary['capture_contract']['validation_profile']}` / "
        f"{summary['capture_contract']['state_save_freq_hz']:g} Hz joint state",
        "",
        "## Formal Results",
        "",
        f"Completed trials: {summary['completed_trials']}/{summary['requested_repeats']}; "
        f"formal pass: {summary['formal_passes']}; formal fail: {summary['formal_failures']}; "
        f"aborted recorder attempts: {summary['aborted_trials']}.",
        "",
        "| Trial | Formal result | Failed mandatory checks | Informational failures |",
        "| --- | --- | --- | --- |",
    ]
    for trial in summary["trials"]:
        if "validation" not in trial:
            lines.append(
                f"| {trial.get('trial_number', '?')} | not saved | {trial.get('error', '-') or '-'} | - |"
            )
            continue
        result = "PASS" if trial["formal_passed"] else "FAIL"
        failed = ", ".join(trial["formal_failed_checks"]) or "-"
        informational = ", ".join(trial["informational_failures"]) or "-"
        lines.append(f"| {trial['trial_number']} | {result} | {failed} | {informational} |")
    lines.extend(
        [
            "",
            "## Source Continuity",
            "",
            "This section is diagnostic only. It does not alter the agreed formal acceptance thresholds.",
            "",
            "| Camera | Min average FPS | Max MCAP gap | Max capture gap | Max host-receive gap | Source skipped frames | Counter events |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for role, entry in summary["source_continuity"].items():
        fps = entry["minimum_average_fps"]
        fps_text = "-" if fps is None else f"{fps:.3f}"
        lines.append(
            f"| {role} | {fps_text} | {entry['max_mcap_gap_ms']:.3f} ms | "
            f"{entry['max_capture_gap_ms']:.3f} ms | {entry['max_host_receive_gap_ms']:.3f} ms | "
            f"{entry['total_skipped_frames']} | {entry['frame_counter_events']} |"
        )
    if summary.get("run_error"):
        lines.extend(["", "## Run Error", "", str(summary["run_error"])])
    lines.extend(
        [
            "",
            "Raw Global Time capture timestamps, host receive timestamps, and RealSense frame numbers are retained "
            "in every MCAP under `/diagnostics/frame_timing`; detailed events are in `summary.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_case_summary(case_dir: Path, summary: Mapping[str, Any]) -> Path:
    summary_path = case_dir / "summary.json"
    _write_json(summary_path, summary)
    (case_dir / "summary.md").write_text(_case_markdown(summary), encoding="utf-8")
    return summary_path


def write_session_comparison(session_dir: str | Path) -> Path:
    """Regenerate the cross-condition A/B/C summary from completed case directories."""
    root = Path(session_dir).expanduser().resolve()
    summaries: list[dict[str, Any]] = []
    if root.is_dir():
        for child in sorted(root.iterdir()):
            path = child / "summary.json"
            if not path.is_file():
                continue
            try:
                with path.open("r", encoding="utf-8") as stream:
                    loaded = json.load(stream)
                if isinstance(loaded, dict):
                    summaries.append(loaded)
            except (OSError, json.JSONDecodeError):
                continue
    comparison = {"session_dir": str(root), "cases": summaries}
    _write_json(root / "comparison.json", comparison)
    lines = [
        "# RealSense A/B/C Comparison",
        "",
        "| Case | Completed trials | Formal pass | Formal fail | High skipped | Left skipped | Right skipped |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        source = summary.get("source_continuity", {})
        skips = [int(source.get(role, {}).get("total_skipped_frames", 0)) for role in CAMERA_TOPICS]
        lines.append(
            f"| {summary.get('case', '?')} | {summary.get('completed_trials', 0)} | "
            f"{summary.get('formal_passes', 0)} | {summary.get('formal_failures', 0)} | "
            f"{skips[0]} | {skips[1]} | {skips[2]} |"
        )
    lines.extend(
        [
            "",
            "Formal results use the unchanged capture profile. Source frame-counter discontinuities are shown "
            "separately as diagnostic evidence, not as a modified acceptance rule.",
            "",
        ]
    )
    path = root / "comparison.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _validate_affinity(cpu_set: Sequence[int]) -> tuple[int, ...]:
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise ExperimentError("CPU affinity experiments require Linux os.sched_setaffinity support")
    available = set(os.sched_getaffinity(0))
    requested = set(cpu_set)
    unavailable = sorted(requested - available)
    if unavailable:
        raise ExperimentError(
            "requested recorder CPUs are outside this shell's allowed affinity set: " + ", ".join(map(str, unavailable))
        )
    return tuple(sorted(available))


def _wait_for_duration(
    duration_s: float,
    printer: Any,
    *,
    abort: Callable[[], bool] | None = None,
) -> bool:
    """Wait for a bounded interval, returning false when an abort is raised.

    The regular interactive collector owns its asynchronous cleanup.  The
    experiment runner must still wake promptly when that cleanup is triggered,
    otherwise a failed long trial would appear hung until its deadline.
    """
    deadline = time.monotonic() + duration_s
    next_progress = time.monotonic() + min(30.0, duration_s)
    while True:
        if abort is not None and abort():
            return False
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            return True
        if time.monotonic() >= next_progress:
            printer(f"Experiment recording continues; about {math.ceil(remaining_s)} s remaining.")
            next_progress += 30.0
        # Keep failure detection responsive without adding measurable load to
        # the camera and telemetry workers.
        time.sleep(min(0.25, remaining_s))


def _collector_background_error(collector: EpisodeCollector) -> BaseException | None:
    """Read the public failure signal while tolerating lightweight test doubles."""
    return getattr(collector, "background_error", None)


def _collector_last_episode_diagnostics(collector: EpisodeCollector) -> dict[str, Any] | None:
    """Read finalized writer evidence without requiring it from test doubles."""
    diagnostics = getattr(collector, "last_episode_diagnostics", None)
    if not isinstance(diagnostics, Mapping):
        return None
    return dict(diagnostics)


def _wait_for_collector_finalization(collector: EpisodeCollector) -> None:
    """Ensure a background discard has captured its evidence before reporting it."""
    waiter = getattr(collector, "wait_for_episode_finalization", None)
    if callable(waiter):
        waiter()


def _collector_recording_active(collector: EpisodeCollector) -> bool:
    state = getattr(collector, "state", None)
    if state is None:
        return True
    return getattr(state, "value", state) == "recording"


def _wait_for_collector_trial(
    duration_s: float, collector: EpisodeCollector, printer: Any
) -> BaseException | None:
    """Wait for one trial and return its asynchronous failure, if any."""
    completed = _wait_for_duration(
        duration_s,
        printer,
        abort=lambda: _collector_background_error(collector) is not None
        or not _collector_recording_active(collector),
    )
    if completed:
        return None
    error = _collector_background_error(collector)
    if error is not None:
        print_status(f"Experiment recording aborted: {error}", "red", printer=printer)
        return error
    error = ExperimentError("experiment recording ended before its requested duration")
    print_status(str(error), "red", printer=printer)
    return error


def run_experiment(
    config: DatasetConfig,
    plan: ExperimentPlan,
    case_name: str,
    session: str,
    *,
    duration_s: float | None = None,
    repeats: int | None = None,
    printer: Any = print,
) -> ExperimentRun:
    """Run duration-bounded full teleoperation recordings for one A/B/C condition."""
    if case_name not in plan.cases:
        choices = ", ".join(sorted(plan.cases))
        raise ExperimentError(f"unknown experiment case {case_name!r}; available: {choices}")
    if _SAFE_NAME_RE.fullmatch(session) is None:
        raise ExperimentError("session may contain only letters, digits, ., _, and -")
    case = plan.cases[case_name]
    selected_duration_s = plan.duration_s if duration_s is None else duration_s
    selected_repeats = plan.repeats if repeats is None else repeats
    if selected_duration_s <= 0 or selected_repeats <= 0:
        raise ExperimentError("duration and repeat count must be positive")
    previous_affinity = _validate_affinity(case.cpu_set)
    session_dir = plan.output_dir / session
    case_dir = session_dir / case.name
    if case_dir.exists():
        raise ExperimentError(f"experiment output already exists: {case_dir}; use a new --session instead of mixing trials")
    case_dir.mkdir(parents=True)
    (case_dir / "trials").mkdir()
    plan_copy = case_dir / "experiment_plan.yaml"
    shutil.copyfile(plan.source_path, plan_copy)

    total_sampled_frames = max(
        config.collection.total_sampled_frames,
        math.ceil((selected_duration_s + 60.0) * config.collection.state_save_freq),
    )
    experiment_config = _redirect_dataset(config, case_dir / "episodes", total_sampled_frames)
    with (case_dir / "resolved_capture_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(dict(experiment_config.raw), stream, allow_unicode=True, sort_keys=False)

    manifest: dict[str, Any] = {
        "started_at_unix_ns": time.time_ns(),
        "session": session,
        "case": case.name,
        "description": case.description,
        "requested_cpu_set": list(case.cpu_set),
        "previous_affinity_cpus": list(previous_affinity),
        "requested_duration_s": selected_duration_s,
        "requested_repeats": selected_repeats,
        "inter_trial_delay_s": plan.inter_trial_delay_s,
        "plan_path": str(plan.source_path),
        "capture_config_path": str(config.source_path),
        "experiment_dataset_dir": str(experiment_config.collection.dataset_dir),
        "experiment_total_sampled_frames": total_sampled_frames,
    }
    _write_json(case_dir / "manifest.json", manifest)
    trials: list[dict[str, Any]] = []
    topology: dict[str, Any] = {}
    run_error: str | None = None
    collector: EpisodeCollector | None = None
    runtime_monitor: RuntimeDiagnosticsMonitor | None = None
    active_episode_id: int | None = None
    try:
        os.sched_setaffinity(0, set(case.cpu_set))
        effective_affinity = sorted(os.sched_getaffinity(0))
        manifest["effective_case_affinity_cpus"] = effective_affinity
        _write_json(case_dir / "manifest.json", manifest)
        before = _write_environment_snapshot(case_dir, "before", experiment_config.cameras)
        topology = dict(before["camera_topology"])
        _, violations = topology_violations(experiment_config.cameras, case)
        if violations:
            manifest["topology_violations"] = violations
            _write_json(case_dir / "manifest.json", manifest)
            raise ExperimentError("topology preflight failed; do not record: " + "; ".join(violations))
        printer(
            f"Experiment {case.name}: CPU affinity {','.join(map(str, effective_affinity))}; "
            f"recording {selected_repeats} x {selected_duration_s:g}s into {case_dir}."
        )
        # This runner writes one case-level diagnostics file below.  Do not
        # also start the regular capture monitor inside the collector, because
        # its extra 200 ms polling would perturb the 60 Hz load experiment.
        collector = EpisodeCollector(experiment_config, enable_runtime_diagnostics=False)
        collector.start()
        collector_snapshot = collector.active_episode_snapshot
        effective_recorder_cpus = collector_snapshot.get("recorder_cpu_set")
        if isinstance(effective_recorder_cpus, list) and all(
            isinstance(cpu, int) for cpu in effective_recorder_cpus
        ):
            manifest["effective_recorder_affinity_cpus"] = effective_recorder_cpus
        configured_encoder_cpus = experiment_config.collection.live_video_cpu_set
        if configured_encoder_cpus is not None:
            manifest["configured_encoder_affinity"] = configured_encoder_cpus
        _write_json(case_dir / "manifest.json", manifest)
        runtime_monitor = RuntimeDiagnosticsMonitor(
            case_dir / "runtime_diagnostics.jsonl", collector, effective_affinity
        )
        if runtime_monitor.start():
            manifest["runtime_diagnostics"] = {
                "path": str(runtime_monitor.path),
                "interval_s": runtime_monitor._INTERVAL_S,
                "schema_version": 1,
            }
        else:
            manifest["runtime_diagnostics"] = {
                "path": str(runtime_monitor.path),
                "error": runtime_monitor.error,
            }
        _write_json(case_dir / "manifest.json", manifest)
        for trial_number in range(1, selected_repeats + 1):
            if trial_number > 1 and plan.inter_trial_delay_s:
                printer(f"Trial {trial_number}/{selected_repeats} starts in {plan.inter_trial_delay_s:g} s.")
                _wait_for_duration(plan.inter_trial_delay_s, printer=lambda _message: None)
            active_episode_id = collector.start_episode()
            episode_id = active_episode_id
            printer(
                f"Trial {trial_number}/{selected_repeats}: recording episode {active_episode_id} for "
                f"{selected_duration_s:g} s. Teleoperate normally."
            )
            started_at_ns = time.time_ns()
            wait_error = _wait_for_collector_trial(selected_duration_s, collector, printer)
            recording_stopped_at_ns = time.time_ns()
            # A worker can have already started its asynchronous discard. Wait
            # for its diagnostic snapshot before calling save_episode, which
            # correctly returns false once no active episode remains.
            if wait_error is not None:
                _wait_for_collector_finalization(collector)
            saved = collector.save_episode()
            finalized_at_ns = time.time_ns()
            recording_elapsed_s = (recording_stopped_at_ns - started_at_ns) / 1_000_000_000
            finalization_elapsed_s = (finalized_at_ns - recording_stopped_at_ns) / 1_000_000_000
            elapsed_s = (finalized_at_ns - started_at_ns) / 1_000_000_000
            if not saved:
                _wait_for_collector_finalization(collector)
            if uses_direct_h264_publication(experiment_config):
                mcap_path = final_mcap_path(experiment_config.collection.dataset_dir, episode_id)
                live_assets_dir = None
            else:
                mcap_path = pending_mcap_path(experiment_config.collection.dataset_dir, episode_id)
                live_assets_dir = live_video_dir(experiment_config.collection.dataset_dir, episode_id)
            active_episode_id = None
            collector_error = wait_error or _collector_background_error(collector)
            if saved and mcap_path.is_file():
                trial = _trial_summary(
                    mcap_path,
                    experiment_config,
                    trial_number,
                    selected_duration_s,
                    live_assets_dir=live_assets_dir,
                )
                trial["episode_id"] = episode_id
                trial["actual_elapsed_s"] = round(elapsed_s, 3)
                trial["recording_elapsed_s"] = round(recording_elapsed_s, 3)
                trial["finalization_elapsed_s"] = round(finalization_elapsed_s, 3)
            else:
                trial = {
                    "trial_number": trial_number,
                    "episode_id": episode_id,
                    "requested_duration_s": selected_duration_s,
                    "actual_elapsed_s": round(elapsed_s, 3),
                    "recording_elapsed_s": round(recording_elapsed_s, 3),
                    "finalization_elapsed_s": round(finalization_elapsed_s, 3),
                    "error": (
                        str(collector_error)
                        if collector_error is not None
                        else "collector did not save an MCAP for this trial"
                    ),
                }
                diagnostics = _collector_last_episode_diagnostics(collector)
                if diagnostics is not None:
                    trial["failure_diagnostics"] = diagnostics
            trials.append(trial)
            _write_json(case_dir / "trials" / f"trial-{trial_number:02d}.json", trial)
            if not saved:
                detail = str(trial.get("error", "collector did not save an MCAP"))
                raise ExperimentError(f"trial {trial_number} was not saved ({detail}); stopping this condition")
    except KeyboardInterrupt:
        run_error = "interrupted by operator"
        if (
            collector is not None
            and active_episode_id is not None
            and _collector_recording_active(collector)
        ):
            collector.discard_episode()
            active_episode_id = None
        raise
    except BaseException as exc:
        run_error = str(exc)
        raise
    finally:
        if collector is not None:
            try:
                if active_episode_id is not None and not any(
                    trial.get("episode_id") == active_episode_id for trial in trials
                ):
                    elapsed_s = (
                        (time.time_ns() - started_at_ns) / 1_000_000_000
                        if "started_at_ns" in locals()
                        else 0.0
                    )
                    active_error = _collector_background_error(collector)
                    trial = {
                        "trial_number": len(trials) + 1,
                        "episode_id": active_episode_id,
                        "requested_duration_s": selected_duration_s,
                        "actual_elapsed_s": round(elapsed_s, 3),
                        "error": str(active_error or run_error or "trial ended before it could be saved"),
                    }
                    diagnostics = _collector_last_episode_diagnostics(collector)
                    if diagnostics is not None:
                        trial["failure_diagnostics"] = diagnostics
                    trials.append(trial)
                    _write_json(case_dir / "trials" / f"trial-{len(trials):02d}.json", trial)
                if active_episode_id is not None and _collector_recording_active(collector):
                    collector.discard_episode()
                collector.close()
            except BaseException as exc:
                if run_error is None:
                    run_error = f"collector shutdown failed: {exc}"
        if runtime_monitor is not None:
            runtime_monitor.stop()
            if runtime_monitor.error is not None:
                diagnostics_entry = manifest.get("runtime_diagnostics")
                if not isinstance(diagnostics_entry, dict):
                    diagnostics_entry = {"path": str(runtime_monitor.path)}
                    manifest["runtime_diagnostics"] = diagnostics_entry
                diagnostics_entry["error"] = runtime_monitor.error
        try:
            _write_environment_snapshot(case_dir, "after", experiment_config.cameras)
        finally:
            os.sched_setaffinity(0, set(previous_affinity))
        manifest["finished_at_unix_ns"] = time.time_ns()
        manifest["run_error"] = run_error
        _write_json(case_dir / "manifest.json", manifest)
        summary = _aggregate_case(
            case=case,
            session=session,
            config=experiment_config,
            requested_duration_s=selected_duration_s,
            requested_repeats=selected_repeats,
            topology=topology,
            trials=trials,
            run_error=run_error,
        )
        summary_path = _write_case_summary(case_dir, summary)
        comparison_path = write_session_comparison(session_dir)
    return ExperimentRun(case_dir=case_dir, summary_path=summary_path, comparison_path=comparison_path)
