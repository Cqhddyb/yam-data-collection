"""Best-effort runtime diagnostics shared by capture and experiment workflows.

The monitor is intentionally outside the time-critical capture callbacks.  It
records host, recorder, and live encoder evidence at a modest cadence, but a
failure to collect that evidence never interrupts a valid recording.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


_NVME_DISK_RE = re.compile(r"nvme\d+n\d+")
_THERMAL_THROTTLE_FIELDS = (
    "core_throttle_count",
    "package_throttle_count",
    "core_throttle_total_time_ms",
    "package_throttle_total_time_ms",
    "core_throttle_max_time_ms",
    "package_throttle_max_time_ms",
)


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def _read_int(path: Path) -> int | None:
    text = _read_text(path)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _read_process_counters(pid: int) -> dict[str, Any] | None:
    """Read inexpensive process counters directly from procfs.

    A process can exit between discovering a writer PID and reading its procfs
    files.  That is diagnostic data being unavailable, not a capture failure.
    """
    proc_dir = Path("/proc") / str(pid)
    stat = _read_text(proc_dir / "stat")
    if stat is None:
        return None
    _prefix, separator, rest = stat.rpartition(")")
    fields = rest.strip().split() if separator else []
    # Fields begin at procfs field 3 (state); utime and stime are 14 and 15.
    if len(fields) < 13:
        return None
    try:
        ticks = int(fields[11]) + int(fields[12])
        clock_ticks = int(os.sysconf("SC_CLK_TCK"))
    except (ValueError, OSError, AttributeError):
        return None
    resident_memory_bytes: int | None = None
    thread_count: int | None = None
    status = _read_text(proc_dir / "status")
    if status is not None:
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                try:
                    resident_memory_bytes = int(parts[1]) * 1024
                except (IndexError, ValueError):
                    pass
            elif line.startswith("Threads:"):
                parts = line.split()
                try:
                    thread_count = int(parts[1])
                except (IndexError, ValueError):
                    pass
    return {
        "pid": pid,
        "state": fields[0],
        "cpu_time_s": round(ticks / clock_ticks, 6),
        "resident_memory_bytes": resident_memory_bytes,
        "thread_count": thread_count,
    }


def _read_system_cpu_counters() -> tuple[int, int] | None:
    """Return cumulative total and busy jiffies from the aggregate CPU line."""
    try:
        first_line = Path("/proc/stat").read_text(encoding="utf-8", errors="replace").splitlines()[0]
    except (IndexError, OSError):
        return None
    fields = first_line.split()
    if not fields or fields[0] != "cpu":
        return None
    try:
        values = [int(value) for value in fields[1:]]
    except ValueError:
        return None
    if len(values) < 5:
        return None
    total = sum(values)
    idle = values[3] + values[4]  # idle and iowait
    return total, total - idle


def _read_available_memory_bytes() -> int | None:
    meminfo = _read_text(Path("/proc/meminfo"))
    if meminfo is None:
        return None
    for line in meminfo.splitlines():
        if not line.startswith("MemAvailable:"):
            continue
        parts = line.split()
        try:
            return int(parts[1]) * 1024
        except (IndexError, ValueError):
            return None
    return None


def _read_nvme_disk_counters() -> dict[str, dict[str, int]]:
    """Read raw whole-device NVMe counters when the kernel exposes them."""
    text = _read_text(Path("/proc/diskstats"))
    if text is None:
        return {}
    counters: dict[str, dict[str, int]] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 14 or _NVME_DISK_RE.fullmatch(fields[2]) is None:
            continue
        try:
            values = [int(value) for value in fields[3:14]]
        except ValueError:
            continue
        counters[fields[2]] = {
            "reads_completed": values[0],
            "reads_merged": values[1],
            "sectors_read": values[2],
            "read_time_ms": values[3],
            "writes_completed": values[4],
            "writes_merged": values[5],
            "sectors_written": values[6],
            "write_time_ms": values[7],
            "io_in_progress": values[8],
            "io_time_ms": values[9],
            "weighted_io_time_ms": values[10],
        }
    return counters


def _disk_counter_deltas(
    current: Mapping[str, Mapping[str, int]], previous: Mapping[str, Mapping[str, int]]
) -> dict[str, dict[str, int]]:
    """Make NVMe counter movement easy to inspect in JSONL."""
    deltas: dict[str, dict[str, int]] = {}
    for device, values in current.items():
        previous_values = previous.get(device)
        if previous_values is None:
            continue
        deltas[device] = {
            name: value - previous_values[name]
            for name, value in values.items()
            if name in previous_values
        }
    return deltas


def _discover_thermal_sensors() -> dict[str, Path]:
    """Resolve readable thermal sysfs files once; samples only read them later."""
    sensors: dict[str, Path] = {}
    try:
        zones = sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp"))
    except OSError:
        zones = []
    for path in zones:
        zone = path.parent
        kind = _read_text(zone / "type") or zone.name
        sensors[f"thermal:{kind}:{zone.name}"] = path
    try:
        hwmon_inputs = sorted(Path("/sys/class/hwmon").glob("hwmon*/temp*_input"))
    except OSError:
        hwmon_inputs = []
    for path in hwmon_inputs:
        hwmon = path.parent
        sensor_id = path.name.removesuffix("_input")
        name = _read_text(hwmon / "name") or hwmon.name
        label = _read_text(hwmon / f"{sensor_id}_label") or sensor_id
        sensors[f"hwmon:{name}:{label}:{hwmon.name}"] = path
    # Do not make a large host sensor inventory expensive at a 200 ms cadence.
    return dict(list(sorted(sensors.items()))[:64])


def _read_temperatures_celsius(sensors: Mapping[str, Path]) -> dict[str, float]:
    temperatures: dict[str, float] = {}
    for name, path in sensors.items():
        text = _read_text(path)
        if text is None:
            continue
        try:
            value = float(text)
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        # Linux thermal/sysfs commonly reports millidegrees C; retain direct
        # degree readings as-is for devices that expose them.
        if abs(value) > 1_000:
            value /= 1_000
        temperatures[name] = round(value, 3)
    return temperatures


def _read_recorder_cpu_frequencies_mhz(cpus: Sequence[int]) -> dict[str, float]:
    frequencies: dict[str, float] = {}
    for cpu in cpus:
        cpu_dir = Path("/sys/devices/system/cpu") / f"cpu{cpu}" / "cpufreq"
        frequency_khz = _read_int(cpu_dir / "scaling_cur_freq")
        if frequency_khz is None:
            frequency_khz = _read_int(cpu_dir / "cpuinfo_cur_freq")
        if frequency_khz is None:
            continue
        # cpufreq sysfs reports kHz. The fallback keeps this useful on an
        # unusual platform that already reports MHz.
        frequency_mhz = frequency_khz / 1_000 if frequency_khz >= 10_000 else float(frequency_khz)
        frequencies[str(cpu)] = round(frequency_mhz, 3)
    return frequencies


def _discover_thermal_throttle_counters(cpus: Sequence[int]) -> dict[str, dict[str, Path]]:
    """Find CPU thermal-throttle counters when this kernel exposes them."""
    root = Path("/sys/devices/system/cpu")
    counters: dict[str, dict[str, Path]] = {}
    for cpu in sorted(set(cpus)):
        directory = root / f"cpu{cpu}" / "thermal_throttle"
        values = {field: directory / field for field in _THERMAL_THROTTLE_FIELDS if (directory / field).is_file()}
        if values:
            counters[str(cpu)] = values
    return counters


def _read_thermal_throttle_counters(paths: Mapping[str, Mapping[str, Path]]) -> dict[str, dict[str, int]]:
    values: dict[str, dict[str, int]] = {}
    for cpu, fields in paths.items():
        readings = {name: value for name, path in fields.items() if (value := _read_int(path)) is not None}
        if readings:
            values[cpu] = readings
    return values


def _collector_active_episode_snapshot(collector: Any) -> dict[str, Any]:
    """Read the collector's optional public live snapshot without coupling it."""
    try:
        snapshot = getattr(collector, "active_episode_snapshot", None)
        if callable(snapshot):
            snapshot = snapshot()
    except BaseException as exc:
        return {"diagnostic_error": f"{type(exc).__name__}: {exc}"}
    if isinstance(snapshot, Mapping):
        return dict(snapshot)
    state = getattr(collector, "state", None)
    return {
        "state": getattr(state, "value", state),
        "episode_id": None,
        "state_samples": 0,
        "camera_frames": {},
        "video_writers": {},
    }


class RuntimeDiagnosticsMonitor:
    """Best-effort 200 ms runtime telemetry kept outside capture callbacks."""

    _INTERVAL_S = 0.2
    _JOIN_TIMEOUT_S = 2.0

    def __init__(self, path: str | Path, collector: Any, recorder_cpus: Sequence[int]) -> None:
        self.path = Path(path)
        self._collector = collector
        self._recorder_cpus = tuple(sorted(set(recorder_cpus)))
        self._thermal_sensors = _discover_thermal_sensors()
        self._thermal_throttle_paths = _discover_thermal_throttle_counters(self._recorder_cpus)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream: Any | None = None
        self._error: str | None = None
        self._sample_index = 0
        self._previous_system_cpu: tuple[int, int] | None = None
        self._previous_process_cpu: dict[int, tuple[int, float]] = {}
        self._previous_disk_counters: dict[str, dict[str, int]] = {}
        self._previous_writer_counters: dict[str, tuple[int, int, int, int]] = {}

    @property
    def error(self) -> str | None:
        return self._error

    def start(self) -> bool:
        """Start diagnostics without allowing a diagnostic failure to stop capture."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = self.path.open("x", encoding="utf-8")
        except OSError as exc:
            self._error = f"could not create runtime diagnostics: {exc}"
            return False
        self._write_sample()
        if self._stream is None:
            return False
        self._thread = threading.Thread(target=self._run, name="runtime-diagnostics", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(self._JOIN_TIMEOUT_S)
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.close()
            except OSError as exc:
                if self._error is None:
                    self._error = f"could not close runtime diagnostics: {exc}"

    def _run(self) -> None:
        deadline = time.monotonic() + self._INTERVAL_S
        while True:
            wait_s = max(0.0, deadline - time.monotonic())
            if self._stop_event.wait(wait_s):
                return
            self._write_sample()
            deadline += self._INTERVAL_S
            if deadline <= time.monotonic():
                deadline = time.monotonic() + self._INTERVAL_S

    def _process_load(self, pid: int, sampled_at_monotonic_ns: int) -> dict[str, Any]:
        counters = _read_process_counters(pid)
        if counters is None:
            return {"pid": pid, "available": False}
        cpu_time_s = float(counters["cpu_time_s"])
        previous = self._previous_process_cpu.get(pid)
        if previous is None:
            cpu_percent: float | None = None
        else:
            previous_ns, previous_cpu_s = previous
            elapsed_s = (sampled_at_monotonic_ns - previous_ns) / 1_000_000_000
            cpu_percent = (
                round(max(0.0, cpu_time_s - previous_cpu_s) / elapsed_s * 100, 3) if elapsed_s > 0 else None
            )
        self._previous_process_cpu[pid] = (sampled_at_monotonic_ns, cpu_time_s)
        counters["available"] = True
        counters["cpu_percent_one_core"] = cpu_percent
        return counters

    def _system_load(self) -> dict[str, Any]:
        cpu_counters = _read_system_cpu_counters()
        busy_percent: float | None = None
        if cpu_counters is not None and self._previous_system_cpu is not None:
            total_delta = cpu_counters[0] - self._previous_system_cpu[0]
            busy_delta = cpu_counters[1] - self._previous_system_cpu[1]
            if total_delta > 0:
                busy_percent = round(max(0.0, busy_delta) / total_delta * 100, 3)
        if cpu_counters is not None:
            self._previous_system_cpu = cpu_counters
        try:
            load_average = [round(value, 3) for value in os.getloadavg()]
        except (AttributeError, OSError):
            load_average = []
        disk_counters = _read_nvme_disk_counters()
        disk_deltas = _disk_counter_deltas(disk_counters, self._previous_disk_counters)
        self._previous_disk_counters = disk_counters
        return {
            "load_average": load_average,
            "cpu_busy_percent": busy_percent,
            "available_memory_bytes": _read_available_memory_bytes(),
            "thermal_celsius": _read_temperatures_celsius(self._thermal_sensors),
            "thermal_throttle_counters": _read_thermal_throttle_counters(self._thermal_throttle_paths),
            "recorder_cpu_frequency_mhz": _read_recorder_cpu_frequencies_mhz(self._recorder_cpus),
            "nvme_disk_counters": disk_counters,
            "nvme_disk_counter_delta": disk_deltas,
        }

    def _writer_throughput(
        self,
        writers: Mapping[str, Any],
        sampled_at_monotonic_ns: int,
    ) -> dict[str, dict[str, Any]]:
        """Derive queue growth and pipe-consumer rates between samples.

        ``frames_submitted_to_ffmpeg`` reaches the encoder's stdin; it is not
        claimed to be a hardware-verified compressed-output count. That exact
        count is checked only after ffmpeg closes and ffprobe decodes the MP4.
        """
        metrics: dict[str, dict[str, Any]] = {}
        for raw_role, raw_writer in writers.items():
            if not isinstance(raw_writer, Mapping):
                continue
            role = str(raw_role)
            accepted = raw_writer.get("accepted_frames")
            submitted = raw_writer.get("frames_submitted_to_ffmpeg")
            queue_depth = raw_writer.get("queue_depth")
            queue_capacity = raw_writer.get("queue_capacity")
            if not all(isinstance(value, int) for value in (accepted, submitted, queue_depth)):
                continue
            previous = self._previous_writer_counters.get(role)
            rates: dict[str, Any] = {
                "accepted_frames": accepted,
                "frames_submitted_to_ffmpeg": submitted,
                "queue_depth": queue_depth,
                "queue_capacity": queue_capacity if isinstance(queue_capacity, int) else None,
                "queue_fill_ratio": round(queue_depth / queue_capacity, 6)
                if isinstance(queue_capacity, int) and queue_capacity > 0
                else None,
                "input_fps": None,
                "ffmpeg_pipe_consume_fps": None,
                "queue_depth_delta": None,
                "queue_growth_frames_per_s": None,
            }
            if previous is not None:
                previous_ns, previous_accepted, previous_submitted, previous_depth = previous
                elapsed_s = (sampled_at_monotonic_ns - previous_ns) / 1_000_000_000
                if elapsed_s > 0:
                    queue_delta = queue_depth - previous_depth
                    rates.update(
                        input_fps=round((accepted - previous_accepted) / elapsed_s, 3),
                        ffmpeg_pipe_consume_fps=round((submitted - previous_submitted) / elapsed_s, 3),
                        queue_depth_delta=queue_delta,
                        queue_growth_frames_per_s=round(queue_delta / elapsed_s, 3),
                    )
            self._previous_writer_counters[role] = (sampled_at_monotonic_ns, accepted, submitted, queue_depth)
            metrics[role] = rates
        return metrics

    def _build_sample(self) -> dict[str, Any]:
        sampled_at_unix_ns = time.time_ns()
        sampled_at_monotonic_ns = time.monotonic_ns()
        collector_snapshot = _collector_active_episode_snapshot(self._collector)
        writers = collector_snapshot.get("video_writers")
        encoder_load: dict[str, dict[str, Any]] = {}
        writer_throughput: dict[str, dict[str, Any]] = {}
        if isinstance(writers, Mapping):
            writer_throughput = self._writer_throughput(writers, sampled_at_monotonic_ns)
            for role, writer in writers.items():
                if not isinstance(writer, Mapping):
                    continue
                pid = writer.get("ffmpeg_pid")
                if isinstance(pid, int):
                    encoder_load[str(role)] = self._process_load(pid, sampled_at_monotonic_ns)
        self._sample_index += 1
        return {
            "schema_version": 1,
            "sample_index": self._sample_index,
            "timestamp_unix_ns": sampled_at_unix_ns,
            "timestamp_monotonic_ns": sampled_at_monotonic_ns,
            "collector": collector_snapshot,
            "process_load": {
                "recorder": self._process_load(os.getpid(), sampled_at_monotonic_ns),
                "encoders": encoder_load,
            },
            "encoder_throughput": writer_throughput,
            "system_load": self._system_load(),
        }

    def _write_sample(self) -> None:
        stream = self._stream
        if stream is None:
            return
        try:
            stream.write(json.dumps(self._build_sample(), ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
        except BaseException as exc:
            # Diagnostics must never take down a valid capture. Disable the
            # monitor after its first I/O or serialization failure.
            self._error = f"runtime diagnostics disabled: {type(exc).__name__}: {exc}"
            self._stream = None
            try:
                stream.close()
            except OSError:
                pass
