from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from yam_data_collection.config import CameraConfig, CameraIntrinsics
from yam_data_collection.experiment import (
    ControllerRule,
    ExperimentCase,
    _collector_background_error,
    _collector_last_episode_diagnostics,
    _collector_recording_active,
    _redirect_dataset,
    _source_discontinuity_events,
    _wait_for_duration,
    _wait_for_collector_trial,
    discover_camera_topology,
    load_experiment_plan,
    parse_cpu_set,
    topology_violations,
    write_session_comparison,
)
from yam_data_collection.quality import CameraTimingDiagnostics
from yam_data_collection.runtime_diagnostics import RuntimeDiagnosticsMonitor


def _cameras() -> dict[str, CameraConfig]:
    intrinsics = CameraIntrinsics.uncalibrated()
    return {
        "cam_high": CameraConfig("cam_high", "high", 1280, 720, 30, "high", intrinsics, "yuyv"),
        "cam_left": CameraConfig("cam_left", "left", 1280, 720, 30, "left", intrinsics, "yuyv"),
        "cam_right": CameraConfig("cam_right", "right", 1280, 720, 30, "right", intrinsics, "yuyv"),
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("16-23", (16, 17, 18, 19, 20, 21, 22, 23)),
        ("0,2-4,4", (0, 2, 3, 4)),
        (" 12 - 19 ", (12, 13, 14, 15, 16, 17, 18, 19)),
    ],
)
def test_parse_cpu_set(value: str, expected: tuple[int, ...]) -> None:
    assert parse_cpu_set(value) == expected


@pytest.mark.parametrize("value", ("", "3-1", "1,,2", "x", "1-2-3"))
def test_parse_cpu_set_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        parse_cpu_set(value)


def test_wait_for_duration_returns_early_when_abort_is_signaled() -> None:
    started = time.monotonic()
    calls = 0

    def abort() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 2

    assert not _wait_for_duration(10.0, printer=lambda _message: None, abort=abort)
    assert time.monotonic() - started < 1.0
    assert calls >= 2


def test_wait_for_duration_returns_true_at_deadline() -> None:
    assert _wait_for_duration(0.0, printer=lambda _message: None)


def test_collector_background_error_accepts_a_lightweight_test_double() -> None:
    assert _collector_background_error(SimpleNamespace()) is None


def test_collector_last_episode_diagnostics_accepts_optional_evidence() -> None:
    evidence = {"video_writers_before_close": {"cam_left": {"dropped_frames": 1}}}
    assert _collector_last_episode_diagnostics(SimpleNamespace(last_episode_diagnostics=evidence)) == evidence
    assert _collector_last_episode_diagnostics(SimpleNamespace()) is None


def test_collector_recording_active_accepts_enum_or_string_state() -> None:
    assert _collector_recording_active(SimpleNamespace())
    assert _collector_recording_active(SimpleNamespace(state="recording"))
    assert not _collector_recording_active(SimpleNamespace(state="idle"))


def test_runtime_diagnostics_monitor_writes_a_nonblocking_jsonl_sample(tmp_path: Path) -> None:
    collector = SimpleNamespace(
        active_episode_snapshot={
            "state": "recording",
            "episode_id": 4,
            "state_samples": 180,
            "camera_frames": {"cam_high": 30},
            "video_writers": {
                "cam_high": {
                    "ffmpeg_pid": 999_999_999,
                    "queue_depth": 2,
                    "queue_capacity": 120,
                    "accepted_frames": 30,
                    "frames_submitted_to_ffmpeg": 28,
                    "max_pipe_write_ms": 4.5,
                }
            },
        }
    )
    path = tmp_path / "runtime_diagnostics.jsonl"
    monitor = RuntimeDiagnosticsMonitor(path, collector, recorder_cpus=(0,))

    assert monitor.start()
    monitor.stop()

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert records
    sample = records[0]
    assert sample["collector"]["episode_id"] == 4
    assert sample["collector"]["video_writers"]["cam_high"]["queue_depth"] == 2
    assert sample["process_load"]["encoders"]["cam_high"] == {"pid": 999_999_999, "available": False}
    assert sample["encoder_throughput"]["cam_high"]["queue_fill_ratio"] == pytest.approx(2 / 120, abs=1e-6)
    assert "recorder" in sample["process_load"]
    assert "available_memory_bytes" in sample["system_load"]
    assert "nvme_disk_counters" in sample["system_load"]
    assert "thermal_throttle_counters" in sample["system_load"]


def test_runtime_diagnostics_monitor_creates_its_output_directory(tmp_path: Path) -> None:
    path = tmp_path / "runtime" / "capture.jsonl"
    monitor = RuntimeDiagnosticsMonitor(path, SimpleNamespace(), recorder_cpus=(0,))

    assert monitor.start()
    monitor.stop()

    assert path.is_file()


def test_wait_for_collector_trial_returns_background_error_promptly() -> None:
    error = RuntimeError("video queue overflow on cam_right")
    collector = SimpleNamespace(background_error=error, state="recording")
    messages: list[str] = []

    assert _wait_for_collector_trial(10.0, collector, messages.append) is error
    assert messages == ["Experiment recording aborted: video queue overflow on cam_right"]


def test_wait_for_collector_trial_detects_an_episode_that_stopped_without_an_error() -> None:
    collector = SimpleNamespace(background_error=None, state="idle")
    messages: list[str] = []

    error = _wait_for_collector_trial(10.0, collector, messages.append)

    assert isinstance(error, Exception)
    assert "ended before" in str(error)


def test_plan_parses_abc_cpu_and_topology_contracts(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(
        """
experiment:
  output_dir: outputs
  duration_s: 300
  repeats: 3
  cases:
    A:
      cpu_set: "16-23"
      camera_controller_rules:
        cam_high: "0000:80:14.0"
    B:
      cpu_set: "12-19"
      camera_controller_rules:
        cam_high: "0000:80:14.0"
    C:
      cpu_set: "16-23"
      camera_controller_rules:
        cam_right:
          one_of: ["0000:00:0d.0", "0000:00:0d.2"]
          different_from: [cam_high, cam_left]
""",
        encoding="utf-8",
    )

    plan = load_experiment_plan(plan_path)

    assert plan.output_dir == tmp_path / "outputs"
    assert plan.cases["A"].cpu_set == tuple(range(16, 24))
    assert plan.cases["B"].cpu_set == tuple(range(12, 20))
    assert plan.cases["C"].camera_controller_rules["cam_right"].one_of == (
        "0000:00:0d.0",
        "0000:00:0d.2",
    )
    assert plan.cases["C"].camera_controller_rules["cam_right"].different_from == ("cam_high", "cam_left")


def test_discover_camera_topology_uses_realsense_physical_ports(tmp_path: Path) -> None:
    usb_root = tmp_path / "usb"
    usb_root.mkdir()
    for port, product in (("4-3", "D405"), ("4-4", "D405"), ("4-5", "D435")):
        device = usb_root / port
        device.mkdir()
        (device / "busnum").write_text("4\n", encoding="utf-8")
        (device / "devnum").write_text("23\n", encoding="utf-8")
        (device / "speed").write_text("5000\n", encoding="utf-8")
        (device / "product").write_text(product, encoding="utf-8")
    # USB descriptor serials intentionally differ from capture serials on the
    # real host, so the injected physical-port mapping is the expected route.
    physical_ports = {
        "high": "/sys/devices/pci0000:80/0000:80:14.0/usb4/4-5/4-5:1.0/video4linux/video16",
        "left": "/sys/devices/pci0000:80/0000:80:14.0/usb4/4-3/4-3:1.0/video4linux/video0",
        "right": "/sys/devices/pci0000:00/0000:00:0d.0/usb1/1-2/1-2:1.0/video4linux/video10",
    }

    topology = discover_camera_topology(_cameras(), usb_root, physical_ports=physical_ports)

    assert topology["cam_high"]["controller"] == "0000:80:14.0"
    assert topology["cam_left"]["usb_port"] == "4-3"
    assert topology["cam_right"]["controller"] == "0000:00:0d.0"


def test_topology_gate_rejects_case_c_when_right_camera_remains_shared(tmp_path: Path) -> None:
    usb_root = tmp_path / "usb"
    usb_root.mkdir()
    for port in ("4-3", "4-4", "4-5"):
        (usb_root / port).mkdir()
    physical_ports = {
        "high": "/sys/devices/pci0000:80/0000:80:14.0/usb4/4-5/4-5:1.0/video4linux/video16",
        "left": "/sys/devices/pci0000:80/0000:80:14.0/usb4/4-3/4-3:1.0/video4linux/video0",
        "right": "/sys/devices/pci0000:80/0000:80:14.0/usb4/4-4/4-4:1.0/video4linux/video10",
    }
    case = ExperimentCase(
        name="C",
        cpu_set=tuple(range(16, 24)),
        camera_controller_rules={
            "cam_high": ControllerRule(equals="0000:80:14.0"),
            "cam_left": ControllerRule(equals="0000:80:14.0"),
            "cam_right": ControllerRule(
                one_of=("0000:00:0d.0", "0000:00:0d.2"), different_from=("cam_high", "cam_left")
            ),
        },
    )

    _topology, violations = topology_violations(
        _cameras(), case, usb_root, physical_ports=physical_ports
    )

    assert any("cam_right: expected one of" in violation for violation in violations)
    assert any("cam_right: must use a controller different" in violation for violation in violations)


def test_source_discontinuity_events_keep_the_three_recorded_clocks() -> None:
    diagnostics = CameraTimingDiagnostics(
        capture_timestamps_ns=(1_000_000_000, 1_033_333_333, 1_133_333_333),
        host_receive_timestamps_ns=(1_000_100_000, 1_033_500_000, 1_133_600_000),
        device_timestamps_ns=(1_000_000_000, 1_033_333_333, 1_133_333_333),
        frame_numbers=(10, 11, 14),
        capture_timestamp_sources={"realsense_global_time": 3},
        device_timestamp_domains={"timestamp_domain.global_time": 3},
    )
    inspection = SimpleNamespace(camera_timing_diagnostics={"cam_left": diagnostics})

    events = _source_discontinuity_events(inspection)

    assert events["cam_left"] == [
        {
            "time_since_first_capture_s": 0.133333,
            "capture_gap_ms": 100.0,
            "host_receive_gap_ms": 100.1,
            "previous_frame_number": 11,
            "frame_number": 14,
            "frame_number_delta": 3,
            "skipped_frames": 2,
            "kind": "skipped",
            "offline_repaired": False,
        }
    ]
    assert events["cam_high"] == []


def test_source_discontinuity_events_cross_an_offline_synthetic_sample() -> None:
    diagnostics = CameraTimingDiagnostics(
        capture_timestamps_ns=(1_000_000_000, 1_033_333_333, 1_100_000_000),
        host_receive_timestamps_ns=(1_000_100_000, 1_100_200_000, 1_100_200_000),
        device_timestamps_ns=(1_000_000_000, None, 1_100_000_000),
        frame_numbers=(10, None, 13),
        capture_timestamp_sources={"realsense_global_time": 2, "offline_duplicate_previous": 1},
        device_timestamp_domains={"timestamp_domain.global_time": 2, "unavailable": 1},
        synthetic_indices=(1,),
        repair_events=({"synthetic_output_index": 1},),
    )
    inspection = SimpleNamespace(camera_timing_diagnostics={"cam_left": diagnostics})

    events = _source_discontinuity_events(inspection)

    assert events["cam_left"][0]["frame_number_delta"] == 3
    assert events["cam_left"][0]["capture_gap_ms"] == 100.0
    assert events["cam_left"][0]["offline_repaired"]


def test_redirect_dataset_only_changes_experiment_output_and_frame_cap(tmp_path: Path) -> None:
    @dataclass(frozen=True)
    class Collection:
        dataset_dir: Path
        total_sampled_frames: int

    @dataclass(frozen=True)
    class Config:
        collection: Collection
        raw: dict[str, dict[str, str]]

    original_collection = Collection(tmp_path / "production", 50_000)
    original = Config(original_collection, {"collection_config": {"dataset_dir": "production"}})

    redirected = _redirect_dataset(original, tmp_path / "experiment", 65_000)

    assert redirected.collection.dataset_dir == tmp_path / "experiment"
    assert redirected.collection.total_sampled_frames == 65_000
    assert original.collection.dataset_dir == tmp_path / "production"
    assert original.collection.total_sampled_frames == 50_000
    assert redirected.raw["collection_config"] == {"dataset_dir": str(tmp_path / "experiment"), "total_sampled_frames": 65_000}


def test_session_comparison_preserves_formal_and_continuity_results(tmp_path: Path) -> None:
    session = tmp_path / "session"
    for case, passed, skips in (("A", 1, 6), ("B", 2, 0)):
        case_dir = session / case
        case_dir.mkdir(parents=True)
        (case_dir / "summary.json").write_text(
            json.dumps(
                {
                    "case": case,
                    "completed_trials": 2,
                    "formal_passes": passed,
                    "formal_failures": 2 - passed,
                    "source_continuity": {
                        "cam_high": {"total_skipped_frames": 0},
                        "cam_left": {"total_skipped_frames": skips},
                        "cam_right": {"total_skipped_frames": 0},
                    },
                }
            ),
            encoding="utf-8",
        )

    comparison_path = write_session_comparison(session)

    text = comparison_path.read_text(encoding="utf-8")
    assert "| A | 2 | 1 | 1 | 0 | 6 | 0 |" in text
    assert "| B | 2 | 2 | 0 | 0 | 0 | 0 |" in text
    assert "not as a modified acceptance rule" in text
