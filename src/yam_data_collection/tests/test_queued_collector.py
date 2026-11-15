from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from yam_data_collection.collector import CaptureError
from yam_data_collection.models import RobotSnapshot
from yam_data_collection.queued_collector import QueuedEpisodeCollector, _StateMcapWriter, _StateWrite


def _item(sequence: int) -> _StateWrite:
    snapshot = RobotSnapshot(
        role="left_leader",
        source_timestamp_ns=sequence,
        sequence=sequence,
        position=(0.0,) * 7,
        velocity=(),
        effort=(),
    )
    return _StateWrite({"left_leader": snapshot}, sequence)


class _RecordingMcap:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def write_joint_state(self, snapshot: RobotSnapshot, log_time_ns: int, *, header_timestamp_ns: int) -> None:
        del snapshot, header_timestamp_ns
        self.calls.append(log_time_ns)


def test_state_writer_drains_all_accepted_samples_before_close() -> None:
    mcap = _RecordingMcap()
    episode = SimpleNamespace(episode_id=1, mcap=mcap)
    writer = _StateMcapWriter(episode, maxsize=16)

    for sequence in range(5):
        writer.submit(_item(sequence))
    writer.close()

    assert mcap.calls == list(range(5))
    assert not writer._thread.is_alive()


def test_state_writer_queue_overflow_is_reported() -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingMcap(_RecordingMcap):
        def write_joint_state(self, snapshot: RobotSnapshot, log_time_ns: int, *, header_timestamp_ns: int) -> None:
            started.set()
            release.wait(timeout=2.0)
            super().write_joint_state(snapshot, log_time_ns, header_timestamp_ns=header_timestamp_ns)

    mcap = BlockingMcap()
    writer = _StateMcapWriter(SimpleNamespace(episode_id=2, mcap=mcap), maxsize=1)
    writer.submit(_item(0))
    assert started.wait(timeout=1.0)
    writer.submit(_item(1))
    with pytest.raises(CaptureError, match="queue overflow"):
        writer.submit(_item(2))
    release.set()
    writer.close()


def test_state_writer_error_does_not_deadlock_close() -> None:
    failed = threading.Event()

    class FailingMcap(_RecordingMcap):
        def write_joint_state(self, snapshot: RobotSnapshot, log_time_ns: int, *, header_timestamp_ns: int) -> None:
            del snapshot, log_time_ns, header_timestamp_ns
            failed.set()
            raise RuntimeError("synthetic writer failure")

    writer = _StateMcapWriter(SimpleNamespace(episode_id=3, mcap=FailingMcap()), maxsize=8)
    writer.submit(_item(0))
    assert failed.wait(timeout=1.0)
    # Simulate samples that were accepted before the worker failure became
    # visible to the sampler. ``close`` must account for their unfinished
    # queue tasks instead of waiting forever.
    for sequence in range(1, 4):
        writer._queue.put_nowait(_item(sequence))

    close_error: list[BaseException] = []

    def close() -> None:
        try:
            writer.close()
        except BaseException as exc:
            close_error.append(exc)

    thread = threading.Thread(target=close)
    thread.start()
    thread.join(timeout=1.0)
    assert not thread.is_alive(), "writer close hung after worker failure"
    assert close_error and isinstance(close_error[0], CaptureError)
    assert "synthetic writer failure" in str(close_error[0])


def test_queued_collector_keeps_preview_on_recorder_cpus_when_partitioned() -> None:
    collector = object.__new__(QueuedEpisodeCollector)
    collector._allowed_cpu_set = tuple(range(16, 24))
    collector._effective_recorder_cpu_set = (16, 17, 18, 19)
    collector.config = SimpleNamespace(
        collection=SimpleNamespace(recorder_cpu_set="16-19", live_video_cpu_set="20-23")
    )

    assert collector._available_preview_cpu_set() == (16, 17, 18, 19)
