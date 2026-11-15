"""Experimental collector with asynchronous joint-state MCAP writes.

This module deliberately subclasses the production collector instead of
changing it. The acquisition thread keeps the existing telemetry timing and
validation guards, while a bounded worker queue moves MCAP serialization off
that timing-critical path. Camera/video handling and all quality thresholds
remain those of :class:`EpisodeCollector`.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

from .collector import CaptureError, CaptureState, EpisodeCollector, _Episode
from .models import RobotSnapshot


@dataclass(frozen=True)
class _StateWrite:
    snapshots: dict[str, RobotSnapshot]
    capture_time_ns: int


class _StateMcapWriter:
    """Drain joint-state writes without blocking the telemetry sampler."""

    _STOP = object()

    def __init__(self, episode: _Episode, *, maxsize: int) -> None:
        self.episode = episode
        self._queue: queue.Queue[_StateWrite | object] = queue.Queue(maxsize=maxsize)
        self._error: BaseException | None = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name=f"state-mcap-writer-{episode.episode_id}",
            daemon=True,
        )
        self._thread.start()

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def error(self) -> BaseException | None:
        return self._error

    def submit(self, item: _StateWrite) -> None:
        if self._closed:
            raise CaptureError("joint-state MCAP writer is already closed")
        if self._error is not None:
            raise CaptureError(f"joint-state MCAP writer failed: {self._error}")
        try:
            self._queue.put_nowait(item)
        except queue.Full as exc:
            raise CaptureError("joint-state MCAP writer queue overflow") from exc

    def _run(self) -> None:
        try:
            while True:
                item = self._queue.get()
                try:
                    if item is self._STOP:
                        return
                    assert isinstance(item, _StateWrite)
                    for role, snapshot in item.snapshots.items():
                        self.episode.mcap.write_joint_state(
                            snapshot,
                            item.capture_time_ns,
                            header_timestamp_ns=snapshot.source_timestamp_ns,
                        )
                finally:
                    self._queue.task_done()
        except BaseException as exc:
            self._error = exc

    def close(self) -> None:
        """Drain all accepted samples, then stop the worker."""
        if self._closed:
            if self._error is not None:
                raise CaptureError(f"joint-state MCAP writer failed: {self._error}") from self._error
            return
        self._closed = True

        # A failed worker no longer consumes the queue.  Use non-blocking stop
        # delivery so a full queue cannot deadlock finalization; the accepted
        # but unwritten items are discarded below and the episode is rejected.
        while self._thread.is_alive() and self._error is None:
            try:
                self._queue.put_nowait(self._STOP)
                break
            except queue.Full:
                time.sleep(0.001)
        self._thread.join()

        # Every queue item contributes to Queue.join's unfinished-task count.
        # If the worker failed, reclaim the items it could not process before
        # joining so the error is reported instead of blocking forever.
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
            else:
                self._queue.task_done()
        self._queue.join()
        if self._error is not None:
            raise CaptureError(f"joint-state MCAP writer failed: {self._error}") from self._error


class QueuedEpisodeCollector(EpisodeCollector):
    """Production collector variant with asynchronous joint-state writes."""

    STATE_WRITER_QUEUE_SIZE = 512

    def _available_preview_cpu_set(self) -> tuple[int, ...]:
        """Keep the experimental preview enabled on fully partitioned hosts."""
        available = super()._available_preview_cpu_set()
        if available:
            return available
        return tuple(getattr(self, "_effective_recorder_cpu_set", ()) or getattr(self, "_allowed_cpu_set", ()))

    def _start_live_preview(self) -> None:
        """Report an immediate Tk/DISPLAY failure while keeping capture alive."""
        super()._start_live_preview()
        preview = getattr(self, "_live_preview", None)
        if preview is None:
            return
        time.sleep(0.05)
        if not preview.running:
            self._report_preview_error(
                "live preview process exited during startup",
                RuntimeError("check DISPLAY and tkinter availability"),
            )

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._state_mcap_writers: dict[int, _StateMcapWriter] = {}

    def _state_writer_for(self, episode: _Episode) -> _StateMcapWriter:
        key = id(episode)
        writer = self._state_mcap_writers.get(key)
        if writer is None:
            writer = _StateMcapWriter(episode, maxsize=self.STATE_WRITER_QUEUE_SIZE)
            self._state_mcap_writers[key] = writer
        return writer

    def _record_state_sample(self) -> None:
        """Read and validate telemetry, then enqueue the MCAP write."""
        with self._state_lock:
            error = self._background_error
        if error is not None:
            self._discard_for_background_error(error)
            return
        preview_snapshots: dict[str, RobotSnapshot] | None = None
        preview_timestamp_ns: int | None = None
        try:
            snapshots = self._read_snapshots(require_fresh=True)
            capture_time_ns = time.time_ns()
            sample_monotonic_ns = time.monotonic_ns()
            self._last_telemetry_success_ns = capture_time_ns
        except Exception as exc:
            self._handle_telemetry_failure(exc)
            return

        error = None
        should_finalize = False
        episode = self._begin_episode_operation()
        if episode is None:
            self._submit_preview_joints(snapshots, sample_monotonic_ns)
            return
        try:
            timing_error = self._episode_telemetry_timing_error(episode, snapshots, sample_monotonic_ns)
            if timing_error is not None:
                error = timing_error
                with self._state_lock:
                    self._background_error = error
            else:
                try:
                    self._state_writer_for(episode).submit(_StateWrite(snapshots, capture_time_ns))
                except Exception as exc:
                    error = exc if isinstance(exc, CaptureError) else CaptureError(str(exc))
                    with self._state_lock:
                        self._background_error = error
                else:
                    self._remember_episode_telemetry_timing(episode, snapshots, sample_monotonic_ns)
                    camera_start_timestamp_ns = min(
                        snapshot.source_timestamp_ns for snapshot in snapshots.values()
                    )
                    with self._state_lock:
                        if getattr(episode, "camera_start_timestamp_ns", None) is None:
                            episode.camera_start_timestamp_ns = camera_start_timestamp_ns
                        recording = self._state is CaptureState.RECORDING
                    episode.state_samples += 1
                    if recording:
                        episode.recording_state_samples = getattr(episode, "recording_state_samples", 0) + 1
                    preview_snapshots = snapshots
                    preview_timestamp_ns = sample_monotonic_ns
                    should_finalize = (
                        recording
                        and getattr(episode, "recording_state_samples", 0)
                        >= self.config.collection.total_sampled_frames
                    )
        finally:
            self._end_episode_operation(episode)
        if error is None and preview_snapshots is not None and preview_timestamp_ns is not None:
            self._submit_preview_joints(preview_snapshots, preview_timestamp_ns)
        if error is not None:
            self._discard_for_background_error(error)
        elif should_finalize:
            self._print("Configured state-frame limit reached; saving episode.")
            self.save_episode()

    def _finish_episode(self, episode: _Episode, commit: bool) -> bool:
        writer = self._state_mcap_writers.pop(id(episode), None)
        if writer is not None:
            try:
                writer.close()
            except BaseException as exc:
                with self._state_lock:
                    if self._background_error is None:
                        self._background_error = exc if isinstance(exc, CaptureError) else CaptureError(str(exc))
                commit = False
        return super()._finish_episode(episode, commit=commit)


__all__ = ["QueuedEpisodeCollector"]
