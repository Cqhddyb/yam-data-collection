from __future__ import annotations

import shutil
import threading
from pathlib import Path

import numpy as np
import pytest

from yam_data_collection.video import TimestampedVideoWriter, close_video_writers, transcode_to_hevc


def test_source_timeline_writers_close_in_parallel() -> None:
    barrier = threading.Barrier(3)
    closed: list[str] = []
    lock = threading.Lock()

    class FakeWriter:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self, *, commit: bool) -> None:
            assert commit
            barrier.wait(timeout=2.0)
            with lock:
                closed.append(self.name)

    errors = close_video_writers(
        (FakeWriter("cam_high"), FakeWriter("cam_left"), FakeWriter("cam_right")),
        commit=True,
        parallel=True,
    )

    assert errors == []
    assert set(closed) == {"cam_high", "cam_left", "cam_right"}


def _decoded_pts(path: Path) -> tuple[list[int], dict[str, str]]:
    av = pytest.importorskip("av")
    with av.open(str(path), mode="r") as container:
        stream = container.streams.video[0]
        return [int(frame.pts) for frame in container.decode(stream)], dict(container.metadata)


def _write_source_timeline_video(path: Path) -> tuple[list[int], int]:
    origin_ns = 1_700_000_000_000_000_000
    offsets_ns = [5_000_000, 21_666_667, 60_123_456, 100_000_111]
    writer = TimestampedVideoWriter(
        path,
        32,
        32,
        60,
        "libx264",
        28,
        preset="ultrafast",
        threads=1,
        queue_size=8,
    )
    for value, offset_ns in enumerate(offsets_ns):
        assert writer.enqueue(
            np.full((32, 32, 3), value, dtype=np.uint8),
            timestamp_ns=origin_ns + offset_ns,
        )
    writer.set_timeline_origin_ns(origin_ns)
    writer.close(commit=True)
    assert writer._timeline_rewrite_mode == "packet-remux"
    return [offset_ns // 1_000 for offset_ns in offsets_ns], origin_ns


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required for the source-timeline smoke test",
)
def test_timestamped_writer_preserves_nonuniform_capture_pts(tmp_path: Path) -> None:
    pytest.importorskip("av")
    expected_pts, origin_ns = _write_source_timeline_video(tmp_path / "cam_left.mp4")

    decoded_pts, metadata = _decoded_pts(tmp_path / "cam_left.mp4")

    assert decoded_pts == expected_pts
    assert metadata["com.yam.video_timeline"] == "source"
    assert metadata["com.yam.timeline_origin_ns"] == str(origin_ns)
    assert metadata["com.yam.timeline_time_base"] == "1/1000000"


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required for the source-timeline smoke test",
)
def test_hevc_transcode_preserves_source_capture_pts(tmp_path: Path) -> None:
    pytest.importorskip("av")
    source = tmp_path / "videos_live" / "cam_right.mp4"
    source.parent.mkdir()
    expected_pts, _origin_ns = _write_source_timeline_video(source)

    destination = tmp_path / "videos" / "cam_right.mp4"
    result = transcode_to_hevc(
        source,
        destination,
        crf=28,
        preset="ultrafast",
        threads=1,
        preserve_timestamps=True,
    )

    decoded_pts, _metadata = _decoded_pts(destination)
    assert decoded_pts == expected_pts
    assert result.source.frame_count == len(expected_pts)
    assert result.output.frame_count == len(expected_pts)
    assert result.output.codec == "hevc"
