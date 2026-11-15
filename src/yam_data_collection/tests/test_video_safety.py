from __future__ import annotations

import io
import json
import shutil
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from yam_data_collection.mcap_writer import CAMERA_TOPICS
from yam_data_collection.quality import ValidationPolicy, validate_video_assets
from yam_data_collection.video import (
    AsyncFfmpegWriter,
    VideoMetadata,
    VideoProbeError,
    VideoTranscodeError,
    VideoWriterError,
    probe_video_frame_count,
    probe_video_metadata,
    transcode_to_h264_with_frame_duplicates,
    transcode_to_hevc,
    transcode_to_hevc_with_frame_duplicates,
)

from .test_mcap_contract import _write_mixed_rate_episode, _write_valid_episode


class _BlockingBrokenStdin:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def write(self, value: bytes) -> int:
        self.started.set()
        self.release.wait(timeout=1.0)
        raise BrokenPipeError("synthetic encoder failure")

    def close(self) -> None:
        return None


class _BrokenEncoderProcess:
    def __init__(self, stdin: _BlockingBrokenStdin) -> None:
        self.stdin = stdin
        self.stderr = io.BytesIO(b"synthetic encoder failure")
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        if self.returncode is None:
            self.returncode = 1
        return self.returncode


def test_failed_encoder_close_never_waits_for_a_full_queue(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    stdin = _BlockingBrokenStdin()
    process = _BrokenEncoderProcess(stdin)
    monkeypatch.setattr("yam_data_collection.video.shutil.which", lambda _: "ffmpeg")
    monkeypatch.setattr("yam_data_collection.video.subprocess.Popen", lambda *args, **kwargs: process)
    writer = AsyncFfmpegWriter(tmp_path / "cam.mp4", 2, 2, 30, "libx265", 23, queue_size=1)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    assert writer.enqueue(frame)
    assert stdin.started.wait(timeout=1.0)
    assert writer.enqueue(frame)  # Fill the queue while the first write is in progress.
    stdin.release.set()
    writer._thread.join(timeout=1.0)

    started = time.monotonic()
    with pytest.raises(VideoWriterError, match="video writer failed"):
        writer.close(commit=True)
    assert time.monotonic() - started < 1.0


def test_diagnostic_snapshot_preserves_queue_overflow_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stdin = _BlockingBrokenStdin()
    process = _BrokenEncoderProcess(stdin)
    monkeypatch.setattr("yam_data_collection.video.shutil.which", lambda _: "ffmpeg")
    monkeypatch.setattr("yam_data_collection.video.subprocess.Popen", lambda *args, **kwargs: process)
    writer = AsyncFfmpegWriter(tmp_path / "cam.mp4", 2, 2, 30, "libx265", 23, queue_size=1)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    assert writer.enqueue(frame)
    assert stdin.started.wait(timeout=1.0)
    assert writer.enqueue(frame)
    assert not writer.enqueue(frame)

    snapshot = writer.diagnostic_snapshot()
    assert snapshot["queue_capacity"] == 1
    assert snapshot["accepted_frames"] == 2
    assert snapshot["dropped_frames"] == 1
    assert snapshot["max_queue_depth"] == 1
    assert snapshot["writer_thread_alive"]

    stdin.release.set()
    writer._thread.join(timeout=1.0)
    writer.close(commit=False)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required only for the encoder smoke test")
def test_async_writer_commits_a_real_h265_video(tmp_path: Path) -> None:
    output_path = tmp_path / "cam.mp4"
    writer = AsyncFfmpegWriter(output_path, 16, 16, 30, "libx265", 28, preset="veryfast", threads=1)
    for value in range(3):
        assert writer.enqueue(np.full((16, 16, 3), value, dtype=np.uint8))

    writer.close(commit=True)

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required only for the encoder smoke test")
def test_async_writer_accepts_packed_yuyv_input(tmp_path: Path) -> None:
    output_path = tmp_path / "cam-yuyv.mp4"
    writer = AsyncFfmpegWriter(
        output_path,
        16,
        16,
        30,
        "libx265",
        28,
        preset="ultrafast",
        threads=1,
        input_pixel_format="yuyv422",
    )
    assert writer.enqueue(np.full((16, 16, 2), 128, dtype=np.uint8))

    writer.close(commit=True)

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_async_writer_uses_x264_specific_thread_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stdin = _BlockingBrokenStdin()
    process = _BrokenEncoderProcess(stdin)
    commands: list[list[str]] = []
    monkeypatch.setattr("yam_data_collection.video.shutil.which", lambda _: "ffmpeg")

    def fake_popen(command: list[str], **kwargs: object) -> _BrokenEncoderProcess:
        del kwargs
        commands.append(command)
        return process

    monkeypatch.setattr("yam_data_collection.video.subprocess.Popen", fake_popen)
    writer = AsyncFfmpegWriter(tmp_path / "live.mp4", 2, 2, 30, "libx264", 23, threads=1)
    writer.close(commit=False)

    assert commands
    command = commands[0]
    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-tag:v") + 1] == "avc1"
    assert command[command.index("-threads") + 1] == "1"
    assert "-x265-params" not in command


def test_async_writer_runs_an_explicit_affinity_encoder_with_taskset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stdin = _BlockingBrokenStdin()
    process = _BrokenEncoderProcess(stdin)
    process.pid = 4242
    commands: list[list[str]] = []
    monkeypatch.setattr("yam_data_collection.video.shutil.which", lambda tool: f"/usr/bin/{tool}")
    monkeypatch.setattr("yam_data_collection.video.os.sched_getaffinity", lambda pid: {20, 21, 22, 23})

    def fake_popen(command: list[str], **kwargs: object) -> _BrokenEncoderProcess:
        del kwargs
        commands.append(command)
        return process

    monkeypatch.setattr("yam_data_collection.video.subprocess.Popen", fake_popen)
    writer = AsyncFfmpegWriter(
        tmp_path / "live.mp4", 2, 2, 30, "libx264", 23, threads=1, encoder_cpu_set="20,21-23"
    )
    snapshot = writer.diagnostic_snapshot()
    writer.close(commit=False)

    assert commands
    assert commands[0][:5] == ["taskset", "--cpu-list", "20-23", "ffmpeg", "-hide_banner"]
    assert snapshot["requested_encoder_cpu_set"] == "20-23"
    assert snapshot["effective_encoder_cpu_set"] == [20, 21, 22, 23]
    assert snapshot["encoder_affinity_mode"] == "taskset"


@pytest.mark.parametrize("cpu_set", ["", "20-", "23-20", "20,,21", 20])
def test_async_writer_rejects_invalid_encoder_cpu_sets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cpu_set: object
) -> None:
    monkeypatch.setattr("yam_data_collection.video.shutil.which", lambda _: "ffmpeg")

    with pytest.raises(VideoWriterError, match="encoder_cpu_set"):
        AsyncFfmpegWriter(tmp_path / "live.mp4", 2, 2, 30, "libx264", 23, encoder_cpu_set=cpu_set)  # type: ignore[arg-type]


def test_probe_video_metadata_prefers_decoded_frame_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "source.mp4"
    path.touch()
    monkeypatch.setattr("yam_data_collection.video.shutil.which", lambda _: "ffprobe")
    monkeypatch.setattr(
        "yam_data_collection.video.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=json.dumps(
                {
                    "streams": [
                        {
                            "codec_name": "h264",
                            "width": 1280,
                            "height": 720,
                            "pix_fmt": "yuv420p",
                            "avg_frame_rate": "30000/1001",
                            "nb_frames": "119",
                            "nb_read_frames": "120",
                        }
                    ]
                }
            ),
        ),
    )

    metadata = probe_video_metadata(path)

    assert metadata.codec == "h264"
    assert metadata.width == 1280
    assert metadata.height == 720
    assert metadata.pixel_format == "yuv420p"
    assert metadata.frame_count == 120
    assert metadata.fps == pytest.approx(30000 / 1001)
    assert probe_video_frame_count(path) == 120


def test_probe_video_frame_count_refuses_to_estimate_when_ffprobe_has_no_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "source.mp4"
    path.touch()
    monkeypatch.setattr("yam_data_collection.video.shutil.which", lambda _: "ffprobe")
    monkeypatch.setattr(
        "yam_data_collection.video.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=json.dumps(
                {
                    "streams": [
                        {
                            "codec_name": "h264",
                            "width": 1280,
                            "height": 720,
                            "avg_frame_rate": "30/1",
                            "nb_frames": "N/A",
                            "nb_read_frames": "N/A",
                        }
                    ]
                }
            ),
        ),
    )

    with pytest.raises(VideoProbeError, match="did not report a decoded frame count"):
        probe_video_frame_count(path)


def test_transcode_refuses_to_publish_a_frame_count_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "videos_live" / "cam_high.mp4"
    source.parent.mkdir()
    source.touch()
    destination = tmp_path / "videos" / "cam_high.mp4"
    destination.parent.mkdir()
    destination.write_bytes(b"previous verified final")
    source_metadata = VideoMetadata(source, "h264", 1280, 720, 30.0, 5, "yuv420p")
    output_metadata = VideoMetadata(
        destination.with_suffix(".partial.mp4"), "hevc", 1280, 720, 30.0, 4, "yuv420p"
    )
    probed: list[Path] = []
    commands: list[list[str]] = []
    monkeypatch.setattr("yam_data_collection.video.shutil.which", lambda _: "ffmpeg")

    def fake_probe(path: Path) -> VideoMetadata:
        probed.append(Path(path))
        return source_metadata if len(probed) == 1 else output_metadata

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        del kwargs
        commands.append(command)
        Path(command[-1]).touch()
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr("yam_data_collection.video.probe_video_metadata", fake_probe)
    monkeypatch.setattr("yam_data_collection.video.subprocess.run", fake_run)

    with pytest.raises(VideoTranscodeError, match="frame-count mismatch"):
        transcode_to_hevc(source, destination, crf=23, preset="ultrafast", threads=1, overwrite=True)

    assert destination.read_bytes() == b"previous verified final"
    assert destination.with_suffix(".partial.mp4").is_file()
    assert commands
    command = commands[0]
    assert command[command.index("-vsync") + 1] == "0"
    assert "-r" not in command


class _MemoryPipe:
    def __init__(self, initial: bytes = b"") -> None:
        self.data = bytearray(initial)
        self.position = 0
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        if self.closed:
            raise ValueError("read from closed pipe")
        if size < 0:
            size = len(self.data) - self.position
        end = min(len(self.data), self.position + size)
        value = bytes(self.data[self.position : end])
        self.position = end
        return value

    def write(self, value: bytes | memoryview) -> int:
        if self.closed:
            raise ValueError("write to closed pipe")
        raw = bytes(value)
        self.data.extend(raw)
        return len(raw)

    def close(self) -> None:
        self.closed = True


class _FakeStreamingProcess:
    def __init__(
        self,
        *,
        stdin: _MemoryPipe | None = None,
        stdout: _MemoryPipe | None = None,
        final_returncode: int = 0,
    ) -> None:
        self.stdin = stdin
        self.stdout = stdout
        self._final_returncode = final_returncode
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        if self.returncode is None:
            self.returncode = self._final_returncode
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


def _mock_streaming_duplicate_transcode(
    monkeypatch: pytest.MonkeyPatch,
    *,
    source: Path,
    destination: Path,
    raw_frames: list[bytes],
    output_frame_count: int,
) -> tuple[list[list[str]], _MemoryPipe]:
    source_metadata = VideoMetadata(source, "h264", 2, 1, 30.0, len(raw_frames), "yuv420p")
    output_metadata = VideoMetadata(
        destination.with_suffix(".partial.mp4"), "hevc", 2, 1, 30.0, output_frame_count, "yuv420p"
    )
    probes = iter((source_metadata, output_metadata))
    commands: list[list[str]] = []
    encoder_pipe = _MemoryPipe()

    monkeypatch.setattr("yam_data_collection.video.shutil.which", lambda _: "ffmpeg")
    monkeypatch.setattr("yam_data_collection.video.probe_video_metadata", lambda _path: next(probes))

    def fake_popen(command: list[str], **kwargs: object) -> _FakeStreamingProcess:
        del kwargs
        commands.append(command)
        if command[-1] == "pipe:1":
            return _FakeStreamingProcess(stdout=_MemoryPipe(b"".join(raw_frames)))
        Path(command[-1]).touch()
        return _FakeStreamingProcess(stdin=encoder_pipe)

    monkeypatch.setattr("yam_data_collection.video.subprocess.Popen", fake_popen)
    return commands, encoder_pipe


def test_streaming_duplicate_transcode_writes_the_previous_frame_once_and_publishes_atomically(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "live.mp4"
    source.write_bytes(b"source remains untouched")
    destination = tmp_path / "final" / "cam.mp4"
    frames = [b"A" * 6, b"B" * 6, b"C" * 6]
    commands, encoder_pipe = _mock_streaming_duplicate_transcode(
        monkeypatch,
        source=source,
        destination=destination,
        raw_frames=frames,
        output_frame_count=4,
    )

    result = transcode_to_hevc_with_frame_duplicates(
        source,
        destination,
        duplicate_after_indices=[1],
        crf=23,
        preset="ultrafast",
        threads=1,
    )

    assert bytes(encoder_pipe.data) == frames[0] + frames[1] + frames[1] + frames[2]
    assert source.read_bytes() == b"source remains untouched"
    assert destination.is_file()
    assert not destination.with_suffix(".partial.mp4").exists()
    assert result.duplicate_after_indices == (1,)
    assert result.source_frame_count == 3
    assert result.output_frame_count == 4
    assert len(commands) == 2
    encoder_command, decoder_command = commands
    assert encoder_command[encoder_command.index("-c:v") + 1] == "libx265"
    assert encoder_command[encoder_command.index("-framerate") + 1] == "30"
    assert decoder_command[-4:] == ["bgr24", "-f", "rawvideo", "pipe:1"]


@pytest.mark.parametrize(
    ("indices", "message"),
    [
        ([-1], "out-of-range"),
        ([3], "out-of-range"),
        ([True], "only integers"),
        ([1.5], "only integers"),
        ([1, 1], "must not contain duplicate"),
    ],
)
def test_streaming_duplicate_transcode_rejects_invalid_source_indices_before_starting_ffmpeg(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    indices: list[object],
    message: str,
) -> None:
    source = tmp_path / "source.mp4"
    source.touch()
    metadata = VideoMetadata(source, "h264", 16, 16, 30.0, 3, "yuv420p")
    monkeypatch.setattr("yam_data_collection.video.shutil.which", lambda _: "ffmpeg")
    monkeypatch.setattr("yam_data_collection.video.probe_video_metadata", lambda _path: metadata)
    monkeypatch.setattr(
        "yam_data_collection.video.subprocess.Popen",
        lambda *_args, **_kwargs: pytest.fail("invalid indices must be rejected before FFmpeg starts"),
    )

    with pytest.raises(VideoTranscodeError, match=message):
        transcode_to_hevc_with_frame_duplicates(
            source,
            tmp_path / "output.mp4",
            duplicate_after_indices=indices,  # type: ignore[arg-type]
            crf=23,
            preset="ultrafast",
        )


def test_streaming_duplicate_transcode_keeps_existing_final_on_validation_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"original source")
    destination = tmp_path / "output.mp4"
    destination.write_bytes(b"previous verified output")
    _mock_streaming_duplicate_transcode(
        monkeypatch,
        source=source,
        destination=destination,
        raw_frames=[b"A" * 6, b"B" * 6, b"C" * 6],
        output_frame_count=3,
    )

    with pytest.raises(VideoTranscodeError, match="frame-count mismatch after repair"):
        transcode_to_hevc_with_frame_duplicates(
            source,
            destination,
            duplicate_after_indices=[1],
            crf=23,
            preset="ultrafast",
            overwrite=True,
        )

    assert source.read_bytes() == b"original source"
    assert destination.read_bytes() == b"previous verified output"
    assert destination.with_suffix(".partial.mp4").is_file()


def test_streaming_duplicate_transcode_never_removes_a_source_named_like_the_partial(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "output.partial.mp4"
    source.write_bytes(b"source")
    metadata = VideoMetadata(source, "h264", 16, 16, 30.0, 1, "yuv420p")
    monkeypatch.setattr("yam_data_collection.video.shutil.which", lambda _: "ffmpeg")
    monkeypatch.setattr("yam_data_collection.video.probe_video_metadata", lambda _path: metadata)

    with pytest.raises(VideoTranscodeError, match="conflicts with the derivative partial path"):
        transcode_to_hevc_with_frame_duplicates(
            source,
            tmp_path / "output.mp4",
            duplicate_after_indices=[0],
            crf=23,
            preset="ultrafast",
        )

    assert source.read_bytes() == b"source"


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required only for the transcode smoke test",
)
def test_offline_transcode_commits_hevc_with_the_same_decoded_frame_count(tmp_path: Path) -> None:
    source = tmp_path / "videos_live" / "cam_high.mp4"
    source.parent.mkdir()
    writer = AsyncFfmpegWriter(source, 16, 16, 30, "libx264", 28, preset="ultrafast", threads=1)
    for value in range(5):
        assert writer.enqueue(np.full((16, 16, 3), value, dtype=np.uint8))
    writer.close(commit=True)

    destination = tmp_path / "videos" / "cam_high.mp4"
    result = transcode_to_hevc(source, destination, crf=28, preset="ultrafast", threads=1)

    assert source.is_file()
    assert destination.is_file()
    assert not destination.with_suffix(".partial.mp4").exists()
    assert result.source.codec == "h264"
    assert result.output.codec == "hevc"
    assert result.frame_count == 5
    assert probe_video_frame_count(destination) == 5


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required only for the transcode smoke test",
)
def test_streaming_duplicate_transcode_commits_a_verified_hevc_derivative(tmp_path: Path) -> None:
    source = tmp_path / "videos_live" / "cam_high.mp4"
    source.parent.mkdir()
    writer = AsyncFfmpegWriter(source, 16, 16, 30, "libx264", 28, preset="ultrafast", threads=1)
    for value in range(3):
        assert writer.enqueue(np.full((16, 16, 3), value, dtype=np.uint8))
    writer.close(commit=True)
    source_bytes = source.read_bytes()

    destination = tmp_path / "videos_repaired" / "cam_high.mp4"
    result = transcode_to_hevc_with_frame_duplicates(
        source,
        destination,
        duplicate_after_indices=[1],
        crf=28,
        preset="ultrafast",
        threads=1,
    )

    assert source.read_bytes() == source_bytes
    assert destination.is_file()
    assert result.source_frame_count == 3
    assert result.output_frame_count == 4
    assert result.output.codec == "hevc"
    assert (result.output.width, result.output.height) == (16, 16)
    assert probe_video_frame_count(destination) == 4


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required only for the transcode smoke test",
)
def test_streaming_duplicate_transcode_commits_a_verified_h264_derivative(tmp_path: Path) -> None:
    source = tmp_path / "videos" / "cam_right.mp4"
    source.parent.mkdir()
    writer = AsyncFfmpegWriter(source, 16, 16, 30, "libx264", 28, preset="ultrafast", threads=1)
    for value in range(3):
        assert writer.enqueue(np.full((16, 16, 3), value, dtype=np.uint8))
    writer.close(commit=True)
    source_bytes = source.read_bytes()

    destination = tmp_path / "videos_repaired" / "cam_right.mp4"
    result = transcode_to_h264_with_frame_duplicates(
        source,
        destination,
        duplicate_after_indices=[1],
        crf=28,
        preset="ultrafast",
        threads=1,
    )

    assert source.read_bytes() == source_bytes
    assert destination.is_file()
    assert result.source_frame_count == 3
    assert result.output_frame_count == 4
    assert result.output.codec == "h264"
    assert (result.output.width, result.output.height) == (16, 16)
    assert probe_video_frame_count(destination) == 4


def _ffprobe_result(
    frame_count: int,
    codec: str = "hevc",
    *,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
) -> SimpleNamespace:
    return SimpleNamespace(
        returncode=0,
        stderr="",
        stdout=json.dumps(
            {
                "streams": [
                    {
                        "codec_name": codec,
                        "width": width,
                        "height": height,
                        "avg_frame_rate": f"{fps}/1",
                        "nb_frames": str(frame_count),
                        "nb_read_frames": str(frame_count),
                    }
                ]
            }
        ),
    )


def test_video_assets_compare_decoded_frame_count_with_camera_info(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mcap_path = tmp_path / "0.mcap"
    _write_valid_episode(mcap_path)
    video_dir = tmp_path / "videos" / "0"
    video_dir.mkdir(parents=True)
    for role in CAMERA_TOPICS:
        (video_dir / f"{role}.mp4").touch()
    monkeypatch.setattr("yam_data_collection.quality.subprocess.run", lambda *args, **kwargs: _ffprobe_result(50))

    result = validate_video_assets(mcap_path, ValidationPolicy(expected_camera_fps=30.0))

    assert not result["passed"]
    assert all(not entry["frame_count_matches"] for entry in result["per_camera"].values())


def test_source_timeline_skips_container_fps_but_keeps_identity_checks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mcap_path = tmp_path / "0.mcap"
    _write_valid_episode(mcap_path)
    video_dir = tmp_path / "videos" / "0"
    video_dir.mkdir(parents=True)
    (video_dir / "source_timeline.json").write_text('{"mode": "source"}\n', encoding="utf-8")
    for role in CAMERA_TOPICS:
        (video_dir / f"{role}.mp4").touch()
    monkeypatch.setattr(
        "yam_data_collection.quality.subprocess.run", lambda *args, **kwargs: _ffprobe_result(61, fps=60)
    )

    result = validate_video_assets(mcap_path, ValidationPolicy(expected_camera_fps=30.0))

    assert result["passed"]
    assert result["source_timeline"]
    assert not result["require_fps_match"]
    assert all(not entry["fps_matches"] for entry in result["per_camera"].values())
    assert all(entry["fps_check_skipped"] for entry in result["per_camera"].values())
    assert all(entry["frame_count_matches"] for entry in result["per_camera"].values())


def test_video_assets_reject_one_frame_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mcap_path = tmp_path / "0.mcap"
    _write_valid_episode(mcap_path)
    video_dir = tmp_path / "videos" / "0"
    video_dir.mkdir(parents=True)
    for role in CAMERA_TOPICS:
        (video_dir / f"{role}.mp4").touch()
    monkeypatch.setattr("yam_data_collection.quality.subprocess.run", lambda *args, **kwargs: _ffprobe_result(62))

    result = validate_video_assets(mcap_path, ValidationPolicy(expected_camera_fps=30.0))

    assert not result["passed"]
    assert all(not entry["frame_count_matches"] for entry in result["per_camera"].values())


def test_video_assets_require_hevc(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mcap_path = tmp_path / "0.mcap"
    _write_valid_episode(mcap_path)
    video_dir = tmp_path / "videos" / "0"
    video_dir.mkdir(parents=True)
    for role in CAMERA_TOPICS:
        (video_dir / f"{role}.mp4").touch()
    monkeypatch.setattr(
        "yam_data_collection.quality.subprocess.run", lambda *args, **kwargs: _ffprobe_result(61, codec="h264")
    )

    result = validate_video_assets(mcap_path, ValidationPolicy(expected_camera_fps=30.0))

    assert not result["passed"]
    assert all(not entry["codec_matches"] for entry in result["per_camera"].values())


def test_video_assets_accept_mixed_configured_frame_rates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mcap_path = tmp_path / "0.mcap"
    cameras = _write_mixed_rate_episode(mcap_path)
    video_dir = tmp_path / "videos" / "0"
    video_dir.mkdir(parents=True)
    profiles = {
        "cam_high": (61, 1280, 720, 30),
        "cam_left": (121, 848, 480, 60),
        "cam_right": (121, 848, 480, 60),
    }
    for role in CAMERA_TOPICS:
        (video_dir / f"{role}.mp4").touch()

    def fake_ffprobe(command: list[str], **kwargs: object) -> SimpleNamespace:
        del kwargs
        frame_count, width, height, fps = profiles[Path(command[-1]).stem]
        return _ffprobe_result(frame_count, width=width, height=height, fps=fps)

    monkeypatch.setattr("yam_data_collection.quality.subprocess.run", fake_ffprobe)

    result = validate_video_assets(mcap_path, ValidationPolicy.for_cameras(cameras))

    assert result["passed"]
    assert result["per_camera"]["cam_high"]["target_fps"] == 30
    assert result["per_camera"]["cam_left"]["target_fps"] == 60


def test_video_assets_reject_resolution_that_differs_from_the_configured_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mcap_path = tmp_path / "0.mcap"
    cameras = _write_mixed_rate_episode(mcap_path)
    video_dir = tmp_path / "videos" / "0"
    video_dir.mkdir(parents=True)
    profiles = {
        "cam_high": (61, 1280, 720, 30),
        "cam_left": (121, 640, 480, 60),
        "cam_right": (121, 848, 480, 60),
    }
    for role in CAMERA_TOPICS:
        (video_dir / f"{role}.mp4").touch()

    def fake_ffprobe(command: list[str], **kwargs: object) -> SimpleNamespace:
        del kwargs
        frame_count, width, height, fps = profiles[Path(command[-1]).stem]
        return _ffprobe_result(frame_count, width=width, height=height, fps=fps)

    monkeypatch.setattr("yam_data_collection.quality.subprocess.run", fake_ffprobe)

    result = validate_video_assets(mcap_path, ValidationPolicy.for_cameras(cameras))

    assert not result["passed"]
    entry = result["per_camera"]["cam_left"]
    assert entry["expected_width"] == 848
    assert entry["expected_height"] == 480
    assert not entry["resolution_matches"]
