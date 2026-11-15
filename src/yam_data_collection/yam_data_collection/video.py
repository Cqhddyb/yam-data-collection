"""Asynchronous live encoding and verified offline video transcoding.

The live recorder deliberately keeps its encoder separate from the final
dataset codec. A lightweight H.264 intermediate can be written while an
episode is active, then converted to the required HEVC asset only after the
episode has finished. This module owns the video-specific safety checks so
the collector can treat a final video as committed only after its decoded
frame count is proven to match the intermediate.
"""

from __future__ import annotations

import json
import math
import os
import queue
import shutil
import struct
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral
from pathlib import Path
from typing import BinaryIO, Iterable, Optional

import numpy as np

from .config import normalize_cpu_set


class VideoWriterError(RuntimeError):
    """ffmpeg could not encode a requested episode video."""


class VideoProbeError(VideoWriterError):
    """ffprobe could not inspect a video stream reliably."""


class VideoTranscodeError(VideoWriterError):
    """An intermediate video could not be safely converted to final HEVC."""


@dataclass(frozen=True)
class VideoMetadata:
    """Decoded metadata for the first video stream in an MP4 asset.

    ``frame_count`` is intentionally optional here because some containers
    cannot report it. Call :func:`probe_video_frame_count` when an exact count
    is required for a data-integrity decision.
    """

    path: Path
    codec: str
    width: int
    height: int
    fps: float | None
    frame_count: int | None
    pixel_format: str | None


@dataclass(frozen=True)
class VideoTranscodeResult:
    """Evidence recorded when an intermediate is promoted to final HEVC."""

    source: VideoMetadata
    output: VideoMetadata

    @property
    def frame_count(self) -> int:
        """The exact, verified frame count shared by source and output."""
        assert self.source.frame_count is not None
        return self.source.frame_count


@dataclass(frozen=True)
class VideoFrameDuplicateResult:
    """Evidence from a verified HEVC derivative with duplicated source frames."""

    source: VideoMetadata
    output: VideoMetadata
    duplicate_after_indices: tuple[int, ...]

    @property
    def source_frame_count(self) -> int:
        assert self.source.frame_count is not None
        return self.source.frame_count

    @property
    def output_frame_count(self) -> int:
        assert self.output.frame_count is not None
        return self.output.frame_count


_HEVC_CODECS = frozenset({"libx265", "hevc"})
_H264_CODECS = frozenset({"libx264", "h264"})
_STDERR_TAIL_BYTES = 4_096


def _normalize_live_codec(codec: str) -> str:
    """Map portable public codec names to FFmpeg software encoders."""
    if not isinstance(codec, str):
        raise VideoWriterError(f"unsupported live video codec {codec!r}")
    normalized = codec.strip().lower()
    if normalized in _HEVC_CODECS:
        return "libx265"
    if normalized in _H264_CODECS:
        return "libx264"
    supported = ", ".join(sorted(_HEVC_CODECS | _H264_CODECS))
    raise VideoWriterError(f"unsupported live video codec {codec!r}; expected one of: {supported}")


def _encoder_options(codec: str, crf: int, preset: str, threads: int) -> list[str]:
    """Build encoder-specific options without leaking x265 flags into x264."""
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 0:
        raise VideoWriterError("video encoder threads must be zero or a positive integer")
    if not isinstance(preset, str) or not preset.strip():
        raise VideoWriterError("video encoder preset must not be empty")
    options = ["-crf", str(crf), "-preset", preset]
    if threads:
        if codec == "libx265":
            # x265's pool setting is the existing recorder control and avoids
            # spawning its default number of worker pools per camera.
            options.extend(["-x265-params", f"pools={threads}"])
        else:
            options.extend(["-threads", str(threads)])
    return options


def partial_video_path(path: str | Path) -> Path:
    """Return the same-directory temporary path used for an atomic MP4 commit."""
    final_path = Path(path)
    return final_path.with_suffix(".partial.mp4")


def _tool_or_error(tool: str, error_type: type[VideoWriterError]) -> None:
    if shutil.which(tool) is None:
        raise error_type(f"{tool} is not installed or not on PATH")


def _tail_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value[-_STDERR_TAIL_BYTES:].strip()


def _parse_fraction(value: object) -> float | None:
    """Parse ffprobe's ``numerator/denominator`` frame-rate form."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text == "N/A":
        return None
    try:
        numerator, denominator = text.split("/", maxsplit=1)
        parsed_numerator = int(numerator)
        parsed_denominator = int(denominator)
    except (TypeError, ValueError):
        return None
    if parsed_numerator <= 0 or parsed_denominator <= 0:
        return None
    return parsed_numerator / parsed_denominator


def _parse_optional_frame_count(value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text == "N/A":
        return None
    try:
        parsed = int(text)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def probe_video_metadata(path: str | Path) -> VideoMetadata:
    """Decode metadata for the first video stream with an exact count if available.

    The command uses ``-count_frames`` so ``nb_read_frames`` represents the
    frames ffprobe actually decoded. Container-level ``nb_frames`` is used
    only as a fallback, because it is unavailable for some otherwise valid
    MP4 files.
    """
    video_path = Path(path)
    if not video_path.is_file():
        raise VideoProbeError(f"video does not exist or is not a regular file: {video_path}")
    _tool_or_error("ffprobe", VideoProbeError)
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=codec_name,width,height,pix_fmt,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise VideoProbeError(f"could not run ffprobe for {video_path}: {exc}") from exc
    if completed.returncode != 0:
        detail = _tail_text(completed.stderr)
        suffix = f": {detail}" if detail else ""
        raise VideoProbeError(f"ffprobe failed for {video_path}{suffix}")
    try:
        streams = json.loads(completed.stdout).get("streams", [])
    except (TypeError, json.JSONDecodeError) as exc:
        raise VideoProbeError(f"ffprobe returned invalid JSON for {video_path}") from exc
    if not isinstance(streams, list) or not streams:
        raise VideoProbeError(f"ffprobe found no video stream in {video_path}")
    stream = streams[0]
    if not isinstance(stream, dict):
        raise VideoProbeError(f"ffprobe returned invalid video stream metadata for {video_path}")
    try:
        codec = str(stream["codec_name"]).strip()
        width = int(stream["width"])
        height = int(stream["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise VideoProbeError(f"ffprobe returned incomplete video metadata for {video_path}") from exc
    if not codec or width <= 0 or height <= 0:
        raise VideoProbeError(f"ffprobe returned invalid video metadata for {video_path}")
    frame_count = _parse_optional_frame_count(stream.get("nb_read_frames"))
    if frame_count is None:
        frame_count = _parse_optional_frame_count(stream.get("nb_frames"))
    fps = _parse_fraction(stream.get("avg_frame_rate")) or _parse_fraction(stream.get("r_frame_rate"))
    pixel_format = stream.get("pix_fmt")
    return VideoMetadata(
        path=video_path,
        codec=codec,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        pixel_format=str(pixel_format) if pixel_format not in {None, ""} else None,
    )


def probe_video_frame_count(path: str | Path) -> int:
    """Return the decoded frame count or fail rather than estimating it."""
    metadata = probe_video_metadata(path)
    if metadata.frame_count is None:
        raise VideoProbeError(f"ffprobe did not report a decoded frame count for {metadata.path}")
    return metadata.frame_count


def transcode_to_hevc(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    crf: int,
    preset: str,
    threads: int = 0,
    overwrite: bool = False,
    preserve_timestamps: bool = False,
) -> VideoTranscodeResult:
    """Convert one completed intermediate MP4 to a verified, atomic HEVC MP4.

    No ``-r`` option is supplied and ``-vsync 0`` asks FFmpeg to pass input
    frames through without cadence-driven duplication or dropping. When
    ``preserve_timestamps`` is true, ``-copyts`` and a microsecond encoder time
    base carry a source-timestamp timeline into the HEVC derivative. The final
    path is only replaced after ffprobe proves that the output's decoded frame
    count exactly equals the intermediate's count. With ``overwrite=True``,
    an existing final is retained until that same atomic replacement. A failed
    run leaves the source and any prior final untouched and keeps any
    ``.partial.mp4`` output for diagnosis.
    """
    source = Path(source_path)
    destination = Path(destination_path)
    if source.resolve(strict=False) == destination.resolve(strict=False):
        raise VideoTranscodeError("intermediate and final video paths must be different")
    if not source.is_file():
        raise VideoTranscodeError(f"intermediate video does not exist or is not a regular file: {source}")
    if destination.exists() and not overwrite:
        raise VideoTranscodeError(f"refusing to overwrite existing final video: {destination}")
    _tool_or_error("ffmpeg", VideoTranscodeError)

    try:
        source_metadata = probe_video_metadata(source)
    except VideoProbeError as exc:
        raise VideoTranscodeError(f"could not inspect intermediate video {source}: {exc}") from exc
    source_frame_count = source_metadata.frame_count
    if source_frame_count is None:
        raise VideoTranscodeError(f"intermediate video has no exact decoded frame count: {source}")
    if source_frame_count <= 0:
        raise VideoTranscodeError(f"refusing to transcode an empty intermediate video: {source}")

    try:
        encoder_options = _encoder_options("libx265", crf, preset, threads)
    except VideoWriterError as exc:
        raise VideoTranscodeError(str(exc)) from exc
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise VideoTranscodeError(f"could not create final video directory {destination.parent}: {exc}") from exc
    partial = partial_video_path(destination)
    try:
        partial.unlink(missing_ok=True)
    except OSError as exc:
        raise VideoTranscodeError(f"could not clear stale partial video {partial}: {exc}") from exc

    timestamp_input_options = ["-copyts"] if preserve_timestamps else []
    timestamp_output_options = ["-enc_time_base", "1:1000000"] if preserve_timestamps else []
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-n",
        *timestamp_input_options,
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        # FFmpeg 4.4 on the deployment host calls this passthrough mode
        # ``-vsync 0``. Do not add ``-r`` or a CFR conversion here.
        "-vsync",
        "0",
        *timestamp_output_options,
        "-c:v",
        "libx265",
        "-tag:v",
        "hvc1",
        "-pix_fmt",
        "yuv420p",
        *encoder_options,
        "-movflags",
        "+faststart",
        str(partial),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise VideoTranscodeError(f"could not run ffmpeg for {source}: {exc}") from exc
    if completed.returncode != 0:
        detail = _tail_text(completed.stderr)
        suffix = f": {detail}" if detail else ""
        raise VideoTranscodeError(f"ffmpeg failed to transcode {source} to {destination}{suffix}")
    if not partial.is_file():
        raise VideoTranscodeError(f"ffmpeg reported success but did not create {partial}")
    try:
        output_metadata = probe_video_metadata(partial)
    except VideoProbeError as exc:
        raise VideoTranscodeError(f"could not inspect transcoded video {partial}: {exc}") from exc
    if output_metadata.codec != "hevc":
        raise VideoTranscodeError(
            f"transcoded video {partial} has codec {output_metadata.codec!r}, expected 'hevc'"
        )
    if output_metadata.frame_count != source_frame_count:
        raise VideoTranscodeError(
            f"frame-count mismatch after transcode for {source}: "
            f"intermediate={source_frame_count}, final={output_metadata.frame_count}"
        )
    if (output_metadata.width, output_metadata.height) != (source_metadata.width, source_metadata.height):
        raise VideoTranscodeError(
            f"resolution changed after transcode for {source}: "
            f"intermediate={source_metadata.width}x{source_metadata.height}, "
            f"final={output_metadata.width}x{output_metadata.height}"
        )
    try:
        partial.replace(destination)
    except OSError as exc:
        raise VideoTranscodeError(f"could not atomically publish final video {destination}: {exc}") from exc
    return VideoTranscodeResult(
        source=source_metadata,
        output=VideoMetadata(
            path=destination,
            codec=output_metadata.codec,
            width=output_metadata.width,
            height=output_metadata.height,
            fps=output_metadata.fps,
            frame_count=output_metadata.frame_count,
            pixel_format=output_metadata.pixel_format,
        ),
    )


def _validated_duplicate_after_indices(values: Iterable[int], source_frame_count: int) -> tuple[int, ...]:
    """Return unique, ordered zero-based source indices that are safe to duplicate."""
    try:
        requested = tuple(values)
    except TypeError as exc:
        raise VideoTranscodeError("duplicate_after_indices must be an iterable of integers") from exc
    normalized: list[int] = []
    for value in requested:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise VideoTranscodeError("duplicate_after_indices must contain only integers")
        normalized.append(int(value))
    if len(set(normalized)) != len(normalized):
        raise VideoTranscodeError("duplicate_after_indices must not contain duplicate indices")
    invalid = sorted(index for index in normalized if index < 0 or index >= source_frame_count)
    if invalid:
        raise VideoTranscodeError(
            "duplicate_after_indices contains out-of-range source indices: "
            + ", ".join(str(index) for index in invalid)
            + f"; valid range is 0..{source_frame_count - 1}"
        )
    return tuple(sorted(normalized))


def _read_raw_frame(stream: BinaryIO, frame_size: int) -> bytes | None:
    """Read one complete raw frame, distinguishing clean EOF from truncation."""
    data = bytearray()
    while len(data) < frame_size:
        chunk = stream.read(frame_size - len(data))
        if not chunk:
            if not data:
                return None
            raise VideoTranscodeError(
                f"ffmpeg decoder returned a truncated raw frame ({len(data)}/{frame_size} bytes)"
            )
        data.extend(chunk)
    return bytes(data)


def _write_raw_frame(stream: BinaryIO, frame: bytes) -> None:
    """Write one complete raw frame even when the pipe accepts a short write."""
    remaining = memoryview(frame)
    while remaining:
        written = stream.write(remaining)
        if written is None or written <= 0:
            raise BrokenPipeError("ffmpeg encoder pipe stopped accepting raw frames")
        remaining = remaining[written:]


def _temporary_file_tail(stream: BinaryIO) -> str:
    """Read only the bounded tail of an FFmpeg stderr temporary file."""
    try:
        stream.flush()
        stream.seek(0, os.SEEK_END)
        length = stream.tell()
        stream.seek(max(0, length - _STDERR_TAIL_BYTES))
        return _tail_text(stream.read())
    except OSError:
        return ""


def _terminate_process(process: subprocess.Popen[bytes] | None) -> None:
    """Best-effort cleanup for one half of the streaming FFmpeg pipeline."""
    if process is None or process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=5.0)
    except (OSError, subprocess.TimeoutExpired):
        try:
            process.kill()
            process.wait(timeout=5.0)
        except (OSError, subprocess.TimeoutExpired):
            pass


def _stream_frames_with_duplicates(
    decoder_command: list[str],
    encoder_command: list[str],
    *,
    frame_size: int,
    duplicate_after_indices: tuple[int, ...],
    encoder_label: str = "HEVC",
) -> tuple[int, int]:
    """Stream decoded frames through Python while retaining at most one frame."""
    decoder: subprocess.Popen[bytes] | None = None
    encoder: subprocess.Popen[bytes] | None = None
    duplicate_indices = set(duplicate_after_indices)
    with tempfile.TemporaryFile(mode="w+b") as decoder_stderr, tempfile.TemporaryFile(mode="w+b") as encoder_stderr:
        try:
            encoder = subprocess.Popen(
                encoder_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=encoder_stderr,
                bufsize=0,
            )
            decoder = subprocess.Popen(
                decoder_command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=decoder_stderr,
                bufsize=0,
            )
            if decoder.stdout is None or encoder.stdin is None:
                raise VideoTranscodeError("ffmpeg did not expose the required raw-video pipes")

            source_frames = 0
            output_frames = 0
            while True:
                frame = _read_raw_frame(decoder.stdout, frame_size)
                if frame is None:
                    break
                source_index = source_frames
                _write_raw_frame(encoder.stdin, frame)
                output_frames += 1
                if source_index in duplicate_indices:
                    _write_raw_frame(encoder.stdin, frame)
                    output_frames += 1
                source_frames += 1

            decoder.stdout.close()
            encoder.stdin.close()
            decoder_returncode = decoder.wait()
            encoder_returncode = encoder.wait()
            if decoder_returncode != 0:
                detail = _temporary_file_tail(decoder_stderr)
                suffix = f": {detail}" if detail else ""
                raise VideoTranscodeError(f"ffmpeg decoder failed{suffix}")
            if encoder_returncode != 0:
                detail = _temporary_file_tail(encoder_stderr)
                suffix = f": {detail}" if detail else ""
                raise VideoTranscodeError(f"ffmpeg {encoder_label} encoder failed{suffix}")
            return source_frames, output_frames
        except VideoTranscodeError:
            raise
        except (BrokenPipeError, OSError) as exc:
            decoder_detail = _temporary_file_tail(decoder_stderr)
            encoder_detail = _temporary_file_tail(encoder_stderr)
            details = "; ".join(detail for detail in (decoder_detail, encoder_detail) if detail)
            suffix = f": {details}" if details else ""
            raise VideoTranscodeError(f"ffmpeg raw-video pipeline failed{suffix}") from exc
        finally:
            if decoder is not None and decoder.stdout is not None and not decoder.stdout.closed:
                decoder.stdout.close()
            if encoder is not None and encoder.stdin is not None and not encoder.stdin.closed:
                try:
                    encoder.stdin.close()
                except OSError:
                    pass
            _terminate_process(decoder)
            _terminate_process(encoder)


def _transcode_with_frame_duplicates(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    duplicate_after_indices: Iterable[int],
    crf: int,
    preset: str,
    threads: int = 0,
    overwrite: bool = False,
    codec: str,
    expected_codec: str,
    codec_tag: str | None,
    codec_label: str,
) -> VideoFrameDuplicateResult:
    """Create an atomic codec derivative by duplicating selected source frames once.

    Indices refer to decoded source frames and are zero-based. Each selected
    source frame is written normally and then written once more immediately
    after itself. The source is opened read-only, Python retains only one raw
    BGR frame, and the destination is published only after exact frame-count,
    resolution, and codec validation.
    """
    source = Path(source_path)
    destination = Path(destination_path)
    if source.resolve(strict=False) == destination.resolve(strict=False):
        raise VideoTranscodeError("source and derivative video paths must be different")
    if not source.is_file():
        raise VideoTranscodeError(f"source video does not exist or is not a regular file: {source}")
    if destination.exists() and not overwrite:
        raise VideoTranscodeError(f"refusing to overwrite existing derivative video: {destination}")
    _tool_or_error("ffmpeg", VideoTranscodeError)

    try:
        source_metadata = probe_video_metadata(source)
    except VideoProbeError as exc:
        raise VideoTranscodeError(f"could not inspect source video {source}: {exc}") from exc
    source_frame_count = source_metadata.frame_count
    if source_frame_count is None:
        raise VideoTranscodeError(f"source video has no exact decoded frame count: {source}")
    if source_frame_count <= 0:
        raise VideoTranscodeError(f"refusing to repair an empty source video: {source}")
    if source_metadata.fps is None or not math.isfinite(source_metadata.fps) or source_metadata.fps <= 0:
        raise VideoTranscodeError(f"source video has no valid frame rate: {source}")
    indices = _validated_duplicate_after_indices(duplicate_after_indices, source_frame_count)
    expected_output_count = source_frame_count + len(indices)

    try:
        encoder_options = _encoder_options(codec, crf, preset, threads)
    except VideoWriterError as exc:
        raise VideoTranscodeError(str(exc)) from exc
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise VideoTranscodeError(f"could not create derivative video directory {destination.parent}: {exc}") from exc
    partial = partial_video_path(destination)
    if source.resolve(strict=False) == partial.resolve(strict=False):
        raise VideoTranscodeError("source video path conflicts with the derivative partial path")
    if partial.exists():
        try:
            if source.samefile(partial):
                raise VideoTranscodeError("source video is the same file as the derivative partial path")
        except OSError as exc:
            raise VideoTranscodeError(f"could not compare source and partial video paths: {exc}") from exc
    try:
        partial.unlink(missing_ok=True)
    except OSError as exc:
        raise VideoTranscodeError(f"could not clear stale partial video {partial}: {exc}") from exc

    decoder_command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-vsync",
        "0",
        "-pix_fmt",
        "bgr24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    encoder_command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-n",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        f"{source_metadata.width}x{source_metadata.height}",
        "-framerate",
        f"{source_metadata.fps:.12g}",
        "-i",
        "pipe:0",
        "-an",
        "-sn",
        "-dn",
        "-vsync",
        "0",
        "-c:v",
        codec,
        *(["-tag:v", codec_tag] if codec_tag else []),
        "-pix_fmt",
        "yuv420p",
        *encoder_options,
        "-movflags",
        "+faststart",
        str(partial),
    ]
    decoded_count, streamed_output_count = _stream_frames_with_duplicates(
        decoder_command,
        encoder_command,
        frame_size=source_metadata.width * source_metadata.height * 3,
        duplicate_after_indices=indices,
        encoder_label=codec_label,
    )
    if decoded_count != source_frame_count:
        raise VideoTranscodeError(
            f"decoded source frame-count changed while repairing {source}: "
            f"probed={source_frame_count}, decoded={decoded_count}"
        )
    if streamed_output_count != expected_output_count:
        raise VideoTranscodeError(
            f"internal repaired frame-count mismatch for {source}: "
            f"expected={expected_output_count}, streamed={streamed_output_count}"
        )
    if not partial.is_file():
        raise VideoTranscodeError(f"ffmpeg reported success but did not create {partial}")
    try:
        output_metadata = probe_video_metadata(partial)
    except VideoProbeError as exc:
        raise VideoTranscodeError(f"could not inspect repaired video {partial}: {exc}") from exc
    if output_metadata.codec != expected_codec:
        raise VideoTranscodeError(
            f"repaired {codec_label} {partial} has codec {output_metadata.codec!r}, "
            f"expected {expected_codec!r}"
        )
    if output_metadata.frame_count != expected_output_count:
        raise VideoTranscodeError(
            f"frame-count mismatch after repair for {source}: "
            f"expected={expected_output_count}, final={output_metadata.frame_count}"
        )
    if (output_metadata.width, output_metadata.height) != (source_metadata.width, source_metadata.height):
        raise VideoTranscodeError(
            f"resolution changed after repair for {source}: "
            f"source={source_metadata.width}x{source_metadata.height}, "
            f"final={output_metadata.width}x{output_metadata.height}"
        )
    try:
        partial.replace(destination)
    except OSError as exc:
        raise VideoTranscodeError(f"could not atomically publish repaired video {destination}: {exc}") from exc
    return VideoFrameDuplicateResult(
        source=source_metadata,
        output=VideoMetadata(
            path=destination,
            codec=output_metadata.codec,
            width=output_metadata.width,
            height=output_metadata.height,
            fps=output_metadata.fps,
            frame_count=output_metadata.frame_count,
            pixel_format=output_metadata.pixel_format,
        ),
        duplicate_after_indices=indices,
    )


def transcode_to_hevc_with_frame_duplicates(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    duplicate_after_indices: Iterable[int],
    crf: int,
    preset: str,
    threads: int = 0,
    overwrite: bool = False,
) -> VideoFrameDuplicateResult:
    """Create a verified HEVC derivative with selected previous-frame holds."""
    return _transcode_with_frame_duplicates(
        source_path,
        destination_path,
        duplicate_after_indices=duplicate_after_indices,
        crf=crf,
        preset=preset,
        threads=threads,
        overwrite=overwrite,
        codec="libx265",
        expected_codec="hevc",
        codec_tag="hvc1",
        codec_label="HEVC",
    )


def transcode_to_h264_with_frame_duplicates(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    duplicate_after_indices: Iterable[int],
    crf: int,
    preset: str,
    threads: int = 0,
    overwrite: bool = False,
) -> VideoFrameDuplicateResult:
    """Create a verified H.264 derivative with selected previous-frame holds."""
    return _transcode_with_frame_duplicates(
        source_path,
        destination_path,
        duplicate_after_indices=duplicate_after_indices,
        crf=crf,
        preset=preset,
        threads=threads,
        overwrite=overwrite,
        codec="libx264",
        expected_codec="h264",
        codec_tag="avc1",
        codec_label="H.264",
    )


class AsyncFfmpegWriter:
    """Encode raw camera frames to a temporary MP4 without blocking callbacks.

    The positional constructor remains compatible with the original recorder.
    ``libx265``/``hevc`` produce HEVC as before; ``libx264``/``h264`` are
    supported for lightweight live intermediates and use the appropriate
    FFmpeg thread option rather than x265-specific options.
    """

    _STOP = object()
    _DRAIN_TIMEOUT_S = 60.0
    _ABORT_JOIN_TIMEOUT_S = 5.0
    _STDERR_TAIL_BYTES = 4_096

    def __init__(
        self,
        path: Path,
        width: int,
        height: int,
        fps: float,
        codec: str,
        crf: int,
        preset: str = "medium",
        threads: int = 0,
        queue_size: int = 120,
        input_pixel_format: str = "bgr24",
        encoder_cpu_set: str | None = None,
        record_source_timestamps: bool = False,
    ) -> None:
        _tool_or_error("ffmpeg", VideoWriterError)
        self.path = Path(path)
        self.partial_path = partial_video_path(self.path)
        self.width = width
        self.height = height
        # Keep the original constructor values intact for the fixed-rate path.
        # The source-timeline writer uses the same values when it rebuilds the
        # presentation timestamps after the live intermediate is complete.
        self.fps = fps
        self.crf = crf
        self.preset = preset
        self.threads = threads
        self.queue_size = queue_size
        self.codec = _normalize_live_codec(codec)
        self._record_source_timestamps = bool(record_source_timestamps)
        self._timestamp_sidecar = self.path.with_name(f".{self.path.name}.timestamps")
        self._timestamp_stream: BinaryIO | None = None
        try:
            self.encoder_cpu_set = (
                normalize_cpu_set(encoder_cpu_set, "encoder_cpu_set") if encoder_cpu_set is not None else None
            )
        except ValueError as exc:
            raise VideoWriterError(str(exc)) from exc
        if input_pixel_format not in {"bgr24", "yuyv422"}:
            raise VideoWriterError(f"unsupported raw-video pixel format: {input_pixel_format}")
        self.input_pixel_format = input_pixel_format
        self._queue: queue.Queue[object] = queue.Queue(maxsize=queue_size)
        # This lock protects only small in-memory counters and error state.
        # It is never held while writing to ffmpeg, waiting for a process, or
        # reading stderr, so camera callbacks cannot be held behind I/O.
        self._diagnostics_lock = threading.Lock()
        self._error: Optional[BaseException] = None
        self._closed = False
        self._last_process_returncode: int | None = None
        self._stderr_tail: str | None = None
        self.accepted_frames = 0
        # This is the number of frames successfully written into ffmpeg's
        # stdin pipe, not an estimate of frames already compressed by its
        # internal encoder. The distinction matters while diagnosing a queue
        # that is growing under thermal or scheduler pressure.
        self.frames_submitted_to_ffmpeg = 0
        self.dropped_frames = 0
        self.max_queue_depth = 0
        self.max_pipe_write_ms = 0.0
        encoder_options = _encoder_options(self.codec, self.crf, self.preset, self.threads)
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            self.input_pixel_format,
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(self.fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            self.codec,
            "-tag:v",
            "hvc1" if self.codec == "libx265" else "avc1",
            "-pix_fmt",
            "yuv420p",
            *encoder_options,
            "-movflags",
            "+faststart",
            str(self.partial_path),
        ]
        if self.encoder_cpu_set is not None:
            _tool_or_error("taskset", VideoWriterError)
            command = ["taskset", "--cpu-list", self.encoder_cpu_set, *command]
        if self._record_source_timestamps:
            try:
                self._timestamp_sidecar.unlink(missing_ok=True)
                self._timestamp_stream = self._timestamp_sidecar.open("wb")
            except OSError as exc:
                raise VideoWriterError(
                    f"could not create source-timestamp sidecar {self._timestamp_sidecar}: {exc}"
                ) from exc
        try:
            self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        except BaseException:
            if self._timestamp_stream is not None:
                self._timestamp_stream.close()
                self._timestamp_stream = None
            if self._record_source_timestamps:
                self._timestamp_sidecar.unlink(missing_ok=True)
            raise
        self._thread = threading.Thread(target=self._write_loop, name=f"video-writer-{self.path.stem}", daemon=True)
        self._thread.start()

    def enqueue(
        self,
        frame_bgr: np.ndarray,
        *,
        copy_frame: bool = True,
        timestamp_ns: int | None = None,
    ) -> bool:
        """Queue a frame. A rejected frame is never represented in the MCAP.

        Callers may transfer an owned, C-contiguous frame with
        ``copy_frame=False``. The default remains defensive for generic camera
        backends and tests that may reuse their input buffer.
        """
        with self._diagnostics_lock:
            closed = self._closed
            error = self._error
        if closed:
            raise VideoWriterError("cannot enqueue into a closed video writer")
        if error is not None:
            raise VideoWriterError(f"video writer failed for {self.path}") from error
        queued_timestamp_ns: int | None = None
        if self._record_source_timestamps:
            if not isinstance(timestamp_ns, Integral):
                raise VideoWriterError(f"{self.path.name} requires a source timestamp for every frame")
            queued_timestamp_ns = int(timestamp_ns)
        elif timestamp_ns is not None:
            raise VideoWriterError(f"{self.path.name} does not accept source timestamps in fixed mode")
        frame = np.asarray(frame_bgr)
        expected_shape = (
            (self.height, self.width, 3) if self.input_pixel_format == "bgr24" else (self.height, self.width, 2)
        )
        if frame.shape != expected_shape or frame.dtype != np.uint8:
            raise VideoWriterError(
                f"{self.path.name} expected uint8 {self.input_pixel_format} {self.width}x{self.height}, "
                f"got {frame.shape} {frame.dtype}"
            )
        if copy_frame:
            queued_frame = np.ascontiguousarray(frame).copy()
        elif not frame.flags.c_contiguous:
            raise VideoWriterError(f"{self.path.name} cannot transfer a non-contiguous frame")
        else:
            queued_frame = frame
        try:
            queued_item: object = (
                (queued_frame, queued_timestamp_ns) if self._record_source_timestamps else queued_frame
            )
            self._queue.put_nowait(queued_item)
        except queue.Full:
            with self._diagnostics_lock:
                self.dropped_frames += 1
            return False
        queue_depth = self._queue.qsize()
        with self._diagnostics_lock:
            self.accepted_frames += 1
            self.max_queue_depth = max(self.max_queue_depth, queue_depth)
        return True

    def _write_loop(self) -> None:
        try:
            while True:
                frame = self._queue.get()
                try:
                    if frame is self._STOP:
                        return
                    timestamp_ns: int | None = None
                    if self._record_source_timestamps:
                        assert isinstance(frame, tuple) and len(frame) == 2
                        frame, timestamp_ns = frame
                        assert isinstance(timestamp_ns, int)
                    assert isinstance(frame, np.ndarray)
                    assert self._process.stdin is not None
                    started_at = time.monotonic_ns()
                    self._process.stdin.write(memoryview(frame).cast("B"))
                    if self._record_source_timestamps:
                        assert timestamp_ns is not None
                        assert self._timestamp_stream is not None
                        self._timestamp_stream.write(struct.pack("<q", timestamp_ns))
                    with self._diagnostics_lock:
                        self.frames_submitted_to_ffmpeg += 1
                        self.max_pipe_write_ms = max(
                            self.max_pipe_write_ms,
                            (time.monotonic_ns() - started_at) / 1_000_000,
                        )
                finally:
                    self._queue.task_done()
        except BaseException as exc:
            with self._diagnostics_lock:
                self._error = exc

    @staticmethod
    def _error_text(error: BaseException | None) -> str | None:
        return None if error is None else f"{type(error).__name__}: {error}"

    def _remember_process_result(self, returncode: int, stderr: bytes | str) -> None:
        if isinstance(stderr, bytes):
            tail = stderr[-self._STDERR_TAIL_BYTES :].decode("utf-8", errors="replace")
        else:
            tail = stderr[-self._STDERR_TAIL_BYTES :]
        with self._diagnostics_lock:
            self._last_process_returncode = returncode
            self._stderr_tail = tail or None

    def _effective_encoder_cpu_set(self) -> list[int] | None:
        """Return the child affinity when the host can report it without blocking."""
        get_affinity = getattr(os, "sched_getaffinity", None)
        pid = getattr(self._process, "pid", None)
        if not callable(get_affinity) or not isinstance(pid, int):
            return None
        try:
            return sorted(get_affinity(pid))
        except (OSError, ValueError):
            return None

    def diagnostic_snapshot(self) -> dict[str, object]:
        """Return a non-blocking, JSON-safe view of encoder health.

        ``poll`` and ``qsize`` do not wait for ffmpeg or the encoder thread.
        stderr is deliberately never read here; it is retained only after the
        process has exited in :meth:`_wait_for_process`.
        """
        queue_depth = self._queue.qsize()
        process_returncode = self._process.poll()
        with self._diagnostics_lock:
            if process_returncode is not None:
                self._last_process_returncode = process_returncode
            return {
                "snapshot_monotonic_ns": time.monotonic_ns(),
                "queue_depth": queue_depth,
                "queue_capacity": self.queue_size,
                "accepted_frames": self.accepted_frames,
                "frames_submitted_to_ffmpeg": self.frames_submitted_to_ffmpeg,
                "dropped_frames": self.dropped_frames,
                "max_queue_depth": self.max_queue_depth,
                "queue_fill_ratio": round(queue_depth / self.queue_size, 6) if self.queue_size else None,
                "max_pipe_write_ms": self.max_pipe_write_ms,
                "ffmpeg_pid": getattr(self._process, "pid", None),
                "requested_encoder_cpu_set": self.encoder_cpu_set,
                "effective_encoder_cpu_set": self._effective_encoder_cpu_set(),
                "encoder_affinity_mode": "taskset" if self.encoder_cpu_set is not None else "inherited",
                "process_returncode": (
                    process_returncode if process_returncode is not None else self._last_process_returncode
                ),
                "writer_thread_alive": self._thread.is_alive(),
                "closed": self._closed,
                "writer_error": self._error_text(self._error),
                "stderr_tail": self._stderr_tail,
            }

    def _discard_queued_frames(self) -> None:
        """Release queued frames after a failed encoder no longer has a consumer."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return
            else:
                self._queue.task_done()

    def _request_stop(self, timeout_s: float) -> bool:
        """Place the writer sentinel without waiting forever for a dead consumer."""
        deadline = time.monotonic() + timeout_s
        while self._thread.is_alive():
            try:
                self._queue.put_nowait(self._STOP)
                return True
            except queue.Full:
                with self._diagnostics_lock:
                    error = self._error
                if error is not None:
                    self._discard_queued_frames()
                elif time.monotonic() >= deadline:
                    return False
                else:
                    time.sleep(0.01)
        return True

    def _terminate_process(self) -> None:
        if self._process.poll() is None:
            self._process.kill()

    def _close_stdin(self) -> None:
        if self._process.stdin is None:
            return
        try:
            self._process.stdin.close()
        except OSError:
            # A failed encoder commonly closes stdin before the writer can.
            pass

    def _wait_for_process(self, timeout_s: float) -> tuple[int, bytes]:
        try:
            result = self._process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self._terminate_process()
            result = self._process.wait()
        stderr = self._process.stderr.read() if self._process.stderr is not None else b""
        self._remember_process_result(result, stderr)
        return result, stderr

    def close(self, commit: bool) -> None:
        with self._diagnostics_lock:
            if self._closed:
                return
            self._closed = True
        if not commit:
            # A discarded episode must not wait for queued H.265 encoding.
            self._discard_queued_frames()
            self._terminate_process()
            self._request_stop(self._ABORT_JOIN_TIMEOUT_S)
            self._thread.join(self._ABORT_JOIN_TIMEOUT_S)
            self._close_stdin()
            self._wait_for_process(self._ABORT_JOIN_TIMEOUT_S)
            if self.partial_path.exists():
                self.partial_path.unlink()
            return

        with self._diagnostics_lock:
            error = self._error
        if error is not None:
            self._discard_queued_frames()
            self._terminate_process()
            self._request_stop(0.0)
        elif not self._request_stop(self._DRAIN_TIMEOUT_S):
            self._terminate_process()
            self._discard_queued_frames()
            self._request_stop(self._ABORT_JOIN_TIMEOUT_S)
        self._thread.join(self._DRAIN_TIMEOUT_S)
        if self._thread.is_alive():
            self._terminate_process()
            self._discard_queued_frames()
            self._request_stop(self._ABORT_JOIN_TIMEOUT_S)
            self._thread.join(self._ABORT_JOIN_TIMEOUT_S)
        self._close_stdin()
        result, stderr = self._wait_for_process(self._DRAIN_TIMEOUT_S)
        with self._diagnostics_lock:
            error = self._error
        if error is not None:
            if self.partial_path.exists():
                self.partial_path.unlink()
            raise VideoWriterError(f"video writer failed for {self.path}") from error
        if self._thread.is_alive():
            if self.partial_path.exists():
                self.partial_path.unlink()
            raise VideoWriterError(f"video writer did not stop for {self.path}")
        if result != 0:
            if self.partial_path.exists():
                self.partial_path.unlink()
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise VideoWriterError(f"ffmpeg failed for {self.path}: {detail}")
        self.partial_path.replace(self.path)


def close_video_writers(
    writers: Iterable[object],
    *,
    commit: bool,
    parallel: bool = False,
) -> list[BaseException]:
    """Close a group of video writers and return every close error.

    Source-timestamp writers do a decode/re-encode pass during ``close``.
    Running those independent camera jobs together reduces finalization time
    without allowing one failed camera to skip cleanup for the other writers.
    The default remains serial for the existing fixed-rate path.
    """
    writer_list = tuple(writers)
    if not writer_list:
        return []

    def close_one(writer: object, writer_commit: bool = commit) -> None:
        close = getattr(writer, "close")
        close(commit=writer_commit)

    errors: list[BaseException] = []
    if parallel and len(writer_list) > 1:
        with ThreadPoolExecutor(
            max_workers=len(writer_list),
            thread_name_prefix="video-finalizer",
        ) as executor:
            futures = [executor.submit(close_one, writer) for writer in writer_list]
            for future in futures:
                try:
                    future.result()
                except BaseException as exc:
                    errors.append(exc)
        return errors

    for writer in writer_list:
        try:
            # Preserve the original fixed-rate behavior: once one writer has
            # failed, later writers are closed in discard mode.
            close_one(writer, commit and not errors)
        except BaseException as exc:
            errors.append(exc)
    return errors


class TimestampedVideoWriter(AsyncFfmpegWriter):
    """Write every accepted frame with a variable source-time MP4 timeline.

    Frames are encoded with the existing lightweight live path first. On a
    successful close, the completed intermediate is normally remuxed with
    each frame's MCAP capture timestamp as its presentation timestamp. This
    keeps the compressed H.264 packets and avoids a second encode. Codecs or
    packet layouts that cannot be remuxed safely fall back to the validated
    decode/re-encode path.
    """

    _TIMESTAMP_TIME_BASE = Fraction(1, 1_000_000)

    def __init__(
        self,
        path: Path,
        width: int,
        height: int,
        fps: float,
        codec: str,
        crf: int,
        preset: str = "medium",
        threads: int = 0,
        queue_size: int = 120,
        input_pixel_format: str = "bgr24",
        encoder_cpu_set: str | None = None,
    ) -> None:
        try:
            import av  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on optional installation
            raise VideoWriterError(
                "source-timestamp video mode requires PyAV; install the optional `timeline` extra"
            ) from exc
        super().__init__(
            path,
            width,
            height,
            fps,
            codec,
            crf,
            preset,
            threads,
            queue_size=queue_size,
            input_pixel_format=input_pixel_format,
            encoder_cpu_set=encoder_cpu_set,
            record_source_timestamps=True,
        )
        self._timeline_origin_ns: int | None = None
        self._timeline_rewritten = False
        self._timeline_finalized = False
        self._timeline_rewrite_mode: str | None = None

    def set_timeline_origin_ns(self, timestamp_ns: int) -> None:
        """Set the shared episode origin used by all source-timestamp videos."""
        if not isinstance(timestamp_ns, Integral):
            raise VideoWriterError("timeline origin must be an integer nanosecond timestamp")
        self._timeline_origin_ns = int(timestamp_ns)

    def _read_source_timestamps(self) -> list[int]:
        try:
            raw = self._timestamp_sidecar.read_bytes()
        except OSError as exc:
            raise VideoWriterError(f"could not read source-timestamp sidecar {self._timestamp_sidecar}: {exc}") from exc
        if len(raw) % struct.calcsize("<q"):
            raise VideoWriterError(f"source-timestamp sidecar is truncated: {self._timestamp_sidecar}")
        return [value[0] for value in struct.iter_unpack("<q", raw)]

    def _expected_timestamps_us(self, timestamps_ns: list[int], origin_ns: int) -> list[int]:
        return [(timestamp_ns - origin_ns) // 1_000 for timestamp_ns in timestamps_ns]

    def _try_remux_with_source_timestamps(
        self,
        timestamps_ns: list[int],
        origin_ns: int,
        rewritten_path: Path,
    ) -> bool:
        """Copy compressed packets while replacing their presentation times.

        The live source encoder is configured as ultrafast H.264, which has
        one packet per frame and no B-frames. We still verify both properties
        at runtime. Returning ``False`` is a controlled request to use the
        slower decode/re-encode fallback for a different codec or layout.
        """
        try:
            import av
        except ImportError:  # pragma: no cover - checked by __init__
            return False

        expected_pts_us = self._expected_timestamps_us(timestamps_ns, origin_ns)
        rewritten_path.unlink(missing_ok=True)
        try:
            with av.open(str(self.path), mode="r") as source_container:
                source_stream = source_container.streams.video[0]
                if int(getattr(source_stream.codec_context, "has_b_frames", 0) or 0) != 0:
                    return False
                source_codec = source_stream.codec_context.codec
                if source_codec is None or source_codec.name not in {"h264", "hevc"}:
                    return False
                with av.open(
                    str(rewritten_path),
                    mode="w",
                    format="mp4",
                    options={"movflags": "+faststart+use_metadata_tags"},
                ) as output_container:
                    output_stream = output_container.add_stream_from_template(source_stream)
                    output_stream.time_base = self._TIMESTAMP_TIME_BASE
                    output_stream.codec_context.time_base = self._TIMESTAMP_TIME_BASE
                    output_container.metadata.update(
                        {
                            "com.yam.video_timeline": "source",
                            "com.yam.timeline_origin_ns": str(origin_ns),
                            "com.yam.timeline_time_base": "1/1000000",
                        }
                    )
                    packet_count = 0
                    for packet in source_container.demux(source_stream):
                        if packet.dts is None:
                            continue
                        if packet_count >= len(expected_pts_us):
                            rewritten_path.unlink(missing_ok=True)
                            return False
                        timestamp_us = expected_pts_us[packet_count]
                        packet.pts = timestamp_us
                        packet.dts = timestamp_us
                        if packet_count + 1 < len(expected_pts_us):
                            packet.duration = max(1, expected_pts_us[packet_count + 1] - timestamp_us)
                        elif packet_count:
                            packet.duration = max(1, timestamp_us - expected_pts_us[packet_count - 1])
                        else:
                            packet.duration = max(1, round(1_000_000 / self.fps))
                        packet.time_base = self._TIMESTAMP_TIME_BASE
                        packet.stream = output_stream
                        output_container.mux(packet)
                        packet_count += 1
                    if packet_count != len(expected_pts_us):
                        rewritten_path.unlink(missing_ok=True)
                        return False
            if not rewritten_path.is_file():
                return False

            # Verify the muxed packet timeline directly. The source encoder is
            # required to have one packet per frame and no B-frames, so packet
            # identity is the relevant invariant here. Decoding every frame
            # just to inspect PTS makes finalization scale with all three video
            # durations and defeats the purpose of packet remuxing.
            with av.open(str(rewritten_path), mode="r") as verification_container:
                verification_stream = verification_container.streams.video[0]
                verified_timestamps: list[tuple[int, int]] = []
                for packet in verification_container.demux(verification_stream):
                    if packet.dts is None or packet.pts is None:
                        continue
                    verified_timestamps.append((int(packet.pts), int(packet.dts)))
            if len(verified_timestamps) != len(expected_pts_us):
                rewritten_path.unlink(missing_ok=True)
                return False
            # Some MP4 muxers quantize the track's edit-list start to a
            # sub-millisecond boundary. Accept one constant offset, but do
            # not accept any change to the per-frame intervals or ordering.
            start_offset_us = verified_timestamps[0][0] - expected_pts_us[0]
            if abs(start_offset_us) > 1_000 or any(
                pts - expected_pts_us[index] != start_offset_us
                or dts - expected_pts_us[index] != start_offset_us
                for index, (pts, dts) in enumerate(verified_timestamps)
            ):
                rewritten_path.unlink(missing_ok=True)
                return False
            rewritten_path.replace(self.path)
            return True
        except Exception:
            rewritten_path.unlink(missing_ok=True)
            return False

    def _rewrite_with_source_timestamps(self) -> None:
        timestamps_ns = self._read_source_timestamps()
        if not timestamps_ns:
            raise VideoWriterError(f"cannot build a source-timestamp video with no frames: {self.path}")
        origin_ns = self._timeline_origin_ns if self._timeline_origin_ns is not None else timestamps_ns[0]
        if origin_ns > timestamps_ns[0]:
            raise VideoWriterError(
                f"timeline origin {origin_ns} is later than the first {self.path.name} frame {timestamps_ns[0]}"
            )
        if any(current <= previous for previous, current in zip(timestamps_ns, timestamps_ns[1:])):
            raise VideoWriterError(f"source timestamps are not strictly increasing for {self.path}")

        try:
            import av
        except ImportError as exc:  # pragma: no cover - exercised on hosts without the optional extra
            raise VideoWriterError(
                "source-timestamp video mode requires PyAV; install it with `uv sync --extra timeline`"
            ) from exc

        rewritten_path = self.path.with_name(f".{self.path.stem}.source-timeline.partial.mp4")
        rewritten_path.unlink(missing_ok=True)
        if self._try_remux_with_source_timestamps(timestamps_ns, origin_ns, rewritten_path):
            self._timeline_rewrite_mode = "packet-remux"
            return

        self._timeline_rewrite_mode = "decode-reencode"
        decoded_count = 0
        timestamp_index = 0
        try:
            with av.open(str(self.path), mode="r") as source_container:
                source_stream = source_container.streams.video[0]
                with av.open(
                    str(rewritten_path),
                    mode="w",
                    format="mp4",
                    options={"movflags": "+faststart+use_metadata_tags"},
                ) as output_container:
                    output_stream = output_container.add_stream(self.codec, rate=Fraction(str(self.fps)))
                    output_stream.width = int(source_stream.codec_context.width or self.width)
                    output_stream.height = int(source_stream.codec_context.height or self.height)
                    output_stream.pix_fmt = "yuv420p"
                    output_stream.time_base = self._TIMESTAMP_TIME_BASE
                    # PyAV otherwise leaves the encoder at its default 1/24 s
                    # time base and silently quantizes the source timestamps.
                    output_stream.codec_context.time_base = self._TIMESTAMP_TIME_BASE
                    output_stream.codec_context.options = {
                        "crf": str(self.crf),
                        "preset": self.preset,
                    }
                    if self.threads > 0:
                        output_stream.codec_context.thread_count = self.threads
                    output_container.metadata.update(
                        {
                            "com.yam.video_timeline": "source",
                            "com.yam.timeline_origin_ns": str(origin_ns),
                            "com.yam.timeline_time_base": "1/1000000",
                        }
                    )
                    for decoded_frame in source_container.decode(source_stream):
                        if timestamp_index >= len(timestamps_ns):
                            raise VideoWriterError(
                                f"decoded more frames than source timestamps for {self.path}: "
                                f"decoded={decoded_count + 1}, timestamps={len(timestamps_ns)}"
                            )
                        timestamp_us = (timestamps_ns[timestamp_index] - origin_ns) // 1_000
                        decoded_frame = decoded_frame.reformat(format="yuv420p")
                        decoded_frame.pts = int(timestamp_us)
                        decoded_frame.time_base = self._TIMESTAMP_TIME_BASE
                        for packet in output_stream.encode(decoded_frame):
                            output_container.mux(packet)
                        decoded_count += 1
                        timestamp_index += 1
                    for packet in output_stream.encode(None):
                        output_container.mux(packet)
            if timestamp_index != len(timestamps_ns):
                raise VideoWriterError(
                    f"decoded fewer frames than source timestamps for {self.path}: "
                    f"decoded={timestamp_index}, timestamps={len(timestamps_ns)}"
                )
            if not rewritten_path.is_file():
                raise VideoWriterError(f"PyAV reported success but did not create {rewritten_path}")
            rewritten_count = probe_video_frame_count(rewritten_path)
            if rewritten_count != decoded_count:
                raise VideoWriterError(
                    f"source-timestamp rewrite changed frame count for {self.path}: "
                    f"decoded={decoded_count}, rewritten={rewritten_count}"
                )
            rewritten_path.replace(self.path)
        except VideoWriterError:
            rewritten_path.unlink(missing_ok=True)
            raise
        except Exception as exc:
            rewritten_path.unlink(missing_ok=True)
            raise VideoWriterError(f"could not rewrite {self.path} with source timestamps: {exc}") from exc

    def close(self, commit: bool) -> None:
        if self._timeline_finalized:
            return
        try:
            super().close(commit)
            if not commit:
                return
            if self._timestamp_stream is not None:
                self._timestamp_stream.flush()
                self._timestamp_stream.close()
                self._timestamp_stream = None
            self._rewrite_with_source_timestamps()
            self._timeline_rewritten = True
        finally:
            if self._timestamp_stream is not None:
                self._timestamp_stream.close()
                self._timestamp_stream = None
            self._timestamp_sidecar.unlink(missing_ok=True)
            self._timeline_finalized = True
