"""Small immutable values shared by the recorder, telemetry, and cameras."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class RobotSnapshot:
    """A read-only telemetry sample exported by one i2rt teleoperation process."""

    role: str
    source_timestamp_ns: int
    sequence: int
    position: Tuple[float, ...]
    velocity: Tuple[float, ...]
    effort: Tuple[float, ...]


@dataclass(frozen=True)
class CameraFrame:
    """One video frame plus host and optional device reception timestamps.

    ``image_bgr`` retains its original name for compatibility. Its byte layout
    is described by ``pixel_format``: normal cameras use ``bgr24`` while the
    bandwidth-conscious D435 path can retain packed ``yuyv422`` until ffmpeg.
    ``owns_image`` permits a capture backend to transfer an already-owned,
    contiguous frame directly to the asynchronous encoder without another full
    frame copy. Backends that leave it false retain defensive copy semantics.
    """

    role: str
    timestamp_ns: int
    image_bgr: np.ndarray
    owns_image: bool = False
    device_timestamp_ns: int | None = None
    # Keep ``pixel_format`` in its established sixth positional slot. The
    # diagnostics below were added later and must not reinterpret callers'
    # existing positional arguments.
    pixel_format: str = "bgr24"
    device_frame_number: int | None = None
    device_timestamp_domain: str | None = None
    # ``timestamp_ns`` is the capture timestamp used by the MCAP contract.
    # Keep the host delivery time separately so queue/scheduler latency remains
    # observable when RealSense exposes a trustworthy global capture clock.
    host_timestamp_ns: int | None = None
    capture_timestamp_source: str = "host_receive"


@dataclass(frozen=True)
class CameraTimingSample:
    """Image-free timing evidence retained until an episode is finalized."""

    role: str
    capture_timestamp_ns: int
    host_receive_timestamp_ns: int
    device_timestamp_ns: int | None
    device_frame_number: int | None
    device_timestamp_domain: str | None
    capture_timestamp_source: str
