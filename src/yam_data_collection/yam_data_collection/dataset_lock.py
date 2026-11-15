"""Advisory dataset-level exclusion for capture and CPU-heavy transcoding."""

from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path
from typing import TextIO


class DatasetBusyError(RuntimeError):
    """Another capture or transcode process already owns the dataset."""


class DatasetOperationLock:
    """Hold an exclusive advisory lock while a dataset is actively used.

    A live capture and offline x265 transcode must never run concurrently on
    this host: doing so would recreate the sustained CPU pressure that caused
    the original queue overflows.  ``flock`` is released automatically by the
    kernel if a process crashes, while the small JSON payload helps an operator
    identify a normal competing process.
    """

    def __init__(self, dataset_dir: str | Path, operation: str) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.operation = operation
        self.path = self.dataset_dir / ".yam_data_collection.lock"
        self._stream: TextIO | None = None

    def acquire(self) -> None:
        if self._stream is not None:
            return
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            stream.seek(0)
            detail = stream.read().strip()
            stream.close()
            suffix = f" ({detail})" if detail else ""
            raise DatasetBusyError(
                f"dataset is busy; do not run capture and offline transcode concurrently{suffix}"
            ) from exc
        payload = {"pid": os.getpid(), "operation": self.operation, "started_at_unix_ns": time.time_ns()}
        stream.seek(0)
        stream.truncate()
        json.dump(payload, stream, ensure_ascii=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        self._stream = stream

    def release(self) -> None:
        stream = self._stream
        if stream is None:
            return
        self._stream = None
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()

    def __enter__(self) -> "DatasetOperationLock":
        self.acquire()
        return self

    def __exit__(self, *args: object) -> None:
        self.release()
