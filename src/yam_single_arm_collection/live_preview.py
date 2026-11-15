"""Wrist-only desktop preview using the recorder's bounded frame publisher."""

from __future__ import annotations

import os
import sys
from typing import Any, Mapping

from yam_data_collection.live_preview import (
    LivePreview,
    LivePreviewConfig,
    _CameraUpdate,
    _TkPreviewApplication,
    _drain_latest,
)


class WristCameraPreview(LivePreview):
    def start(self) -> None:
        with self._lifecycle_lock:
            if self._started:
                return
            if self._closed:
                raise RuntimeError("a stopped live preview cannot be restarted")
            if self._context is None:
                import multiprocessing

                self._context = multiprocessing.get_context("spawn")
            self._camera_queues = {
                role: self._context.Queue(maxsize=self.config.queue_size) for role in self.config.camera_roles
            }
            self._stop_event = self._context.Event()
            self._process = self._context.Process(
                target=_run_wrist_preview,
                args=(self.config, self._camera_queues, self._stop_event),
                name="yam-wrist-preview",
                daemon=True,
            )
            try:
                self._process.start()
            except BaseException:
                self._close_queues()
                self._process = None
                self._stop_event = None
                raise
            self._started = True


class _WristPreviewApplication(_TkPreviewApplication):
    def _build(self) -> None:
        self._root.title(self._config.window_title)
        self._root.configure(background="#15171a")
        self._root.protocol("WM_DELETE_WINDOW", self.close)
        role = self._config.camera_roles[0]
        panel = self._tk.Frame(self._root, background="#202328")
        panel.pack(padx=8, pady=8)
        self._tk.Label(
            panel,
            text=f"{role.replace('_', ' ').title()} (wrist)",
            foreground="#f1f3f5",
            background="#202328",
            font=("TkDefaultFont", 12, "bold"),
        ).pack(fill="x", pady=(4, 6))
        canvas = self._tk.Canvas(
            panel,
            width=self._config.camera_width,
            height=self._config.camera_height,
            background="#08090a",
            highlightthickness=0,
        )
        canvas.pack(padx=4, pady=(0, 4))
        canvas.create_text(
            self._config.camera_width / 2,
            self._config.camera_height / 2,
            text="Waiting for wrist camera frames",
            fill="#8b929a",
            tags=("camera",),
        )
        self._camera_canvases[role] = canvas

    def _refresh(self) -> None:
        if self._closed:
            return
        if self._stop_event.is_set():
            self.close()
            return
        for role, source in self._camera_queues.items():
            update = _drain_latest(source)
            if isinstance(update, _CameraUpdate):
                self._draw_camera(role, update)
        self._root.after(self._refresh_ms, self._refresh)


def _run_wrist_preview(
    config: LivePreviewConfig,
    camera_queues: Mapping[str, Any],
    stop_event: Any,
) -> None:
    try:
        if config.cpu_set is not None and hasattr(os, "sched_setaffinity"):
            try:
                os.sched_setaffinity(0, set(config.cpu_set))
            except OSError as exc:
                print(f"Wrist preview CPU affinity was not applied: {exc}", file=sys.stderr)
        import tkinter as tk

        root = tk.Tk()
        _WristPreviewApplication(tk, root, config, camera_queues, None, stop_event).run()
    except Exception as exc:
        print(f"Wrist preview unavailable: {exc}", file=sys.stderr)
    finally:
        stop_event.set()
