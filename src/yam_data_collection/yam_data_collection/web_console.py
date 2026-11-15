"""Small, local-only web console primitives for YAM data collection.

This module is deliberately independent from the recorder hot path.  It uses
only the Python standard library and does not import Portal, RealSense, NumPy,
or Tk.  A caller can feed low-rate camera bytes and joint snapshots into
``WebConsoleState`` while a ``WebConsoleServer`` serves a browser dashboard.

The optional :class:`ArmProcessSupervisor` is intentionally explicit: command
argv and readiness probes must be supplied by the deployment layer.  No CAN
interface or hardware command is guessed by this module.  ``start_all`` starts
followers before leaders and waits for every configured probe before moving to
the next process; a partial failure is stopped in reverse order.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import signal
import socket
import struct
import subprocess
import threading
import time
import zlib
from collections import deque
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import unquote, urlparse


DEFAULT_CAMERA_ROLES = ("cam_high", "cam_left", "cam_right", "cam_aux")
DEFAULT_ARM_ROLES = ("left_leader", "right_leader", "left_follower", "right_follower")


# The browser has no framework dependency.  The layout intentionally keeps
# visualization on the left and controls/messages on the right, while CSS
# grid and canvas sizing make both columns adapt to the viewport.
WEB_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>YAM 采集控制台</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  <main class="shell">
    <section class="visual-pane" aria-label="Live visualization">
      <header class="pane-header"><h1>YAM 采集控制台</h1><div class="state-group"><span id="capture-state" class="state">采集：空闲</span><span id="arms-state" class="state">机械臂：已停止</span></div></header>
      <div id="camera-grid" class="camera-grid"></div>
      <div id="joint-grid" class="joint-grid"></div>
    </section>
    <aside class="control-pane">
      <section class="controls" aria-label="Controls">
        <h2>控制</h2>
        <div class="button-row">
          <button id="start-arms" type="button" disabled>启动机械臂</button>
          <button id="stop-arms" type="button" class="secondary" disabled>停止机械臂</button>
        </div>
        <div class="button-row">
          <button id="start-capture" type="button" disabled>开始采集</button>
          <button id="stop-capture" type="button" class="danger" disabled>结束采集</button>
        </div>
        <div class="readout"><span>CPU 温度</span><strong id="cpu-temp">--</strong></div>
      </section>
      <section class="messages" aria-label="Messages">
        <div class="messages-header"><h2>提示信息</h2><button id="clear-messages" type="button" class="icon-button" title="清空提示" aria-label="清空提示">×</button></div>
        <ol id="message-list"></ol>
      </section>
    </aside>
  </main>
  <script src="/static/app.js"></script>
</body>
</html>
"""


WEB_CSS = r""":root {
  color-scheme: dark;
  --bg: #101316;
  --panel: #1a1f24;
  --panel-2: #20262c;
  --line: #34404a;
  --text: #e7edf2;
  --muted: #99a5af;
  --good: #63d391;
  --bad: #ff7373;
  --accent: #73b7ff;
}
* { box-sizing: border-box; }
html, body { margin: 0; min-height: 100%; background: var(--bg); color: var(--text); font: 14px/1.4 system-ui, sans-serif; }
body { min-height: 100vh; }
.shell { display: grid; grid-template-columns: minmax(0, 1fr) minmax(280px, 360px); min-height: 100vh; gap: 12px; padding: 12px; }
.visual-pane, .control-pane { min-width: 0; }
.visual-pane { display: flex; flex-direction: column; gap: 10px; }
.pane-header, .messages-header { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.state-group { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 8px; }
h1, h2 { margin: 0; letter-spacing: 0; }
h1 { font-size: 1.2rem; }
h2 { font-size: .95rem; }
.state { color: var(--muted); text-transform: uppercase; font-size: .75rem; }
.state.good { color: var(--good); }
.state.bad { color: var(--bad); }
.camera-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
.joint-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; flex: 1 1 auto; min-height: 280px; }
.camera-card, .joint-card, .controls, .messages { border: 1px solid var(--line); background: var(--panel); border-radius: 6px; overflow: hidden; }
.camera-card h3, .joint-card h3 { margin: 0; padding: 6px 8px; font-size: .8rem; font-weight: 600; color: var(--muted); }
.camera-wrap { position: relative; width: 100%; aspect-ratio: 16 / 9; background: #090b0d; display: grid; place-items: center; }
.camera-frame { display: block; width: 100%; height: 100%; aspect-ratio: 16 / 9; object-fit: contain; background: #090b0d; }
.camera-placeholder { color: var(--muted); font-size: .8rem; }
.joint-card { min-height: 150px; display: flex; flex-direction: column; }
.joint-chart { display: block; width: 100%; height: 100%; min-height: 120px; flex: 1; background: #12161a; }
.control-pane { display: grid; grid-template-rows: auto minmax(0, 1fr); gap: 12px; min-height: 0; }
.controls { padding: 12px; }
.button-row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 10px; }
button { border: 1px solid #528bc2; border-radius: 5px; background: #2b659a; color: white; min-height: 36px; padding: 7px 10px; font: inherit; cursor: pointer; }
button:hover { filter: brightness(1.12); }
button:disabled { opacity: .45; cursor: wait; }
button.secondary { background: #2a333b; border-color: #5b6873; }
button.danger { background: #873e44; border-color: #c76067; }
.readout { display: flex; justify-content: space-between; align-items: center; margin-top: 12px; color: var(--muted); }
.readout strong { color: var(--text); font-variant-numeric: tabular-nums; }
.messages { padding: 10px; min-height: 0; display: flex; flex-direction: column; }
.icon-button { width: 28px; min-height: 28px; padding: 0; background: transparent; border-color: var(--line); color: var(--muted); font-size: 1.1rem; }
#message-list { list-style: none; padding: 0; margin: 8px 0 0; overflow: auto; min-height: 0; }
#message-list li { border-top: 1px solid #293139; padding: 7px 2px; color: var(--muted); overflow-wrap: anywhere; }
#message-list li.success { color: var(--good); }
#message-list li.error { color: var(--bad); }
#message-list time { display: block; color: #6f7b85; font-size: .72rem; }
@media (max-width: 860px) {
  .shell { grid-template-columns: 1fr; }
  .control-pane { grid-template-rows: auto minmax(180px, 35vh); }
}
@media (max-width: 520px) {
  .camera-grid, .joint-grid { grid-template-columns: 1fr; }
  .button-row { grid-template-columns: 1fr; }
}
"""


WEB_JS = r"""(() => {
  const cameraGrid = document.querySelector('#camera-grid');
  const jointGrid = document.querySelector('#joint-grid');
  const messages = document.querySelector('#message-list');
  const stateLabel = document.querySelector('#capture-state');
  const armsLabel = document.querySelector('#arms-state');
  const cpuTemp = document.querySelector('#cpu-temp');
  const charts = new Map();
  const cameraImages = new Map();
  const pendingActions = new Set();
  let lastStatus = null;
  let lastMessageIds = '';

  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  function makeCameras(roles) {
    cameraGrid.innerHTML = roles.map(role => `<article class="camera-card"><h3>${esc(role)}</h3><div class="camera-wrap"><img class="camera-frame" data-role="${esc(role)}" alt="${esc(role)} 画面" hidden><span class="camera-placeholder" data-placeholder="${esc(role)}">未配置</span></div></article>`).join('');
    roles.forEach(role => cameraImages.set(role, cameraGrid.querySelector(`img[data-role="${CSS.escape(role)}"]`)));
  }
  function makeCharts(roles) {
    jointGrid.innerHTML = roles.map(role => `<article class="joint-card"><h3>${esc(role)}</h3><canvas class="joint-chart" data-role="${esc(role)}"></canvas></article>`).join('');
    roles.forEach(role => charts.set(role, jointGrid.querySelector(`canvas[data-role="${CSS.escape(role)}"]`)));
  }
  function drawChart(canvas, samples) {
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(120, Math.floor(rect.width * ratio));
    const height = Math.max(100, Math.floor(rect.height * ratio));
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    const ctx = canvas.getContext('2d'); ctx.clearRect(0, 0, width, height);
    const left = 34 * ratio, right = 8 * ratio, top = 12 * ratio, bottom = 18 * ratio;
    ctx.strokeStyle = '#4a565f'; ctx.lineWidth = ratio; ctx.beginPath(); ctx.moveTo(left, top); ctx.lineTo(left, height-bottom); ctx.lineTo(width-right, height-bottom); ctx.stroke();
    if (!samples || !samples.length) { ctx.fillStyle = '#89949e'; ctx.fillText('等待关节数据', left + 8*ratio, height/2); return; }
    const count = Math.min(...samples.map(s => (s.values || []).length)); if (!count) return;
    const vals = samples.flatMap(s => (s.values || []).slice(0, count)); let lo = Math.min(...vals), hi = Math.max(...vals); if (hi-lo < 1e-9) { lo -= .5; hi += .5; } else { const pad = (hi-lo)*.08; lo -= pad; hi += pad; }
    const first = Number(samples[0].timestamp_ns), last = Number(samples[samples.length-1].timestamp_ns), span = Math.max(1, last-first);
    const colors = ['#e15759','#4e79a7','#59a14f','#f28e2b','#b07aa1','#76b7b2','#edc948'];
    for (let j=0;j<count;j++) { ctx.strokeStyle=colors[j%colors.length]; ctx.beginPath(); samples.forEach((s,i) => { const x=left+(Number(s.timestamp_ns)-first)*(width-left-right)/span; const y=top+(hi-Number(s.values[j]))*(height-top-bottom)/(hi-lo); if(i===0)ctx.moveTo(x,y); else ctx.lineTo(x,y); }); ctx.stroke(); }
  }
  async function request(path, options={}) {
    const response = await fetch(path, {headers: {'Content-Type':'application/json'}, ...options});
    const body = await response.json().catch(() => ({})); if (!response.ok) throw new Error(body.error || response.statusText); return body;
  }
  function renderMessages(items) {
    const id = JSON.stringify(items || []); if (id === lastMessageIds) return; lastMessageIds = id;
    messages.innerHTML = (items || []).slice().reverse().map(item => `<li class="${item.level === 'error' ? 'error' : item.level === 'success' ? 'success' : ''}"><time>${esc(item.time || '')}</time>${esc(item.message)}</li>`).join('');
  }
  function updateButtonStates(status) {
    const arms = status.arms || {};
    const armControlAvailable = Object.keys(arms).length > 0;
    const armState = status.arms_state || 'stopped';
    const captureAvailable = Object.prototype.hasOwnProperty.call(status, 'capture');
    const captureState = status.capture_state || 'idle';
    const captureOperation = status.capture && status.capture.operation;
    const armActionPending = pendingActions.has('arms/start') || pendingActions.has('arms/stop');
    const captureActionPending = pendingActions.has('capture/start') || pendingActions.has('capture/stop');

    document.querySelector('#start-arms').disabled = !armControlAvailable || armActionPending || armState === 'ready' || armState === 'starting';
    document.querySelector('#stop-arms').disabled = !armControlAvailable || armActionPending || !['ready', 'starting', 'error'].includes(armState);
    document.querySelector('#start-capture').disabled = !captureAvailable || captureActionPending || Boolean(captureOperation) || !['idle', 'error'].includes(captureState);
    document.querySelector('#stop-capture').disabled = !captureAvailable || captureActionPending || Boolean(captureOperation) || !['arming', 'recording'].includes(captureState);
  }
  async function refresh() {
    try {
      const status = await request('/api/status');
      lastStatus = status;
      if (!cameraGrid.children.length) makeCameras(status.camera_roles || []);
      if (!jointGrid.children.length) makeCharts(status.arm_roles || []);
      const captureNames = {idle: '空闲', starting: '启动中', arming: '预检中', recording: '采集中', stopping: '结束中', finalizing: '保存中', error: '错误', closed: '已关闭'};
      const armNames = {stopped: '已停止', starting: '启动中', ready: '已就绪', error: '错误'};
      const captureState = status.capture_state || 'idle';
      const armState = status.arms_state || 'stopped';
      updateButtonStates(status);
      stateLabel.textContent = `采集：${captureNames[captureState] || captureState}`; stateLabel.className = `state ${captureState === 'recording' ? 'good' : captureState === 'error' ? 'bad' : ''}`;
      armsLabel.textContent = `机械臂：${armNames[armState] || armState}`; armsLabel.className = `state ${armState === 'ready' ? 'good' : armState === 'error' ? 'bad' : ''}`;
      cpuTemp.textContent = status.cpu_temperature_c == null ? '--' : `${Number(status.cpu_temperature_c).toFixed(1)} °C`;
      renderMessages(status.messages);
      (status.camera_roles || []).forEach(role => { const meta = status.cameras && status.cameras[role]; const img = cameraImages.get(role); const placeholder = cameraGrid.querySelector(`[data-placeholder="${CSS.escape(role)}"]`); if (meta && img && meta.sequence !== Number(img.dataset.sequence || 0)) { img.src = `/api/preview/camera/${encodeURIComponent(role)}?v=${meta.sequence}`; img.dataset.sequence = meta.sequence; img.hidden = false; if (placeholder) placeholder.hidden = true; } else if (!meta && img) { img.hidden = true; if (placeholder) placeholder.hidden = false; } });
      const joints = await request('/api/preview/joints'); window.__yamJoints = joints; (status.arm_roles || []).forEach(role => drawChart(charts.get(role), joints[role] || []));
    } catch (error) { renderMessages([{level:'error', message: error.message, time: new Date().toLocaleTimeString()}]); }
  }
  async function action(path) {
    pendingActions.add(path);
    updateButtonStates(lastStatus || {});
    try {
      await request(`/api/control/${path}`, {method:'POST', body:'{}'});
    } catch (error) {
      renderMessages([{level:'error', message:error.message, time:new Date().toLocaleTimeString()}]);
    } finally {
      pendingActions.delete(path);
      await refresh();
    }
  }
  document.querySelector('#start-arms').onclick = () => action('arms/start');
  document.querySelector('#stop-arms').onclick = () => action('arms/stop');
  document.querySelector('#start-capture').onclick = () => action('capture/start');
  document.querySelector('#stop-capture').onclick = () => action('capture/stop');
  document.querySelector('#clear-messages').onclick = async () => { try { await request('/api/messages/clear', {method:'POST',body:'{}'}); await refresh(); } catch (_) {} };
  window.addEventListener('resize', () => charts.forEach((canvas, role) => drawChart(canvas, window.__yamJoints?.[role] || [])));
  refresh(); setInterval(refresh, 1000);
})();
"""


@dataclass(frozen=True)
class CameraPreview:
    """One immutable camera preview payload."""

    role: str
    data: bytes
    content_type: str = "image/jpeg"
    timestamp_ns: int = 0
    sequence: int = 0


def encode_camera_preview(
    image: Any,
    pixel_format: str = "bgr24",
    *,
    max_width: int = 320,
    max_height: int = 240,
) -> tuple[bytes, str]:
    """Encode a compact browser image without making OpenCV mandatory.

    When OpenCV is installed we use its JPEG encoder.  The fallback is a small
    standards-compliant RGB PNG encoder implemented with ``zlib``; this keeps
    the web module usable in the project's minimal Python environment.  Input
    arrays are expected to have ``shape`` and integer indexing like NumPy
    arrays, but no NumPy import is performed here.
    """
    if max_width <= 0 or max_height <= 0:
        raise ValueError("preview bounds must be positive")
    normalized = str(pixel_format).lower()
    # JPEG is substantially cheaper to transfer when the optional dependency
    # is already present.  Import lazily so serving the UI never requires cv2.
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]

        array = np.asarray(image)
        if normalized == "rgb24":
            array = array[..., ::-1]
        elif normalized == "yuyv422":
            array = cv2.cvtColor(array, cv2.COLOR_YUV2BGR_YUY2)
        if getattr(array, "ndim", 0) != 3 or array.shape[2] != 3:
            raise ValueError("camera image must have three color channels")
        height, width = int(array.shape[0]), int(array.shape[1])
        scale = min(1.0, max_width / width, max_height / height)
        if scale < 1.0:
            target = (max(1, round(width * scale)), max(1, round(height * scale)))
            array = cv2.resize(array, target, interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", array, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if ok:
            return bytes(encoded), "image/jpeg"
    except Exception:
        pass

    # NumPy is already a required collector dependency.  Use a vectorized
    # fallback when OpenCV is absent so preview conversion remains cheap enough
    # for the camera worker's low-rate best-effort callback.
    try:
        import numpy as np  # type: ignore[import-not-found]

        array = np.asarray(image)
        height, width = int(array.shape[0]), int(array.shape[1])
        scale = min(1.0, max_width / width, max_height / height)
        target_width = max(1, round(width * scale))
        target_height = max(1, round(height * scale))
        rows = np.rint(np.linspace(0, height - 1, target_height)).astype(np.intp)
        columns = np.rint(np.linspace(0, width - 1, target_width)).astype(np.intp)
        sampled = array[rows[:, None], columns[None, :]]
        if normalized == "yuyv422":
            if sampled.shape[2] != 2:
                raise ValueError("YUYV camera image must have two channels")
            y = sampled[..., 0].astype(np.int32)
            u = np.repeat(sampled[:, 0::2, 1], 2, axis=1)[:, :target_width].astype(np.int32) - 128
            v_source = sampled[:, 1::2, 1]
            v = np.repeat(v_source, 2, axis=1)[:, :target_width].astype(np.int32) - 128
            luminance = np.maximum(y - 16, 0)
            rgb_array = np.stack(
                (
                    (298 * luminance + 409 * v + 128) >> 8,
                    (298 * luminance - 100 * u - 208 * v + 128) >> 8,
                    (298 * luminance + 516 * u + 128) >> 8,
                ),
                axis=-1,
            )
        elif normalized == "bgr24":
            rgb_array = sampled[..., ::-1]
        elif normalized == "rgb24":
            rgb_array = sampled
        else:
            raise ValueError(f"unsupported camera pixel format: {pixel_format}")
        rgb_array = np.clip(rgb_array, 0, 255).astype(np.uint8, copy=False)
        return _png_bytes(np.ascontiguousarray(rgb_array).tobytes(), target_width, target_height), "image/png"
    except Exception:
        pass

    rgb, width, height = _preview_rgb_bytes(image, normalized, max_width, max_height)
    return _png_bytes(rgb, width, height), "image/png"


def _preview_rgb_bytes(image: Any, pixel_format: str, max_width: int, max_height: int) -> tuple[bytes, int, int]:
    shape = getattr(image, "shape", None)
    if shape is None or len(shape) != 3:
        raise ValueError("camera image must be an HxWxC array")
    source_height, source_width, channels = (int(shape[0]), int(shape[1]), int(shape[2]))
    if source_width <= 0 or source_height <= 0:
        raise ValueError("camera image dimensions must be positive")
    if pixel_format in {"bgr24", "rgb24"} and channels != 3:
        raise ValueError("24-bit camera image must have three channels")
    if pixel_format == "yuyv422" and channels != 2:
        raise ValueError("YUYV camera image must have two channels")
    scale = min(1.0, max_width / source_width, max_height / source_height)
    width = max(1, round(source_width * scale))
    height = max(1, round(source_height * scale))
    output = bytearray(width * height * 3)
    for out_y in range(height):
        src_y = min(source_height - 1, round(out_y * (source_height - 1) / max(1, height - 1)))
        for out_x in range(width):
            src_x = min(source_width - 1, round(out_x * (source_width - 1) / max(1, width - 1)))
            pixel = image[src_y, src_x]
            if pixel_format == "yuyv422":
                y = int(pixel[0])
                even = image[src_y, src_x - (src_x % 2)]
                odd = image[src_y, min(source_width - 1, src_x + (1 if src_x % 2 == 0 else 0))]
                u = int(even[1]) - 128
                v = int(odd[1]) - 128
                luminance = max(y - 16, 0)
                red = (298 * luminance + 409 * v + 128) >> 8
                green = (298 * luminance - 100 * u - 208 * v + 128) >> 8
                blue = (298 * luminance + 516 * u + 128) >> 8
                rgb = (red, green, blue)
            elif pixel_format == "bgr24":
                rgb = (int(pixel[2]), int(pixel[1]), int(pixel[0]))
            else:
                rgb = (int(pixel[0]), int(pixel[1]), int(pixel[2]))
            offset = (out_y * width + out_x) * 3
            output[offset : offset + 3] = bytes(max(0, min(255, value)) for value in rgb)
    return bytes(output), width, height


def _png_bytes(rgb: bytes, width: int, height: int) -> bytes:
    if len(rgb) != width * height * 3:
        raise ValueError("RGB buffer size does not match dimensions")

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)

    rows = b"".join(b"\x00" + rgb[row * width * 3 : (row + 1) * width * 3] for row in range(height))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(rows, 3)) + chunk(b"IEND", b"")


class PreviewBuffer:
    """Thread-safe, bounded latest-camera plus finite joint history storage."""

    def __init__(
        self,
        camera_roles: Iterable[str] = DEFAULT_CAMERA_ROLES,
        arm_roles: Iterable[str] = DEFAULT_ARM_ROLES,
        *,
        joint_history: int = 240,
        camera_fps: float = 5.0,
        joint_fps: float = 10.0,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.camera_roles = tuple(dict.fromkeys(str(role) for role in camera_roles if str(role)))
        self.arm_roles = tuple(dict.fromkeys(str(role) for role in arm_roles if str(role)))
        if not self.camera_roles:
            raise ValueError("camera_roles must not be empty")
        if not self.arm_roles:
            raise ValueError("arm_roles must not be empty")
        if int(joint_history) <= 0:
            raise ValueError("joint_history must be positive")
        if float(camera_fps) <= 0 or float(joint_fps) <= 0:
            raise ValueError("preview frame rates must be positive")
        self._joint_history_limit = int(joint_history)
        self.camera_fps = float(camera_fps)
        self.joint_fps = float(joint_fps)
        self._clock_ns = clock_ns
        self._camera_interval_ns = max(1, round(1_000_000_000 / self.camera_fps))
        self._joint_interval_ns = max(1, round(1_000_000_000 / self.joint_fps))
        self._last_camera_submit_ns: dict[str, int] = {}
        self._last_joint_submit_ns = 0
        self._lock = threading.RLock()
        self._cameras: dict[str, CameraPreview | None] = {role: None for role in self.camera_roles}
        self._camera_sequences = {role: 0 for role in self.camera_roles}
        self._joints: dict[str, deque[dict[str, Any]]] = {
            role: deque(maxlen=self._joint_history_limit) for role in self.arm_roles
        }

    def put_camera(
        self,
        role: str,
        data: bytes | bytearray | memoryview,
        *,
        content_type: str = "image/jpeg",
        timestamp_ns: int | None = None,
    ) -> CameraPreview:
        role = str(role)
        if role not in self._cameras:
            raise KeyError(f"unknown camera role: {role}")
        payload = bytes(data)
        if not payload:
            raise ValueError("camera preview payload must not be empty")
        if not str(content_type).startswith("image/"):
            raise ValueError("camera preview content_type must be an image MIME type")
        with self._lock:
            self._camera_sequences[role] += 1
            preview = CameraPreview(role, payload, str(content_type), int(timestamp_ns or time.time_ns()), self._camera_sequences[role])
            self._cameras[role] = preview
            return preview

    def submit_camera(self, frame: Any) -> bool:
        """Accept either raw bytes or an existing ``CameraFrame``-like object.

        ``EpisodeCollector`` invokes preview sinks with a ``CameraFrame``.  The
        conversion is deliberately best effort: malformed optional preview
        data returns ``False`` and never propagates into the recorder callback.
        Callers that already have browser-ready bytes can pass a mapping with
        ``role`` and ``data`` keys or a ``(role, data)`` pair.
        """
        try:
            role_hint = str(frame[0]) if isinstance(frame, (tuple, list)) and len(frame) == 2 else str(
                frame.get("role") if isinstance(frame, Mapping) else getattr(frame, "role")
            )
            now_ns = self._clock_ns()
            previous_ns = self._last_camera_submit_ns.get(role_hint)
            if previous_ns is not None and now_ns - previous_ns < self._camera_interval_ns:
                return False
            self._last_camera_submit_ns[role_hint] = now_ns
            if isinstance(frame, (tuple, list)) and len(frame) == 2:
                role, data = frame
                self.put_camera(str(role), data)
                return True
            if isinstance(frame, Mapping) and "role" in frame and "data" in frame:
                self.put_camera(
                    str(frame["role"]),
                    frame["data"],
                    content_type=str(frame.get("content_type", "image/jpeg")),
                    timestamp_ns=frame.get("timestamp_ns"),
                )
                return True
            role = str(getattr(frame, "role"))
            image = getattr(frame, "image_bgr")
            pixel_format = str(getattr(frame, "pixel_format", "bgr24"))
            timestamp_ns = int(getattr(frame, "timestamp_ns", time.time_ns()))
            data, content_type = encode_camera_preview(image, pixel_format)
            self.put_camera(role, data, content_type=content_type, timestamp_ns=timestamp_ns)
            return True
        except Exception:
            if "role_hint" in locals():
                self._last_camera_submit_ns.pop(role_hint, None)
            return False

    def get_camera(self, role: str) -> CameraPreview | None:
        with self._lock:
            if role not in self._cameras:
                raise KeyError(f"unknown camera role: {role}")
            return self._cameras[role]

    def camera_metadata(self) -> dict[str, dict[str, Any] | None]:
        with self._lock:
            return {
                role: None
                if preview is None
                else {"content_type": preview.content_type, "timestamp_ns": preview.timestamp_ns, "sequence": preview.sequence}
                for role, preview in self._cameras.items()
            }

    def put_joints(
        self,
        samples: Mapping[str, Sequence[float]],
        *,
        timestamp_ns: int | None = None,
    ) -> None:
        stamp = int(timestamp_ns or time.time_ns())
        with self._lock:
            for role, values in samples.items():
                if role not in self._joints:
                    continue
                normalized = tuple(float(value) for value in values)
                if not normalized:
                    continue
                # JSON-friendly immutable values; reject NaN/Inf rather than
                # allowing JavaScript chart rendering to become undefined.
                if any(not isinstance(value, (int, float)) or not _finite(value) for value in normalized):
                    continue
                self._joints[role].append({"timestamp_ns": stamp, "values": list(normalized)})

    def submit_joints(self, snapshots: Mapping[str, Any], *, timestamp_ns: int | None = None) -> bool:
        """Rate-limit and append a collector-style snapshot mapping."""
        now_ns = self._clock_ns()
        if self._last_joint_submit_ns and now_ns - self._last_joint_submit_ns < self._joint_interval_ns:
            return False
        normalized: dict[str, Sequence[float]] = {}
        try:
            for role, snapshot in snapshots.items():
                values = getattr(snapshot, "position", snapshot)
                normalized[role] = tuple(float(value) for value in values)
            if not normalized:
                return False
            self.put_joints(normalized, timestamp_ns=timestamp_ns or now_ns)
        except (TypeError, ValueError):
            return False
        self._last_joint_submit_ns = now_ns
        return True

    def get_joints(self) -> dict[str, list[dict[str, Any]]]:
        with self._lock:
            return {role: list(history) for role, history in self._joints.items()}


def _finite(value: float) -> bool:
    try:
        return bool(value == value and abs(value) != float("inf"))
    except Exception:
        return False


@dataclass(frozen=True)
class ReadinessProbe:
    host: str
    port: int
    timeout_s: float = 0.2

    def __post_init__(self) -> None:
        if not str(self.host):
            raise ValueError("readiness probe host must not be empty")
        if not 1 <= int(self.port) <= 65535:
            raise ValueError("readiness probe port must be in 1..65535")
        if float(self.timeout_s) <= 0:
            raise ValueError("readiness probe timeout must be positive")


@dataclass(frozen=True)
class ArmCommand:
    """A fully specified, non-shell arm process command."""

    role: str
    argv: tuple[str, ...]
    readiness: tuple[ReadinessProbe, ...] = ()
    cwd: str | Path | None = None
    env: Mapping[str, str] | None = None
    startup_timeout_s: float = 15.0

    def __post_init__(self) -> None:
        if not str(self.role):
            raise ValueError("arm command role must not be empty")
        if not self.argv or any(not str(part) for part in self.argv):
            raise ValueError("arm command argv must not be empty")
        if float(self.startup_timeout_s) <= 0:
            raise ValueError("arm startup timeout must be positive")


@dataclass
class ArmProcessStatus:
    role: str
    pid: int | None = None
    state: str = "stopped"
    ready: bool = False
    error: str | None = None
    returncode: int | None = None
    started_at: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "pid": self.pid,
            "state": self.state,
            "ready": self.ready,
            "error": self.error,
            "returncode": self.returncode,
            "started_at": self.started_at,
        }


class ArmStartupError(RuntimeError):
    """An arm process failed to start or become ready."""


class ArmProcessSupervisor:
    """Start/stop configured follower and leader processes safely.

    ``commands`` are never interpreted by a shell.  Roles containing the word
    ``follower`` are ordered before roles containing ``leader``; otherwise the
    supplied order is retained.  The deployment can therefore use explicit
    left/right ordering while still enforcing the critical follower-first
    dependency.
    """

    def __init__(
        self,
        commands: Sequence[ArmCommand],
        *,
        on_event: Callable[[str, str], None] | None = None,
        popen: Callable[..., Any] = subprocess.Popen,
        connect: Callable[[ReadinessProbe], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not commands:
            raise ValueError("at least one arm command is required")
        roles = [command.role for command in commands]
        if len(set(roles)) != len(roles):
            raise ValueError("arm command roles must be unique")
        self.commands = tuple(commands)
        self._ordered = tuple(sorted(enumerate(commands), key=lambda item: (self._priority(item[1].role), item[0])))
        self._on_event = on_event
        self._popen = popen
        self._connect = connect or _probe_tcp
        self._clock = clock
        self._sleeper = sleeper
        self._lock = threading.RLock()
        self._processes: dict[str, Any] = {}
        self._statuses = {command.role: ArmProcessStatus(role=command.role) for command in commands}
        self._starting = False

    def set_event_handler(self, handler: Callable[[str, str], None] | None) -> None:
        """Replace the optional event sink used for UI status messages."""
        with self._lock:
            self._on_event = handler

    @staticmethod
    def _priority(role: str) -> int:
        normalized = role.lower()
        # Keep the documented physical startup order deterministic even if a
        # caller supplies commands in a different sequence.
        exact = {
            "left_follower": 0,
            "right_follower": 1,
            "right_leader": 2,
            "left_leader": 3,
        }
        if normalized in exact:
            return exact[normalized]
        if "follower" in normalized:
            return 10
        if "leader" in normalized:
            return 11
        return 12

    @property
    def ordered_roles(self) -> tuple[str, ...]:
        return tuple(command.role for _, command in self._ordered)

    def statuses(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            for role, process in self._processes.items():
                status = self._statuses[role]
                returncode = getattr(process, "poll", lambda: None)()
                if returncode is not None:
                    status.returncode = returncode
                    if status.state not in {"failed", "stopped"}:
                        status.state = "exited"
                        status.ready = False
        return {role: status.as_dict() for role, status in self._statuses.items()}

    @property
    def running(self) -> bool:
        with self._lock:
            return any(getattr(process, "poll", lambda: 0)() is None for process in self._processes.values())

    def start_all(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            if self._starting:
                raise ArmStartupError("arm startup is already in progress")
            if self.running:
                raise ArmStartupError("one or more arm processes are already running")
            self._starting = True
            for status in self._statuses.values():
                status.state, status.ready, status.error, status.returncode = "stopped", False, None, None
        try:
            for _, command in self._ordered:
                self._start_one(command)
            self._emit("success", "All arm processes are ready")
            return self.statuses()
        except BaseException as exc:
            self._emit("error", f"Arm startup failed: {exc}")
            self.stop_all(announce=False)
            raise
        finally:
            with self._lock:
                self._starting = False

    def _start_one(self, command: ArmCommand) -> None:
        self._emit("info", f"Starting {command.role}")
        try:
            process = self._popen(
                list(command.argv),
                cwd=str(command.cwd) if command.cwd is not None else None,
                env=(dict(os.environ) | {str(key): str(value) for key, value in command.env.items()})
                if command.env is not None
                else None,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        except Exception as exc:
            self._set_failed(command.role, str(exc))
            raise ArmStartupError(f"could not launch {command.role}: {exc}") from exc
        with self._lock:
            self._processes[command.role] = process
            status = self._statuses[command.role]
            status.pid = getattr(process, "pid", None)
            status.state = "starting"
            status.started_at = self._clock()
        self._start_output_reader(command.role, process)
        deadline = self._clock() + command.startup_timeout_s
        while True:
            returncode = getattr(process, "poll", lambda: None)()
            if returncode is not None:
                self._set_failed(command.role, f"process exited with code {returncode}", returncode)
                raise ArmStartupError(f"{command.role} exited before readiness (code {returncode})")
            try:
                ready = all(self._connect(probe) for probe in command.readiness)
            except Exception:
                ready = False
            if ready:
                with self._lock:
                    self._statuses[command.role].state = "ready"
                    self._statuses[command.role].ready = True
                self._emit("success", f"{command.role} ready")
                return
            if self._clock() >= deadline:
                self._set_failed(command.role, "readiness timeout")
                raise ArmStartupError(f"{command.role} did not become ready within {command.startup_timeout_s:.1f}s")
            self._sleeper(0.05)

    def _start_output_reader(self, role: str, process: Any) -> None:
        stream = getattr(process, "stdout", None)
        if stream is None:
            return

        def read_lines() -> None:
            try:
                for line in iter(stream.readline, ""):
                    line = str(line).strip()
                    if line:
                        lowered = line.lower()
                        level = "error" if any(
                            marker in lowered for marker in ("error", "failed", "failure", "exception", "traceback")
                        ) else "info"
                        self._emit(level, f"{role}: {line}")
            except Exception as exc:
                self._emit("error", f"{role} output reader: {exc}")

        threading.Thread(target=read_lines, name=f"yam-web-{role}-output", daemon=True).start()

    def _set_failed(self, role: str, error: str, returncode: int | None = None) -> None:
        with self._lock:
            status = self._statuses[role]
            status.state, status.ready, status.error, status.returncode = "failed", False, error, returncode

    def stop_all(self, timeout_s: float = 2.0, *, announce: bool = True) -> dict[str, dict[str, Any]]:
        with self._lock:
            items = [(role, self._processes.get(role)) for role in reversed(self.ordered_roles)]
        for role, process in items:
            if process is None or getattr(process, "poll", lambda: 0)() is not None:
                continue
            self._emit("info", f"Stopping {role}")
            try:
                _terminate_process_group(process)
                process.wait(timeout=max(0.0, timeout_s))
            except Exception:
                try:
                    _kill_process_group(process)
                    process.wait(timeout=0.5)
                except Exception as exc:
                    self._emit("error", f"{role} stop failed: {exc}")
            with self._lock:
                status = self._statuses[role]
                # Preserve a failed role's diagnostic state after rollback;
                # the next start attempt resets it explicitly.
                if status.state != "failed":
                    status.state = "stopped"
                status.ready = False
                status.returncode = getattr(process, "poll", lambda: None)()
        if announce:
            self._emit("success", "Arm processes stopped")
        return self.statuses()

    def _emit(self, level: str, message: str) -> None:
        if self._on_event is not None:
            try:
                self._on_event(level, message)
            except Exception:
                pass


def _probe_tcp(probe: ReadinessProbe) -> bool:
    try:
        with socket.create_connection((probe.host, probe.port), timeout=probe.timeout_s):
            return True
    except (OSError, ValueError):
        return False


def _terminate_process_group(process: Any) -> None:
    """Gracefully stop a process and children spawned by minimum_gello."""
    pid = getattr(process, "pid", None)
    if pid is not None and hasattr(os, "killpg"):
        try:
            os.killpg(os.getpgid(int(pid)), signal.SIGTERM)
            return
        except (OSError, ProcessLookupError, ValueError):
            pass
    terminate = getattr(process, "terminate", None)
    if callable(terminate):
        terminate()


def _kill_process_group(process: Any) -> None:
    pid = getattr(process, "pid", None)
    if pid is not None and hasattr(os, "killpg"):
        try:
            os.killpg(os.getpgid(int(pid)), signal.SIGKILL)
            return
        except (OSError, ProcessLookupError, ValueError):
            pass
    kill = getattr(process, "kill", None)
    if callable(kill):
        kill()


@dataclass
class WebConsoleState:
    """Shared state and event log used by HTTP handlers and producers."""

    preview: PreviewBuffer = field(default_factory=PreviewBuffer)
    cpu_temperature_reader: Callable[[], float | None] = lambda: read_cpu_temperature_c()
    message_limit: int = 200
    temperature_interval_s: float = 1.0
    capture_state: str = "idle"
    arms_state: str = "stopped"

    def __post_init__(self) -> None:
        if self.message_limit <= 0:
            raise ValueError("message_limit must be positive")
        if float(self.temperature_interval_s) <= 0:
            raise ValueError("temperature_interval_s must be positive")
        self._lock = threading.RLock()
        self._messages: deque[dict[str, Any]] = deque(maxlen=self.message_limit)
        self._message_sequence = 0
        self._temperature_lock = threading.Lock()
        self._temperature_last_monotonic = 0.0
        self._temperature_value: float | None = None

    @property
    def camera_roles(self) -> tuple[str, ...]:
        return self.preview.camera_roles

    @property
    def arm_roles(self) -> tuple[str, ...]:
        return self.preview.arm_roles

    def message(self, level: str, text: str) -> dict[str, Any]:
        normalized = str(level).lower()
        if normalized not in {"info", "success", "error"}:
            normalized = "info"
        with self._lock:
            self._message_sequence += 1
            item = {"id": self._message_sequence, "level": normalized, "message": str(text), "time": time.strftime("%H:%M:%S")}
            self._messages.append(item)
            return dict(item)

    def clear_messages(self) -> None:
        with self._lock:
            self._messages.clear()

    def set_capture_state(self, state: str, *, message: str | None = None, level: str = "info") -> None:
        with self._lock:
            self.capture_state = str(state)
        if message:
            self.message(level, message)

    def set_arms_state(self, state: str, *, message: str | None = None, level: str = "info") -> None:
        with self._lock:
            self.arms_state = str(state)
        if message:
            self.message(level, message)

    def status(self, *, arm_status: Mapping[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            messages = list(self._messages)
            capture_state = self.capture_state
            arms_state = self.arms_state
        now = time.monotonic()
        with self._temperature_lock:
            if now - self._temperature_last_monotonic >= self.temperature_interval_s:
                try:
                    self._temperature_value = self.cpu_temperature_reader()
                except Exception as exc:
                    self._temperature_value = None
                    self.message("error", f"CPU temperature unavailable: {exc}")
                self._temperature_last_monotonic = now
            temperature = self._temperature_value
        return {
            "capture_state": capture_state,
            "arms_state": arms_state,
            "camera_roles": list(self.camera_roles),
            "arm_roles": list(self.arm_roles),
            "cameras": self.preview.camera_metadata(),
            "cpu_temperature_c": temperature,
            "arms": dict(arm_status or {}),
            "messages": messages,
        }


def read_cpu_temperature_c() -> float | None:
    """Return a CPU temperature, falling back to the host maximum if needed."""
    values: list[float] = []
    cpu_values: list[float] = []
    cpu_markers = ("coretemp", "x86_pkg_temp", "tcpu", "cpu")

    def record(path: Path, descriptor: str) -> None:
        try:
            raw = path.read_text(encoding="ascii").strip()
            value = float(raw)
            if value > 1000:
                value /= 1000.0
            if _finite(value):
                values.append(value)
                if any(marker in descriptor.lower() for marker in cpu_markers):
                    cpu_values.append(value)
        except (OSError, ValueError):
            return

    for root, pattern in (
        (Path("/sys/class/thermal"), "thermal_zone*/temp"),
        (Path("/sys/class/hwmon"), "hwmon*/temp*_input"),
    ):
        try:
            paths = list(root.glob(pattern))
        except OSError:
            continue
        for path in paths:
            descriptor = [path.as_posix()]
            if path.parent.name.startswith("thermal_zone"):
                try:
                    descriptor.append((path.parent / "type").read_text(encoding="ascii"))
                except OSError:
                    pass
            elif path.parent.name.startswith("hwmon"):
                try:
                    descriptor.append((path.parent / "name").read_text(encoding="ascii"))
                except OSError:
                    pass
                label = path.with_name(path.name.removesuffix("_input") + "_label")
                try:
                    descriptor.append(label.read_text(encoding="ascii"))
                except OSError:
                    pass
            record(path, " ".join(descriptor))
    return max(cpu_values or values) if (cpu_values or values) else None


@dataclass(frozen=True)
class WebConsoleCallbacks:
    """Optional application hooks invoked by control endpoints."""

    start_arms: Callable[[], Any] | None = None
    stop_arms: Callable[[], Any] | None = None
    start_capture: Callable[[], Any] | None = None
    stop_capture: Callable[[], Any] | None = None


class WebCaptureController:
    """Serialize browser capture actions around an existing collector.

    The factory receives ``preview_sink=...`` and should return an
    ``EpisodeCollector`` (or a compatible collector).  Starting the collector
    and beginning an episode happen in one worker operation; ending an episode
    calls the collector's normal ``save_episode`` path.  No camera, Portal, or
    CAN object is constructed by this class itself.
    """

    def __init__(
        self,
        collector_factory: Callable[..., Any],
        preview: PreviewBuffer,
        *,
        state: WebConsoleState | None = None,
        preview_sink: Any | None = None,
    ) -> None:
        self.state = state or WebConsoleState(preview=preview)
        self.preview = preview
        # ``preview`` is the browser state store.  A deployment may provide a
        # separate asynchronous encoder/queue as the collector callback sink.
        self.preview_sink = preview if preview_sink is None else preview_sink
        self._collector_factory = collector_factory
        self._collector: Any | None = None
        self._lock = threading.RLock()
        self._operation: str | None = None
        self._operation_thread: threading.Thread | None = None
        self._closed = False
        self._last_background_error: str | None = None

    @property
    def collector(self) -> Any | None:
        with self._lock:
            return self._collector

    @property
    def operation(self) -> str | None:
        with self._lock:
            return self._operation

    def wait_idle(self, timeout_s: float | None = None) -> bool:
        """Wait for the current asynchronous button operation to finish."""
        with self._lock:
            thread = self._operation_thread
        if thread is None:
            return True
        thread.join(timeout_s)
        return not thread.is_alive()

    def start_capture(self) -> dict[str, Any]:
        with self._lock:
            self._ensure_open()
            self._ensure_idle_operation()
            collector = self._collector
            state_value = getattr(getattr(collector, "state", None), "value", None)
            if state_value in {"arming", "recording", "finalizing"}:
                raise RuntimeError(f"capture is already active ({state_value})")
            self._operation = "start_capture"
            self.state.set_capture_state("starting")
            self._launch_locked(self._start_worker)
            return {"accepted": True, "operation": self._operation}

    def stop_capture(self) -> dict[str, Any]:
        with self._lock:
            self._ensure_open()
            self._ensure_idle_operation()
            collector = self._collector
            if collector is None:
                raise RuntimeError("capture has not been started")
            state_value = getattr(getattr(collector, "state", None), "value", "idle")
            if state_value not in {"arming", "recording"}:
                raise RuntimeError(f"no active episode to end ({state_value})")
            self._operation = "stop_capture"
            self.state.set_capture_state("stopping")
            self._launch_locked(self._stop_worker)
            return {"accepted": True, "operation": self._operation}

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            operation_thread = self._operation_thread
        # Do not race a save/start worker with collector.close(). Finalization
        # owns the camera, Portal, and writer lifetimes, so wait for it first.
        if operation_thread is not None and operation_thread is not threading.current_thread():
            self.state.set_capture_state("finalizing")
            self.state.message("info", "正在等待当前采集操作收尾")
            operation_thread.join()
        # The start worker can create the collector after close() first takes
        # the controller lock. Re-read it after the worker exits so that
        # startup and shutdown never leak a newly-created collector.
        with self._lock:
            collector = self._collector
        if collector is not None:
            try:
                collector.close()
            except Exception as exc:
                self.state.message("error", f"Collector close failed: {exc}")
        self.state.set_capture_state("closed")

    def close_collector(self) -> None:
        """Close the current idle collector while keeping the controller reusable."""
        with self._lock:
            self._ensure_idle_operation()
            collector = self._collector
            self._collector = None
        if collector is not None:
            collector.close()

    def status(self) -> dict[str, Any]:
        with self._lock:
            collector = self._collector
            operation = self._operation
        snapshot: dict[str, Any] = {}
        if collector is not None:
            try:
                snapshot = dict(getattr(collector, "active_episode_snapshot", {}) or {})
            except Exception as exc:
                snapshot = {"background_error": str(exc)}
            validation = getattr(collector, "last_post_validation", None)
            if isinstance(validation, Mapping):
                snapshot["post_validation"] = dict(validation)
            state_value = getattr(getattr(collector, "state", None), "value", None)
            if state_value and operation is None:
                self.state.set_capture_state(str(state_value))
            background_error = snapshot.get("background_error")
            if background_error:
                message = str(background_error)
                if message != self._last_background_error:
                    self._last_background_error = message
                    self.state.message("error", message)
        return {"operation": operation, "collector": snapshot}

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("capture controller is closed")

    def _ensure_idle_operation(self) -> None:
        if self._operation is not None:
            raise RuntimeError(f"capture operation already running: {self._operation}")

    def _launch_locked(self, target: Callable[[], None]) -> None:
        thread = threading.Thread(target=target, name=f"yam-web-{self._operation}", daemon=True)
        self._operation_thread = thread
        thread.start()

    def _finish_operation(self, *, state: str, message: str, level: str = "success") -> None:
        with self._lock:
            self._operation = None
            self._operation_thread = None
        self.state.set_capture_state(state)
        self.state.message(level, message)

    def _start_worker(self) -> None:
        collector: Any | None = None
        try:
            with self._lock:
                collector = self._collector
                if collector is None:
                    collector = self._collector_factory(preview_sink=self.preview_sink)
                    self._collector = collector
            collector.start()
            collector.start_episode()
            state_value = getattr(getattr(collector, "state", None), "value", "recording")
            self._finish_operation(state=str(state_value), message="Capture started")
        except Exception as exc:
            if collector is not None:
                try:
                    collector.close()
                except Exception:
                    pass
            with self._lock:
                # A failed start leaves the collector closed and unusable.
                # Drop it so the next button press creates a fresh instance.
                if self._collector is collector:
                    self._collector = None
            self._finish_operation(state="error", message=f"Capture start failed: {exc}", level="error")

    def _stop_worker(self) -> None:
        try:
            collector = self.collector
            if collector is None:
                raise RuntimeError("capture has not been started")
            state_value = getattr(getattr(collector, "state", None), "value", "idle")
            if state_value == "recording":
                saved = collector.save_episode()
                if saved is False:
                    raise RuntimeError("episode could not be saved")
            else:
                collector.discard_episode()
            validation = getattr(collector, "last_post_validation", None)
            if isinstance(validation, Mapping) and validation.get("passed") is False:
                detail = validation.get("error") or validation.get("failed_checks") or "post-save validation failed"
                self._finish_operation(state="idle", message=f"Capture ended; validation failed: {detail}", level="error")
            elif isinstance(validation, Mapping) and validation.get("informational_failures"):
                detail = validation["informational_failures"]
                self._finish_operation(state="idle", message=f"Capture ended; validation warnings: {detail}", level="info")
            else:
                self._finish_operation(state="idle", message="Capture ended", level="success")
        except Exception as exc:
            self._finish_operation(state="error", message=f"Capture stop failed: {exc}", level="error")


def load_arm_commands(path: str | Path) -> tuple[ArmCommand, ...]:
    """Load explicit arm command/readiness definitions from a JSON file.

    The file format is a list of objects with ``role``, ``argv`` and optional
    ``readiness`` (``[{"host": "127.0.0.1", "port": 1235}]``), ``cwd``,
    ``env`` and ``startup_timeout_s`` fields.  JSON keeps this loader in the
    standard library and makes shell expansion impossible.
    """
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read arm command file {path}: {exc}") from exc
    if isinstance(raw, Mapping):
        raw = raw.get("commands")
    if not isinstance(raw, list) or not raw:
        raise ValueError("arm command file must contain a non-empty JSON list")
    commands: list[ArmCommand] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"arm command {index} must be an object")
        readiness_raw = item.get("readiness", ())
        if not isinstance(readiness_raw, (list, tuple)):
            raise ValueError(f"arm command {index}.readiness must be a list")
        probes: list[ReadinessProbe] = []
        for probe_index, probe in enumerate(readiness_raw):
            if not isinstance(probe, Mapping):
                raise ValueError(f"arm command {index}.readiness[{probe_index}] must be an object")
            probes.append(ReadinessProbe(str(probe.get("host", "127.0.0.1")), int(probe["port"]), float(probe.get("timeout_s", .2))))
        argv = item.get("argv")
        if not isinstance(argv, (list, tuple)):
            raise ValueError(f"arm command {index}.argv must be a list")
        env = item.get("env")
        if env is not None and not isinstance(env, Mapping):
            raise ValueError(f"arm command {index}.env must be an object")
        commands.append(
            ArmCommand(
                role=str(item.get("role", "")),
                argv=tuple(str(value) for value in argv),
                readiness=tuple(probes),
                cwd=item.get("cwd"),
                env={str(key): str(value) for key, value in env.items()} if env is not None else None,
                startup_timeout_s=float(item.get("startup_timeout_s", 15.0)),
            )
        )
    return tuple(commands)


class WebConsoleApplication:
    """Application object kept separate from HTTP transport for testing."""

    def __init__(
        self,
        state: WebConsoleState | None = None,
        *,
        callbacks: WebConsoleCallbacks | None = None,
        arm_supervisor: ArmProcessSupervisor | None = None,
        capture_controller: WebCaptureController | None = None,
    ) -> None:
        self.state = state or WebConsoleState()
        self.callbacks = callbacks or WebConsoleCallbacks()
        self.arm_supervisor = arm_supervisor
        self.capture_controller = capture_controller
        self._action_lock = threading.Lock()
        if capture_controller is not None and state is None:
            self.state = capture_controller.state
        if capture_controller is not None:
            self.callbacks = WebConsoleCallbacks(
                start_arms=self.callbacks.start_arms,
                stop_arms=self.callbacks.stop_arms,
                start_capture=self.callbacks.start_capture or capture_controller.start_capture,
                stop_capture=self.callbacks.stop_capture or capture_controller.stop_capture,
            )
        if arm_supervisor is not None:
            arm_supervisor.set_event_handler(self._arm_event)

    def _arm_event(self, level: str, message: str) -> None:
        self.state.set_arms_state("ready" if level == "success" and message == "All arm processes are ready" else self.state.arms_state)
        self.state.message(level, message)

    def status(self) -> dict[str, Any]:
        arm_status = self.arm_supervisor.statuses() if self.arm_supervisor else {}
        if arm_status:
            if any(item.get("state") in {"failed", "exited"} for item in arm_status.values()):
                self.state.set_arms_state("error")
            elif all(item.get("ready") for item in arm_status.values()):
                self.state.set_arms_state("ready")
            elif any(item.get("state") == "starting" for item in arm_status.values()):
                self.state.set_arms_state("starting")
            else:
                self.state.set_arms_state("stopped")
        result = self.state.status(arm_status=arm_status)
        if self.capture_controller is not None:
            result["capture"] = self.capture_controller.status()
            snapshot = result["capture"].get("collector", {})
            if isinstance(snapshot, Mapping):
                result["collector"] = snapshot
        return result

    def action(self, action: str) -> dict[str, Any]:
        # ThreadingHTTPServer dispatches each request independently.  Keep
        # physical arm and collector transitions strictly one-at-a-time even
        # when a browser or operator sends duplicate clicks.
        if not self._action_lock.acquire(blocking=False):
            raise RuntimeError("another control action is already running")
        try:
            return self._action_unlocked(action)
        finally:
            self._action_lock.release()

    def _action_unlocked(self, action: str) -> dict[str, Any]:
        callbacks = self.callbacks
        if action == "arms/start":
            try:
                self._ensure_arms_action_allowed()
                if self.capture_controller is not None and callbacks.start_arms is None:
                    self.capture_controller.close_collector()
                self.state.set_arms_state("starting")
                result = (
                    _call(callbacks.start_arms)
                    if callbacks.start_arms is not None
                    else _call(self.arm_supervisor.start_all if self.arm_supervisor else None)
                )
            except Exception as exc:
                self.state.set_arms_state("error")
                self.state.message("error", f"Arm start failed: {exc}")
                raise
            self.state.set_arms_state("ready")
            self.state.message("success", "Arm startup completed")
            return {"ok": True, "result": result}
        if action == "arms/stop":
            try:
                self._ensure_arms_action_allowed()
                if self.capture_controller is not None and callbacks.stop_arms is None:
                    self.capture_controller.close_collector()
                result = (
                    _call(callbacks.stop_arms)
                    if callbacks.stop_arms is not None
                    else _call(self.arm_supervisor.stop_all if self.arm_supervisor else None)
                )
            except Exception as exc:
                self.state.message("error", f"Arm stop failed: {exc}")
                raise
            self.state.set_arms_state("stopped")
            self.state.message("success", "Arm processes stopped")
            return {"ok": True, "result": result}
        if action == "capture/start":
            try:
                if self.arm_supervisor is not None:
                    statuses = self.arm_supervisor.statuses()
                    if not statuses or not all(item.get("ready") for item in statuses.values()):
                        raise RuntimeError("all four arm processes must be ready before capture")
                self.state.set_capture_state("starting")
                result = _call(callbacks.start_capture)
            except Exception as exc:
                self.state.set_capture_state("error")
                self.state.message("error", f"Capture start failed: {exc}")
                raise
            if not isinstance(result, Mapping) or not result.get("accepted"):
                self.state.set_capture_state("recording")
                self.state.message("success", "Capture started")
            return {"ok": True, "result": result}
        if action == "capture/stop":
            self.state.set_capture_state("stopping")
            try:
                result = _call(callbacks.stop_capture)
            except Exception as exc:
                self.state.set_capture_state("error")
                self.state.message("error", f"Capture stop failed: {exc}")
                raise
            if not isinstance(result, Mapping) or not result.get("accepted"):
                self.state.set_capture_state("idle")
                self.state.message("success", "Capture ended")
            return {"ok": True, "result": result}
        raise KeyError(f"unknown action: {action}")

    def _ensure_arms_action_allowed(self) -> None:
        controller = self.capture_controller
        if controller is None:
            return
        if controller.operation is not None:
            raise RuntimeError(f"capture operation is still running: {controller.operation}")
        collector = controller.collector
        state = getattr(getattr(collector, "state", None), "value", "idle")
        if state in {"arming", "recording", "finalizing"}:
            raise RuntimeError("end or discard the active capture before changing arms")


def _call(callback: Callable[[], Any] | None) -> Any:
    if callback is None:
        raise RuntimeError("control action is not configured")
    return callback()


class WebConsoleHandler(BaseHTTPRequestHandler):
    """HTTP transport for :class:`WebConsoleApplication`."""

    server_version = "YAMWeb/0.1"
    max_body_bytes = 64 * 1024

    @property
    def application(self) -> WebConsoleApplication:
        return getattr(self.server, "application")  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep the local console quiet by default; callers may replace this.
        logger = getattr(self.server, "logger", None)  # type: ignore[attr-defined]
        if logger is not None:
            logger(fmt % args)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path in {"", "/"}:
            self._send_bytes(WEB_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/static/style.css":
            self._send_bytes(WEB_CSS.encode("utf-8"), "text/css; charset=utf-8")
            return
        if path == "/static/app.js":
            self._send_bytes(WEB_JS.encode("utf-8"), "application/javascript; charset=utf-8")
            return
        if path == "/api/status":
            self._send_json(self.application.status())
            return
        if path == "/api/preview/joints":
            self._send_json(self.application.state.preview.get_joints())
            return
        prefix = "/api/preview/camera/"
        if path.startswith(prefix):
            role = unquote(path[len(prefix) :])
            try:
                preview = self.application.state.preview.get_camera(role)
            except KeyError:
                self._send_json({"error": "unknown camera role"}, HTTPStatus.NOT_FOUND)
                return
            if preview is None:
                self._send_json({"error": "no preview frame"}, HTTPStatus.NOT_FOUND)
                return
            self._send_bytes(preview.data, preview.content_type)
            return
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/messages/clear":
            try:
                self._read_json()
            except ValueError as exc:
                self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            self.application.state.clear_messages()
            self._send_json({"ok": True})
            return
        aliases = {
            "/api/control/start-arms": "arms/start",
            "/api/control/stop-arms": "arms/stop",
            "/api/control/start-capture": "capture/start",
            "/api/control/stop-capture": "capture/stop",
        }
        action_prefix = "/api/control/"
        action = aliases.get(path)
        if action is None and path.startswith(action_prefix):
            action = path[len(action_prefix) :]
        if action in {"arms/start", "arms/stop", "capture/start", "capture/stop"}:
            try:
                self._read_json()
                result = self.application.action(action)
            except (KeyError, RuntimeError, ValueError) as exc:
                self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.CONFLICT)
                return
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json(result)
            return
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def _read_json(self) -> dict[str, Any]:
        length_text = self.headers.get("Content-Length", "0")
        try:
            length = int(length_text)
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length < 0 or length > self.max_body_bytes:
            raise ValueError("request body is too large")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("request body must be valid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._send_bytes(data, "application/json; charset=utf-8", status)

    def _send_bytes(self, data: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


class WebConsoleServer(ThreadingHTTPServer):
    """Threaded local web server carrying an application object."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int] = ("127.0.0.1", 8765),
        application: WebConsoleApplication | None = None,
        *,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        _validate_local_bind_host(server_address[0])
        self.application = application or WebConsoleApplication()
        self.logger = logger
        super().__init__(server_address, WebConsoleHandler)


def create_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    application: WebConsoleApplication | None = None,
    logger: Callable[[str], None] | None = None,
) -> WebConsoleServer:
    """Build (but do not start) a local console server."""
    return WebConsoleServer((host, int(port)), application, logger=logger)


def _validate_local_bind_host(host: str) -> None:
    """Keep the unauthenticated arm-control API on the local machine."""
    normalized = str(host).strip().lower()
    if normalized == "localhost":
        return
    try:
        address = ipaddress.ip_address(normalized)
        is_loopback = address.version == 4 and address.is_loopback
    except ValueError:
        is_loopback = False
    if not is_loopback:
        raise ValueError(
            "the web console only permits IPv4 loopback bind addresses (127.0.0.1 or localhost)"
        )


def serve_forever(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    application: WebConsoleApplication | None = None,
) -> None:
    server = create_server(host, port, application=application)
    try:
        server.serve_forever()
    finally:
        server.server_close()


class WebPreviewSink:
    """Small synchronous compatibility sink for callers without an encoder worker.

    The production adapter in :mod:`web_runtime` adds asynchronous downsampling
    and image encoding.  This wrapper keeps the collector-facing sink contract
    usable for tests and lightweight integrations while retaining the same
    bounded :class:`PreviewBuffer` storage.
    """

    def __init__(self, preview: PreviewBuffer) -> None:
        self.preview = preview

    def start(self) -> None:
        return None

    def stop(self, timeout_s: float = 0.0) -> None:
        del timeout_s

    def submit_camera(self, frame: Any) -> bool:
        return self.preview.submit_camera(frame)

    def submit_joints(self, snapshots: Mapping[str, Any], *, timestamp_ns: int | None = None) -> bool:
        return self.preview.submit_joints(snapshots, timestamp_ns=timestamp_ns)


# Friendly name for integration code that treats the bounded preview object as
# a broker/hub rather than a browser-specific implementation.
PreviewHub = PreviewBuffer


def _demo_application() -> WebConsoleApplication:
    """Create a harmless empty demo app for ``python -m ...web_console``."""
    app = WebConsoleApplication()
    app.state.message("info", "Web console is running; no hardware commands are configured")
    return app


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the isolated YAM local web console")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    server = create_server(args.host, args.port, application=_demo_application())
    print(f"YAM web console: http://{args.host}:{server.server_address[1]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
