#!/usr/bin/env python3
"""Calibrate one RealSense color stream with an ordinary chessboard.

The tool captures accepted views automatically while the board is visible.  A
view is accepted only after the requested interval and when its board position
or scale differs from the previous accepted view.  The result is written as
JSON plus a YAML fragment suitable for ``capture_modes.<mode>.cameras``.

This is intentionally a standalone tool. It does not modify any collection
configuration or existing dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def _imports() -> tuple[Any, Any, Any]:
    try:
        import cv2
        import numpy as np
        import pyrealsense2 as rs
    except ImportError as exc:  # pragma: no cover - depends on deployment host
        raise SystemExit(
            "Calibration requires OpenCV and pyrealsense2. Install with "
            "`uv pip install opencv-python` inside yam_data_collection/.venv."
        ) from exc
    return cv2, np, rs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True, help="RealSense serial number")
    parser.add_argument("--role", default="cam_high", help="Configuration role used in output (default: cam_high)")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--corners-cols", type=int, required=True, help="Chessboard inner corners across")
    parser.add_argument("--corners-rows", type=int, required=True, help="Chessboard inner corners down")
    parser.add_argument("--square-size-m", type=float, default=0.026, help="Square edge in metres (default: 0.026)")
    parser.add_argument("--views", type=int, default=25, help="Accepted views before calibration (default: 25)")
    parser.add_argument("--capture-interval-s", type=float, default=0.7, help="Minimum time between views")
    parser.add_argument("--output-dir", type=Path, default=Path("camera_calibrations"))
    parser.add_argument("--no-preview", action="store_true", help="Capture without an OpenCV preview window")
    parser.add_argument("--timeout-ms", type=int, default=2000)
    return parser.parse_args()


def _to_bgr(cv2: Any, np: Any, frame: Any, width: int, height: int) -> Any:
    image = np.asanyarray(frame.get_data())
    if image.ndim == 2 and image.dtype == np.uint16 and image.shape == (height, width):
        # pyrealsense2 exposes packed YUYV as one uint16 per pixel on this
        # host. Viewing the same bytes preserves the Y0 U Y1 V ordering.
        image = image.view(np.uint8).reshape(height, width, 2)
    elif image.ndim == 2 and image.shape == (height, width * 2):
        image = image.reshape(height, width, 2)
    if image.ndim == 3 and image.shape[2] == 2:
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUY2)
    if image.ndim == 3 and image.shape[2] == 3:
        # This branch is useful when a device/profile is changed to BGR/RGB
        # during a smoke test; calibration only needs a grayscale image.
        return image.copy()
    raise RuntimeError(f"unsupported color frame shape {image.shape!r}; expected YUYV or 3-channel image")


def _board_detection(cv2: Any, gray: Any, pattern: tuple[int, int]) -> Any | None:
    # findChessboardCornersSB is more tolerant of perspective and lighting.
    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(gray, pattern, flags=cv2.CALIB_CB_EXHAUSTIVE)
        if found:
            return corners.astype("float32")
    found, corners = cv2.findChessboardCorners(
        gray,
        pattern,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)


def _view_is_different(np: Any, corners: Any, previous: Any | None, width: int, height: int) -> bool:
    if previous is None:
        return True
    current_center = corners.reshape(-1, 2).mean(axis=0)
    previous_center = previous.reshape(-1, 2).mean(axis=0)
    center_shift = float(np.linalg.norm(current_center - previous_center)) / max(width, height)
    current_span = float(np.linalg.norm(corners[0, 0] - corners[-1, 0]))
    previous_span = float(np.linalg.norm(previous[0, 0] - previous[-1, 0]))
    scale_change = abs(current_span - previous_span) / max(previous_span, 1.0)
    return center_shift >= 0.035 or scale_change >= 0.06


def _reprojection_errors(cv2: Any, np: Any, object_points: list[Any], image_points: list[Any], rvecs: Any, tvecs: Any, matrix: Any, distortion: Any) -> list[float]:
    errors: list[float] = []
    for obj, image, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, matrix, distortion)
        errors.append(float(np.mean(np.linalg.norm(image.reshape(-1, 2) - projected.reshape(-1, 2), axis=1))))
    return errors


def _yaml_fragment(result: dict[str, Any]) -> str:
    def values(items: list[float]) -> str:
        return "[" + ", ".join(f"{value:.9g}" for value in items) + "]"

    return "\n".join(
        [
            f"{result['role']}:",
            f"  width: {result['width']}",
            f"  height: {result['height']}",
            f"  fps: {result['fps']}",
            "  intrinsics:",
            "    distortion_model: plumb_bob",
            f"    D: {values(result['distortion'])}",
            f"    K: {values(result['K'])}",
            f"    R: {values(result['R'])}",
            f"    P: {values(result['P'])}",
        ]
    ) + "\n"


def main() -> int:
    args = _parse_args()
    if args.corners_cols < 2 or args.corners_rows < 2:
        raise SystemExit("--corners-cols and --corners-rows must be at least 2")
    if args.square_size_m <= 0 or args.views < 8:
        raise SystemExit("--square-size-m must be positive and --views must be at least 8")
    cv2, np, rs = _imports()
    pattern = (args.corners_cols, args.corners_rows)
    object_template = np.zeros((args.corners_cols * args.corners_rows, 3), np.float32)
    object_template[:, :2] = np.mgrid[0 : args.corners_cols, 0 : args.corners_rows].T.reshape(-1, 2)
    object_template *= args.square_size_m

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.output_dir / f"{args.role}-{args.serial}-{args.width}x{args.height}@{args.fps}-{stamp}"
    run_dir.mkdir()
    image_dir = run_dir / "accepted_views"
    image_dir.mkdir()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.yuyv, args.fps)
    try:
        pipeline_profile = pipeline.start(config)
    except Exception as exc:  # pragma: no cover - hardware dependent
        raise SystemExit(f"Could not start {args.serial} at {args.width}x{args.height}@{args.fps} YUYV: {exc}") from exc

    active_profile = pipeline_profile.get_stream(rs.stream.color).as_video_stream_profile()
    active_video = active_profile.get_video_stream_profile() if hasattr(active_profile, "get_video_stream_profile") else active_profile
    active_intrinsics = active_video.get_intrinsics()
    actual_profile = {
        "width": int(active_intrinsics.width),
        "height": int(active_intrinsics.height),
        "fps": int(active_video.fps()),
        "model": str(active_intrinsics.model),
        "coeffs": [float(value) for value in active_intrinsics.coeffs],
        "fx": float(active_intrinsics.fx),
        "fy": float(active_intrinsics.fy),
        "ppx": float(active_intrinsics.ppx),
        "ppy": float(active_intrinsics.ppy),
    }
    if (actual_profile["width"], actual_profile["height"], actual_profile["fps"]) != (args.width, args.height, args.fps):
        pipeline.stop()
        raise SystemExit(
            "RealSense started a different active profile: "
            f"requested {args.width}x{args.height}@{args.fps}, "
            f"active {actual_profile['width']}x{actual_profile['height']}@{actual_profile['fps']}"
        )

    object_points: list[Any] = []
    image_points: list[Any] = []
    previous: Any | None = None
    last_capture = 0.0
    print(f"Calibrating {args.serial}: board={args.corners_cols}x{args.corners_rows} inner corners, square=2.6 cm")
    print("Move the board through the whole image, changing distance and tilt. q exits; accepted views are saved automatically.")
    try:
        while len(image_points) < args.views:
            frames = pipeline.wait_for_frames(args.timeout_ms)
            color = frames.get_color_frame()
            if not color:
                continue
            bgr = _to_bgr(cv2, np, color, args.width, args.height)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            corners = _board_detection(cv2, gray, pattern)
            now = time.monotonic()
            accepted = corners is not None and now - last_capture >= args.capture_interval_s and _view_is_different(np, corners, previous, args.width, args.height)
            display = bgr.copy()
            if corners is not None:
                cv2.drawChessboardCorners(display, pattern, corners, True)
            if accepted:
                object_points.append(object_template.copy())
                image_points.append(corners.copy())
                previous = corners.copy()
                last_capture = now
                view_path = image_dir / f"view-{len(image_points):03d}.png"
                cv2.imwrite(str(view_path), display)
                print(f"Accepted view {len(image_points)}/{args.views}: {view_path}")
            if not args.no_preview:
                cv2.putText(display, f"views {len(image_points)}/{args.views} | q=quit", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if corners is not None else (0, 0, 255), 2)
                cv2.imshow("RealSense chessboard calibration", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        pipeline.stop()
        if not args.no_preview:
            cv2.destroyAllWindows()

    if len(image_points) < 8:
        raise SystemExit(f"Only {len(image_points)} accepted views; need at least 8 (target was {args.views})")
    image_size = (args.width, args.height)
    rms, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, None, None)
    errors = _reprojection_errors(cv2, np, object_points, image_points, rvecs, tvecs, matrix, distortion)
    result = {
        "role": args.role,
        "serial": args.serial,
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "square_size_m": args.square_size_m,
        "board_inner_corners": [args.corners_cols, args.corners_rows],
        "active_device_profile": actual_profile,
        "views": len(image_points),
        "rms_reprojection_error_px": float(rms),
        "mean_reprojection_error_px": float(np.mean(errors)),
        "max_view_reprojection_error_px": float(max(errors)),
        "distortion_model": "plumb_bob",
        "distortion": distortion.reshape(-1).tolist(),
        "K": matrix.reshape(-1).tolist(),
        "R": np.eye(3, dtype=float).reshape(-1).tolist(),
        "P": np.concatenate([matrix, np.zeros((3, 1))], axis=1).reshape(-1).tolist(),
    }
    (run_dir / "calibration.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (run_dir / "intrinsics.yaml").write_text(_yaml_fragment(result), encoding="utf-8")
    print(f"Calibration complete: RMS={rms:.4f}px, mean={result['mean_reprojection_error_px']:.4f}px, max-view={result['max_view_reprojection_error_px']:.4f}px")
    print(f"Artifacts: {run_dir}")
    print(f"YAML fragment: {run_dir / 'intrinsics.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
