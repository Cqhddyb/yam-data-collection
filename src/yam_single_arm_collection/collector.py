"""Queued single-arm collection with scoped validation and H.264 publication."""

from __future__ import annotations

import math
import shutil
from pathlib import Path

from yam_data_collection.collector import CaptureError, CaptureState, _Episode
from yam_data_collection.config import DatasetConfig
from yam_data_collection.live_preview import LivePreviewConfig
from yam_data_collection.queued_collector import QueuedEpisodeCollector
from yam_data_collection.video import close_video_writers

from .config import camera_roles, robot_roles
from .live_preview import WristCameraPreview
from .quality import validate_mcap, validate_video_assets
from .video_pipeline import publish_live_episode


def parse_preview_fps(value: str | float) -> float:
    fps = float(value)
    if not math.isfinite(fps) or not 0 < fps <= 10:
        raise ValueError("preview FPS must be finite and greater than 0, up to 10")
    return fps


class SingleArmCollector(QueuedEpisodeCollector):
    """Reuse acquisition guards, but never call the four-arm publication gate."""

    def __init__(
        self,
        config: DatasetConfig,
        *,
        enable_runtime_diagnostics: bool = True,
        enable_preview: bool = False,
        preview_fps: float = 10.0,
    ) -> None:
        preview_fps = parse_preview_fps(preview_fps)
        selection = config.raw.get("single_arm", {})
        side, setup = selection.get("side"), selection.get("cameras")
        if set(config.telemetry) != set(robot_roles(side)) or set(config.cameras) != set(camera_roles(side, setup)):
            raise ValueError("SingleArmCollector requires exactly the selected arm pair and cameras")
        if config.collection.video_codec not in {"libx264", "h264"}:
            raise ValueError("SingleArmCollector only publishes H.264-final episodes")
        super().__init__(
            config,
            enable_runtime_diagnostics=enable_runtime_diagnostics,
            auto_delete_failed_validation=True,
            enable_post_save_review=True,
            video_timeline="source",
        )
        if enable_preview:
            preview_cpus = self._available_preview_cpu_set()
            if preview_cpus:
                wrist_role = f"cam_{side}"
                wrist_camera = config.cameras[wrist_role]
                self._live_preview = WristCameraPreview(
                    LivePreviewConfig(
                        camera_roles=(wrist_role,),
                        arm_roles=robot_roles(side),
                        camera_fps=preview_fps,
                        camera_width=wrist_camera.width,
                        camera_height=wrist_camera.height,
                        cpu_set=preview_cpus,
                        window_title=f"YAM wrist camera preview ({side})",
                    )
                )
            else:
                self._print("Warning: camera preview disabled; no CPU is available outside recorder/encoder CPU sets.")

    def _start_live_preview(self) -> None:
        preview = self._live_preview
        if preview is None:
            return
        try:
            preview.start()
            self._print(
                f"Live preview requested: {', '.join(preview.config.camera_roles)}, "
                f"{preview.config.camera_fps:g} Hz. Use the capture terminal for recording keys."
            )
        except Exception as exc:
            self._report_preview_error("could not start live preview", exc)
            try:
                preview.stop(timeout_s=0.1)
            except Exception:
                pass

    def save_episode(self) -> bool:
        saved = super().save_episode()
        if saved:
            self._print(
                f"本条采集时长（episode {self._last_saved_episode_id}）："
                f"{self._last_saved_episode_duration_s:.2f} 秒（不含预热和保存耗时）。"
            )
        return saved

    def _finish_episode(self, episode: _Episode, commit: bool) -> bool:
        errors: list[BaseException] = []
        saved = False
        before_close: dict[str, dict[str, object]] | None = None
        with self._state_lock:
            finalized_error = self._background_error
            self._last_post_validation = None
        try:
            if not episode.operations_drained.wait(timeout=self._FINALIZATION_WAIT_S):
                self._print("Waiting for in-flight camera/state writes before finalization.")
                episode.operations_drained.wait()
            state_writer = self._state_mcap_writers.pop(id(episode), None)
            if state_writer is not None:
                try:
                    state_writer.close()
                except BaseException as exc:
                    errors.append(exc)
                    commit = False
            with self._state_lock:
                if self._background_error is not None:
                    commit = False
            before_close = {role: self._video_writer_snapshot(writer) for role, writer in episode.videos.items()}
            self._remember_episode_diagnostics(episode, before_close=before_close)
            if commit:
                try:
                    for role, samples in episode.camera_timing_samples.items():
                        episode.mcap.write_camera_timing_diagnostics(role, samples)
                except BaseException as exc:
                    errors.append(exc)
            try:
                episode.mcap.close()
            except BaseException as exc:
                errors.append(exc)
            origin_ns = episode.camera_start_timestamp_ns
            if commit:
                if origin_ns is None:
                    timestamps = [
                        sample.capture_timestamp_ns
                        for samples in episode.camera_timing_samples.values()
                        for sample in samples
                    ]
                    origin_ns = min(timestamps) if timestamps else None
                if origin_ns is None:
                    errors.append(CaptureError("source-timestamp video mode recorded no camera timestamps"))
                else:
                    for writer in episode.videos.values():
                        try:
                            writer.set_timeline_origin_ns(origin_ns)
                        except BaseException as exc:
                            errors.append(exc)
            errors.extend(close_video_writers(episode.videos.values(), commit=commit, parallel=True))
            if commit and not errors:
                try:
                    self._write_source_timeline_manifest(episode, origin_ns)
                    episode.partial_mcap_path.replace(episode.final_mcap_path)
                    episode.video_dir.replace(episode.live_video_dir)
                    published = publish_live_episode(episode.final_mcap_path, self.config)
                    episode.committed_mcap_path = published.final_mcap_path
                    saved = True
                    if not self._report_cached_post_episode_validation(
                        dict(published.final_validation), video_label="Video asset validation"
                    ):
                        self._run_post_episode_validation(published.final_mcap_path)
                except BaseException as exc:
                    errors.append(exc)
            if saved:
                duration_s = self._recording_duration_s(episode)
                frames = ", ".join(f"{role}={episode.camera_frames.get(role, 0)}" for role in self.config.cameras)
                self._print(
                    f"Saved H.264 single-arm episode {episode.episode_id}: "
                    f"{episode.state_samples} state samples, {frames}, {duration_s:.1f}s."
                )
                self._print(f"Published episode: {episode.committed_mcap_path}. No offline transcode is required.")
            else:
                retained = commit and episode.final_mcap_path.is_file() and episode.live_video_dir.is_dir()
                if retained:
                    self._print(
                        f"Episode {episode.episode_id} was not published. Pending MCAP/videos_live retained "
                        f"for diagnosis: {'; '.join(str(error) for error in errors)}"
                    )
                else:
                    try:
                        episode.partial_mcap_path.unlink(missing_ok=True)
                    except OSError as exc:
                        errors.append(exc)
                    for directory in (episode.pending_dir, episode.video_dir, episode.live_video_dir):
                        shutil.rmtree(directory, ignore_errors=True)
                    detail = "; ".join(str(error) for error in errors)
                    self._print(f"Discarded episode {episode.episode_id}. {detail}")
            return saved
        finally:
            if before_close is not None:
                after_close = {role: self._video_writer_snapshot(writer) for role, writer in episode.videos.items()}
                self._remember_episode_diagnostics(episode, after_close=after_close)
            with self._state_lock:
                if self._state is not CaptureState.CLOSED:
                    self._state = CaptureState.IDLE
                if finalized_error is not None and self._background_error is finalized_error:
                    self._background_error = None
            self._finalization_done.set()

    def _run_post_episode_validation(self, mcap_path: Path, *, live_video_dir: Path | None = None) -> None:
        try:
            self._store_post_episode_validation(
                validate_mcap(mcap_path, self.config),
                validate_video_assets(mcap_path, self.config, video_dir=live_video_dir),
                video_label="Live video asset validation" if live_video_dir is not None else "Video asset validation",
            )
        except Exception as exc:
            self._print(f"Post-episode validation could not run: {exc}")
            with self._state_lock:
                self._last_post_validation = {"passed": False, "error": str(exc)}
