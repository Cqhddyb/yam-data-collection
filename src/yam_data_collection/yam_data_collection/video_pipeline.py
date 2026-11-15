"""Verified video publication for real-time YAM acquisition.

Live capture always writes a low-cost MP4 beside a pending MCAP. Existing
HEVC configurations keep the serial x265 publication path. Explicit H.264
final configurations validate those live assets and atomically publish them
without decoding or re-encoding their video bytes.
"""

from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .config import DatasetConfig
from .dataset_lock import DatasetOperationLock
from .mcap_writer import CAMERA_TOPICS
from .quality import ValidationPolicy, build_validation_policy, validate_mcap, validate_video_assets
from .video import VideoTranscodeResult, transcode_to_hevc


LIVE_VIDEO_ROOT = "videos_live"
PENDING_ROOT = ".pending"


class VideoPipelineError(RuntimeError):
    """A live episode cannot be safely promoted to its final dataset layout."""


@dataclass(frozen=True)
class TranscodedEpisode:
    """Evidence from a successfully published, final HEVC episode."""

    episode_id: int
    pending_mcap_path: Path
    final_mcap_path: Path
    live_video_dir: Path
    final_video_dir: Path
    live_validation: Mapping[str, Any]
    final_validation: Mapping[str, Any]
    videos: Mapping[str, VideoTranscodeResult]


@dataclass(frozen=True)
class DirectPublishedEpisode:
    """Evidence from a verified H.264 live episode published without re-encoding."""

    episode_id: int
    pending_mcap_path: Path
    final_mcap_path: Path
    live_video_dir: Path
    final_video_dir: Path
    live_validation: Mapping[str, Any]
    final_validation: Mapping[str, Any]


def live_video_dir(dataset_dir: str | Path, episode_id: int) -> Path:
    return Path(dataset_dir) / LIVE_VIDEO_ROOT / str(episode_id)


def episode_dir(dataset_dir: str | Path, episode_id: int) -> Path:
    """Return the self-contained public directory for one episode."""
    return Path(dataset_dir) / str(episode_id)


def final_mcap_path(dataset_dir: str | Path, episode_id: int) -> Path:
    root = episode_dir(dataset_dir, episode_id)
    if root.is_dir():
        candidates = sorted(path for path in root.glob("*.mcap") if not path.name.endswith(".partial.mcap"))
        if len(candidates) == 1:
            return candidates[0]
    return root / f"{episode_id}.mcap"


def pending_episode_dir(dataset_dir: str | Path, episode_id: int) -> Path:
    return Path(dataset_dir) / PENDING_ROOT / str(episode_id)


def pending_mcap_path(dataset_dir: str | Path, episode_id: int, *, filename: str | None = None) -> Path:
    selected_filename = filename or f"{episode_id}.mcap"
    if Path(selected_filename).name != selected_filename or not selected_filename.endswith(".mcap"):
        raise ValueError(f"pending MCAP filename must be a plain .mcap filename: {selected_filename!r}")
    return pending_episode_dir(dataset_dir, episode_id) / selected_filename


def timestamped_mcap_filename(timestamp_ns: int) -> str:
    """Create a local-time episode filename with minute precision."""
    timestamp = datetime.fromtimestamp(timestamp_ns / 1_000_000_000).astimezone()
    return timestamp.strftime("%Y-%m-%d_%H-%M.mcap")


def final_video_dir(dataset_dir: str | Path, episode_id: int) -> Path:
    return episode_dir(dataset_dir, episode_id) / "videos"


def remove_episode_assets(mcap_path: str | Path) -> None:
    """Remove one episode from either the public or pending capture layout."""
    path = Path(mcap_path)
    if path.suffix != ".mcap":
        raise ValueError(f"episode path must be an .mcap file, got {path}")
    root = path.parent
    if root.name.isdigit() and root.parent.name == PENDING_ROOT:
        dataset = root.parent.parent
        episode_id = int(root.name)
        if root.exists():
            shutil.rmtree(root)
        live_dir = live_video_dir(dataset, episode_id)
        if live_dir.exists():
            shutil.rmtree(live_dir)
        return
    if root.name.isdigit():
        if root.exists():
            shutil.rmtree(root)
        return
    if not path.stem.isdigit():
        raise ValueError(f"cannot identify episode directory for MCAP: {path}")
    episode_id = path.stem
    path.unlink(missing_ok=True)
    for directory in (
        root / "videos" / episode_id,
        root / LIVE_VIDEO_ROOT / episode_id,
        root / PENDING_ROOT / episode_id,
    ):
        if directory.exists():
            shutil.rmtree(directory)


def codec_family(codec: str) -> tuple[str, ...]:
    """Translate configured FFmpeg codec aliases to ffprobe codec names."""
    normalized = codec.strip().lower()
    if normalized in {"libx264", "h264"}:
        return ("h264",)
    if normalized in {"libx265", "hevc"}:
        return ("hevc",)
    raise VideoPipelineError(f"unsupported configured video codec {codec!r}")


def live_codec_family(codec: str) -> tuple[str, ...]:
    """Compatibility alias for callers validating a temporary live video."""
    return codec_family(codec)


def final_codec_family(config: DatasetConfig) -> tuple[str, ...]:
    """Return the ffprobe codec expected for this configuration's final assets."""
    return codec_family(config.collection.video_codec)


def uses_direct_h264_publication(config: DatasetConfig) -> bool:
    """Whether this configuration publishes its verified live H.264 assets directly."""
    return final_codec_family(config) == ("h264",)


def _policy(config: DatasetConfig) -> ValidationPolicy:
    return build_validation_policy(config.cameras, config.collection.validation_profile)


def validate_live_episode(mcap_path: str | Path, config: DatasetConfig) -> dict[str, Any]:
    """Validate pending MCAP and live MP4 frame identity before publication.

    This gate is deliberately narrower than formal quality acceptance. A
    completed recording that has a real source-rate or synchronization failure
    still needs faithful final assets so the normal ``validate`` command
    can report that failure. The promotion gate therefore rejects only an
    unreadable MCAP or a video identity mismatch; it never fabricates a pass
    or discards a completed episode merely because its quality report failed.
    """
    path = Path(mcap_path)
    episode_id = _episode_id(path)
    dataset = config.collection.dataset_dir
    _require_pending_path(path, dataset, episode_id)
    policy = _policy(config)
    mcap_result = validate_mcap(path, policy)
    live_dir = live_video_dir(dataset, episode_id)
    result = validate_video_assets(
        path,
        policy,
        video_dir=live_dir,
        expected_video_codecs=codec_family(config.collection.live_video_codec),
        require_fps_match=False,
    )
    if not result["passed"]:
        details = "; ".join(
            f"{role}: {entry.get('error', 'asset validation failed')}"
            for role, entry in result["per_camera"].items()
            if not entry.get("passed")
        )
        raise VideoPipelineError(f"live video validation failed for episode {episode_id}: {details}")
    result["formal_mcap_passed"] = bool(mcap_result["output_mcap_passed"])
    result["formal_failed_checks"] = list(mcap_result["failed_checks"])
    # Keep the complete formal report with the live identity report.  Direct
    # H.264 publication moves these exact bytes into their final paths, so a
    # source-timeline collector can report this already completed validation
    # without reading the MCAP and decoding/counting all videos a second time.
    result["mcap_validation"] = mcap_result
    return result


def _episode_id(path: Path) -> int:
    if path.suffix != ".mcap":
        raise VideoPipelineError(f"episode path must be an .mcap file, got {path}")
    if path.parent.name.isdigit():
        return int(path.parent.name)
    if path.stem.isdigit():
        return int(path.stem)
    raise VideoPipelineError(f"cannot identify numeric episode directory for MCAP: {path}")


def _require_pending_path(path: Path, dataset: Path, episode_id: int) -> None:
    expected_dir = pending_episode_dir(dataset, episode_id)
    if path.parent.resolve(strict=False) != expected_dir.resolve(strict=False):
        raise VideoPipelineError(
            f"publication accepts only a verified MCAP directly under {expected_dir}, not {path}; "
            "a published episode cannot be republished as pending"
        )
    if not path.is_file():
        raise VideoPipelineError(f"pending MCAP does not exist or is not a regular file: {path}")


def _stage_dir(dataset: Path, episode_id: int) -> Path:
    root = episode_dir(dataset, episode_id)
    root.mkdir(parents=True, exist_ok=True)
    directory = root / f".videos-transcoding-{uuid.uuid4().hex}"
    directory.mkdir()
    return directory


def _publish_directory(staged: Path, destination: Path, *, overwrite: bool) -> None:
    backup: Path | None = None
    if destination.exists():
        if not overwrite:
            raise VideoPipelineError(f"final video directory already exists: {destination}")
        backup = destination.with_name(f".{destination.name}.replaced-{uuid.uuid4().hex}")
        try:
            destination.replace(backup)
        except OSError as exc:
            raise VideoPipelineError(f"could not stage existing final video directory {destination}: {exc}") from exc
    try:
        staged.replace(destination)
    except OSError as exc:
        if backup is not None and not destination.exists():
            try:
                backup.replace(destination)
            except OSError:
                pass
        raise VideoPipelineError(f"could not publish final video directory {destination}: {exc}") from exc
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)


def _publish_mcap(source: Path, destination: Path, *, overwrite: bool) -> None:
    backup: Path | None = None
    if destination.exists():
        if not overwrite:
            raise VideoPipelineError(f"final MCAP already exists: {destination}")
        backup = destination.with_name(f".{destination.name}.replaced-{uuid.uuid4().hex}")
        try:
            destination.replace(backup)
        except OSError as exc:
            raise VideoPipelineError(f"could not stage existing final MCAP {destination}: {exc}") from exc
    try:
        source.replace(destination)
    except OSError as exc:
        if backup is not None and not destination.exists():
            try:
                backup.replace(destination)
            except OSError:
                pass
        raise VideoPipelineError(f"could not publish final MCAP {destination}: {exc}") from exc
    if backup is not None:
        try:
            backup.unlink()
        except OSError:
            pass


@dataclass(frozen=True)
class _PublicationBackups:
    """Existing public assets temporarily hidden during an overwrite."""

    video_dir: Path | None
    mcap: Path | None


def _backup_path(destination: Path) -> Path:
    return destination.with_name(f".{destination.name}.replaced-{uuid.uuid4().hex}")


def _stage_existing_public_assets(
    final_dir: Path,
    final_mcap: Path,
    *,
    overwrite: bool,
) -> _PublicationBackups:
    """Hide old public assets only after all replacement videos are verified.

    The episode directory is the publication marker. During replacement it
    is preferable to have no public episode briefly than to expose a new
    video directory beside an old MCAP. Both old paths are retained for
    rollback until the new video directory and MCAP have been published.
    """
    existing = (final_dir.exists(), final_mcap.exists())
    if any(existing) and not overwrite:
        existing_names = ", ".join(
            name for name, present in ((str(final_dir), existing[0]), (str(final_mcap), existing[1])) if present
        )
        raise VideoPipelineError(f"final assets already exist: {existing_names}; refusing to overwrite without --overwrite")

    video_backup: Path | None = None
    mcap_backup: Path | None = None
    try:
        if final_dir.exists():
            video_backup = _backup_path(final_dir)
            final_dir.replace(video_backup)
        if final_mcap.exists():
            mcap_backup = _backup_path(final_mcap)
            final_mcap.replace(mcap_backup)
    except OSError as exc:
        if mcap_backup is not None and mcap_backup.exists() and not final_mcap.exists():
            try:
                mcap_backup.replace(final_mcap)
            except OSError:
                pass
        if video_backup is not None and video_backup.exists() and not final_dir.exists():
            try:
                video_backup.replace(final_dir)
            except OSError:
                pass
        raise VideoPipelineError(f"could not stage existing final assets for replacement: {exc}") from exc
    return _PublicationBackups(video_dir=video_backup, mcap=mcap_backup)


def _restore_public_assets(
    backups: _PublicationBackups,
    final_dir: Path,
    final_mcap: Path,
    *,
    staged: Path,
    pending_mcap: Path,
    published_video: bool,
    published_mcap: bool,
) -> None:
    """Best-effort rollback that keeps the pending/live episode retryable."""
    if published_mcap and final_mcap.exists() and not pending_mcap.exists():
        try:
            final_mcap.replace(pending_mcap)
        except OSError:
            pass
    if published_video and final_dir.exists() and not staged.exists():
        try:
            final_dir.replace(staged)
        except OSError:
            pass
    if backups.mcap is not None and backups.mcap.exists() and not final_mcap.exists():
        try:
            backups.mcap.replace(final_mcap)
        except OSError:
            pass
    if backups.video_dir is not None and backups.video_dir.exists() and not final_dir.exists():
        try:
            backups.video_dir.replace(final_dir)
        except OSError:
            pass


def _discard_public_backups(backups: _PublicationBackups) -> None:
    if backups.video_dir is not None:
        shutil.rmtree(backups.video_dir, ignore_errors=True)
    if backups.mcap is not None:
        try:
            backups.mcap.unlink()
        except OSError:
            pass


def _publish_verified_episode(
    staged: Path,
    final_dir: Path,
    pending_mcap: Path,
    final_mcap: Path,
    *,
    overwrite: bool,
) -> None:
    """Publish a complete video directory before making its MCAP visible."""
    backups = _stage_existing_public_assets(final_dir, final_mcap, overwrite=overwrite)
    published_video = False
    published_mcap = False
    try:
        # Existing assets were moved aside above, so each individual helper
        # still has its simple atomic non-overwrite behavior.
        _publish_directory(staged, final_dir, overwrite=False)
        published_video = True
        _publish_mcap(pending_mcap, final_mcap, overwrite=False)
        published_mcap = True
    except BaseException:
        _restore_public_assets(
            backups,
            final_dir,
            final_mcap,
            staged=staged,
            pending_mcap=pending_mcap,
            published_video=published_video,
            published_mcap=published_mcap,
        )
        raise
    _discard_public_backups(backups)


def publish_live_episode(
    mcap_path: str | Path,
    config: DatasetConfig,
    *,
    overwrite: bool = False,
) -> DirectPublishedEpisode:
    """Publish a verified H.264 live episode without an offline encode.

    The live directory itself is the staged directory. It is first verified
    while still under ``videos_live`` and is then atomically renamed into the
    public ``videos`` tree before the pending MCAP is made visible. If MCAP
    publication fails, the shared publication helper returns the directory to
    ``videos_live`` and leaves the pending MCAP retryable.

    This helper intentionally does not acquire ``DatasetOperationLock``: the
    collector already owns that lock while finalizing a live episode.
    """
    if not uses_direct_h264_publication(config):
        raise VideoPipelineError(
            "direct live publication requires an H.264-final configuration; "
            "HEVC configurations must use yam-data-collect transcode"
        )
    if codec_family(config.collection.live_video_codec) != ("h264",):
        raise VideoPipelineError("H.264-final publication requires an H.264 live video codec")

    path = Path(mcap_path)
    episode_id = _episode_id(path)
    dataset = config.collection.dataset_dir
    _require_pending_path(path, dataset, episode_id)
    live_dir = live_video_dir(dataset, episode_id)
    final_dir = final_video_dir(dataset, episode_id)
    final_mcap = episode_dir(dataset, episode_id) / path.name
    if (final_dir.exists() or final_mcap.exists()) and not overwrite:
        raise VideoPipelineError(
            f"episode {episode_id} already has final assets; refusing to overwrite without --overwrite"
        )

    live_validation = validate_live_episode(path, config)
    # HEVC creates this parent through its private staging directory. Direct
    # H.264 reuses videos_live as the staging directory, so create the public
    # parent explicitly before its atomic rename.
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        # The validation above proves these exact H.264 bytes match the MCAP;
        # rename does not alter bytes, so no second encode or copy is needed.
        _publish_verified_episode(live_dir, final_dir, path, final_mcap, overwrite=overwrite)
    except BaseException:
        raise

    final_validation = dict(live_validation)
    final_validation["video_dir"] = str(final_dir)
    final_validation["expected_video_codecs"] = list(final_codec_family(config))
    try:
        pending_episode_dir(dataset, episode_id).rmdir()
    except OSError:
        pass
    return DirectPublishedEpisode(
        episode_id=episode_id,
        pending_mcap_path=pending_mcap_path(dataset, episode_id, filename=path.name),
        final_mcap_path=final_mcap,
        live_video_dir=live_dir,
        final_video_dir=final_dir,
        live_validation=live_validation,
        final_validation=final_validation,
    )


def publish_pending_episode(
    mcap_path: str | Path,
    config: DatasetConfig,
    *,
    overwrite: bool = False,
) -> DirectPublishedEpisode:
    """Retry direct publication of a verified pending H.264 episode.

    The live recorder invokes :func:`publish_live_episode` while it already
    owns the dataset lock. This public entry point is for recovery after a
    process or filesystem failure left a complete H.264-final episode under
    ``.pending`` and ``videos_live``.
    """
    if not uses_direct_h264_publication(config):
        raise VideoPipelineError(
            "publish-pending is only available for H.264-final configurations; "
            "HEVC configurations must use yam-data-collect transcode"
        )
    path = Path(mcap_path)
    with DatasetOperationLock(config.collection.dataset_dir, "publish-pending"):
        return publish_live_episode(path, config, overwrite=overwrite)


def _transcode_episode(mcap_path: Path, config: DatasetConfig, *, overwrite: bool) -> TranscodedEpisode:
    if uses_direct_h264_publication(config):
        raise VideoPipelineError(
            "H.264-final episodes are published during capture; no offline transcode is required"
        )
    episode_id = _episode_id(mcap_path)
    dataset = config.collection.dataset_dir
    _require_pending_path(mcap_path, dataset, episode_id)
    live_dir = live_video_dir(dataset, episode_id)
    final_dir = final_video_dir(dataset, episode_id)
    final_mcap = episode_dir(dataset, episode_id) / mcap_path.name
    if (final_dir.exists() or final_mcap.exists()) and not overwrite:
        raise VideoPipelineError(
            f"episode {episode_id} already has final assets; refusing to overwrite without --overwrite"
        )

    live_validation = validate_live_episode(mcap_path, config)
    staged = _stage_dir(dataset, episode_id)
    timeline_manifest = live_dir / "source_timeline.json"
    transcoded: dict[str, VideoTranscodeResult] = {}
    try:
        for role in CAMERA_TOPICS:
            transcoded[role] = transcode_to_hevc(
                live_dir / f"{role}.mp4",
                staged / f"{role}.mp4",
                crf=config.collection.video_crf,
                preset=config.collection.video_preset,
                threads=config.collection.video_threads,
                preserve_timestamps=timeline_manifest.is_file(),
            )
        if timeline_manifest.is_file():
            shutil.copy2(timeline_manifest, staged / timeline_manifest.name)
        policy = _policy(config)
        final_validation = validate_video_assets(
            mcap_path,
            policy,
            video_dir=staged,
            expected_video_codecs=final_codec_family(config),
            require_fps_match=False,
        )
        if not final_validation["passed"]:
            details = "; ".join(
                f"{role}: {entry.get('error', 'asset validation failed')}"
                for role, entry in final_validation["per_camera"].items()
                if not entry.get("passed")
            )
            raise VideoPipelineError(f"final HEVC validation failed for episode {episode_id}: {details}")
        # Publish videos before the numeric MCAP. A root-level MCAP is thus
        # never visible before all three final videos have passed validation.
        # Overwrites retain the previous complete pair until the replacement
        # can be published as a complete pair as well.
        _publish_verified_episode(
            staged,
            final_dir,
            mcap_path,
            final_mcap,
            overwrite=overwrite,
        )
    except BaseException:
        raise
    try:
        pending_episode_dir(dataset, episode_id).rmdir()
    except OSError:
        pass
    return TranscodedEpisode(
        episode_id=episode_id,
        pending_mcap_path=pending_mcap_path(dataset, episode_id, filename=mcap_path.name),
        final_mcap_path=final_mcap,
        live_video_dir=live_dir,
        final_video_dir=final_dir,
        live_validation=live_validation,
        final_validation=final_validation,
        videos=transcoded,
    )


def transcode_episode(mcap_path: str | Path, config: DatasetConfig, *, overwrite: bool = False) -> TranscodedEpisode:
    """Serially transcode and publish one verified pending episode."""
    if uses_direct_h264_publication(config):
        raise VideoPipelineError(
            "H.264-final episodes are published during capture; no offline transcode is required"
        )
    path = Path(mcap_path)
    with DatasetOperationLock(config.collection.dataset_dir, "transcode"):
        return _transcode_episode(path, config, overwrite=overwrite)


def transcode_dataset(config: DatasetConfig, *, overwrite: bool = False) -> tuple[TranscodedEpisode, ...]:
    """Serially process all complete pending MCAP files for one dataset."""
    if uses_direct_h264_publication(config):
        raise VideoPipelineError(
            "H.264-final episodes are published during capture; no offline transcode is required"
        )
    dataset = config.collection.dataset_dir
    pending_root = dataset / PENDING_ROOT
    mcap_paths = sorted(
        (
            path
            for path in pending_root.glob("*/*.mcap")
            if path.parent.name.isdigit() and not path.name.endswith(".partial.mcap")
        ),
        key=lambda path: (int(path.parent.name), path.name),
    )
    if not mcap_paths:
        raise VideoPipelineError(f"no pending episode MCAP files found under {pending_root}")
    with DatasetOperationLock(dataset, "transcode-dataset"):
        return tuple(_transcode_episode(path, config, overwrite=overwrite) for path in mcap_paths)
