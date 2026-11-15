"""Publish scoped H.264 episodes using the original atomic publication helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from yam_data_collection.config import DatasetConfig
from yam_data_collection.video_pipeline import (
    DirectPublishedEpisode,
    VideoPipelineError,
    _episode_id,
    _publish_verified_episode,
    _require_pending_path,
    episode_dir,
    final_video_dir,
    live_video_dir,
    pending_episode_dir,
    pending_mcap_path,
)

from .quality import validate_mcap, validate_video_assets


def validate_live_episode(path: str | Path, config: DatasetConfig) -> dict[str, Any]:
    path = Path(path)
    episode_id = _episode_id(path)
    dataset = config.collection.dataset_dir
    _require_pending_path(path, dataset, episode_id)
    mcap_result = validate_mcap(path, config)
    result = validate_video_assets(path, config, video_dir=live_video_dir(dataset, episode_id))
    if not result["passed"]:
        raise VideoPipelineError(f"single-arm live video identity validation failed: {result}")
    result["formal_mcap_passed"] = mcap_result["output_mcap_passed"]
    result["formal_failed_checks"] = list(mcap_result["failed_checks"])
    result["mcap_validation"] = mcap_result
    return result


def publish_live_episode(path: str | Path, config: DatasetConfig) -> DirectPublishedEpisode:
    """Promote verified bytes without overwriting an existing episode or transcoding."""
    path = Path(path)
    episode_id = _episode_id(path)
    dataset = config.collection.dataset_dir
    _require_pending_path(path, dataset, episode_id)
    live_dir = live_video_dir(dataset, episode_id)
    final_dir = final_video_dir(dataset, episode_id)
    final_mcap = episode_dir(dataset, episode_id) / path.name
    if final_dir.exists() or final_mcap.exists():
        raise VideoPipelineError(f"episode {episode_id} already exists; refusing to overwrite")
    validation = validate_live_episode(path, config)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    _publish_verified_episode(live_dir, final_dir, path, final_mcap, overwrite=False)
    final_validation = {**validation, "video_dir": str(final_dir)}
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
        live_validation=validation,
        final_validation=final_validation,
    )
