#!/usr/bin/env python3
"""Run the experimental queued-state YAM capture collector.

The normal ``yam-data-collect capture`` command intentionally remains on the
production collector. This entry point only moves joint-state MCAP writes to a
bounded worker queue; camera capture, video encoding, and validation thresholds
remain unchanged. Like the terminal production entry point, failed validation
episodes are removed and passed episodes can be reviewed with ``q``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from yam_data_collection.config import capture_mode_cli_choices, load_config
from yam_data_collection.queued_collector import QueuedEpisodeCollector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Host-specific YAML configuration")
    parser.add_argument(
        "--mode",
        choices=capture_mode_cli_choices(),
        required=True,
        help="Formal capture contract to resolve from the configuration",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show low-rate camera previews and four-arm joint trends",
    )
    parser.add_argument(
        "--video-timeline",
        choices=("fixed", "source"),
        default="fixed",
        help="Video timing: fixed uses the existing configured FPS; source uses MCAP capture timestamps",
    )
    args = parser.parse_args()
    config = load_config(args.config, capture_mode=args.mode)
    collector = QueuedEpisodeCollector(
        config,
        enable_preview=args.preview,
        auto_delete_failed_validation=True,
        enable_post_save_review=True,
        video_timeline=args.video_timeline,
    )
    collector.run_interactive()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
