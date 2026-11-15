#!/usr/bin/env python3
"""Run Dream's official, key-free local dataset checks; never upload data.

Run this with ~/dream-env/bin/python, independently of the capture environment.
"""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path


PARTS = {
    "right_arm": [0, 6],
    "right_gripper": [6, 7],
    "left_arm": [7, 13],
    "left_gripper": [13, 14],
}
DEFAULT_CONFIG = {
    "state": PARTS,
    "action": PARTS.copy(),
    "video": {
        "overhead": "observation.images.cam_high",
        "wrist_l": "observation.images.cam_left",
        "wrist_r": "observation.images.cam_right",
    },
    "head_camera": "overhead",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--config", type=Path, help="optional Dream embodiment config JSON")
    parser.add_argument("--report", type=Path, help="save a new report outside the dataset")
    args = parser.parse_args()
    dataset = args.dataset.expanduser().resolve()
    if not dataset.is_dir():
        parser.error(f"dataset directory does not exist: {dataset}")
    destination = args.report.expanduser().resolve() if args.report else None
    if destination is not None:
        if destination == dataset or dataset in destination.parents:
            parser.error("save reports outside the dataset directory")
        if destination.exists():
            parser.error(f"refusing to overwrite existing report: {destination}")
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else DEFAULT_CONFIG

    # Limit these libraries independently of the recorder; no hardware code is imported.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    import pyarrow as pa
    from dream import datasets

    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    # Official standalone API: unlike constructing DreamClient, no credentials needed.
    report = datasets.validate_dataset(dataset, config)
    result = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "sdk": {"dreamlabs-client": importlib.metadata.version("dreamlabs-client")},
        "validation_api": "dream.datasets.validate_dataset",
        "validation_scope": "official local format/content checks, no upload or robot access",
        "embodiment_config": config,
        "report": dataclasses.asdict(report),
    }
    try:
        result["sdk"]["dreamlabs"] = importlib.metadata.version("dreamlabs")
    except importlib.metadata.PackageNotFoundError:
        pass
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(rendered)
        print(f"Report saved: {destination}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
