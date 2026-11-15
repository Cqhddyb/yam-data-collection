#!/usr/bin/env python3
"""Read only non-episode camera-test MP4 headers; never opens hardware."""
import argparse
import json
from pathlib import Path

import av


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    rows = []
    for project in ("yam_data_collection", "yam_single_arm_collection"):
        base = root / project / "camera_tests"
        if args.out.resolve().is_relative_to(base):
            raise SystemExit("Output must be outside the input camera-test directories")
        for path in sorted(base.rglob("*.mp4")) if base.exists() else []:
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                rows.append({
                    "sample": f"T{len(rows)+1:02d}", "project": project,
                    "role": path.stem, "bytes": path.stat().st_size,
                    "codec": stream.codec_context.name, "width": stream.width,
                    "height": stream.height, "frames_header": stream.frames,
                    "container_average_rate_hz": float(stream.average_rate) if stream.average_rate else None,
                    "duration_s": float(stream.duration * stream.time_base) if stream.duration is not None else None,
                    "pixel_format": stream.codec_context.format.name,
                    "source_relative_path": str(path.relative_to(root)),
                })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"videos": len(rows), "frames_header": sum(r["frames_header"] for r in rows),
                      "bytes": sum(r["bytes"] for r in rows)}))


if __name__ == "__main__":
    main()
