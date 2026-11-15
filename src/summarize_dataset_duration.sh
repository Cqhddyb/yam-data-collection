#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
collector_python="$root/yam_data_collection/.venv/bin/python"
if [[ ! -x "$collector_python" ]]; then
    printf 'Collector environment not found: %s\n' "$collector_python" >&2
    exit 1
fi
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="/opt/ros/humble/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$collector_python" "$root/yam_data_collection/scripts/summarize_dataset_duration.py" "$@"
