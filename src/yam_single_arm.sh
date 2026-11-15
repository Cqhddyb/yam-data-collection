#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python="$root/yam_data_collection/.venv/bin/python"
if [[ ! -x "$python" ]]; then
    printf 'Collector environment not found: %s\n' "$python" >&2
    exit 1
fi
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="/opt/ros/humble/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$root"
exec "$python" -m yam_single_arm_collection "$@"
