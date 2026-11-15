#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
command -v uv >/dev/null
for program in ffmpeg ffprobe ip ss taskset; do
    command -v "$program" >/dev/null || { printf 'Missing system tool: %s\n' "$program" >&2; exit 1; }
done
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$root/src/i2rt"
uv sync --frozen --python 3.11
cd "$root/src/yam_data_collection"
uv sync --frozen --python 3.11 --extra camera --extra timeline --extra dev
printf '%s\n' 'Environments created. Device templates still require local configuration.'
