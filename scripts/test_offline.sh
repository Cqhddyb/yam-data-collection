#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$root/src"
yam_data_collection/.venv/bin/python -m pytest yam_data_collection/tests yam_single_arm_collection/tests single_arm_transfer/tests -q -c yam_data_collection/pyproject.toml -p no:cacheprovider --import-mode=importlib
cd "$root/src/i2rt"
.venv/bin/python -m pytest -q -c pyproject.toml -p no:cacheprovider \
  i2rt/robots/tests/test_minimum_gello_health.py \
  i2rt/robots/tests/test_minimum_gello_lag.py \
  i2rt/robots/tests/test_minimum_gello_telemetry.py \
  i2rt/robots/tests/test_minimum_gello_preprogrammed.py \
  i2rt/robots/tests/test_motor_chain_health.py \
  i2rt/robots/tests/test_motor_chain_recording.py \
  i2rt/robots/tests/test_passive_encoder.py \
  i2rt/robots/tests/test_read_gripper_motor_angle.py
.venv/bin/python -m pytest -q -c pyproject.toml -p no:cacheprovider \
  i2rt/robots/tests/test_robot_variants.py \
  i2rt/robots/tests/test_assembly.py
