#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -ne 2 || "$2" != --confirm-motion ]]; then
    printf 'Usage: bash %s/leader_motion.sh left|right --confirm-motion\n' "$root/yam_single_arm_collection" >&2
    exit 2
fi
case "$1" in
    left) interface=can_leader_l; serial=REPLACE_LEADER_L_CAN_SERIAL; port=1235; telemetry_port=1245 ;;
    right) interface=can_leader_r; serial=REPLACE_LEADER_R_CAN_SERIAL; port=1234; telemetry_port=1244 ;;
    *) printf 'Select left or right explicitly.\n' >&2; exit 2 ;;
esac
if [[ ! -e "/sys/class/net/$interface/device" ]]; then
    printf 'CAN interface missing: %s\n' "$interface" >&2
    exit 1
fi
device="$(readlink -f -- "/sys/class/net/$interface/device")"
if [[ "$(cat -- "$device/../serial")" != "$serial" ]]; then
    printf 'Unexpected USB serial for %s; refusing motor startup.\n' "$interface" >&2
    exit 1
fi
link_json="$(ip -details -json link show dev "$interface")"
"$root/yam_data_collection/.venv/bin/python" -c '
import json, sys
link = json.loads(sys.argv[1])[0]
info = link.get("linkinfo", {}).get("info_data", {})
if not ("UP" in link["flags"] and info.get("state") == "ERROR-ACTIVE" and info.get("bittiming", {}).get("bitrate") == 1000000 and info.get("restart_ms") == 0 and link.get("txqlen") == 2000):
    raise SystemExit("CAN interface must be UP / ERROR-ACTIVE, 1 Mbps, restart-ms 0, txqueuelen 2000")
' "$link_json"
if [[ -n "$(ss -H -ltn "sport = :$telemetry_port")" ]]; then
    printf 'Port %s already occupied; refusing duplicate leader controller.\n' "$telemetry_port" >&2
    exit 1
fi
if [[ -z "$(ss -H -ltn "sport = :$port")" ]]; then
    printf 'Follower port %s is not listening; start the selected follower first.\n' "$port" >&2
    exit 1
fi
arguments=(--mode leader --gripper yam_teaching_handle --can-channel "$interface"
    --server-host 127.0.0.1 --server-port "$port" --telemetry-port "$telemetry_port"
    --bilateral-kp 0.0 --enable-preprogrammed-motion)
if [[ "$1" == right ]]; then
    arguments+=(--gripper-command-closed REPLACE_RIGHT_GRIPPER_CLOSED --gripper-command-open REPLACE_RIGHT_GRIPPER_OPEN)
fi
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="/opt/ros/humble/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$root/i2rt"
exec taskset -c 0-11 .venv/bin/python examples/minimum_gello/minimum_gello.py "${arguments[@]}"
