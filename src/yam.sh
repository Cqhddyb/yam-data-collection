#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
collector="$root/yam_data_collection"
config="$collector/config.host.yaml"
capture_mode="${YAM_MODE:-60hz}"
if [[ "$capture_mode" != 60hz && "$capture_mode" != 30hz ]]; then
    printf 'YAM_MODE must be 30hz or 60hz, not %s\n' "$capture_mode" >&2
    exit 2
fi
command_name="${1:-help}"
if [[ $# -gt 0 ]]; then
    shift
fi
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="/opt/ros/humble/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
case "$command_name" in
    check)
        cd "$collector"
        exec .venv/bin/yam-data-collect "$command_name" --config "$config" --mode "$capture_mode" "$@"
        ;;
    camera-test)
        cd "$collector"
        exec taskset -c 12-19 .venv/bin/yam-data-collect camera-test --config "$config" --mode "$capture_mode" "$@"
        ;;
    capture)
        cd "$collector"
        .venv/bin/yam-data-collect check --config "$config" --mode "$capture_mode"
        exec .venv/bin/python scripts/capture_queued.py --config "$config" --mode "$capture_mode" --video-timeline source "$@"
        ;;
    follower-r|follower-l|leader-r|leader-l)
        if [[ "${1:-}" != --confirm-motion || $# -ne 1 ]]; then
            printf '%s\n' 'Robot startup can enable motors and calibrate the gripper.' \
                'Clear the workspace, verify role mapping and gripper endpoints, and keep the emergency stop ready.' \
                "Then run: bash $root/yam.sh $command_name --confirm-motion" >&2
            exit 2
        fi
        case "$command_name" in
            follower-r) interface=can_follower_r; serial=REPLACE_FOLLOWER_R_CAN_SERIAL; port=1234; telemetry_port=1234; mode=follower ;;
            follower-l) interface=can_follower_l; serial=REPLACE_FOLLOWER_L_CAN_SERIAL; port=1235; telemetry_port=1235; mode=follower ;;
            leader-r) interface=can_leader_r; serial=REPLACE_LEADER_R_CAN_SERIAL; port=1234; telemetry_port=1244; mode=leader ;;
            leader-l) interface=can_leader_l; serial=REPLACE_LEADER_L_CAN_SERIAL; port=1235; telemetry_port=1245; mode=leader ;;
        esac
        if [[ ! -e "/sys/class/net/$interface/device" ]]; then
            printf 'CAN interface missing: %s; run sudo bash %s/deploy/install_can.sh first.\n' "$interface" "$root" >&2
            exit 1
        fi
        device="$(readlink -f -- "/sys/class/net/$interface/device")"
        if [[ "$(cat -- "$device/../serial")" != "$serial" ]]; then
            printf 'Unexpected USB serial for %s; refusing motor startup.\n' "$interface" >&2
            exit 1
        fi
        link_json="$(ip -details -json link show dev "$interface")"
        "$collector/.venv/bin/python" -c '
import json, sys
link = json.loads(sys.argv[1])[0]
info = link.get("linkinfo", {}).get("info_data", {})
if not ("UP" in link["flags"] and info.get("state") == "ERROR-ACTIVE" and info.get("bittiming", {}).get("bitrate") == 1000000 and info.get("restart_ms") == 0 and link.get("txqlen") == 2000):
    raise SystemExit("CAN interface must be UP / ERROR-ACTIVE, 1 Mbps, restart-ms 0, txqueuelen 2000")
' "$link_json"
        if ss -H -ltn "sport = :$telemetry_port" | grep -q .; then
            printf 'Port %s is already occupied; refusing duplicate controller.\n' "$telemetry_port" >&2
            exit 1
        fi
        arguments=(--mode "$mode" --can-channel "$interface" --server-port "$port")
        if [[ "$mode" == follower ]]; then
            arguments+=(--gripper linear_4310)
        else
            arguments+=(--gripper yam_teaching_handle --server-host 127.0.0.1 --telemetry-port "$telemetry_port" --bilateral-kp 0.0)
            if [[ "$command_name" == leader-r ]]; then
                arguments+=(--gripper-command-closed REPLACE_RIGHT_GRIPPER_CLOSED --gripper-command-open REPLACE_RIGHT_GRIPPER_OPEN)
            fi
        fi
        cd "$root/i2rt"
        exec taskset -c 0-11 .venv/bin/python examples/minimum_gello/minimum_gello.py "${arguments[@]}"
        ;;
    help|--help|-h)
        printf '%s\n' 'Usage: bash yam.sh COMMAND [OPTIONS]' \
            '  check                          Read-only Portal telemetry check' \
            '  camera-test --duration 180 --no-video' \
            '  camera-test --duration 30       Includes H.264/MP4 verification' \
            '  follower-r --confirm-motion    Terminal 1 (can enable motors)' \
            '  follower-l --confirm-motion    Terminal 2 (can enable motors)' \
            '  leader-r --confirm-motion      Terminal 3 (can enable motors)' \
            '  leader-l --confirm-motion      Terminal 4 (can enable motors)' \
            '  capture                        Queued H.264, source timeline' \
            'Set YAM_MODE=30hz for the 30 Hz profile; the default remains 60hz.'
        ;;
    *) printf 'Unknown command: %s\n' "$command_name" >&2; exit 2 ;;
esac
