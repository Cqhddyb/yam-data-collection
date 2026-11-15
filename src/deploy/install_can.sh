#!/usr/bin/env bash
set -euo pipefail

deployment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$EUID" -ne 0 ]]; then
    printf '%s\n' "Run: sudo bash $deployment_dir/install_can.sh" >&2
    exit 1
fi
if pgrep -f '[p]ython.*minimum_gello.py' >/dev/null; then
    printf '%s\n' 'Stop all robot control processes before changing CAN interfaces.' >&2
    exit 1
fi
declare -A expected=(
    [REPLACE_LEADER_L_CAN_SERIAL]=can_leader_l
    [REPLACE_FOLLOWER_L_CAN_SERIAL]=can_follower_l
    [REPLACE_LEADER_R_CAN_SERIAL]=can_leader_r
    [REPLACE_FOLLOWER_R_CAN_SERIAL]=can_follower_r
)
declare -A current=()
for interface in /sys/class/net/*; do
    [[ -e "$interface/device/driver" ]] || continue
    [[ "$(basename -- "$(readlink -f -- "$interface/device/driver")")" == gs_usb ]] || continue
    device="$(readlink -f -- "$interface/device")"
    serial="$(cat -- "$device/../serial")"
    [[ -n "${expected[$serial]:-}" ]] || continue
    if [[ -n "${current[$serial]:-}" ]]; then
        printf 'Duplicate CAN serial: %s\n' "$serial" >&2
        exit 1
    fi
    current[$serial]="$(basename -- "$interface")"
done
for serial in "${!expected[@]}"; do
    if [[ -z "${current[$serial]:-}" ]]; then
        printf 'Missing CAN adapter: %s (%s)\n' "$serial" "${expected[$serial]}" >&2
        exit 1
    fi
    target="${expected[$serial]}"
    if [[ "${current[$serial]}" != "$target" && -e "/sys/class/net/$target" ]]; then
        printf 'Refusing to replace occupied interface: %s\n' "$target" >&2
        exit 1
    fi
    printf '%s: %s -> %s\n' "$serial" "${current[$serial]}" "$target"
done
for module in can can_raw can_dev gs_usb; do
    modprobe "$module"
done
timestamp="$(date +%Y%m%d_%H%M%S)"
for destination in /etc/udev/rules.d/90-yam-can.rules /etc/systemd/system/yam-can@.service; do
    if [[ -e "$destination" ]]; then
        cp -a -- "$destination" "$destination.backup.$timestamp"
    fi
done
install -m 0644 "$deployment_dir/90-yam-can.rules" /etc/udev/rules.d/90-yam-can.rules
install -m 0644 "$deployment_dir/yam-can@.service" /etc/systemd/system/yam-can@.service
systemctl daemon-reload
udevadm control --reload-rules
for serial in "${!expected[@]}"; do
    target="${expected[$serial]}"
    systemctl stop "yam-can@$target.service"
    systemctl reset-failed "yam-can@$target.service"
    ip link set dev "${current[$serial]}" down
    if [[ "${current[$serial]}" != "$target" ]]; then
        ip link set dev "${current[$serial]}" name "$target"
    fi
done
for target in "${expected[@]}"; do
    udevadm trigger --action=add --subsystem-match=net --sysname-match="$target"
done
udevadm settle
for target in "${expected[@]}"; do
    if ! systemctl start "yam-can@$target.service"; then
        systemctl status "yam-can@$target.service" --no-pager -l || true
        journalctl -b -u "yam-can@$target.service" --no-pager -n 20 || true
        exit 1
    fi
    ip -details link show dev "$target"
done
printf '%s\n' 'CAN configuration installed. No motor commands were sent.'
printf '%s\n' 'Automatic Bus-Off restart is disabled for this gs_usb driver (restart-ms 0).'
