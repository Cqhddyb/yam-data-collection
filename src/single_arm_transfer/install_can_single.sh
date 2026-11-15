#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 1 || $# -gt 2 || ( "$1" != left && "$1" != right ) || ( $# -eq 2 && "$2" != --preview ) ]]; then
    printf 'Usage: bash %s/single_arm_transfer/install_can_single.sh left|right [--preview]\n' "$root" >&2
    exit 2
fi
if [[ ! -f "$root/.yam-single-arm-transfer" ]]; then
    printf '%s\n' 'Run inside the extracted transfer package.' >&2
    exit 1
fi
if [[ "$EUID" -ne 0 && "${2:-}" != --preview ]]; then
    printf '%s\n' 'CAN installation requires sudo, with robot controllers stopped.' >&2
    exit 1
fi
if [[ "$1" == left ]]; then
    suffix=_l
else
    suffix=_r
fi
staging_dir="$(mktemp -d -t yam-single-can.XXXXXXXX)"
trap 'rm -rf -- "$staging_dir"' EXIT
awk -v suffix="$suffix" '
    /^declare -A expected=\(/ { in_expected=1; print; next }
    in_expected && /^\)/ { in_expected=0; print; next }
    !in_expected || $0 ~ ("=can_(leader|follower)" suffix "[[:space:]]*$") { print }
' "$root/deploy/install_can.sh" > "$staging_dir/install_can.sh"
awk -v suffix="$suffix" '
    index($0, "NAME=\"can_leader" suffix "\"") || index($0, "NAME=\"can_follower" suffix "\"")
' "$root/deploy/90-yam-can.rules" > "$staging_dir/90-yam-can.rules"
if [[ "$(wc -l < "$staging_dir/90-yam-can.rules")" -ne 2 ]]; then
    printf '%s\n' 'Expected exactly two selected-side CAN naming rules.' >&2
    exit 1
fi
cp -- "$root/deploy/yam-can@.service" "$staging_dir/yam-can@.service"
if [[ "${2:-}" == --preview ]]; then
    sed -n '/^declare -A expected=(/,/^)/p' "$staging_dir/install_can.sh"
    cat "$staging_dir/90-yam-can.rules" "$staging_dir/yam-can@.service"
    printf '%s\n' 'Preview only; no system or hardware state was changed.'
    exit 0
fi
printf 'Installing only the %s leader/follower CAN interfaces.\n' "$1"
bash "$staging_dir/install_can.sh"
