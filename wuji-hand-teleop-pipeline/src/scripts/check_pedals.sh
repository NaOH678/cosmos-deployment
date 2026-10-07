#!/usr/bin/env bash
set -euo pipefail

INPUT_BY_ID="/dev/input/by-id"
required_ready=true

check_pedal() {
    local pedal_id="$1"
    local serial="$2"
    local key_name="$3"
    local required="$4"
    local path="${INPUT_BY_ID}/usb-LinTx_LinTx_Keyboard_${serial}-if01-event-kbd"

    if [[ -e "${path}" ]]; then
        printf 'Pedal %s  %-3s  %s  ONLINE   %s\n' \
            "${pedal_id}" "${key_name}" "${serial}" "$(readlink -f "${path}")"
        return
    fi

    printf 'Pedal %s  %-3s  %s  OFFLINE\n' \
        "${pedal_id}" "${key_name}" "${serial}"
    if [[ "${required}" == "required" ]]; then
        required_ready=false
    fi
}

check_pedal 1 BE11FDB0 F7 required
check_pedal 2 BE11E330 F8 required
check_pedal 3 BE7FFADA k optional

if [[ "${required_ready}" == "true" ]]; then
    echo "Required pedals: READY"
    exit 0
fi

echo "Required pedals: NOT READY"
exit 1
