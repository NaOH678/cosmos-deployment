#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PI05_CONFIG="/home/wuji/ros2_ws/src/wuji_data_pipeline/config/pi05_protocol_v2.yaml"

for argument in "$@"; do
    if [[ "${argument}" == "--config" ]]; then
        echo "ERROR: start_pi05_deployment_session.sh fixes the PI05 config; do not pass --config." >&2
        exit 2
    fi
done

exec "${SCRIPT_DIR}/start_deployment_session.sh" \
    "$@" \
    --config "${PI05_CONFIG}"
