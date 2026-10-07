#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [both|left|right]" >&2
    exit 2
fi

ACTIVE_ARM="${1:-both}"
case "${ACTIVE_ARM}" in
    both|left|right) ;;
    *)
        echo "Usage: $0 [both|left|right]" >&2
        exit 2
        ;;
esac

if ! docker ps --format '{{.Names}}' | grep -qx 'wuji-hand-teleop'; then
    echo "ERROR: wuji-hand-teleop container is not running." >&2
    exit 1
fi

if ! docker ps --format '{{.Names}}' | grep -qx 'wuji-openvr-input'; then
    echo "ERROR: wuji-openvr-input is not running." >&2
    echo "Start it first: ./src/scripts/start_openvr_input.sh" >&2
    exit 1
fi

OPENVR_ENV="$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' wuji-openvr-input)"
if ! grep -qx 'ROS_DOMAIN_ID=112' <<<"${OPENVR_ENV}"; then
    echo "ERROR: wuji-openvr-input was not started with ROS_DOMAIN_ID=112." >&2
    exit 1
fi
if ! grep -qx 'RMW_IMPLEMENTATION=rmw_cyclonedds_cpp' <<<"${OPENVR_ENV}"; then
    echo "ERROR: wuji-openvr-input is not using the documented CycloneDDS RMW." >&2
    exit 1
fi

exec docker exec -it \
    -e ROS_DOMAIN_ID=112 \
    -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    -e ROS2CLI_DISABLE_DAEMON=1 \
    -e WUJI_ACTIVE_ARM="${ACTIVE_ARM}" \
    wuji-hand-teleop bash -lc '
unset CYCLONEDDS_URI
source /opt/ros/humble/setup.bash
source /home/wuji/ros2_ws/install/setup.bash
exec ros2 run wuji_data_pipeline arm_teleop_session "${WUJI_ACTIVE_ARM}"
'
