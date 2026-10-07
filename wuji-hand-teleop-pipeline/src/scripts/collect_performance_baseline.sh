#!/usr/bin/env bash
set -euo pipefail

LABEL="${1:-baseline}"
DURATION="${2:-60}"

if ! [[ "${LABEL}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
    echo "ERROR: label must match [A-Za-z0-9][A-Za-z0-9_-]*" >&2
    exit 2
fi
if ! [[ "${DURATION}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
    echo "ERROR: duration must be a non-negative number" >&2
    exit 2
fi
if ! docker ps --format '{{.Names}}' | grep -qx 'wuji-hand-teleop'; then
    echo "ERROR: wuji-hand-teleop container is not running." >&2
    exit 1
fi

docker exec -it \
    -e ROS_DOMAIN_ID=112 \
    -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    -e ROS2CLI_DISABLE_DAEMON=1 \
    wuji-hand-teleop bash -lc '
unset CYCLONEDDS_URI
source /opt/ros/humble/setup.bash
source /home/wuji/ros2_ws/install/setup.bash
exec ros2 run controller tianji_performance_collector "$@"
' bash --duration "${DURATION}" --label "${LABEL}"
