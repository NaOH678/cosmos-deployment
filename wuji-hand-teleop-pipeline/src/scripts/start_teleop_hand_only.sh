#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 2 ]]; then
    echo "Usage: $0 [right|left|both] [ramp-seconds]" >&2
    exit 2
fi

ACTIVE_HAND="${1:-right}"
RAMP_SECONDS="${2:-5.0}"

case "${ACTIVE_HAND}" in
    right)
        ENABLE_LEFT=false
        ENABLE_RIGHT=true
        ;;
    left)
        ENABLE_LEFT=true
        ENABLE_RIGHT=false
        ;;
    both)
        ENABLE_LEFT=true
        ENABLE_RIGHT=true
        ;;
    *)
        echo "ERROR: active hand must be right, left, or both." >&2
        echo "Usage: $0 [right|left|both] [ramp-seconds]" >&2
        exit 2
        ;;
esac

if [[ ! "${RAMP_SECONDS}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "ERROR: ramp-seconds must be a non-negative number." >&2
    exit 2
fi

if ! docker ps --format '{{.Names}}' | grep -qx 'wuji-hand-teleop'; then
    echo "ERROR: wuji-hand-teleop container is not running." >&2
    echo "Start it first from the repository root:" >&2
    echo "  cd docker && docker compose up -d" >&2
    exit 1
fi

if ! docker exec wuji-hand-teleop bash -lc '
test -f /home/wuji/ros2_ws/install/setup.bash
source /opt/ros/humble/setup.bash
source /home/wuji/ros2_ws/install/setup.bash
ros2 pkg prefix wuji_teleop_bringup >/dev/null
'; then
    echo "ERROR: the container ROS workspace is not ready or is not built." >&2
    exit 1
fi

HAND_PROCESS_PATTERN='[w]ujihand_driver_node|[w]ujihand_controller|[m]anus_data_publisher'
if docker exec wuji-hand-teleop pgrep -f "${HAND_PROCESS_PATTERN}" \
    >/dev/null 2>&1; then
    echo "ERROR: a WujiHand or MANUS process is already running:" >&2
    docker exec wuji-hand-teleop pgrep -af "${HAND_PROCESS_PATTERN}" >&2 || true
    echo "Stop the existing teleop/record/deployment session before starting hand-only teleop." >&2
    exit 1
fi

echo "Starting supervised WujiHand-only teleoperation"
echo "  active hand: ${ACTIVE_HAND}"
echo "  input:       MANUS (from wujihand_ik.yaml)"
echo "  ramp:        ${RAMP_SECONDS}s"
echo "  controls:    r Recovery, a Enable, x Disable, q/Ctrl+C Exit"

cleanup_owned_hand_processes() {
    if ! docker ps --format '{{.Names}}' | grep -qx 'wuji-hand-teleop'; then
        return
    fi
    docker exec wuji-hand-teleop bash -lc '
pattern="[w]ujihand_driver_node|[w]ujihand_controller|[m]anus_data_publisher"
pids="$(pgrep -f "${pattern}" || true)"
if [[ -z "${pids}" ]]; then
    exit 0
fi
kill -INT ${pids} 2>/dev/null || true
sleep 1
pids="$(pgrep -f "${pattern}" || true)"
if [[ -n "${pids}" ]]; then
    kill -TERM ${pids} 2>/dev/null || true
    sleep 1
fi
pids="$(pgrep -f "${pattern}" || true)"
if [[ -n "${pids}" ]]; then
    kill -KILL ${pids} 2>/dev/null || true
fi
' >/dev/null 2>&1 || true
}

trap cleanup_owned_hand_processes EXIT
SESSION_STATUS=0
docker exec -it \
    -e ROS_DOMAIN_ID=112 \
    -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    -e ROS2CLI_DISABLE_DAEMON=1 \
    -e WUJI_ENABLE_LEFT_HAND="${ENABLE_LEFT}" \
    -e WUJI_ENABLE_RIGHT_HAND="${ENABLE_RIGHT}" \
    -e WUJI_ACTIVE_HAND="${ACTIVE_HAND}" \
    -e WUJI_HAND_RAMP_SECONDS="${RAMP_SECONDS}" \
    wuji-hand-teleop bash -lc '
unset CYCLONEDDS_URI
source /opt/ros/humble/setup.bash
source /home/wuji/ros2_ws/install/setup.bash
exec ros2 run wuji_data_pipeline hand_teleop_session \
    --active-hand "${WUJI_ACTIVE_HAND}" \
    --ramp-seconds "${WUJI_HAND_RAMP_SECONDS}"
' || SESSION_STATUS=$?

trap - EXIT
cleanup_owned_hand_processes
exit "${SESSION_STATUS}"
