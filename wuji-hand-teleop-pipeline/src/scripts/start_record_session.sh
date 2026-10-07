#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

ACTIVE_HAND="right"
WITH_CAMERA=false
TASK_NAME=""
OUTPUT_DIR=""
HANDOFF_RAMP_SEC=""
CAMERA_TRANSPORT="${WUJI_CAMERA_TRANSPORT:-direct}"
OPENVR_STARTED_BY_SESSION=false

cleanup_openvr() {
    if [[ "${OPENVR_STARTED_BY_SESSION}" != true ]]; then
        return
    fi
    if docker ps --format '{{.Names}}' | grep -qx 'wuji-openvr-input'; then
        echo "Stopping session-owned OpenVR Tracker input..."
        docker stop --time 3 wuji-openvr-input >/dev/null 2>&1 || true
    fi
}
trap cleanup_openvr EXIT

if [[ $# -gt 0 && "${1}" != --* ]]; then
    ACTIVE_HAND="${1}"
    shift
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-camera)
            WITH_CAMERA=true
            shift
            ;;
        --task)
            [[ $# -ge 2 ]] || {
                echo "ERROR: --task requires a task name" >&2
                exit 2
            }
            TASK_NAME="$2"
            shift 2
            ;;
        --output-dir)
            [[ $# -ge 2 ]] || {
                echo "ERROR: --output-dir requires a container path" >&2
                exit 2
            }
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --handoff-ramp-sec)
            [[ $# -ge 2 ]] || {
                echo "ERROR: --handoff-ramp-sec requires seconds" >&2
                exit 2
            }
            HANDOFF_RAMP_SEC="$2"
            shift 2
            ;;
        --camera-transport)
            [[ $# -ge 2 ]] || {
                echo "ERROR: --camera-transport requires direct or ros" >&2
                exit 2
            }
            CAMERA_TRANSPORT="$2"
            shift 2
            ;;
        *)
            echo "Usage: $0 [right|left|both] [--with-camera] [--task NAME] [--output-dir PATH] [--handoff-ramp-sec SEC] [--camera-transport direct|ros]" >&2
            exit 2
            ;;
    esac
done

case "${ACTIVE_HAND}" in
    both|left|right) ;;
    *)
        echo "Usage: $0 [right|left|both] [--with-camera] [--task NAME] [--output-dir PATH] [--handoff-ramp-sec SEC]" >&2
        exit 2
        ;;
esac

case "${CAMERA_TRANSPORT}" in
    direct|ros) ;;
    *)
        echo "ERROR: --camera-transport must be direct or ros" >&2
        exit 2
        ;;
esac

if [[ -n "${TASK_NAME}" ]]; then
    if [[ ! "${TASK_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
        echo "ERROR: task name must match [A-Za-z0-9][A-Za-z0-9_-]*" >&2
        exit 2
    fi
    if [[ -z "${OUTPUT_DIR}" ]]; then
        OUTPUT_DIR="/home/wuji/datasets/tianji_wuji/${TASK_NAME}"
    fi
fi

if [[ -n "${HANDOFF_RAMP_SEC}" ]] && \
   [[ ! "${HANDOFF_RAMP_SEC}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
    echo "ERROR: --handoff-ramp-sec must be a non-negative number" >&2
    exit 2
fi

if ! docker ps --format '{{.Names}}' | grep -qx 'wuji-hand-teleop'; then
    echo "ERROR: wuji-hand-teleop container is not running." >&2
    exit 1
fi

if ! docker ps --format '{{.Names}}' | grep -qx 'wuji-openvr-input'; then
    echo "OpenVR Tracker input is not running; starting it now..."
    if ! ROS_DOMAIN_ID=112 WUJI_RMW=rmw_cyclonedds_cpp \
        "${SCRIPT_DIR}/start_openvr_input.sh" --detach; then
        echo "ERROR: failed to start OpenVR Tracker input." >&2
        exit 1
    fi
    OPENVR_STARTED_BY_SESSION=true
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

echo "Waiting for /openvr_input..."
OPENVR_READY=false
for _ in $(seq 1 30); do
    if docker exec \
        -e ROS_DOMAIN_ID=112 \
        -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
        -e ROS2CLI_DISABLE_DAEMON=1 \
        wuji-hand-teleop bash -lc '
unset CYCLONEDDS_URI
source /opt/ros/humble/setup.bash
source /home/wuji/ros2_ws/install/setup.bash
ros2 node list 2>/dev/null | grep -qx /openvr_input
'; then
        OPENVR_READY=true
        break
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx 'wuji-openvr-input'; then
        break
    fi
    sleep 0.25
done
if [[ "${OPENVR_READY}" != true ]]; then
    echo "ERROR: OpenVR container started but /openvr_input did not become ready." >&2
    docker logs --tail 40 wuji-openvr-input 2>&1 || true
    exit 1
fi
echo "OpenVR Tracker input is ready."

SESSION_ARGS=(
    --active-hand "${ACTIVE_HAND}"
    --active-arm "${ACTIVE_HAND}"
)
SESSION_ARGS+=(--camera-transport "${CAMERA_TRANSPORT}")
if [[ "${WITH_CAMERA}" != true ]]; then
    SESSION_ARGS+=(--no-camera)
fi
if [[ -n "${OUTPUT_DIR}" ]]; then
    SESSION_ARGS+=(--output-dir "${OUTPUT_DIR}")
fi
if [[ -n "${TASK_NAME}" ]]; then
    SESSION_ARGS+=(--task-name "${TASK_NAME}")
fi
if [[ -n "${HANDOFF_RAMP_SEC}" ]]; then
    SESSION_ARGS+=(--handoff-ramp-sec "${HANDOFF_RAMP_SEC}")
fi

docker exec -it \
    -e ROS_DOMAIN_ID=112 \
    -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    -e ROS2CLI_DISABLE_DAEMON=1 \
    wuji-hand-teleop bash -lc '
unset CYCLONEDDS_URI
source /opt/ros/humble/setup.bash
source /home/wuji/ros2_ws/install/setup.bash
exec ros2 run wuji_data_pipeline record_session "$@"
' bash "${SESSION_ARGS[@]}"
