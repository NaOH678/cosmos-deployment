#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
    echo "ERROR: host ROS 2 Humble was not found at /opt/ros/humble." >&2
    exit 1
fi

# ROS 2's generated setup scripts intentionally inspect variables that may
# not exist yet, so nounset must be suspended only while sourcing them.
set +u
source /opt/ros/humble/setup.bash
set -u

if ! /usr/bin/python3 -c 'import PyQt5, rclpy' >/dev/null 2>&1; then
    echo "ERROR: host Python needs PyQt5 and ROS 2 rclpy." >&2
    echo "Install python3-pyqt5 and the ROS 2 Humble desktop packages first." >&2
    exit 1
fi

export ROS_DOMAIN_ID=112
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS2CLI_DISABLE_DAEMON=1
unset CYCLONEDDS_URI
export WUJI_REPOSITORY_ROOT="${REPOSITORY_ROOT}"
export WUJI_DATASET_ROOT="${REPOSITORY_ROOT}/datasets/tianji_wuji"
export WUJI_CAMERA_TRANSPORT="${WUJI_CAMERA_TRANSPORT:-direct}"
export WUJI_CAMERA_SHARED_MEMORY_DIR="${WUJI_CAMERA_SHARED_MEMORY_DIR:-/dev/shm/wuji_camera_v1}"
export PYTHONPATH="${REPOSITORY_ROOT}/src/wuji_teleop_monitor:${REPOSITORY_ROOT}/src/camera:${PYTHONPATH:-}"

exec /usr/bin/python3 -m wuji_teleop_monitor.ui.run_record
