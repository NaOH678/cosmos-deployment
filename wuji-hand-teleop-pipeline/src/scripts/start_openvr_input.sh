#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_SRC="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DETACH=false

if [[ $# -gt 1 ]] || [[ $# -eq 1 && "$1" != "--detach" ]]; then
    echo "Usage: $0 [--detach]" >&2
    exit 2
fi
if [[ $# -eq 1 ]]; then
    DETACH=true
fi

IMAGE="${WUJI_TELEOP_IMAGE:-wuji-hand-teleop:latest}"
STEAM_ROOT="${STEAM_ROOT:-${HOME}/.local/share/Steam}"
ROS_DOMAIN="${ROS_DOMAIN_ID:-0}"
TRACKER_CONFIG="${REPO_SRC}/input_devices/openvr_input/config/openvr_input.yaml"
STEAMVR_RUNTIME="${STEAM_ROOT}/steamapps/common/SteamVR"

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker is not installed or not in PATH." >&2
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Cannot access the Docker daemon." >&2
    echo "Run this script as a host user with Docker access." >&2
    exit 1
fi

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "ERROR: Docker image not found: ${IMAGE}" >&2
    exit 1
fi

if [[ ! -d "${STEAMVR_RUNTIME}" ]]; then
    echo "ERROR: SteamVR runtime not found: ${STEAMVR_RUNTIME}" >&2
    exit 1
fi

if [[ ! -f "${TRACKER_CONFIG}" ]]; then
    echo "ERROR: Tracker config not found: ${TRACKER_CONFIG}" >&2
    exit 1
fi

if ! pgrep -x vrserver >/dev/null 2>&1; then
    echo "ERROR: SteamVR vrserver is not running on the host." >&2
    echo "Start SteamVR first, then run this script again." >&2
    exit 1
fi

if docker ps --format '{{.Names}}' | grep -qx 'wuji-openvr-input'; then
    echo "ERROR: wuji-openvr-input is already running." >&2
    exit 1
fi

echo "Starting OpenVR Tracker TF publisher"
echo "  SteamVR: ${STEAMVR_RUNTIME}"
echo "  Config:  ${TRACKER_CONFIG}"
echo "  Domain:  ${ROS_DOMAIN}"
if [[ "${DETACH}" == true ]]; then
    echo "  Mode:    detached (owned by the record session)"
    DOCKER_RUN_MODE=(--rm -d)
else
    echo "Press Ctrl-C to stop."
    DOCKER_RUN_MODE=(--rm -it)
fi

exec docker run "${DOCKER_RUN_MODE[@]}" \
    --name wuji-openvr-input \
    --network host \
    --ipc host \
    --pid host \
    --user wuji \
    --entrypoint bash \
    -e HOME=/home/wuji \
    -e "ROS_DOMAIN_ID=${ROS_DOMAIN}" \
    -e "RMW_IMPLEMENTATION=${WUJI_RMW:-rmw_cyclonedds_cpp}" \
    -e PYTHONPATH=/src/input_devices/openvr_input \
    -v "${STEAM_ROOT}:/steam:ro" \
    -v /tmp:/tmp:rw \
    -v "${REPO_SRC}:/src:ro" \
    "${IMAGE}" -lc '
mkdir -p /home/wuji/.config/openvr /home/wuji/.openvr-logs

cat > /home/wuji/.config/openvr/openvrpaths.vrpath <<EOF
{
  "config": ["/steam/config"],
  "external_drivers": null,
  "jsonid": "vrpathreg",
  "log": ["/home/wuji/.openvr-logs"],
  "runtime": ["/steam/steamapps/common/SteamVR"],
  "version": 1
}
EOF

source /opt/ros/humble/setup.bash

cat > /home/wuji/run_openvr_input.py <<PY
from openvr_input.openvr_input_node import main
from rclpy._rclpy_pybind11 import RCLError

try:
    main([
        "-c",
        "/src/input_devices/openvr_input/config/openvr_input.yaml",
    ])
except RCLError as exc:
    if "rcl_shutdown already called" not in str(exc):
        raise
PY

exec python3 /home/wuji/run_openvr_input.py
'
