#!/usr/bin/env bash

# Containerized, supervised LingBot-VA FDM deployment. This is intentionally
# independent from start_deployment_session.sh so the retained Pi entrypoint
# and its PI05_HTTP_API_KEY handling remain unchanged.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CONTAINER_SERVICE="teleop"
CONTAINER_REF=""
SERVER="${LINGBOT_VA_SERVER:-}"
CONTAINER_CONFIG="/home/wuji/ros2_ws/src/wuji_data_pipeline/config/lingbot_va_fdm.yaml"
CHECK_ONLY=false

usage() {
    echo "Usage: $0 [--server http(s)://HOST] [--check]"
    echo
    echo "Starts the existing robot Docker environment and opens a supervised"
    echo "right-arm/right-WujiHand LingBot-VA FDM deployment session."
    echo "The server is required for deployment and may also be set with"
    echo "LINGBOT_VA_SERVER. The --check preflight needs neither server nor API key."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --server)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --server requires a URL" >&2
                exit 2
            fi
            SERVER="$2"
            shift 2
            ;;
        --check)
            CHECK_ONLY=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "${CHECK_ONLY}" != true && -z "${SERVER}" ]]; then
    echo "ERROR: set LINGBOT_VA_SERVER or pass --server http(s)://HOST" >&2
    exit 2
fi

if [[ -n "${SERVER}" && ! "${SERVER}" =~ ^https?://[^/]+/?$ ]]; then
    echo "ERROR: server must be an http(s) base URL without a path: ${SERVER}" >&2
    exit 2
fi

if [[ ! -f "${REPO_ROOT}/src/wuji_data_pipeline/config/lingbot_va_fdm.yaml" ]]; then
    echo "ERROR: host FDM config is missing" >&2
    exit 1
fi

if [[ "${CHECK_ONLY}" != true && -z "${LINGBOT_VA_API_KEY:-}" ]]; then
    if [[ ! -t 0 ]]; then
        echo "ERROR: LINGBOT_VA_API_KEY is unset and stdin is not interactive" >&2
        exit 1
    fi
    read -r -s -p "LingBot-VA API key: " LINGBOT_VA_API_KEY
    echo
    if [[ -z "${LINGBOT_VA_API_KEY}" ]]; then
        echo "ERROR: API key must not be empty" >&2
        exit 1
    fi
    export LINGBOT_VA_API_KEY
fi

cd "${REPO_ROOT}/docker"
docker compose up -d

# Resolve the container from the Compose service instead of assuming the
# configured container_name was applied. During replacement, newer Compose
# versions may leave the running container under a temporary generated name.
CONTAINER_REF="$(docker compose ps -q "${CONTAINER_SERVICE}")"
if [[ -z "${CONTAINER_REF}" ]]; then
    echo "ERROR: Compose service ${CONTAINER_SERVICE} has no running container" >&2
    docker compose ps -a >&2 || true
    exit 1
fi

# On the first container start, entrypoint.sh builds the bind-mounted ROS
# workspace. Wait for that build instead of racing a partial install tree.
CONTAINER_READY_TIMEOUT_S="${WUJI_CONTAINER_READY_TIMEOUT_S:-1800}"
if docker exec "${CONTAINER_REF}" \
    pgrep -f '^/bin/bash /entrypoint\.sh( |$)' >/dev/null 2>&1; then
    echo "Container initialization is running; waiting for its ROS build..."
    ready_deadline=$((SECONDS + CONTAINER_READY_TIMEOUT_S))
    while docker exec "${CONTAINER_REF}" \
        pgrep -f '^/bin/bash /entrypoint\.sh( |$)' >/dev/null 2>&1; do
        if ((SECONDS >= ready_deadline)); then
            echo "ERROR: container initialization exceeded ${CONTAINER_READY_TIMEOUT_S}s" >&2
            docker logs --tail 80 "${CONTAINER_REF}" >&2 || true
            exit 1
        fi
        sleep 2
    done
fi

# Read-only preflight inside the environment that actually owns the hardware
# SDK and ROS packages. Nothing is installed or rebuilt by this script.
if ! docker exec "${CONTAINER_REF}" bash -lc '
test -f /home/wuji/ros2_ws/install/setup.bash
test -f /home/wuji/ros2_ws/src/wuji_data_pipeline/config/lingbot_va_fdm.yaml
source /opt/ros/humble/setup.bash
source /home/wuji/ros2_ws/install/setup.bash
ros2 pkg prefix wujihand_driver >/dev/null
ros2 pkg prefix wuji_data_pipeline >/dev/null
/usr/bin/python3 -c '\''from wuji_data_pipeline.fdm_async import FDM_PROTOCOL_MODE; assert FDM_PROTOCOL_MODE == "fdm_async"'\''
'; then
    echo "ERROR: the existing container ROS environment is not FDM-ready" >&2
    echo "Inspect it with: docker logs --tail 120 ${CONTAINER_REF}" >&2
    exit 1
fi

if [[ "${CHECK_ONLY}" == true ]]; then
    echo "Container FDM preflight passed; deployment was not started."
    exit 0
fi

SESSION_ARGS=(
    --active-hand right
    --server "${SERVER}"
    --config "${CONTAINER_CONFIG}"
)

echo "Starting containerized LingBot-VA FDM deployment session"
echo "  server: ${SERVER}"
echo "  config: ${CONTAINER_CONFIG}"
echo "  active side: right"
echo "  arm command mode: joint"
echo "  cameras: head + right_wrist (required by FDM feedback)"
echo "  safety: use r/a/x in the session for Recovery/Enable/Disable"

# Supplying only the environment variable name makes Docker copy its value
# without exposing the secret in argv or logs.
exec docker exec -it \
    -e ROS_DOMAIN_ID=112 \
    -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    -e ROS2CLI_DISABLE_DAEMON=1 \
    --env LINGBOT_VA_API_KEY \
    "${CONTAINER_REF}" bash -lc '
unset CYCLONEDDS_URI
source /opt/ros/humble/setup.bash
source /home/wuji/ros2_ws/install/setup.bash
exec ros2 run wuji_data_pipeline deployment_session "$@"
' bash "${SESSION_ARGS[@]}"
