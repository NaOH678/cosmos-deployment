#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SERVER="${WUJI_POLICY_SERVER:-}"
ACTIVE_HAND="right"
NO_CAMERA=false
PIPELINE_CONFIG=""
HTTP_POLICY=false
POLICY_API_KEY_ENVS=()

if [[ $# -gt 0 && "${1}" != --* ]]; then
    SERVER="$1"
    shift
fi
if [[ $# -gt 0 && "${1}" != --* ]]; then
    ACTIVE_HAND="$1"
    shift
fi
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-camera)
            NO_CAMERA=true
            shift
            ;;
        --config)
            [[ $# -ge 2 ]] || {
                echo "ERROR: --config requires a container path" >&2
                exit 2
            }
            PIPELINE_CONFIG="$2"
            shift 2
            ;;
        --api-key-env)
            [[ $# -ge 2 ]] || {
                echo "ERROR: --api-key-env requires an environment variable name" >&2
                exit 2
            }
            [[ "$2" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || {
                echo "ERROR: invalid API key environment variable name: $2" >&2
                exit 2
            }
            POLICY_API_KEY_ENVS+=("$2")
            shift 2
            ;;
        *)
            echo "Usage: $0 [tcp://host:port|http(s)://host] [right|left|both] [--no-camera] [--config CONTAINER_PATH] [--api-key-env NAME]" >&2
            exit 2
            ;;
    esac
done

if [[ -n "${SERVER}" ]]; then
    case "${SERVER}" in
        tcp://*) ;;
        http://*|https://*) HTTP_POLICY=true ;;
        *)
            echo "ERROR: policy endpoint must start with tcp://, http://, or https://" >&2
            exit 2
            ;;
    esac
fi

# Preserve the existing key while allowing model-neutral and model-specific
# profiles to select their own environment variable. Values are copied by
# name through docker exec and are never placed in the command line.
for candidate in PI05_HTTP_API_KEY WUJI_POLICY_API_KEY COSMOS_POLICY_API_KEY; do
    if [[ -n "${!candidate:-}" ]]; then
        POLICY_API_KEY_ENVS+=("${candidate}")
    fi
done

if [[ "${HTTP_POLICY}" == true && ${#POLICY_API_KEY_ENVS[@]} -eq 0 ]]; then
    echo "ERROR: HTTP policy endpoint requires an API key environment variable." >&2
    echo "Set the variable named by deployment.policy_http_api_key_env, or pass --api-key-env NAME." >&2
    exit 2
fi

case "${ACTIVE_HAND}" in
    both|left|right) ;;
    *)
        echo "ERROR: active side must be right, left, or both" >&2
        exit 2
        ;;
esac

cd "${REPO_ROOT}/docker"
docker compose up -d

# docker compose returns as soon as the container process starts, while the
# image entrypoint may still be performing its first rosdep/colcon build.  Do
# not race that build by sourcing an install space that does not exist yet.
CONTAINER_READY_TIMEOUT_S="${WUJI_CONTAINER_READY_TIMEOUT_S:-1800}"
if docker exec wuji-hand-teleop pgrep -f '^/bin/bash /entrypoint\.sh( |$)' \
    >/dev/null 2>&1; then
    echo "Container first-start initialization is still running; waiting for the ROS workspace build..."
    ready_deadline=$((SECONDS + CONTAINER_READY_TIMEOUT_S))
    while docker exec wuji-hand-teleop \
        pgrep -f '^/bin/bash /entrypoint\.sh( |$)' >/dev/null 2>&1; do
        if ((SECONDS >= ready_deadline)); then
            echo "ERROR: container initialization did not finish within ${CONTAINER_READY_TIMEOUT_S}s." >&2
            docker logs --tail 80 wuji-hand-teleop >&2 || true
            exit 1
        fi
        sleep 2
    done
fi

if ! docker exec wuji-hand-teleop bash -lc '
test -f /home/wuji/ros2_ws/install/setup.bash
source /opt/ros/humble/setup.bash
source /home/wuji/ros2_ws/install/setup.bash
ros2 pkg prefix wuji_data_pipeline >/dev/null
'; then
    echo "ERROR: container initialization finished, but wuji_data_pipeline is not built." >&2
    echo "Inspect the build failure with: docker logs --tail 120 wuji-hand-teleop" >&2
    exit 1
fi

SESSION_ARGS=(
    --active-hand "${ACTIVE_HAND}"
)
if [[ -n "${SERVER}" ]]; then
    SESSION_ARGS+=(--server "${SERVER}")
fi
if [[ "${NO_CAMERA}" == true ]]; then
    SESSION_ARGS+=(--no-camera)
fi
if [[ -n "${PIPELINE_CONFIG}" ]]; then
    SESSION_ARGS+=(--config "${PIPELINE_CONFIG}")
fi

DOCKER_ENV_ARGS=()
declare -A SEEN_POLICY_API_KEY_ENVS=()
for api_key_env in "${POLICY_API_KEY_ENVS[@]}"; do
    if [[ -n "${SEEN_POLICY_API_KEY_ENVS[${api_key_env}]:-}" ]]; then
        continue
    fi
    if [[ -z "${!api_key_env:-}" ]]; then
        echo "ERROR: API key environment variable ${api_key_env} is empty." >&2
        exit 2
    fi
    SEEN_POLICY_API_KEY_ENVS["${api_key_env}"]=1
    # Supplying only the variable name asks Docker to copy its value from this
    # shell without placing the secret in the command line or logs.
    DOCKER_ENV_ARGS+=(--env "${api_key_env}")
done

exec docker exec -it \
    -e ROS_DOMAIN_ID=112 \
    -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    -e ROS2CLI_DISABLE_DAEMON=1 \
    "${DOCKER_ENV_ARGS[@]}" \
    wuji-hand-teleop bash -lc '
unset CYCLONEDDS_URI
source /opt/ros/humble/setup.bash
source /home/wuji/ros2_ws/install/setup.bash
exec ros2 run wuji_data_pipeline deployment_session "$@"
' bash "${SESSION_ARGS[@]}"
