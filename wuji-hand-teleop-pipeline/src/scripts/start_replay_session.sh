#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

usage() {
    cat >&2 <<EOF
Usage: $0 <right|left|both> <container-episode-dir> [playback-rate-hz] [replay-options...]

Examples:
  $0 right /home/wuji/datasets/tianji_wuji/TASK/episode_0000_TIMESTAMP
  $0 right /home/wuji/datasets/tianji_wuji/TASK/episode_0000_TIMESTAMP --playback-rate-hz 6.0 --arm-command-mode joint
  $0 both  /home/wuji/datasets/tianji_wuji/TASK/episode_0000_TIMESTAMP --arm-hardware-mode impedance
EOF
}

if (( $# < 2 )); then
    usage
    exit 2
fi

ACTIVE_HAND="$1"
EPISODE="$2"
shift 2
PLAYBACK_RATE_HZ="6.0"
# Preserve the original third positional rate while also allowing the clearer
# --playback-rate-hz option to pass straight through to replay_session.
if (( $# > 0 )) && [[ "$1" != --* ]]; then
    PLAYBACK_RATE_HZ="$1"
    shift
fi
EXTRA_ARGS=("$@")

case "${ACTIVE_HAND}" in
    both|left|right) ;;
    *)
        usage
        exit 2
        ;;
esac

cd "${REPO_ROOT}/docker"
docker compose up -d

exec docker exec -it \
    -e ROS_DOMAIN_ID=112 \
    -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    -e ROS2CLI_DISABLE_DAEMON=1 \
    wuji-hand-teleop bash -lc '
unset CYCLONEDDS_URI
source /opt/ros/humble/setup.bash
source /home/wuji/ros2_ws/install/setup.bash
exec ros2 run wuji_data_pipeline replay_session "$@"
' bash \
    --active-hand "${ACTIVE_HAND}" \
    --episode-dir "${EPISODE}" \
    --playback-rate-hz "${PLAYBACK_RATE_HZ}" \
    "${EXTRA_ARGS[@]}"
