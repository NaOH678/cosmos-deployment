#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIAGNOSTICS_DIR="${REPO_ROOT}/datasets/tianji_wuji/diagnostics"
SIDE="${1:-auto}"

case "${SIDE}" in
    auto|left|right|both) ;;
    *)
        echo "Usage: $0 [auto|left|right|both] [STATE_TRACE]" >&2
        exit 2
        ;;
esac

STATE_TRACE="${2:-}"
if [[ -z "${STATE_TRACE}" ]]; then
    STATE_TRACE="$(find "${DIAGNOSTICS_DIR}" -maxdepth 1 -type f \
        -name 'deployment_state_*.jsonl' -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr | sed -n '1p' | cut -d' ' -f2-)"
fi

if [[ -z "${STATE_TRACE}" || ! -f "${STATE_TRACE}" ]]; then
    echo "ERROR: no deployment_state_*.jsonl trace found in ${DIAGNOSTICS_DIR}" >&2
    exit 1
fi

export MPLCONFIGDIR="/tmp/wuji-deployment-matplotlib"
export PYTHONPATH="${REPO_ROOT}/src/wuji_data_pipeline${PYTHONPATH:+:${PYTHONPATH}}"

exec /usr/bin/python3 -m wuji_data_pipeline.plot_deployment_trace \
    --state-trace "${STATE_TRACE}" \
    --side "${SIDE}"
