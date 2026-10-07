#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# The checkpoint is the only selector normally changed for this dropper run.
# Example: COSMOS_CKPT_STEP=32500 bash script/start_cosmos_dropper_policy_server.sh
export COSMOS_CKPT_STEP="${COSMOS_CKPT_STEP:-30000}"
export COSMOS_RUN_NAME="${COSMOS_RUN_NAME:-singlerighthand-dropper-edge-droid-50k-aot}"
export COSMOS_DEPLOYMENT_CONFIG="${COSMOS_DEPLOYMENT_CONFIG:-${REPO_ROOT}/examples/deployment/cosmos_singlerighthand_dropper_edge_protocol_v2.yaml}"

exec /bin/bash "${SCRIPT_DIR}/start_cosmos_policy_server.sh"
