#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# This is the only selector normally changed when switching Nano checkpoints.
# Example: COSMOS_CKPT_STEP=25000 bash script/start_cosmos_nano_policy_server.sh
COSMOS_CKPT_STEP="${COSMOS_CKPT_STEP:-27500}"
if [[ ! "${COSMOS_CKPT_STEP}" =~ ^[0-9]+$ ]]; then
  echo "COSMOS_CKPT_STEP must be a non-negative integer: ${COSMOS_CKPT_STEP}" >&2
  exit 2
fi
printf -v CHECKPOINT_ITER 'iter_%09d' "$((10#${COSMOS_CKPT_STEP}))"

NANO_RUN_ROOT="${COSMOS_NANO_RUN_ROOT:-/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/runs/cosmos/singlerighthand-nano-policy-droid}"
RUN_DIR="${NANO_RUN_ROOT}/cosmos3_action/action_sft/action_policy_singlerighthand_nano"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints/${CHECKPOINT_ITER}"
MODEL_CONFIG_FILE="${RUN_DIR}/config.yaml"
NANO_ASSET_DIR="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/models/cosmos3-nano-policy-droid"
MODEL_ID="${COSMOS_MODEL_ID:-singlerighthand-nano-policy-droid-${CHECKPOINT_ITER//_/-}}"

DEPLOYMENT_CONFIG="${REPO_ROOT}/examples/deployment/cosmos_singlerighthand_nano_protocol_v2.yaml"
PYTHON_BIN="${COSMOS_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
SERVICE_MODE="${COSMOS_SERVICE_MODE:-full}"
SERVICE_HOST="${COSMOS_SERVICE_HOST:-0.0.0.0}"
SERVICE_PORT="${SERVICE_PORT:-8000}"
GUIDANCE="${COSMOS_GUIDANCE:-3.0}"
NUM_STEPS="${COSMOS_NUM_STEPS:-4}"
SHIFT="${COSMOS_SHIFT:-5.0}"
TRAJECTORY_SMOOTHING="${COSMOS_TRAJECTORY_SMOOTHING:-binomial5}"

export COSMOS_POLICY_API_KEY="${H_API_KEY:-${COSMOS_POLICY_API_KEY:-}}"
if [[ -z "${COSMOS_POLICY_API_KEY}" ]]; then
  echo "H_API_KEY (or COSMOS_POLICY_API_KEY) must be supplied by the platform" >&2
  exit 2
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Cosmos Python does not exist or is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -f "${CHECKPOINT_DIR}/model/.metadata" ]]; then
  echo "Invalid Nano DCP checkpoint: missing ${CHECKPOINT_DIR}/model/.metadata" >&2
  exit 2
fi
if [[ ! -f "${MODEL_CONFIG_FILE}" ]]; then
  echo "Nano training config does not exist: ${MODEL_CONFIG_FILE}" >&2
  exit 2
fi
if [[ ! -f "${NANO_ASSET_DIR}/config.json" || ! -f "${NANO_ASSET_DIR}/tokenizer_config.json" ]]; then
  echo "Nano tokenizer/model assets are incomplete: ${NANO_ASSET_DIR}" >&2
  exit 2
fi
if [[ ! -f "${DEPLOYMENT_CONFIG}" ]]; then
  echo "Nano deployment config does not exist: ${DEPLOYMENT_CONFIG}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Model variant: Nano 16B"
echo "Repository: ${REPO_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Service: http://${SERVICE_HOST}:${SERVICE_PORT}/v1/robot-policy"
echo "Service mode: ${SERVICE_MODE}"
echo "Sampler: UniPC guidance=${GUIDANCE} num_steps=${NUM_STEPS} shift=${SHIFT}"
echo "Trajectory smoothing: ${TRAJECTORY_SMOOTHING}"
echo "Checkpoint step: ${COSMOS_CKPT_STEP}"
echo "Model ID: ${MODEL_ID}"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Model config: ${MODEL_CONFIG_FILE}"
echo "Nano assets: ${NANO_ASSET_DIR}"

cd "${REPO_ROOT}"
exec env -u LD_LIBRARY_PATH "${PYTHON_BIN}" \
  -m cosmos_framework.scripts.action_policy_server_protocol_v2 \
  --config "${DEPLOYMENT_CONFIG}" \
  --model-id "${MODEL_ID}" \
  --checkpoint-path "${CHECKPOINT_DIR}" \
  --model-config-file "${MODEL_CONFIG_FILE}" \
  --service-mode "${SERVICE_MODE}" \
  --guidance "${GUIDANCE}" \
  --num-steps "${NUM_STEPS}" \
  --shift "${SHIFT}" \
  --trajectory-smoothing "${TRAJECTORY_SMOOTHING}" \
  --host "${SERVICE_HOST}" \
  --port "${SERVICE_PORT}"
