#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# Change these two independent selectors through the deployment environment.
# Changing the run name updates the run directory and default model_id; changing
# the step updates the DCP directory and default model_id.
COSMOS_RUN_NAME="${COSMOS_RUN_NAME:-singlerighthand-edge-droid-50k-retrain-v1}"
COSMOS_CKPT_STEP="${COSMOS_CKPT_STEP:-40000}"
if [[ -z "${COSMOS_RUN_NAME}" || "${COSMOS_RUN_NAME}" == */* || "${COSMOS_RUN_NAME}" == "." || "${COSMOS_RUN_NAME}" == ".." ]]; then
  echo "COSMOS_RUN_NAME must be one directory name: ${COSMOS_RUN_NAME}" >&2
  exit 2
fi
if [[ ! "${COSMOS_CKPT_STEP}" =~ ^[0-9]+$ ]]; then
  echo "COSMOS_CKPT_STEP must be a non-negative integer: ${COSMOS_CKPT_STEP}" >&2
  exit 2
fi
printf -v CHECKPOINT_ITER 'iter_%09d' "$((10#${COSMOS_CKPT_STEP}))"

RUNS_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/runs/cosmos"
RUN_DIR="${RUNS_ROOT}/${COSMOS_RUN_NAME}/cosmos3_action/action_sft/action_policy_singlerighthand_edge"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints/${CHECKPOINT_ITER}"
MODEL_CONFIG_FILE="${RUN_DIR}/config.yaml"
MODEL_ID="${COSMOS_MODEL_ID:-${COSMOS_RUN_NAME}-${CHECKPOINT_ITER//_/-}}"

DEPLOYMENT_CONFIG="${COSMOS_DEPLOYMENT_CONFIG:-${REPO_ROOT}/examples/deployment/cosmos_singlerighthand_protocol_v2.yaml}"
PYTHON_BIN="${COSMOS_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
# Production uses the model's complete action chunk. Set
# COSMOS_SERVICE_MODE=small_motion explicitly for bounded diagnostics.
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
  echo "Invalid DCP checkpoint: missing ${CHECKPOINT_DIR}/model/.metadata" >&2
  exit 2
fi
if [[ ! -f "${MODEL_CONFIG_FILE}" ]]; then
  echo "Training config does not exist: ${MODEL_CONFIG_FILE}" >&2
  exit 2
fi
if [[ ! -f "${DEPLOYMENT_CONFIG}" ]]; then
  echo "Deployment config does not exist: ${DEPLOYMENT_CONFIG}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Repository: ${REPO_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Service: http://${SERVICE_HOST}:${SERVICE_PORT}/v1/robot-policy"
echo "Service mode: ${SERVICE_MODE}"
echo "Sampler: UniPC guidance=${GUIDANCE} num_steps=${NUM_STEPS} shift=${SHIFT}"
echo "Trajectory smoothing: ${TRAJECTORY_SMOOTHING}"
echo "Run name: ${COSMOS_RUN_NAME}"
echo "Model ID: ${MODEL_ID}"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Model config: ${MODEL_CONFIG_FILE}"
echo "Deployment config: ${DEPLOYMENT_CONFIG}"

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
