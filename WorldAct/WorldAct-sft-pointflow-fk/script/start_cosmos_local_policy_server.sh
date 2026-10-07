#!/usr/bin/env bash

# Local (single-machine) launcher for the Cosmos protocol-v2 robot policy server.
# Unlike script/start_cosmos_policy_server.sh, which derives checkpoints from the
# cluster's shared-storage layout, this script takes explicit local paths.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

CHECKPOINT_DIR=""
MODEL_CONFIG_FILE=""
MODEL_ID=""
DEPLOYMENT_CONFIG="${COSMOS_DEPLOYMENT_CONFIG:-${REPO_ROOT}/examples/deployment/cosmos_singlerighthand_dropper_edge_protocol_v2.yaml}"
PYTHON_BIN="${COSMOS_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
# Production uses the model's complete action chunk. Set --service-mode
# small_motion explicitly for bounded diagnostics, hold for protocol checks.
SERVICE_MODE="${COSMOS_SERVICE_MODE:-full}"
SERVICE_HOST="${COSMOS_SERVICE_HOST:-127.0.0.1}"
SERVICE_PORT="${SERVICE_PORT:-8000}"
GUIDANCE="${COSMOS_GUIDANCE:-3.0}"
NUM_STEPS="${COSMOS_NUM_STEPS:-4}"
SHIFT="${COSMOS_SHIFT:-5.0}"
TRAJECTORY_SMOOTHING="${COSMOS_TRAJECTORY_SMOOTHING:-binomial5}"
MODEL_PACKAGE="${EDGE_DROID_MODEL_PATH:-}"
SOURCE_ROOT="${COSMOS_INFERENCE_REPO:-${REPO_ROOT}}"
CHECK_ONLY=false
BACKEND="native"
OMNI_ROOT="$(dirname "${REPO_ROOT}")/omni-wam-lab"
OMNI_MODEL=""
OMNI_QUANTIZATION="fp8"

usage() {
  cat <<'EOF'
Usage: start_cosmos_local_policy_server.sh --checkpoint-dir DIR --model-config-file FILE [options]

Required:
  --checkpoint-dir DIR       DCP iteration directory containing model/.metadata
  --model-config-file FILE   frozen training config.yaml from the same run

Options:
  --backend native|omni     inference engine (default: native)
  --omni-root DIR           isolated Omni runtime directory
  --omni-model DIR          converted EMA model directory
  --omni-quantization MODE  fp8 | none (default: fp8)
  --inference-repo DIR       source checkout to import (Python environment stays
                             COSMOS_PYTHON or this launcher's .venv)
  --check-only               validate paths and imported source; no server/GPU
  --model-id ID              served model_id (default: keep the deployment YAML's)
  --deployment-config FILE   deployment manifest (default: examples/deployment/
                             cosmos_singlerighthand_dropper_edge_protocol_v2.yaml)
  --model-package DIR        Edge model package; exported as EDGE_DROID_MODEL_PATH
  --service-mode MODE        full | hold | small_motion (default: full)
  --host HOST                bind address (default: 127.0.0.1)
  --port PORT                bind port (default: 8000)
  --guidance X --num-steps N --shift X --trajectory-smoothing M
                             sampler overrides (defaults: 3.0 / 4 / 5.0 / binomial5)
  -h, --help                 show this help

Environment: COSMOS_POLICY_API_KEY must be set and non-empty.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2 ;;
    --omni-root) OMNI_ROOT="$2"; shift 2 ;;
    --omni-model) OMNI_MODEL="$2"; shift 2 ;;
    --omni-quantization) OMNI_QUANTIZATION="$2"; shift 2 ;;
    --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
    --model-config-file) MODEL_CONFIG_FILE="$2"; shift 2 ;;
    --model-id) MODEL_ID="$2"; shift 2 ;;
    --deployment-config) DEPLOYMENT_CONFIG="$2"; shift 2 ;;
    --model-package) MODEL_PACKAGE="$2"; shift 2 ;;
    --inference-repo) SOURCE_ROOT="$2"; shift 2 ;;
    --check-only) CHECK_ONLY=true; shift ;;
    --service-mode) SERVICE_MODE="$2"; shift 2 ;;
    --host) SERVICE_HOST="$2"; shift 2 ;;
    --port) SERVICE_PORT="$2"; shift 2 ;;
    --guidance) GUIDANCE="$2"; shift 2 ;;
    --num-steps) NUM_STEPS="$2"; shift 2 ;;
    --shift) SHIFT="$2"; shift 2 ;;
    --trajectory-smoothing) TRAJECTORY_SMOOTHING="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "${BACKEND}" in
  native) ;;
  omni)
    OMNI_ROOT="$(realpath "${OMNI_ROOT}")"
    OMNI_MODEL="$(realpath "${OMNI_MODEL:-${OMNI_ROOT}/artifacts/4w-ema-omni}")"
    PYTHON_BIN="${OMNI_ROOT}/.venv/bin/python"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
    [[ -f "${SOURCE_ROOT}/cosmos_framework/inference/robot_policy/omni_http.py" ]] || {
      echo "Omni adapter missing from inference checkout: ${SOURCE_ROOT}" >&2; exit 2;
    }
    ;;
  *) echo "Invalid --backend: ${BACKEND}" >&2; exit 2 ;;
esac

if [[ -z "${CHECKPOINT_DIR}" || -z "${MODEL_CONFIG_FILE}" ]]; then
  echo "--checkpoint-dir and --model-config-file are both required" >&2
  usage >&2
  exit 2
fi
case "${SERVICE_MODE}" in
  full|hold|small_motion) ;;
  *) echo "Invalid --service-mode: ${SERVICE_MODE}" >&2; exit 2 ;;
esac

export COSMOS_POLICY_API_KEY="${H_API_KEY:-${COSMOS_POLICY_API_KEY:-}}"
if [[ -z "${COSMOS_POLICY_API_KEY}" && "${CHECK_ONLY}" != true ]]; then
  echo "COSMOS_POLICY_API_KEY must be set (shared with the robot-side client)" >&2
  exit 2
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Cosmos Python does not exist or is not executable: ${PYTHON_BIN}" >&2
  echo "Create the environment first: uv sync --all-extras --group=cu130-train" >&2
  exit 2
fi
if [[ ! -f "${CHECKPOINT_DIR}/model/.metadata" && "${SERVICE_MODE}" != "hold" ]]; then
  echo "Invalid DCP checkpoint: missing ${CHECKPOINT_DIR}/model/.metadata" >&2
  exit 2
fi
if [[ ! -f "${MODEL_CONFIG_FILE}" && "${SERVICE_MODE}" != "hold" ]]; then
  echo "Training config does not exist: ${MODEL_CONFIG_FILE}" >&2
  exit 2
fi
if [[ ! -f "${DEPLOYMENT_CONFIG}" ]]; then
  echo "Deployment config does not exist: ${DEPLOYMENT_CONFIG}" >&2
  exit 2
fi
if [[ ! -f "${SOURCE_ROOT}/cosmos_framework/scripts/action_policy_server_protocol_v2.py" ]]; then
  echo "Inference checkout is missing the protocol-v2 entry point: ${SOURCE_ROOT}" >&2
  exit 2
fi
# Resolve user-supplied relative paths before changing into the source checkout.
SOURCE_ROOT="$(realpath "${SOURCE_ROOT}")"
PYTHON_BIN="$(realpath -s "${PYTHON_BIN}")"
DEPLOYMENT_CONFIG="$(realpath "${DEPLOYMENT_CONFIG}")"
CHECKPOINT_DIR="$(realpath -m "${CHECKPOINT_DIR}")"
MODEL_CONFIG_FILE="$(realpath -m "${MODEL_CONFIG_FILE}")"
if [[ -n "${MODEL_PACKAGE}" ]]; then
  if [[ ! -d "${MODEL_PACKAGE}" ]]; then
    echo "Model package directory does not exist: ${MODEL_PACKAGE}" >&2
    exit 2
  fi
  export EDGE_DROID_MODEL_PATH="$(realpath "${MODEL_PACKAGE}")"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Launcher repository: ${REPO_ROOT}"
echo "Inference repository: ${SOURCE_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Service: http://${SERVICE_HOST}:${SERVICE_PORT}/v1/robot-policy"
echo "Service mode: ${SERVICE_MODE}"
echo "Sampler: UniPC guidance=${GUIDANCE} num_steps=${NUM_STEPS} shift=${SHIFT}"
echo "Trajectory smoothing: ${TRAJECTORY_SMOOTHING}"
echo "Model ID: ${MODEL_ID:-<from deployment config>}"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Model config: ${MODEL_CONFIG_FILE}"
echo "Deployment config: ${DEPLOYMENT_CONFIG}"
echo "EDGE_DROID_MODEL_PATH: ${EDGE_DROID_MODEL_PATH:-<unset>}"

cd "${SOURCE_ROOT}"
# The complete package (model, packing, attention and inference) must come from
# this checkout, even when reusing a venv installed in a different worktree.
export PYTHONPATH="${SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
env -u LD_LIBRARY_PATH "${PYTHON_BIN}" - "${SOURCE_ROOT}" <<'PY'
import importlib.util
from pathlib import Path
import sys

expected = Path(sys.argv[1]).resolve() / "cosmos_framework" / "__init__.py"
spec = importlib.util.find_spec("cosmos_framework")
actual = Path(spec.origin).resolve() if spec and spec.origin else None
if actual != expected:
    raise SystemExit(f"Wrong inference import: expected {expected}, got {actual}")
print(f"Verified inference import: {actual}", flush=True)
PY
if [[ "${CHECK_ONLY}" == true && "${BACKEND}" == omni ]]; then
  exec env -u LD_LIBRARY_PATH "${PYTHON_BIN}" -m cosmos_framework.inference.robot_policy.omni_http \
    --config "${DEPLOYMENT_CONFIG}" --omni-root "${OMNI_ROOT}" --omni-model "${OMNI_MODEL}" \
    --omni-quantization "${OMNI_QUANTIZATION}" --checkpoint-path "${CHECKPOINT_DIR}" \
    --model-config-file "${MODEL_CONFIG_FILE}" --service-mode "${SERVICE_MODE}" --check-only
fi
if [[ "${CHECK_ONLY}" == true ]]; then
  echo "Source/path checks passed; no model, server or robot was started."
  exit 0
fi

EXTRA_ARGS=()
if [[ -n "${MODEL_ID}" ]]; then
  EXTRA_ARGS+=(--model-id "${MODEL_ID}")
fi

ENTRY_MODULE="cosmos_framework.scripts.action_policy_server_protocol_v2"
if [[ "${BACKEND}" == omni ]]; then
  ENTRY_MODULE="cosmos_framework.inference.robot_policy.omni_http"
  EXTRA_ARGS+=(--omni-root "${OMNI_ROOT}" --omni-model "${OMNI_MODEL}" --omni-quantization "${OMNI_QUANTIZATION}")
  echo "Backend: Omni; quantization=${OMNI_QUANTIZATION}; export=${OMNI_MODEL}"
fi

# NGC images ship an LD_LIBRARY_PATH that breaks torch imports; strip it.
exec env -u LD_LIBRARY_PATH "${PYTHON_BIN}" \
  -m "${ENTRY_MODULE}" \
  --config "${DEPLOYMENT_CONFIG}" \
  --checkpoint-path "${CHECKPOINT_DIR}" \
  --model-config-file "${MODEL_CONFIG_FILE}" \
  --service-mode "${SERVICE_MODE}" \
  --guidance "${GUIDANCE}" \
  --num-steps "${NUM_STEPS}" \
  --shift "${SHIFT}" \
  --trajectory-smoothing "${TRAJECTORY_SMOOTHING}" \
  --host "${SERVICE_HOST}" \
  --port "${SERVICE_PORT}" \
  "${EXTRA_ARGS[@]}"
