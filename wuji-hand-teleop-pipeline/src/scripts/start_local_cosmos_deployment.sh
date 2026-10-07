#!/usr/bin/env bash
# Local all-in-one Cosmos deployment: start the on-machine protocol-v2 policy
# server, wait for readiness, then run the interactive ROS deployment session.
#
# The policy server runs on the host (uv venv in the WorldAct checkout); the
# ROS graph runs in the wuji-hand-teleop container (network_mode: host), so
# http://127.0.0.1:8000 inside the container reaches the host server.
#
# Usage:
#   ./start_local_cosmos_deployment.sh \
#       --checkpoint-dir ~/code/ckpt/<run>/checkpoints/iter_000030000 \
#       --model-config-file ~/code/ckpt/<run>/config.yaml \
#       [--model-package ~/code/ckpt/cosmos3-edge-droid] \
#       [--service-mode full|hold|small_motion] [--model-id ID] [--port 8000] \
#       [--attach-existing] [--config CONTAINER_PATH]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORLDACT_ROOT="${WORLDACT_ROOT:-$(dirname "${REPO_ROOT}")/WorldAct/WorldAct-sft-pointflow-fk}"
SERVER_SCRIPT="${WORLDACT_ROOT}/script/start_cosmos_local_policy_server.sh"
CONTAINER_CONFIG="/home/wuji/ros2_ws/src/wuji_data_pipeline/config/cosmos_protocol_v2.yaml"

CHECKPOINT_DIR=""
MODEL_CONFIG_FILE=""
MODEL_ID=""
MODEL_PACKAGE=""
SERVICE_MODE="full"
PORT="8000"
ATTACH_EXISTING=false
PIPELINE_CONFIG=""
INFERENCE_REPO="${COSMOS_INFERENCE_REPO:-}"
CHECK_ONLY=false
BACKEND="native"
OMNI_ROOT=""
OMNI_MODEL=""
OMNI_QUANTIZATION="fp8"
RECORD_VIDEO=false
RECORDING_ENABLED=true
RUN_DIR=""
SESSION_STARTED=false
SERVER_READY_TIMEOUT_S="${COSMOS_SERVER_READY_TIMEOUT_S:-900}"

usage() {
  cat <<'EOF'
Usage: start_local_cosmos_deployment.sh --checkpoint-dir DIR --model-config-file FILE [options]

Required (unless --attach-existing):
  --checkpoint-dir DIR       local DCP iteration directory (model/.metadata)
  --model-config-file FILE   frozen training config.yaml from the same run

Options:
  --backend native|omni     inference engine (default: native)
  --omni-root DIR           isolated Omni runtime directory
  --omni-model DIR          converted EMA model (default: validated 4w export)
  --omni-quantization MODE  fp8 | none (default: fp8)
  --record-video           save generated video latents for offline decoding (Omni)
  --inference-repo DIR       complete inference source checkout; 50k-retrain-v1-0831
                             defaults to the historical WorldAct-sft checkout
  --check-only               check server source/paths without starting GPU, HTTP,
                             Docker, ROS or hardware (not with --attach-existing)
  --no-recording             disable the per-run bundle and server capture
  --model-package DIR        Edge model package (exported as EDGE_DROID_MODEL_PATH)
  --model-id ID              served model_id (default: deployment YAML's value)
  --service-mode MODE        full | hold | small_motion (default: full)
  --port PORT                policy server port (default: 8000)
  --attach-existing          use an already-running server on --port; do not
                             start or stop it
  --config CONTAINER_PATH    pipeline YAML path inside the container
                             (default: /home/wuji/ros2_ws/src/wuji_data_pipeline/config/cosmos_protocol_v2.yaml)
  -h, --help                 show this help

COSMOS_POLICY_API_KEY: if unset, a random local key is generated for this run
and shared by both sides (loopback only).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2 ;;
    --omni-root) OMNI_ROOT="$2"; shift 2 ;;
    --omni-model) OMNI_MODEL="$2"; shift 2 ;;
    --omni-quantization) OMNI_QUANTIZATION="$2"; shift 2 ;;
    --record-video) RECORD_VIDEO=true; shift ;;
    --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
    --model-config-file) MODEL_CONFIG_FILE="$2"; shift 2 ;;
    --model-package) MODEL_PACKAGE="$2"; shift 2 ;;
    --model-id) MODEL_ID="$2"; shift 2 ;;
    --service-mode) SERVICE_MODE="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --attach-existing) ATTACH_EXISTING=true; shift ;;
    --config) PIPELINE_CONFIG="$2"; shift 2 ;;
    --inference-repo) INFERENCE_REPO="$2"; shift 2 ;;
    --check-only) CHECK_ONLY=true; shift ;;
    --no-recording) RECORDING_ENABLED=false; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "${BACKEND}" in native|omni) ;; *) echo "Invalid --backend: ${BACKEND}" >&2; exit 2 ;; esac
if [[ "${RECORD_VIDEO}" == true && ( "${BACKEND}" != omni || "${ATTACH_EXISTING}" == true || "${RECORDING_ENABLED}" != true ) ]]; then
  echo "--record-video requires --backend omni, managed server and recording enabled." >&2
  exit 2
fi

if [[ "${CHECK_ONLY}" == true && "${ATTACH_EXISTING}" == true ]]; then
  echo "ERROR: --check-only cannot verify the source of --attach-existing." >&2
  exit 2
fi

if [[ -z "${INFERENCE_REPO}" ]]; then
  case "${CHECKPOINT_DIR}" in
    */singlerighthand-edge-droid-50k-retrain-v1-0831/*)
      # Restore the entire historical inference implementation for this run,
      # not just its unchanged HTTP adapter. Keep the installed local venv.
      INFERENCE_REPO="$(dirname "${WORLDACT_ROOT}")/WorldAct-sft"
      ;;
    *) INFERENCE_REPO="${WORLDACT_ROOT}" ;;
  esac
fi

if [[ "${ATTACH_EXISTING}" == false ]]; then
  if [[ -z "${CHECKPOINT_DIR}" || -z "${MODEL_CONFIG_FILE}" ]]; then
    echo "ERROR: --checkpoint-dir and --model-config-file are required (or use --attach-existing)" >&2
    exit 2
  fi
  if [[ ! -x "${SERVER_SCRIPT}" ]]; then
    echo "ERROR: server launcher missing or not executable: ${SERVER_SCRIPT}" >&2
    exit 2
  fi
fi

SERVER_ARGS=(
  --backend "${BACKEND}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --model-config-file "${MODEL_CONFIG_FILE}"
  --service-mode "${SERVICE_MODE}"
  --host 127.0.0.1
  --port "${PORT}"
  --inference-repo "${INFERENCE_REPO}"
)
if [[ "${BACKEND}" == omni ]]; then
  SERVER_ARGS+=(--omni-quantization "${OMNI_QUANTIZATION}")
  if [[ -n "${OMNI_ROOT}" ]]; then SERVER_ARGS+=(--omni-root "${OMNI_ROOT}"); fi
  if [[ -n "${OMNI_MODEL}" ]]; then SERVER_ARGS+=(--omni-model "${OMNI_MODEL}"); fi
fi
if [[ -n "${MODEL_ID}" ]]; then
  SERVER_ARGS+=(--model-id "${MODEL_ID}")
fi
if [[ -n "${MODEL_PACKAGE}" ]]; then
  SERVER_ARGS+=(--model-package "${MODEL_PACKAGE}")
fi
if [[ "${CHECK_ONLY}" == true ]]; then
  exec bash "${SERVER_SCRIPT}" "${SERVER_ARGS[@]}" --check-only
fi
if [[ "${SERVICE_MODE}" == "hold" || "${SERVICE_MODE}" == "small_motion" ]]; then
  echo "NOTE: service-mode=${SERVICE_MODE} is a bounded diagnostics mode, not task execution." >&2
fi

# Prepare a run-scoped client config and server capture before starting anything.
# --check-only above remains free of file writes, GPU, Docker and ROS.
if [[ "${RECORDING_ENABLED}" == true ]]; then
  if [[ "${ATTACH_EXISTING}" == false && "${SERVICE_MODE}" == full && ! -f "${INFERENCE_REPO}/cosmos_framework/inference/robot_policy/recording.py" ]]; then
    echo "ERROR: selected inference source lacks request recording support: ${INFERENCE_REPO}" >&2
    echo "Use a recording-capable checkout, or explicitly pass --no-recording." >&2
    exit 2
  fi
  RECORD_ARGS=(
    --repo "${REPO_ROOT}" --config "${PIPELINE_CONFIG:-${CONTAINER_CONFIG}}"
    --inference-repo "${INFERENCE_REPO}" --model-config "${MODEL_CONFIG_FILE}" --service-mode "${SERVICE_MODE}"
    --checkpoint "${CHECKPOINT_DIR}" --model-package "${MODEL_PACKAGE}"
    --deployment-config "${COSMOS_DEPLOYMENT_CONFIG:-${WORLDACT_ROOT}/examples/deployment/cosmos_singlerighthand_dropper_edge_protocol_v2.yaml}"
  )
  if [[ "${ATTACH_EXISTING}" == true ]]; then
    RECORD_ARGS+=(--attached)
    echo "NOTE: attaching records client data only; an existing server's recording cannot be enabled here."
  fi
  RECORDING_INFO="$(/usr/bin/python3 "${REPO_ROOT}/src/scripts/prepare_cosmos_recording.py" "${RECORD_ARGS[@]}")"
  mapfile -t RECORDING_PATHS <<< "${RECORDING_INFO}"
  export COSMOS_RECORDING_RUN_ID="${RECORDING_PATHS[0]}"
  RUN_DIR="${RECORDING_PATHS[1]}"
  PIPELINE_CONFIG="${RECORDING_PATHS[2]}"
  export COSMOS_RECORDING_DIR="${RUN_DIR}/server"
  export COSMOS_RECORDING_QUEUE_SIZE="${COSMOS_RECORDING_QUEUE_SIZE:-16}"
  echo "Run recording: ${RUN_DIR}"
else
  unset COSMOS_RECORDING_DIR COSMOS_RECORDING_RUN_ID
fi

if [[ "${RECORD_VIDEO}" == true ]]; then
  export COSMOS_VIDEO_LATENT_DIR="${RUN_DIR}/video_latents"
fi

# One shared key for server and robot client. Loopback-only, so a per-run
# random key is fine and never touches the command line of other processes.
if [[ -z "${COSMOS_POLICY_API_KEY:-}" ]]; then
  COSMOS_POLICY_API_KEY="$(head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n')"
  export COSMOS_POLICY_API_KEY
  echo "Generated a per-run COSMOS_POLICY_API_KEY (loopback only)."
fi

BASE_URL="http://127.0.0.1:${PORT}"
SERVER_PID=""
GPU_LOG_PID=""

cleanup() {
  local exit_code=$?
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Stopping local Cosmos policy server (pid ${SERVER_PID})..."
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  if [[ -n "${GPU_LOG_PID}" ]]; then
    kill "${GPU_LOG_PID}" 2>/dev/null || true
    wait "${GPU_LOG_PID}" 2>/dev/null || true
    GPU_LOG_PID=""
  fi
  if [[ -n "${RUN_DIR}" ]]; then
    if [[ "${SESSION_STARTED}" == true ]]; then
      # The ROS session writes inside the container, not the host /tmp.
      if ! docker cp wuji-hand-teleop:/tmp/wuji_cloud_deployment.log "${RUN_DIR}/deployment.log" 2>"${RUN_DIR}/deployment_log_copy_error.log"; then
        echo "WARNING: could not archive the container deployment log; see deployment_log_copy_error.log." >&2
      fi
    fi
    /usr/bin/python3 "${REPO_ROOT}/src/scripts/audit_cosmos_recording_bundle.py" "${RUN_DIR}" \
      --finalized --output "${RUN_DIR}/recording_audit.json" || true
    /usr/bin/python3 "${REPO_ROOT}/src/scripts/prepare_cosmos_recording.py" \
      --finalize-dir "${RUN_DIR}" --exit-code "${exit_code}" || true
    echo "Run recording saved: ${RUN_DIR} (check component drop/error/close summaries)."
  fi
}
trap cleanup EXIT INT TERM

if [[ -n "${RUN_DIR}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm \
    --format=csv --loop-ms=1000 >"${RUN_DIR}/gpu.csv" 2>"${RUN_DIR}/gpu_telemetry_error.log" &
  GPU_LOG_PID=$!
fi

if [[ "${ATTACH_EXISTING}" == true ]]; then
  curl -sf "${BASE_URL}/healthz" >/dev/null || {
    echo "ERROR: --attach-existing but no healthy server on ${BASE_URL}" >&2
    exit 1
  }
  echo "Attaching to existing policy server at ${BASE_URL} (lifecycle not managed)."
else
  echo "Starting local Cosmos policy server (mode=${SERVICE_MODE})..."
  echo "Inference source: ${INFERENCE_REPO}"
  LOG_DIR="${WORLDACT_ROOT}/logs"
  mkdir -p "${LOG_DIR}"
  SERVER_LOG="${LOG_DIR}/cosmos_policy_server_$(date +%Y%m%d_%H%M%S).log"
  if [[ -n "${RUN_DIR}" ]]; then SERVER_LOG="${RUN_DIR}/server.log"; fi
  nohup bash "${SERVER_SCRIPT}" "${SERVER_ARGS[@]}" >"${SERVER_LOG}" 2>&1 &
  SERVER_PID=$!
  echo "Server pid ${SERVER_PID}, log: ${SERVER_LOG}"

  echo "Waiting for policy server readiness (model load + warmup may take minutes)..."
  deadline=$((SECONDS + SERVER_READY_TIMEOUT_S))
  while true; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "ERROR: policy server exited during startup; last log lines:" >&2
      tail -40 "${SERVER_LOG}" >&2 || true
      exit 1
    fi
    if curl -sf "${BASE_URL}/readyz" >/dev/null 2>&1; then
      echo "Policy server is ready."
      break
    fi
    if (( SECONDS >= deadline )); then
      echo "ERROR: policy server not ready within ${SERVER_READY_TIMEOUT_S}s; last log lines:" >&2
      tail -40 "${SERVER_LOG}" >&2 || true
      exit 1
    fi
    sleep 5
  done
fi

if [[ -n "${RUN_DIR}" && "${ATTACH_EXISTING}" == false && "${SERVICE_MODE}" == full ]]; then
  /usr/bin/python3 "${REPO_ROOT}/src/scripts/prepare_cosmos_recording.py" --verify-server-dir "${RUN_DIR}"
fi

# The ROS graph runs in the container; verify the container reaches the host
# server (network_mode: host makes 127.0.0.1 shared). Not fatal if the
# container is still starting: the deployment node retries the handshake.
docker compose -f "${REPO_ROOT}/docker/docker-compose.yml" up -d >/dev/null 2>&1 || true
if docker exec wuji-hand-teleop bash -lc "curl -sf ${BASE_URL}/healthz >/dev/null" 2>/dev/null; then
  echo "Container wuji-hand-teleop reaches the policy server at ${BASE_URL}."
else
  echo "WARNING: container cannot reach ${BASE_URL} yet; the session will keep retrying the handshake." >&2
fi

CONFIG_ARGS=()
if [[ -n "${PIPELINE_CONFIG}" ]]; then
  CONFIG_ARGS=(--config "${PIPELINE_CONFIG}")
else
  CONFIG_ARGS=(--config "${CONTAINER_CONFIG}")
fi

echo "Starting deployment session (single arm: right)..."
SESSION_STARTED=true
set +e
bash "${REPO_ROOT}/src/scripts/start_deployment_session.sh" \
  "${BASE_URL}" \
  right \
  "${CONFIG_ARGS[@]}"
SESSION_RC=$?
set -e

echo "Deployment session exited (rc=${SESSION_RC})."
exit "${SESSION_RC}"
