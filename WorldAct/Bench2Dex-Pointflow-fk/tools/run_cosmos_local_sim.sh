#!/usr/bin/env bash
# Two local processes, one GPU: model in its own environment, Isaac in env_isaaclab.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
: "${COSMOS_PYTHON:?Set COSMOS_PYTHON to the Python executable of your Cosmos inference environment}"
: "${COSMOS_CONFIG:?Set COSMOS_CONFIG to your completed local policy YAML}"
: "${BASELINE_ANCHOR_HDF5:?Set BASELINE_ANCHOR_HDF5 to the RGB HDF5 matching the baseline training appearance}"
run_dir="${COSMOS_RUN_DIR:-outputs/cosmos_local/run_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$run_dir"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export DEX2BENCH_RECORD_RES="${DEX2BENCH_RECORD_RES:-640}"
export DEX2BENCH_RECORD_JPEG="${DEX2BENCH_RECORD_JPEG:-90}"
export DEX2BENCH_RECORD_STRIDE="${DEX2BENCH_RECORD_STRIDE:-1}"
# Do not inherit FK injection from an earlier point-FK training shell.
unset FK_ENCODER_CHECKPOINT
"$COSMOS_PYTHON" - <<'PY'
import socket
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 9000))  # Refuse to connect the simulator to an unrelated existing service.
PY
"$COSMOS_PYTHON" -m script.policy_model_server --config "$COSMOS_CONFIG" --host 127.0.0.1 --port 9000 \
  > "$run_dir/model.log" 2>&1 &
model_pid=$!
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv,nounits -l 1 > "$run_dir/gpu.csv" &
monitor_pid=$!
trap 'kill "$model_pid" "$monitor_pid" 2>/dev/null || true; wait "$model_pid" "$monitor_pid" 2>/dev/null || true' EXIT
"$COSMOS_PYTHON" - "$model_pid" <<'PY'
import os, socket, sys, time
from pathlib import Path
pid = int(sys.argv[1])
deadline = time.monotonic() + 600
while time.monotonic() < deadline:
    os.kill(pid, 0)
    stat = Path(f'/proc/{pid}/stat')
    if stat.exists() and stat.read_text().split(') ', 1)[1].startswith('Z'):
        raise RuntimeError('Local model process exited; inspect model.log')
    try:
        with socket.create_connection(('127.0.0.1', 9000), timeout=1):
            break
    except OSError:
        time.sleep(1)
else:
    raise TimeoutError('Local model startup timed out; inspect model.log')
PY
bash tools/run_isaaclab.sh run_policy.py \
  --policy-type REMOTE --policy-name Cosmos --remote-host 127.0.0.1 --remote-port 9000 \
  --remote-timeout-s 600 --task scenes/21_condiment_box_loading.yaml \
  --robot-key multi_ur5_wuji_with_flange --active-dof \
  --cameras cam_overhead cam_wrist_left cam_wrist_right \
  --collect-config configs/collect/default.yaml --enable-rgb --headless --enable_cameras \
  --enable-generalization --generalization-profile none --anchor-hdf5 "$BASELINE_ANCHOR_HDF5" \
  --num-episodes 1 --episode-steps 1000 --record-all \
  --record-dir "$run_dir/episodes" --output-dir "$run_dir/metrics" "$@" \
  > "$run_dir/sim.log" 2>&1
