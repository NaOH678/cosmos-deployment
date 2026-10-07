#!/usr/bin/env bash
# Run on a GPU node. CPU mode prepares real windows but does not run Sonata.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export LD_LIBRARY_PATH=''
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
DEVICE="${DEVICE:-cuda}"
EXTRA_ARGS=()
if [[ "$DEVICE" == cuda ]]; then EXTRA_ARGS+=(--backward); fi
exec "${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}" -u \
  -m cosmos_framework.scripts.validate_pointflow_sonata \
  --data-root "${DATA_ROOT:-$REPO_ROOT/../datasets/sandwich_dense_fullseq_10_0298_20260908/outputs}" \
  --checkpoint "${SONATA_CHECKPOINT:-$REPO_ROOT/../checkpoints/ptv3/sonata_small.pth}" \
  --output "${OUTPUT_DIR:-$REPO_ROOT/pointflow_outputs/sonata_smoke}" \
  --device "$DEVICE" --seed "${SEED:-0}" \
  --start-frame "${START_FRAME:-0}" --max-points "${MAX_POINTS:-8192}" \
  --voxel-size "${VOXEL_SIZE:-0.02}" "${EXTRA_ARGS[@]}" "$@"
