#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export LD_LIBRARY_PATH=''
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
exec "${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}" -u -m cosmos_framework.scripts.validate_pointflow_codec \
  --data-root "${DATA_ROOT:-$REPO_ROOT/../datasets/sandwich_dense_fullseq_10_0298_20260908/outputs}" \
  --checkpoint "${SONATA_CHECKPOINT:-$REPO_ROOT/../checkpoints/ptv3/sonata_small.pth}" \
  --output "${OUTPUT_DIR:-$REPO_ROOT/pointflow_outputs/task3}" \
  --stage "${STAGE:-3}" --seed "${SEED:-0}" "$@"
