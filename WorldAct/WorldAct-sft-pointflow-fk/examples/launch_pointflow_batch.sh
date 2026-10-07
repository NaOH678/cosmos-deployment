#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export LD_LIBRARY_PATH=''
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
exec "${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}" -m cosmos_framework.scripts.validate_pointflow_batch \
  --manifest "${POINTFLOW_MANIFEST:-$REPO_ROOT/pointflow_outputs/task5/mixed_manifest.json}" \
  --output "${OUTPUT_DIR:-$REPO_ROOT/pointflow_outputs/task6}" --seed "${SEED:-0}" "$@"
