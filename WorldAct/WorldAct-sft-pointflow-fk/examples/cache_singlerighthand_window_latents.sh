#!/usr/bin/env bash
# Cache per-window Wan2.2 VAE latents for the sandwich episodes (the exact
# latents the online path computes per training window). One worker per GPU,
# jobs pinned to their card; dead workers are retried automatically by the
# tool itself (look for [retry] lines).
#
#   bash examples/cache_singlerighthand_window_latents.sh
#
# Env overrides: VIDEO_CACHE_ROOT, WAN_VAE_PATH, ALLOWLIST, OUTPUT_ROOT,
# WORKERS, DEVICES, BATCH_SIZE, OVERWRITE=1, PYTHON_BIN.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export LD_LIBRARY_PATH=''
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
: "${PYTHON_BIN:=$ROOT/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN="$(command -v python)"
: "${VIDEO_CACHE_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/datasets/singlerighthand-sandwich-100-cosmos-cache}"
: "${WAN_VAE_PATH:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/models/cosmos3-edge-droid/vae/Wan2.2_VAE.pth}"
: "${ALLOWLIST:=$ROOT/examples/pointflow_sandwich_all_101_episodes.txt}"
: "${OUTPUT_ROOT:=$VIDEO_CACHE_ROOT/vae_window_latents}"
: "${WORKERS:=8}"
: "${DEVICES:=0,1,2,3,4,5,6,7}"
: "${BATCH_SIZE:=1}"
: "${OVERWRITE:=1}"

[[ -f "$ALLOWLIST" ]] || { echo "missing allowlist: $ALLOWLIST" >&2; exit 1; }
[[ -f "$WAN_VAE_PATH" ]] || { echo "missing VAE: $WAN_VAE_PATH" >&2; exit 1; }

extra=()
[[ "$OVERWRITE" == "1" ]] && extra+=(--overwrite)

echo "cache-root: $VIDEO_CACHE_ROOT"
echo "allowlist:  $ALLOWLIST"
echo "output:     $OUTPUT_ROOT"
echo "workers:    $WORKERS on $DEVICES (batch-size $BATCH_SIZE, overwrite=$OVERWRITE)"

exec "$PYTHON_BIN" "$ROOT/tools/cache_window_vae_latents.py" \
  --cache-root "$VIDEO_CACHE_ROOT" \
  --vae-path "$WAN_VAE_PATH" \
  --episode-allowlist "$ALLOWLIST" \
  --output-root "$OUTPUT_ROOT" \
  --workers "$WORKERS" \
  --devices "$DEVICES" \
  --batch-size "$BATCH_SIZE" \
  "${extra[@]}"
