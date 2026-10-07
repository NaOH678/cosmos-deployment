#!/usr/bin/env bash
# Cache per-window Wan2.2 VAE latents for the 101 dropper episodes.
# Thin wrapper around cache_singlerighthand_window_latents.sh with the dropper
# paths pinned, so nothing long has to be pasted.
#
#   bash examples/cache_dropper_window_latents.sh
#
# Env overrides: WORKERS, DEVICES, BATCH_SIZE, OVERWRITE=1, PYTHON_BIN.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export VIDEO_CACHE_ROOT="${VIDEO_CACHE_ROOT:-/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/datasets/singlerighthand-dropper-100-cosmos-cache}"
export ALLOWLIST="${ALLOWLIST:-$ROOT/examples/pointflow_dropper_all_101_episodes.txt}"

exec bash "$ROOT/examples/cache_singlerighthand_window_latents.sh"
