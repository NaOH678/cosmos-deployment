#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
: "${PYTHON_BIN:=$ROOT/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN="$(command -v python)"
: "${VIDEO_CACHE_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/datasets/singlerighthand-sandwich-100-cosmos-cache}"
: "${ALLOWLIST:=$ROOT/examples/pointflow_sandwich_10_episodes.txt}"
: "${WAN_VAE_PATH:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/models/cosmos3-edge-droid/vae/Wan2.2_VAE.pth}"
: "${OUTPUT_ROOT:=$VIDEO_CACHE_ROOT/vae_latents}"

[[ -f "$ALLOWLIST" ]] || { echo "missing allowlist: $ALLOWLIST" >&2; exit 1; }
[[ -f "$WAN_VAE_PATH" ]] || { echo "missing VAE: $WAN_VAE_PATH" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT"

while IFS= read -r ep; do
  [[ -z "$ep" || "$ep" == \#* ]] && continue
  input="$VIDEO_CACHE_ROOT/video_frames/$ep.npy"
  output="$OUTPUT_ROOT/$ep.pt"
  [[ -f "$input" ]] || { echo "missing video cache: $input" >&2; exit 1; }
  if [[ -f "$output" ]]; then
    echo "skip $ep (already cached)"
    continue
  fi
  tmp="/tmp/${ep}.pt"
  read -r target_h target_w < <("$PYTHON_BIN" - "$VIDEO_CACHE_ROOT/video_manifest.json" "$ep" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
r=next(x for x in d["episodes"] if x["name"]==sys.argv[2])
print(int(r["image_size"][0]), int(r["image_size"][1]))
PY
  )
  "$PYTHON_BIN" - "$input" "$tmp" <<'PY'
import sys, torch, numpy as np
torch.save(torch.from_numpy(np.load(sys.argv[1])), sys.argv[2])
PY
  "$PYTHON_BIN" "$ROOT/tools/cache_wan22_latents.py" --input "$tmp" --output "$output" --vae-path "$WAN_VAE_PATH" --image-size "$target_h" "$target_w"
  rm -f "$tmp"
done < "$ALLOWLIST"

echo "VAE latent cache ready: $OUTPUT_ROOT"
