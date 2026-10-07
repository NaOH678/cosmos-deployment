#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${BENCH2DEX_CACHE_ROOT:?Set converted cache path}"
: "${EDGE_DROID_MODEL_PATH:?Set local Edge processor/model path}"
: "${BASE_CHECKPOINT_PATH:?Set base DCP checkpoint path}"
: "${WAN_VAE_PATH:=$EDGE_DROID_MODEL_PATH/vae/Wan2.2_VAE.pth}"
: "${OUTPUT_ROOT:?Set a dedicated training output path}"
: "${BENCH2DEX_LATENT_ROOT:=$BENCH2DEX_CACHE_ROOT/vae_window_latents}"
export BENCH2DEX_CACHE_ROOT EDGE_DROID_MODEL_PATH BASE_CHECKPOINT_PATH WAN_VAE_PATH BENCH2DEX_LATENT_ROOT
export IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export COSMOS_FLASH2_VARLEN="${COSMOS_FLASH2_VARLEN:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
python -c 'import pathlib,cosmos_framework; assert pathlib.Path(cosmos_framework.__file__).resolve().parent == pathlib.Path.cwd()/"cosmos_framework", "Wrong editable package"'
test -f "$BENCH2DEX_LATENT_ROOT/window_manifest.json"
test -f "$BENCH2DEX_CACHE_ROOT/manifest.json"
test -f "$BENCH2DEX_CACHE_ROOT/video_manifest.json"
test -f "$BASE_CHECKPOINT_PATH/model/.metadata"
python -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" \
  -m cosmos_framework.scripts.train --sft-toml=examples/toml/sft_config/action_policy_bench2dex_edge.toml "$@"
