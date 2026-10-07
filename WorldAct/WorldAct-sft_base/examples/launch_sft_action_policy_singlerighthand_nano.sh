#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Direct torchrun launcher for Cosmos3-Nano-Policy-DROID (16B). The Edge (4B)
# launcher remains independent and keeps its existing defaults.

TOML_FILE="examples/toml/sft_config/action_policy_singlerighthand_nano.toml"

: "${SINGLERIGHTHAND_RAW_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/raw_data/singlerighthand_sandwich_100}"
: "${SINGLERIGHTHAND_CACHE_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/datasets/singlerighthand-sandwich-100-cosmos-cache}"
: "${NANO_DROID_MODEL_PATH:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/models/cosmos3-nano-policy-droid}"
: "${BASE_CHECKPOINT_PATH:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/models/cosmos3-nano-policy-droid-dcp}"
# The policy snapshot carries a Diffusers VAE, while training needs the Wan
# tokenizer's original .pth file. Reuse the already-local copy by default.
: "${WAN_VAE_PATH:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/models/cosmos3-edge-droid/vae/Wan2.2_VAE.pth}"
: "${OUTPUT_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/runs/cosmos/singlerighthand-nano-policy-droid}"
: "${COSMOS_GPU_VIDEO_AUGMENTATION:=false}"
if [[ -z "${SINGLERIGHTHAND_USE_VIDEO_CACHE:-}" ]]; then
    [[ -f "$SINGLERIGHTHAND_CACHE_ROOT/video_manifest.json" ]] \
        && SINGLERIGHTHAND_USE_VIDEO_CACHE=true \
        || SINGLERIGHTHAND_USE_VIDEO_CACHE=false
fi

PYTHON_BASE_PREFIX="$(python -c 'import sys; print(sys.base_prefix)')"
NPP_LIB="$(python -c 'import nvidia.npp, pathlib; print(pathlib.Path(nvidia.npp.__path__[0]) / "lib")')"
export PATH="$PATH:$PYTHON_BASE_PREFIX/bin"
export LD_LIBRARY_PATH="$PYTHON_BASE_PREFIX/lib:$NPP_LIB:${LD_LIBRARY_PATH:-}"

export SINGLERIGHTHAND_RAW_ROOT SINGLERIGHTHAND_CACHE_ROOT NANO_DROID_MODEL_PATH
export SINGLERIGHTHAND_USE_VIDEO_CACHE COSMOS_GPU_VIDEO_AUGMENTATION
export BASE_CHECKPOINT_PATH WAN_VAE_PATH OUTPUT_ROOT

export COSMOS_CPU_AFFINITY_MODE="${COSMOS_CPU_AFFINITY_MODE:-partition}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

DATASET_PATH="$SINGLERIGHTHAND_RAW_ROOT"
EXTRA_DATASET_CHECK='
[[ -f "$SINGLERIGHTHAND_CACHE_ROOT/manifest.json" ]] || { echo "ERROR: missing $SINGLERIGHTHAND_CACHE_ROOT/manifest.json; run tools/prepare_singlerighthand_raw.py first" >&2; exit 1; }
[[ -f "$BASE_CHECKPOINT_PATH/model/.metadata" ]] || { echo "ERROR: invalid Nano DCP checkpoint: missing $BASE_CHECKPOINT_PATH/model/.metadata" >&2; exit 1; }
[[ -f "$NANO_DROID_MODEL_PATH/config.json" && -f "$NANO_DROID_MODEL_PATH/tokenizer_config.json" && -f "$NANO_DROID_MODEL_PATH/tokenizer.json" && -f "$NANO_DROID_MODEL_PATH/vocab.json" && -f "$NANO_DROID_MODEL_PATH/merges.txt" ]] || { echo "ERROR: incomplete local Nano-Policy-DROID tokenizer bundle: $NANO_DROID_MODEL_PATH" >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo "ERROR: ffmpeg not found under $PYTHON_BASE_PREFIX/bin" >&2; exit 1; }
python -c "import torchcodec" || { echo "ERROR: TorchCodec runtime dependencies are unavailable" >&2; exit 1; }
'

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
