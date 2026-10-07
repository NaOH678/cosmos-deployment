#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Direct torchrun launcher for the current CUDA 12.8 cluster. No Slurm or
# Apptainer is used. Export NPROC_PER_NODE to match the visible GPU count.

TOML_FILE="examples/toml/sft_config/action_policy_singlerighthand_edge.toml"

: "${SINGLERIGHTHAND_RAW_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/raw_data/singlerighthand_sandwich_100}"
: "${SINGLERIGHTHAND_CACHE_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/datasets/singlerighthand-sandwich-100-cosmos-cache}"
: "${EDGE_DROID_MODEL_PATH:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/models/cosmos3-edge-droid}"
: "${BASE_CHECKPOINT_PATH:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/models/cosmos3-edge-droid-dcp}"
: "${WAN_VAE_PATH:=$EDGE_DROID_MODEL_PATH/vae/Wan2.2_VAE.pth}"
: "${OUTPUT_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/runs/cosmos/singlerighthand-edge-droid}"
: "${COSMOS_GPU_VIDEO_AUGMENTATION:=false}"
if [[ -z "${SINGLERIGHTHAND_USE_VIDEO_CACHE:-}" ]]; then
    [[ -f "$SINGLERIGHTHAND_CACHE_ROOT/video_manifest.json" ]] \
        && SINGLERIGHTHAND_USE_VIDEO_CACHE=true \
        || SINGLERIGHTHAND_USE_VIDEO_CACHE=false
fi

# The uv venv reuses a Conda Python prefix. TorchCodec needs FFmpeg from that
# prefix and NPP from the CUDA wheel directory at dynamic-link time.
PYTHON_BASE_PREFIX="$(python -c 'import sys; print(sys.base_prefix)')"
NPP_LIB="$(python -c 'import nvidia.npp, pathlib; print(pathlib.Path(nvidia.npp.__path__[0]) / "lib")')"
export PATH="$PATH:$PYTHON_BASE_PREFIX/bin"
export LD_LIBRARY_PATH="$PYTHON_BASE_PREFIX/lib:$NPP_LIB:${LD_LIBRARY_PATH:-}"

export SINGLERIGHTHAND_RAW_ROOT SINGLERIGHTHAND_CACHE_ROOT EDGE_DROID_MODEL_PATH
export SINGLERIGHTHAND_USE_VIDEO_CACHE
export COSMOS_GPU_VIDEO_AUGMENTATION
export BASE_CHECKPOINT_PATH WAN_VAE_PATH
export OUTPUT_ROOT

# The cluster exposes 64 CPUs for eight GPUs, while NVML reports host CPU IDs
# outside the container cpuset. Give every rank a disjoint eight-CPU partition.
export COSMOS_CPU_AFFINITY_MODE="${COSMOS_CPU_AFFINITY_MODE:-partition}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

DATASET_PATH="$SINGLERIGHTHAND_RAW_ROOT"
EXTRA_DATASET_CHECK='
[[ -f "$SINGLERIGHTHAND_CACHE_ROOT/manifest.json" ]] || { echo "ERROR: missing $SINGLERIGHTHAND_CACHE_ROOT/manifest.json; run tools/prepare_singlerighthand_raw.py first" >&2; exit 1; }
[[ -f "$BASE_CHECKPOINT_PATH/model/.metadata" ]] || { echo "ERROR: invalid DCP checkpoint: missing $BASE_CHECKPOINT_PATH/model/.metadata" >&2; exit 1; }
[[ -f "$EDGE_DROID_MODEL_PATH/processor_config.json" && -f "$EDGE_DROID_MODEL_PATH/preprocessor_config.json" && -f "$EDGE_DROID_MODEL_PATH/video_preprocessor_config.json" && -f "$EDGE_DROID_MODEL_PATH/tokenizer.json" && -f "$EDGE_DROID_MODEL_PATH/chat_template.jinja" ]] || { echo "ERROR: incomplete local Edge-DROID processor bundle: $EDGE_DROID_MODEL_PATH" >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo "ERROR: ffmpeg not found under $PYTHON_BASE_PREFIX/bin" >&2; exit 1; }
python -c "import torchcodec" || { echo "ERROR: TorchCodec runtime dependencies are unavailable" >&2; exit 1; }
'

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
