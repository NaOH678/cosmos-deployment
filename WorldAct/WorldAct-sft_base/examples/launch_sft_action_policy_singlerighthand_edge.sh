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
: "${SINGLERIGHTHAND_VAE_LATENT_ROOT:=$SINGLERIGHTHAND_CACHE_ROOT/vae_latents}"
: "${OUTPUT_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/runs/cosmos/pointflow_test}"
: "${NPROC_PER_NODE:=1}"
: "${COSMOS_GPU_VIDEO_AUGMENTATION:=false}"
: "${POINTFLOW_SONATA_CHECKPOINT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/checkpoints/ptv3/sonata_small.pth}"
: "${POINTFLOW_MANIFEST:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/WorldAct-cosmos3-edge-droid-sft/pointflow_outputs/task5/mixed_manifest.json}"
: "${POINTFLOW_VIDEO_DECODER:=torchcodec}"
: "${SINGLERIGHTHAND_VIDEO_DECODER:=opencv}"
# One name for the episode allowlist, shared by both modalities -- the recipe reads
# SINGLERIGHTHAND_EPISODE_ALLOWLIST.  POINTFLOW_EPISODE_ALLOWLIST is still accepted
# as a fallback so existing PointFlow invocations keep working: setting only the old
# name used to leave the recipe's value empty, which reads as "allowlist off" and
# silently trains on the FULL episode set instead of the file named here.
: "${SINGLERIGHTHAND_EPISODE_ALLOWLIST:=${POINTFLOW_EPISODE_ALLOWLIST:-$PWD/examples/pointflow_sandwich_10_episodes.txt}}"
# GT-ranked restriction to the fastest-moving fraction of anchor points (0 = off,
# keep the full cloud). Diagnostic only: the ranking reads future labels, so a model
# trained this way cannot be conditioned the same way at inference. Use it to check
# that the PointFlow branch can fit motion at all, on a few clusters instead of 8192
# points. 0.05 measured 409 points at 4-10 mm spacing; see tools/scan_pointflow_selection.py.
: "${POINTFLOW_SELECT_MOTION_FRACTION:=0}"
# Keep exactly N points by the same GT-motion ranking, NO floor (the fraction floors at
# 3).  Set to 1 to collapse the cloud to a single point: that removes the "one cluster
# token must describe many points" question and tests the rest of the pipeline alone.
# Takes precedence over the fraction.  0 = off.
: "${POINTFLOW_SELECT_TOP_N:=0}"
# Guard for every motion ranking: a point whose 2 cm voxel holds fewer than this many
# points is ranked last, because a lost track is always alone. 0 = off.
: "${POINTFLOW_MIN_VOXEL_MEMBERS:=0}"
# Supervise only N points, from the most-moving voxel that passes the guard; every other
# point is masked out of the loss.  The cloud is NOT reduced, so the geometry encoder
# still sees a full surface.  0 supervises everything.
: "${POINTFLOW_SUPERVISE_CLUSTER_N:=0}"
# Token granularity of the PointFlow branch: "cluster" (one token per Sonata
# cluster, the task-7 default) or "per_point" (every original point is its own
# token; nothing is pooled across points). per_point with select_top_n=300 is the
# one-point-one-token fitting experiment.
: "${POINTFLOW_TOKEN_MODE:=cluster}"
# Diagnostic: run each eval case a second time with the action zeroed and report how
# far the prediction moves (action_ablation_dependence; ~0 = the branch ignores it).
: "${POINTFLOW_ABLATE_ACTION:=false}"
: "${SINGLERIGHTHAND_VAE_LATENT_ROOT:=$SINGLERIGHTHAND_CACHE_ROOT/vae_latents}"
# Per-window VAE latents (tools/cache_window_vae_latents.py): each window's own
# encode, bit-identical to a cache-less run. Takes precedence over the
# whole-episode vae_latents above when the directory exists.
: "${SINGLERIGHTHAND_VAE_WINDOW_LATENT_ROOT:=$SINGLERIGHTHAND_CACHE_ROOT/vae_window_latents}"
if [[ -z "${SINGLERIGHTHAND_USE_VIDEO_CACHE:-}" ]]; then
    [[ -f "$SINGLERIGHTHAND_CACHE_ROOT/video_manifest.json" ]] \
        && SINGLERIGHTHAND_USE_VIDEO_CACHE=true \
        || SINGLERIGHTHAND_USE_VIDEO_CACHE=false
fi

# Pin the interpreter instead of inheriting whatever `python` the ambient PATH
# resolves to: on this node that is a Conda prefix with no torch, and the failure
# mode is an ImportError several minutes into the launch.
#
# The venv is the one built for the sibling worktree. It is reusable because it
# carries `cosmos_framework` as an EDITABLE install whose .pth hard-codes that
# worktree's path, and PYTHONPATH (set below by the launcher common script) takes
# precedence over .pth -- so this worktree's sources win. That precedence is
# implicit, hence the check immediately below: without it, a launch from any other
# directory would silently train THIS worktree's config against the SIBLING
# worktree's code, and the symptom would be "my edits have no effect".
SIBLING_VENV="/mnt/afs/WorldAct-cosmos3-edge-droid-sft/.venv"
PYTHON_BIN="${PYTHON_BIN:-$SIBLING_VENV/bin/python}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN="$(command -v python)"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$(pwd)" != "$REPO_ROOT" ]]; then
    echo "ERROR: run this launcher from its own worktree root ($REPO_ROOT), not $(pwd)." >&2
    exit 1
fi

PYTHONPATH=. "$PYTHON_BIN" - <<'PY' || exit 1
import os, sys
import cosmos_framework
resolved, expected = os.path.realpath(cosmos_framework.__file__), os.path.realpath(os.getcwd())
if not resolved.startswith(expected + os.sep):
    sys.exit(f"ERROR: cosmos_framework resolved to {resolved}, outside {expected}. "
             f"Run this launcher from the repository root of the worktree you mean to train.")
PY

export SINGLERIGHTHAND_RAW_ROOT SINGLERIGHTHAND_CACHE_ROOT EDGE_DROID_MODEL_PATH
export WAN_VAE_PATH SINGLERIGHTHAND_VAE_LATENT_ROOT
export POINTFLOW_SONATA_CHECKPOINT POINTFLOW_MANIFEST POINTFLOW_VIDEO_DECODER SINGLERIGHTHAND_VIDEO_DECODER
export SINGLERIGHTHAND_EPISODE_ALLOWLIST SINGLERIGHTHAND_VAE_LATENT_ROOT SINGLERIGHTHAND_VAE_WINDOW_LATENT_ROOT
# Kept in sync for anything still reading the pre-merge PointFlow name.
export POINTFLOW_EPISODE_ALLOWLIST="$SINGLERIGHTHAND_EPISODE_ALLOWLIST"
export POINTFLOW_SELECT_MOTION_FRACTION POINTFLOW_SELECT_TOP_N POINTFLOW_ABLATE_ACTION
export POINTFLOW_MIN_VOXEL_MEMBERS POINTFLOW_SUPERVISE_CLUSTER_N POINTFLOW_TOKEN_MODE
export SINGLERIGHTHAND_USE_VIDEO_CACHE
export COSMOS_GPU_VIDEO_AUGMENTATION
export BASE_CHECKPOINT_PATH WAN_VAE_PATH
export OUTPUT_ROOT
export NPROC_PER_NODE

# With [model.compile] enabled=true this Edge recipe dies in Triton codegen at
# the first backward:
#   InductorError: No valid triton configs. OutOfMemoryError: out of resource:
#   triton_per_fused_..._backward_sum_19 Required: 286784 Hardware limit:232448
# The kernel fuses two reductions (`mean` and `sum`, from two `slice`s) into one
# persistent-reduction kernel. That is inductor's mix-order reduction path:
# codegen/simd.py::_generate_kernel_code_for_mix_order_reduction builds the
# kernel with override_persistent_reduction=True and then asserts it, so it
# bypasses config.triton.persistent_reductions entirely -- which is why setting
# TORCHINDUCTOR_PERSISTENT_REDUCTIONS=0 did not change the kernel name. The gate
# that does control it is config.triton.mix_order_reduction (scheduler.py::
# MixOrderReduction.can_fuse), default on outside fbcode.
export TORCHINDUCTOR_MIX_ORDER_REDUCTION="${TORCHINDUCTOR_MIX_ORDER_REDUCTION:-0}"

# Belt and braces for the same failure mode, and kept from the previous attempt:
# once mix-order is disabled the two reductions below compile separately, and
# either could still land on the persistent-reduction template. Revisit both
# once the mix-order hypothesis is confirmed or ruled out.
export TORCHINDUCTOR_PERSISTENT_REDUCTIONS="${TORCHINDUCTOR_PERSISTENT_REDUCTIONS:-0}"

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
'

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
