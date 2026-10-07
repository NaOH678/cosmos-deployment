#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# FK-modality launcher: v1 trains the robot-hand keypoints as a modality of their
# own, with the PointFlow branch NOT loaded.  The two branches are gated by
# separate env vars so neither implies the other; see docs/fk_modality_design.md.
#
# Direct torchrun launcher for the current CUDA 12.8 cluster. No Slurm or
# Apptainer is used. Export NPROC_PER_NODE to match the visible GPU count.

TOML_FILE="examples/toml/sft_config/action_policy_fk_singlerighthand_edge.toml"

: "${SINGLERIGHTHAND_RAW_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/raw_data/singlerighthand_sandwich_100}"
: "${SINGLERIGHTHAND_CACHE_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/datasets/singlerighthand-sandwich-100-cosmos-cache}"
: "${EDGE_DROID_MODEL_PATH:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/models/cosmos3-edge-droid}"
: "${BASE_CHECKPOINT_PATH:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/models/cosmos3-edge-droid-dcp}"
: "${WAN_VAE_PATH:=$EDGE_DROID_MODEL_PATH/vae/Wan2.2_VAE.pth}"
: "${OUTPUT_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/runs/cosmos/fk-singlerighthand-edge}"
: "${FK_ANNOTATION_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/raw_data/sandwich_fk21}"
# Per-window VAE latents: the window's own encode, bit-identical to a cache-less
# run (verified 2026-09-13, max|diff| = 0.000000 on all 10 episodes).  Without it
# the frozen Wan2.2 VAE re-encodes every pixel frame of every window each step.
# Its manifest pins fps/chunk_length/sample_stride, and the dataset refuses to
# start if they disagree with the run -- so a stale cache fails loudly, not silently.
: "${SINGLERIGHTHAND_VAE_WINDOW_LATENT_ROOT:=$SINGLERIGHTHAND_CACHE_ROOT/vae_window_latents}"
: "${SINGLERIGHTHAND_EPISODE_ALLOWLIST:=$PWD/examples/pointflow_sandwich_10_episodes.txt}"
# FK is opt-in and independent of POINTFLOW_SONATA_CHECKPOINT: v1 trains FK alone,
# so this must never be set from the PointFlow recipe's variables.
export FK_ENCODER_CHECKPOINT="${FK_ENCODER_CHECKPOINT:-1}"
# v1 does not reshape FK into the DA3 canvas (see design §1.2), so the annotation
# tree and the camera-frame conversion are the only geometry inputs.
# Validate before the first optimizer step, so a defect in the eval path surfaces
# in the first minute instead of at validation_iter. Must be non-empty: the config
# decodes this with oc.decode, which errors on an empty string rather than
# falling back to its default.
: "${FK_VAL_ON_START:=true}"
: "${FK_FPS:=15}"
: "${FK_STEPS:=32}"
: "${FK_STEPS_PER_TOKEN:=4}"
: "${COSMOS_GPU_VIDEO_AUGMENTATION:=false}"

# torchrun's own default for --nproc_per_node is 8, so leaving this unset on a
# single-GPU box starts eight ranks on one card.  The PointFlow recipe sidesteps
# it by hard-coding 1; detect instead, so the same script is correct on either
# machine, and let an explicit NPROC_PER_NODE still win.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    _VISIBLE_GPUS="$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")"
else
    _VISIBLE_GPUS="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')"
fi
: "${NPROC_PER_NODE:=${_VISIBLE_GPUS:-1}}"
if ! [[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: NPROC_PER_NODE must be a positive integer, got '$NPROC_PER_NODE' (detected ${_VISIBLE_GPUS:-0})" >&2
    exit 1
fi
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
export SINGLERIGHTHAND_USE_VIDEO_CACHE
export COSMOS_GPU_VIDEO_AUGMENTATION
export BASE_CHECKPOINT_PATH WAN_VAE_PATH
export NPROC_PER_NODE
export OUTPUT_ROOT FK_ANNOTATION_ROOT SINGLERIGHTHAND_EPISODE_ALLOWLIST
export FK_ENCODER_CHECKPOINT FK_FPS FK_STEPS FK_STEPS_PER_TOKEN FK_VAL_ON_START
export SINGLERIGHTHAND_VAE_WINDOW_LATENT_ROOT

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
# FK needs its annotations and the allowlist up front: a missing annotation tree
# otherwise surfaces as an empty FK batch several minutes in, and a missing
# allowlist as a 101-episode run that silently is not the small-sample one.
[[ -d "$FK_ANNOTATION_ROOT" ]] || { echo "ERROR: missing FK annotation root: $FK_ANNOTATION_ROOT" >&2; exit 1; }
# Fail here rather than after the model is built: a missing window latent cache
# would otherwise surface as "run tools/cache_window_vae_latents.py" from inside
# the dataset constructor, several minutes into the launch.
[[ -f "$SINGLERIGHTHAND_VAE_WINDOW_LATENT_ROOT/window_manifest.json" ]] || { echo "ERROR: missing window latent manifest: $SINGLERIGHTHAND_VAE_WINDOW_LATENT_ROOT/window_manifest.json; run tools/cache_window_vae_latents.py" >&2; exit 1; }
[[ -f "$SINGLERIGHTHAND_EPISODE_ALLOWLIST" ]] || { echo "ERROR: missing FK episode allowlist: $SINGLERIGHTHAND_EPISODE_ALLOWLIST" >&2; exit 1; }
# The count used to be pinned to exactly 10, which caught a silently-wrong
# allowlist.  It cannot be pinned any more: the recipe runs either the small-sample
# ten or the full 101 (examples/singlerighthand_101_episodes.txt), and the caller
# decides which.  What still has to hold is that the file is not
# empty and that every episode in it is annotated -- the loop below checks the
# second, this line the first -- so the count is echoed instead, to put the
# distribution this run trained on into its own log.
# ``|| true`` because grep exits 1 on a zero count, and under set -e that would
# abort on the assignment instead of reaching the message below.
_ALLOWLIST_COUNT="$(grep -c . "$SINGLERIGHTHAND_EPISODE_ALLOWLIST" || true)"
[[ "${_ALLOWLIST_COUNT:-0}" -ge 1 ]] || { echo "ERROR: empty FK episode allowlist: $SINGLERIGHTHAND_EPISODE_ALLOWLIST" >&2; exit 1; }
echo "[fk] episode allowlist: $_ALLOWLIST_COUNT episodes from $SINGLERIGHTHAND_EPISODE_ALLOWLIST"
while read -r episode; do
    [[ -z "$episode" ]] && continue
    [[ -f "$FK_ANNOTATION_ROOT/$episode/annotations/wuji_fk21.npz" ]] || { echo "ERROR: no FK annotation for $episode" >&2; exit 1; }
done < "$SINGLERIGHTHAND_EPISODE_ALLOWLIST"
'

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

echo "[fk] NPROC_PER_NODE=$NPROC_PER_NODE  (override with NPROC_PER_NODE=<n>)"
# max_samples_per_batch is PER RANK, so the effective global batch is
# NPROC_PER_NODE x 32.  Two runs with different GPU counts are therefore not
# directly comparable -- compare ADE against the same-machine zero baseline, not
# across machines.
echo "[fk] effective global batch = $((NPROC_PER_NODE * 32)) samples/step"

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
