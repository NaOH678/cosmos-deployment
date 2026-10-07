#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Four-modality launcher: trains video + action + FK + PointFlow together.
# Both branch gates are set below; they are independent env vars and neither
# implies the other, so a single-modality run is this launcher with one of them
# overridden to empty.  See docs/fk_modality_design.md (FK) and
# docs/fk_pointflow_merge.md (how the two coexist -- in particular the flat
# ordering [vision | action | sound | pointflow | FK]).
#
# Direct torchrun launcher for the current A800 cluster.  No Slurm or Apptainer.
# NPROC_PER_NODE is detected from the visible GPUs; export it to override.

TOML_FILE="examples/toml/sft_config/action_policy_fk_point_singlerighthand_edge.toml"

: "${SINGLERIGHTHAND_RAW_ROOT:=/data/shichaojian/raw_data/singlerighthand_sandwich_100}"
: "${SINGLERIGHTHAND_CACHE_ROOT:=/data/shichaojian/datasets/singlerighthand-sandwich-100-cosmos-cache}"
: "${EDGE_DROID_MODEL_PATH:=/data/shichaojian/models/cosmos3-edge-droid}"
: "${BASE_CHECKPOINT_PATH:=/data/shichaojian/models/cosmos3-edge-droid-dcp}"
: "${WAN_VAE_PATH:=$EDGE_DROID_MODEL_PATH/vae/Wan2.2_VAE.pth}"
: "${OUTPUT_ROOT:=/data/shichaojian/runs/cosmos/fk-point-singlerighthand-edge}"
: "${FK_ANNOTATION_ROOT:=/data/shichaojian/raw_data/sandwich_fk21}"
# Per-window VAE latents: the window's own encode, bit-identical to a cache-less
# run (verified 2026-09-13, max|diff| = 0.000000 on all 10 episodes).  Without it
# the frozen Wan2.2 VAE re-encodes every pixel frame of every window each step.
# Its manifest pins fps/chunk_length/sample_stride, and the dataset refuses to
# start if they disagree with the run -- so a stale cache fails loudly, not silently.
: "${SINGLERIGHTHAND_VAE_WINDOW_LATENT_ROOT:=$SINGLERIGHTHAND_CACHE_ROOT/vae_window_latents}"
: "${SINGLERIGHTHAND_EPISODE_ALLOWLIST:=$PWD/examples/singlerighthand_101_episodes.txt}"
# --- the two branch gates -------------------------------------------------------
# Despite the name, FK_ENCODER_CHECKPOINT is a boolean switch, not a path: the FK
# encoder is learned from scratch (there is no pretrained keypoint backbone).
export FK_ENCODER_CHECKPOINT="${FK_ENCODER_CHECKPOINT:-1}"
# Unlike FK, PointFlow DOES need a checkpoint here: the Sonata/PTv3 geometry
# encoder is pretrained and frozen.
: "${POINTFLOW_SONATA_CHECKPOINT:=/data/shichaojian/checkpoints/ptv3/sonata_small.pth}"

# --- PointFlow data + selection -------------------------------------------------
# The manifest and the selection knobs below are ONE decision, because
# ``pointflow_displacement_scale`` in the recipe is the pooled per-element std of
# the displacement that selection actually yields.  Change either without
# re-measuring and the branch trains against a wrongly-scaled target -- silently,
# since a scale error produces no shape error, only predictions scaled overall.
# 0.0528 m: sandwich stratified top500 (40/45/15), voxel/valid/phantom guards.
# Reuses the existing sandwich cache; token representation stays cluster.
: "${POINTFLOW_MANIFEST:=/mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow/pointflow_outputs/sandwich_924_20260928/manifest.json}"
: "${POINTFLOW_TOKEN_MODE:=cluster}"
: "${POINTFLOW_SELECT_TOP_N:=500}"
: "${POINTFLOW_MIN_VOXEL_MEMBERS:=3}"
: "${POINTFLOW_SELECT_MIN_VALID_STEPS:=16}"
: "${POINTFLOW_SELECT_MOTION_FRACTION:=0}"
: "${POINTFLOW_SELECT_REGIONS:=}"
: "${POINTFLOW_SELECT_REGION_QUOTAS:=2:0.40,3:0.45,4:0.15}"
: "${POINTFLOW_SELECT_PHANTOM_GUARD:=true}"
# Explicit empty value disables the cache for an online comparison.
POINTFLOW_WINDOW_CACHE_ROOT="${POINTFLOW_WINDOW_CACHE_ROOT-$SINGLERIGHTHAND_CACHE_ROOT/pointflow_windows}"
: "${POINTFLOW_DISPLACEMENT_SCALE:=0.0528}"
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
# Hard fail, NOT a fallback to `command -v python`.  That fallback used to sit here
# and it re-created the exact failure it was meant to prevent: on this node the
# ambient `python` is /mnt/afs/miniconda3/bin/python -- Python 3.14 with no torch --
# so a missing venv would have swapped in the wrong interpreter silently and died
# minutes later with an unreadable ImportError, which is precisely the symptom the
# pin above exists to avoid.  It also dragged torchrun along: PATH would then lead
# with miniconda (no torchrun in it), so line 99's bare ``torchrun`` would fall
# through to /usr/local/bin/torchrun, whose shebang is a third interpreter entirely
# (/usr/bin/python3.12).  A venv that is gone is a deployment error, not a reason to
# guess -- and "the venv went missing" is not hypothetical here: this launcher's
# sibling venv had to be re-pointed once already.  Failing on line 1 costs one
# second; failing at model-build time costs the minutes spent reading the traceback.
[[ -x "$PYTHON_BIN" ]] || {
    echo "ERROR: interpreter is missing or not executable: $PYTHON_BIN" >&2
    echo "       expected the shared venv at $SIBLING_VENV" >&2
    echo "       (set PYTHON_BIN=... to point at another one; it must have torch)" >&2
    exit 1
}
# Prepending the interpreter's own bin/ is what makes the common script's bare
# ``torchrun`` deterministic: it resolves to that venv's torchrun, whose shebang is
# that venv's python, so the launcher and the ranks it spawns share one interpreter.
export PATH="$(dirname "$PYTHON_BIN"):$PATH"

# Clear LD_LIBRARY_PATH, and do it HERE rather than in _sft_launcher_common.sh.
#
# Required: in this NGC-style container LD_LIBRARY_PATH leads with
# /usr/local/nvidia/lib*:/usr/local/cuda/lib64, and that last entry is a CUDA 13.2
# toolkit whose libcublasLt is 13.4.0.1 -- while PyTorch's own nvidia-cublas wheel is
# 13.1.0.3.  Only ONE of the pair is shadowed; measured on the dev node by reading
# /proc/self/maps after a matmul:
#     libcublas.so.13   -> venv nvidia/cu13    (13.1.0.3)
#     libcublasLt.so.13 -> /usr/local/cuda-13.2 (13.4.0.1)   <-- mismatched minor
# and with the variable cleared both come from the venv.  Run -10-3 died on
# ``CUBLAS_STATUS_NOT_INITIALIZED ... cublasLtMatmulAlgoGetHeuristic`` -- a
# cublasLt call -- on its very first FK validation step.  Causal link not proven
# (plain matmuls survive the mismatch), but this is the same workaround docs/faq.md:29
# and AGENTS.md:104 already mandate and the same one script/start_cosmos_*_policy_server.sh
# applies via ``exec env -u``; the training path was the one place it was missing.
#
# HERE, not in the shared file: that file serves ~18 recipes and
# launch_sft_action_policy_singlerighthand_nano.sh sets LD_LIBRARY_PATH on purpose, so
# unsetting it centrally would silently overrule that recipe's intent.
#
# Set KEEP_LD_LIBRARY_PATH=1 to keep the inherited value -- e.g. on a node whose
# toolkit actually matches the wheels.
if [[ -z "${KEEP_LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH=
fi

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
export POINTFLOW_SONATA_CHECKPOINT POINTFLOW_MANIFEST POINTFLOW_TOKEN_MODE
export POINTFLOW_SELECT_TOP_N POINTFLOW_MIN_VOXEL_MEMBERS POINTFLOW_SELECT_MIN_VALID_STEPS
export POINTFLOW_SELECT_MOTION_FRACTION POINTFLOW_SELECT_REGIONS
export POINTFLOW_SELECT_REGION_QUOTAS POINTFLOW_SELECT_PHANTOM_GUARD POINTFLOW_WINDOW_CACHE_ROOT

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

# expandable_segments against fragmentation, which is what actually killed runs
# -10-4 and -10-5.  Both reached iteration 1 and died on iteration 2 with
#     69.26 GiB allocated by PyTorch, and 7.07 GiB reserved by PyTorch but unallocated
# out of a 79.25 GiB card -- i.e. ~7 GiB the allocator was holding and could not
# reuse, which is the single largest reclaimable block available.  PyTorch names this
# setting in its own OOM message.  -10-2 ran the SAME recipe to 10000 steps at
# max_samples_per_batch=32 on the old cluster, and a full config diff against this
# run shows no difference but paths (plus the newer fk_rollout ``joint_action`` key),
# so nothing in the recipe changed to push it over -- the new environment simply
# lands closer to the edge, and the fragmentation is what tips it over.
#
# This changes only how the allocator hands out virtual address space: no numerical
# effect, nothing to invalidate comparisons against -10/-10-2.  Set
# PYTORCH_CUDA_ALLOC_CONF yourself to override.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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
# PointFlow side.  The manifest must exist AND every episode in the allowlist must
# be listed in it -- the dataset asserts exactly that ("PointFlow manifest must
# explicitly list selected episodes, including unlabeled"), several minutes in.
[[ -f "$POINTFLOW_MANIFEST" ]] || { echo "ERROR: missing PointFlow manifest: $POINTFLOW_MANIFEST" >&2; exit 1; }
[[ -f "$POINTFLOW_SONATA_CHECKPOINT" ]] || { echo "ERROR: missing Sonata checkpoint: $POINTFLOW_SONATA_CHECKPOINT" >&2; exit 1; }
"$PYTHON_BIN" - "$POINTFLOW_MANIFEST" "$SINGLERIGHTHAND_EPISODE_ALLOWLIST" <<'PYPRE' || exit 1
import json, sys
from pathlib import Path
manifest, allowlist = Path(sys.argv[1]), Path(sys.argv[2])
names = {e["name"] for e in json.loads(manifest.read_text())["episodes"]}
want = {ln.strip() for ln in allowlist.read_text().splitlines() if ln.strip()}
missing = sorted(want - names)
if missing:
    sys.exit(f"ERROR: {len(missing)} allowlist episodes are absent from the manifest, e.g. {missing[:3]}")
print(f"[pointflow] manifest covers all {len(want)} allowlisted episodes")
PYPRE
'

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
    "model.config.rectified_flow_training_config.pointflow_displacement_scale=$POINTFLOW_DISPLACEMENT_SCALE"
)

echo "[fk] NPROC_PER_NODE=$NPROC_PER_NODE  (override with NPROC_PER_NODE=<n>)"
# max_samples_per_batch is PER RANK, so the effective global batch is
# NPROC_PER_NODE x 32.  Two runs with different GPU counts are therefore not
# directly comparable -- compare ADE against the same-machine zero baseline, not
# across machines.
# max_samples_per_batch is set by the wrapper (run_fk_point_101.sh) through
# EXTRA_TAIL_OVERRIDES, so the global batch is decided there, not here.
echo "[fk-point] NPROC_PER_NODE=$NPROC_PER_NODE; see the wrapper for batch/topology"

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
