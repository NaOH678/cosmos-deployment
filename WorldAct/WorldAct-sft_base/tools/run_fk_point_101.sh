#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Four-modality training wrapper: video + action + FK + PointFlow at once.
#
# Submit exactly the same command on every node; do not distinguish ranks:
#
#   OUTPUT_ROOT=/data/shichaojian/runs/cosmos/fk-point-101-<date> NNODES=2 \
#     bash /mnt/afs/WorldAct-cosmos3-edge-droid-sft_base/tools/run_fk_point_101.sh
#
# Plan first, launch later:  add DRY_RUN=1 to print the resolved plan and exit.
# Continue an interrupted run: add RESUME=1 (the OUTPUT_ROOT itself is reused).
#
# Overridable (export before the command): OUTPUT_ROOT (REQUIRED, fresh),
# NNODES, NPROC_PER_NODE, PER_RANK_BATCH, MAX_ITER, RESUME, and every POINTFLOW_*
# / FK_* / SINGLERIGHTHAND_* variable the launcher reads.
#
# Shape of the chain:
#   tools/run_fk_point_101.sh                                          <- this file
#     └─ examples/launch_sft_action_policy_fk_point_singlerighthand_edge.sh
#          └─ examples/_sft_launcher_common.sh   (torchrun argument assembly)
#               └─ torchrun -m cosmos_framework.scripts.train --sft-toml=...

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REL="cosmos3_action/action_sft/action_policy_fk_point_singlerighthand_edge"
RUNS_ROOT="/data/shichaojian/runs/cosmos"
ALLOWLIST="${ALLOWLIST:-$ROOT/examples/singlerighthand_101_episodes.txt}"
CONFIG="$ROOT/cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_singlerighthand_edge.py"

# --- environment the A800/cu130 stack needs -------------------------------------
# LD_LIBRARY_PATH in NGC-style containers shadows the venv's nvidia-cublas wheel
# with the CUDA toolkit's libcublasLt (13.4.0.1 vs 13.1.0.3); the mismatch shows up
# as CUBLAS_STATUS_NOT_INITIALIZED inside the first eval, not at startup.  Both
# single-modality lines export this in their own launcher; this wrapper only needs
# it so its own pre-flight `python` calls are consistent with the training process.
export LD_LIBRARY_PATH=""
# Pin the interpreter.  This worktree has no .venv of its own: cosmos_framework is
# an editable install in the sibling venv whose .pth hard-codes that worktree, and
# PYTHONPATH=. (set by the launcher) is what makes THIS worktree's sources win.
# Never fall back to `command -v python`: job pods ship an old python3, and the
# failure would be a confusing ImportError minutes into the launch.
export PYTHON_BIN="${PYTHON_BIN:-/mnt/afs/WorldAct-cosmos3-edge-droid-sft/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || {
    echo "ERROR: interpreter is missing or not executable: $PYTHON_BIN" >&2
    echo "       set PYTHON_BIN=... to point at another one; it must have torch" >&2
    exit 1
}
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE="${WANDB_MODE:-offline}"
# flash2's varlen path is banned upstream; on sm80 (A800) that leaves natten as the
# only varlen backend and costs ~1.85x on step time.  Set inside the script rather
# than required on the command line: this prefix has been dropped from a submission
# before, and the only symptom is a single early log line.
export COSMOS_FLASH2_VARLEN="${COSMOS_FLASH2_VARLEN:-1}"

export EDGE_DROID_MODEL_PATH="${EDGE_DROID_MODEL_PATH:-/data/shichaojian/models/cosmos3-edge-droid}"
export BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-/data/shichaojian/models/cosmos3-edge-droid-dcp}"
export SINGLERIGHTHAND_RAW_ROOT="${SINGLERIGHTHAND_RAW_ROOT:-/data/shichaojian/raw_data/singlerighthand_sandwich_100}"
export SINGLERIGHTHAND_CACHE_ROOT="${SINGLERIGHTHAND_CACHE_ROOT:-/data/shichaojian/datasets/singlerighthand-sandwich-100-cosmos-cache}"
export FK_ANNOTATION_ROOT="${FK_ANNOTATION_ROOT:-/data/shichaojian/raw_data/sandwich_fk21}"
# FK displacement scale is a property of the data selection: 0.083745 is the
# sandwich 101-episode value, 0.040438 the dropper 101-episode value (both
# measured with tools/scan_fk_displacement_scale.py). The recipe reads this env.
export FK_DISPLACEMENT_SCALE="${FK_DISPLACEMENT_SCALE:-0.083745}"
export POINTFLOW_SONATA_CHECKPOINT="${POINTFLOW_SONATA_CHECKPOINT:-/data/shichaojian/checkpoints/ptv3/sonata_small.pth}"
export POINTFLOW_MANIFEST="${POINTFLOW_MANIFEST:-/mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow/pointflow_outputs/sandwich_924_20260928/manifest.json}"

# --- topology -------------------------------------------------------------------
export NPROC_PER_NODE="${NPROC_PER_NODE:-${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}}"
_GPUS_PER_NODE="$NPROC_PER_NODE"
# NODE_RANK is deliberately NOT defaulted here: the launcher derives it from the
# Kubeflow pod name, and that derivation only fires while the variable is empty.
# Defaulting it to 0 makes every worker believe it is the master and they all hang
# at rendezvous.
_NODES="${SENSECORE_PYTORCH_NNODES:-${NNODES:-1}}"
PER_RANK_BATCH="${PER_RANK_BATCH:-16}"
[[ "$PER_RANK_BATCH" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: PER_RANK_BATCH must be a positive integer, got '$PER_RANK_BATCH'" >&2
    exit 1
}
[[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: NPROC_PER_NODE must be a positive integer, got '$NPROC_PER_NODE'" >&2
    exit 1
}

# --- input guards: fail here, not minutes into the model build -------------------
[[ -f "$ALLOWLIST" ]] || { echo "ERROR: missing allowlist: $ALLOWLIST" >&2; exit 1; }
_ALLOWLIST_COUNT="$(grep -c . "$ALLOWLIST" || true)"
[[ "${_ALLOWLIST_COUNT:-0}" -ge 1 ]] || { echo "ERROR: empty allowlist: $ALLOWLIST" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "ERROR: missing recipe: $CONFIG" >&2; exit 1; }
[[ -f "$POINTFLOW_MANIFEST" ]] || { echo "ERROR: missing PointFlow manifest: $POINTFLOW_MANIFEST" >&2; exit 1; }

# --- output root ----------------------------------------------------------------
: "${OUTPUT_ROOT:?set OUTPUT_ROOT to a fresh directory, e.g. OUTPUT_ROOT=/data/shichaojian/runs/cosmos/fk-point-101-<date>}"
[[ "$OUTPUT_ROOT" == /* ]] || OUTPUT_ROOT="$RUNS_ROOT/$OUTPUT_ROOT"
if [[ -n "${RESUME:-}" ]]; then
    [[ -d "$OUTPUT_ROOT/$REL/checkpoints" ]] || {
        echo "ERROR: RESUME=1 but $OUTPUT_ROOT has no $REL/checkpoints" >&2; exit 1; }
    # Old 300-point runs used a different scale; resume only a matching distribution.
    _SAVED_CONFIG="$OUTPUT_ROOT/$REL/config.yaml"
    _FK_CAMERA_PROFILE="${FK_CAMERA_PROFILE:-legacy}"
    if [[ "$_FK_CAMERA_PROFILE" != "legacy" ]] || grep -q 'fk_camera_profile:' "$_SAVED_CONFIG"; then
        grep -Eq "^[[:space:]]+fk_camera_profile: ${_FK_CAMERA_PROFILE}[[:space:]]*$" "$_SAVED_CONFIG" || {
            echo "ERROR: saved FK camera profile differs from $_FK_CAMERA_PROFILE; use a fresh OUTPUT_ROOT." >&2
            exit 1
        }
    fi
    grep -Eq "^[[:space:]]+pointflow_displacement_scale: ${POINTFLOW_DISPLACEMENT_SCALE:-0.0528}[[:space:]]*$" "$_SAVED_CONFIG" || {
        echo "ERROR: saved PointFlow scale differs or config.yaml is missing; use a fresh OUTPUT_ROOT." >&2
        exit 1
    }
    grep -Eq "^[[:space:]]+fk_displacement_scale: ${FK_DISPLACEMENT_SCALE}[[:space:]]*$" "$_SAVED_CONFIG" || {
        echo "ERROR: saved FK scale differs from FK_DISPLACEMENT_SCALE=$FK_DISPLACEMENT_SCALE; use a fresh OUTPUT_ROOT." >&2
        exit 1
    }
else
    # Guard on checkpoints/, NOT on the directory.  Every node runs this script at
    # the same time, and _sft_launcher_common.sh does `mkdir -p "$LOG_DIR"` -- so a
    # directory-existence guard is a race the master always wins and the workers
    # always lose.  checkpoints/ only exists once a run has actually trained.
    [[ ! -d "$OUTPUT_ROOT/$REL/checkpoints" ]] || {
        echo "ERROR: $OUTPUT_ROOT already holds a finished run's checkpoints;" >&2
        echo "       pick a fresh directory or pass RESUME=1 to continue it." >&2
        exit 1; }
fi
export OUTPUT_ROOT
export IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT"
export SINGLERIGHTHAND_EPISODE_ALLOWLIST="$ALLOWLIST"

# --- HSDP + the run's shape ------------------------------------------------------
# shard = GPUs per node (intra-node FSDP over NVLink, one all-gather per layer),
# replicate = node count (inter-node gradient all-reduce only).  Do NOT shard across
# nodes: that pushes a per-layer all-gather onto the interconnect.
_TAIL="model.config.parallelism.data_parallel_shard_degree=${_GPUS_PER_NODE}"
_TAIL="$_TAIL model.config.parallelism.data_parallel_replicate_degree=${_NODES}"
_TAIL="$_TAIL dataloader_train.max_samples_per_batch=${PER_RANK_BATCH}"
_TAIL="$_TAIL model.config.rectified_flow_training_config.fk_loss_objective=charbonnier"
_TAIL="$_TAIL trainer.validation_iter=500"
if [[ -n "${MAX_ITER:-}" ]]; then
    _TAIL="trainer.max_iter=$MAX_ITER scheduler.cycle_lengths=[$MAX_ITER] $_TAIL"
fi
export EXTRA_TAIL_OVERRIDES="$_TAIL ${EXTRA_TAIL_OVERRIDES:-}"

# --- both eval arms on -----------------------------------------------------------
# The three callbacks sample on EVERY rank (FSDP collectives must stay symmetric)
# and write on rank 0, so at 16 ranks the eval costs 16x the GPU time of one rank.
# FK_ROLLOUT_EVERY counts validation events, not steps.
export FK_VAL_ON_START="${FK_VAL_ON_START:-true}"
export FK_EVAL_EVERY="${FK_EVAL_EVERY:-1}"
export FK_ROLLOUT="${FK_ROLLOUT:-1}"
export FK_ROLLOUT_EVERY="${FK_ROLLOUT_EVERY:-2}"
export FK_ROLLOUT_SECONDS="${FK_ROLLOUT_SECONDS:-16}"
export FK_ROLLOUT_GENERATED="${FK_ROLLOUT_GENERATED:-1}"
export FK_ROLLOUT_JOINT_ACTION="${FK_ROLLOUT_JOINT_ACTION:-1}"
export POINTFLOW_EVAL_EVERY="${POINTFLOW_EVAL_EVERY:-1}"
export POINTFLOW_EVAL_JOINT="${POINTFLOW_EVAL_JOINT:-true}"

# --- plan -----------------------------------------------------------------------
_GLOBAL_BATCH=$((PER_RANK_BATCH * _GPUS_PER_NODE * _NODES))
echo "=============================================================================="
echo " fk + point four-modality training"
echo "   output root : $OUTPUT_ROOT"
echo "   episodes    : $_ALLOWLIST_COUNT  ($ALLOWLIST)"
echo "   gpus        : $_GPUS_PER_NODE per node x $_NODES node(s) = $((_GPUS_PER_NODE * _NODES)) ranks"
echo "   batch       : $PER_RANK_BATCH per rank -> global $_GLOBAL_BATCH samples/step"
echo "   topology    : shard=$_GPUS_PER_NODE replicate=$_NODES"
echo "   pointflow   : ${POINTFLOW_TOKEN_MODE:-cluster}, up to ${POINTFLOW_SELECT_TOP_N:-500} points, scale=${POINTFLOW_DISPLACEMENT_SCALE:-0.0528}"
echo "   fk          : scale=$FK_DISPLACEMENT_SCALE, annotations $FK_ANNOTATION_ROOT, camera=${FK_CAMERA_PROFILE:-legacy}"
echo "   point cache : ${POINTFLOW_WINDOW_CACHE_ROOT-$SINGLERIGHTHAND_CACHE_ROOT/pointflow_windows}"
echo "   attention   : COSMOS_FLASH2_VARLEN=$COSMOS_FLASH2_VARLEN"
echo "   eval        : fk_eval + fk_rollout + joint arms, pointflow_eval + joint"
echo "   overrides   : $EXTRA_TAIL_OVERRIDES model.config.rectified_flow_training_config.pointflow_displacement_scale=${POINTFLOW_DISPLACEMENT_SCALE:-0.0528}"
if [[ -n "${RESUME:-}" ]]; then echo "   RESUME      : yes"; fi
echo "=============================================================================="
env | grep -E '^(SENSECORE|MASTER_|NNODES|NODE_RANK|WORLD_SIZE|GROUP_RANK|RANK|PET_|TORCH_|LOCAL_RANK)' \
    | sort | sed 's/^/[topology-env] /' || true

if [[ -n "${DRY_RUN:-}" ]]; then
    echo "(DRY_RUN: not launching)"
    exit 0
fi

cd "$ROOT"
exec bash "$ROOT/examples/launch_sft_action_policy_fk_point_singlerighthand_edge.sh"
