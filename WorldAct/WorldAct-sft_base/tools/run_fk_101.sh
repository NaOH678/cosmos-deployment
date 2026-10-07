#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Train the FK branch on the full 101-episode set.
#
#     bash tools/run_fk_101.sh
#
# Why a script rather than a command: the invocation needs five environment
# variables and a long single-line override string.  Every one of them is a
# multi-line paste hazard -- a wrapped override becomes two arguments, and a lost
# line silently drops a setting (an OUTPUT_ROOT lost that way is how two runs
# ended up writing into the same directory and overwriting each other's
# checkpoints).  Keeping them in a file removes the hazard.
#
# Two settings here are not defaults and would be wrong to omit:
#
#   * IMAGINAIRE_OUTPUT_ROOT, not OUTPUT_ROOT.  OUTPUT_ROOT is only an
#     intermediary: cosmos_framework/utils/config.py reads IMAGINAIRE_OUTPUT_ROOT,
#     and _sft_launcher_common.sh sets it from OUTPUT_ROOT *only if it is unset*
#     (`${IMAGINAIRE_OUTPUT_ROOT:-$OUTPUT_ROOT}`).  If the ambient environment
#     already exports it -- and on this host it does -- then OUTPUT_ROOT is
#     ignored and the run lands in whatever the ambient value points at.
#
#   * The 101-episode allowlist.  The launcher defaults to the small-sample ten.
#
# fk_displacement_scale is deliberately NOT overridden: the experiment config
# already carries 0.083745, which is the pooled per-element std measured over
# exactly these 101 episodes (tools/scan_fk_displacement_scale.py, n=202,138,272).
# The check below re-derives nothing; it only refuses if the config has drifted
# back to the ten-episode value, which would train the branch against the wrong
# unit and is not something a run should discover 6 hours in.
#
# Overridable, all optional:
#   MAX_ITER    default below; lowering it is the cheap fix when disk is tight
#   RUN_NAME    base name; a fresh directory is always chosen (see below)

set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKDIR"

REL="cosmos3_action/action_sft/action_policy_fk_singlerighthand_edge"
RUNS_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/runs/cosmos"
ALLOWLIST="$WORKDIR/examples/singlerighthand_101_episodes.txt"
CONFIG="$WORKDIR/cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_singlerighthand_edge.py"

MAX_ITER="${MAX_ITER:-20000}"
SAVE_ITER=1000
# Overridable: this is exported as NPROC_PER_NODE below, which OVERRIDES the
# launcher's own "${NPROC_PER_NODE:=${_VISIBLE_GPUS:-1}}" auto-detection.  A
# hardcoded 8 therefore breaks the moment the node has fewer visible GPUs --
# `CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 bash tools/run_fk_101.sh` is the shape,
# and without the override torchrun spawns 8 ranks against 4 devices and dies
# with "CUDA error: invalid device ordinal" (2026-09-24 14:45).
NPROC="${NPROC:-8}"

[[ -f "$ALLOWLIST" ]] || { echo "ERROR: missing allowlist $ALLOWLIST" >&2; exit 1; }
[[ "$(grep -c . "$ALLOWLIST" || true)" == "101" ]] || {
    echo "ERROR: $ALLOWLIST should hold 101 episodes, found $(grep -c . "$ALLOWLIST" || true)" >&2; exit 1; }

# The scale is a property of the episode selection.  If the config still says
# 0.074450 it is the ten-episode value and this run would train against a unit
# 12% off -- see the header.  Refuse rather than start.
grep -q 'fk_displacement_scale"\] = 0.083745' "$CONFIG" || {
    echo "ERROR: $CONFIG does not set fk_displacement_scale = 0.083745." >&2
    echo "       That is the 101-episode value; re-run tools/scan_fk_displacement_scale.py" >&2
    echo "       if the episode selection changed." >&2
    exit 1; }

if [[ -n "${RESUME:-}" ]]; then
    # Continue the newest 101 run in place.  Safe to reuse THIS directory -- and
    # only this one -- because the dataset is unchanged, so the fixed_cases.json
    # already there still indexes the same windows.  That is the opposite of the
    # fresh-run case below.
    OUTPUT_ROOT="$(ls -dt "$RUNS_ROOT"/fk-singlerighthand-edge-101* 2>/dev/null | while read -r d; do
        [[ -f "$d/$REL/checkpoints/latest_checkpoint.txt" ]] && { echo "$d"; break; }
    done)"
    [[ -n "$OUTPUT_ROOT" ]] || { echo "ERROR: RESUME=1 but no 101 run with a checkpoint found" >&2; exit 1; }
    echo "  resuming    : $(cat "$OUTPUT_ROOT/$REL/checkpoints/latest_checkpoint.txt") in $(basename "$OUTPUT_ROOT")"
else
    # A fresh directory.  A reused one still holds the previous run's
    # fk_eval/fixed_cases.json, and fixed_cases() indexes the new dataset with the
    # old indices -- it detects the mismatch and aborts ("Fixed eval identity
    # changed"), so a collision with a *different* dataset is a guaranteed crash,
    # not a silent wrong answer.
    BASE_NAME="${RUN_NAME:-fk-singlerighthand-edge-101}"
    RUN_NAME="$BASE_NAME"
    suffix=1
    while [[ -e "$RUNS_ROOT/$RUN_NAME" ]]; do
        suffix=$((suffix + 1))
        RUN_NAME="$BASE_NAME-$suffix"
    done
    OUTPUT_ROOT="$RUNS_ROOT/$RUN_NAME"
fi

CHECKPOINTS=$(( MAX_ITER / SAVE_ITER ))
NEED_GB=$(( CHECKPOINTS * 30 + 50 ))
AVAIL_GB=$(df -BG --output=avail "$RUNS_ROOT" 2>/dev/null | tail -1 | tr -dc '0-9')

echo "=== FK 101-episode train ==="
echo "  output root : $OUTPUT_ROOT"
echo "  episodes    : 101   (10x the small-sample recipe)"
echo "  scale       : 0.083745   (from the config)"
echo "  objective   : charbonnier"
echo "  gpus        : $NPROC   (global batch = $((NPROC * 32)) samples/step)"
echo "  max_iter    : $MAX_ITER   save every $SAVE_ITER, eval every 500"
echo "  checkpoints : $CHECKPOINTS x ~30 GB = ~$((CHECKPOINTS * 30)) GB"
echo "  runs root   : ${AVAIL_GB:-?} GB free, need >= ${NEED_GB} GB"
echo "  rollout     : middle 16s (17.1s of frames) every 1000 steps, action+video+FK"
echo "                jointly denoised -- overridable via FK_ROLLOUT*/FK_ROLLOUT_JOINT_ACTION*"
echo

# No disk guard.  The line above is information, not a gate: there is no retention
# policy, so a long run does need periodic pruning, but deciding when is the
# operator's call and a refusal here would only be in the way.

# Set BOTH.  IMAGINAIRE_OUTPUT_ROOT is the one that decides (config.py); OUTPUT_ROOT
# is set so the launcher's own logging and its `${IMAGINAIRE_OUTPUT_ROOT:-$OUTPUT_ROOT}`
# fallback agree rather than contradicting each other.
export IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT"
export OUTPUT_ROOT="$OUTPUT_ROOT"
export SINGLERIGHTHAND_EPISODE_ALLOWLIST="$ALLOWLIST"
export NPROC_PER_NODE="$NPROC"
# scheduler.cycle_lengths MUST track max_iter.  The LR schedule is a list of cycles
# and LambdaWarmUpCosineScheduler.find_in_interval() falls off its end returning
# None for any step past the last cycle; schedule() then does
# `lr_warm_up_steps[None]` and dies with omegaconf's "ListConfig indices must be
# integers or slices, not NoneType".  The recipe pins cycle_lengths = [10000], so
# raising only max_iter trains 10000 steps and then crashes on step 10001 -- which
# is exactly what happened to run -101 on 2026-09-23 (log 04:29:30).  Setting both
# from $MAX_ITER here is the reason this is a script and not a pasted command.
export EXTRA_TAIL_OVERRIDES="trainer.max_iter=$MAX_ITER scheduler.cycle_lengths=[$MAX_ITER] model.config.rectified_flow_training_config.fk_loss_objective=charbonnier checkpoint.save_iter=$SAVE_ITER trainer.validation_iter=500"

# ---------------------------------------------------------------- rollout ----
# The middle 16 s of each of the two val episodes -- 8 windows x 32 model frames
# = 256 frames = 17.1 s at the dataset's 15 Hz, which is the "two 17 s videos"
# the -10-2 arm produces.  Writes fk_rollout_joint_action/step_*/<episode>/
# rollout.npz; tools/render_fk_rollout.py draws them offline (the animation, and
# the two-panel projection: real recording on the left, the model's own
# generation on the right).
#
# FK_ROLLOUT_JOINT_ACTION is the deployed configuration: the future ACTION and
# the video are denoised jointly with FK, so the anchor frame is the only thing
# still handed over.  It REQUIRES FK_ROLLOUT_GENERATED -- the callback rejects it
# alone rather than implying the parent, so both are set below.  This is a SAMPLING
# switch, not a training one -- it changes what the eval draws, never what the
# model learns, which is what makes resuming onto it training-equivalent.  (The
# sibling PointFlow worktree reached the same conclusion from the other side:
# independent_*_schedule is the training-side fix that puts this joint form in
# the training distribution.  We already have independent_fk_schedule=True; for
# ACTION the coupled schedule already covers the joint diagonal, so nothing on
# the training side has to change either.)
#
# FK_ROLLOUT_EVERY counts validation PASSES, and validation_iter=500 here, so 2
# gives one rollout every 1000 steps -- 10 over the 9000..20000 that remain.
export FK_ROLLOUT="${FK_ROLLOUT:-1}"
export FK_ROLLOUT_EVERY="${FK_ROLLOUT_EVERY:-2}"
export FK_ROLLOUT_SECONDS="${FK_ROLLOUT_SECONDS:-16}"
# Both switches, explicitly, even though joint_action now IMPLIES generated_video
# in the callback.  The implication is the documented contract and the callback
# honours it, but a 15-hour run should not be the thing that tests it: setting only
# FK_ROLLOUT_JOINT_ACTION is what killed the first attempt at 02:50 on 2026-09-24,
# when the callback merely *required* the parent switch instead of implying it.
export FK_ROLLOUT_GENERATED="${FK_ROLLOUT_GENERATED:-1}"
export FK_ROLLOUT_JOINT_ACTION="${FK_ROLLOUT_JOINT_ACTION:-1}"

printf '%s\n' "$OUTPUT_ROOT" > /tmp/fk_last_run

exec bash examples/launch_sft_action_policy_fk_singlerighthand_edge.sh
