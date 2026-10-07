#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Sample the JOINT video+FK arm on a checkpoint that already exists.  No training.
#
#     bash tools/run_fk_joint_eval.sh
#     SOURCE_RUN=fk-singlerighthand-edge-101 ITER=10000 bash tools/run_fk_joint_eval.sh
#
# What this is for: the joint arm -- generate the video from noise (i2v) and denoise
# FK alongside it, instead of handing FK a clean GT video -- is an INFERENCE-ONLY
# change.  It lives entirely in the sampler (``_sample_joint`` /
# ``_make_joint_velocity`` in omni_mot_model.py) and touches no weight, because
# training already puts both modalities into one packed sequence and noises them
# together; the model has always been able to denoise them jointly.  So a finished
# run's checkpoint can answer the question directly, and re-training to ask it would
# be spending GPU-days on a switch.
#
# How it avoids training: the source run's checkpoint is SYMLINKED (30 GB, the
# source run is left untouched) into a fresh output root as latest_checkpoint.txt,
# and ``trainer.max_iter`` is pinned to the checkpoint's own iteration.  The loop's
# first check already satisfies the exit condition, so it runs the initial
# validation -- which is where the eval callbacks fire -- and stops.  This is the
# pattern tools/run_fk_sampler_probe.sh established; the alternative of pointing
# ``checkpoint.load_path`` at the iter_* directory needs the right key set and
# load_training_state, and getting that subtly wrong looks exactly like a broken
# sampler.
#
# ⚠️  The scale must match.  ``fk_displacement_scale`` is metres-per-model-unit and
# the sampler multiplies FK's prediction by whatever the CONFIG says, not by
# whatever the checkpoint was trained under -- so evaluating a checkpoint from a run
# with a different scale silently shrinks or stretches every predicted displacement
# with no error message.  This is not hypothetical: run -10 trained under 0.083745
# while its ten-episode data measured 0.074450.  The check below refuses to launch
# on a mismatch rather than produce a plausible-looking wrong trajectory.
#
# Overridable:
#   SOURCE_RUN  run to take the checkpoint from (default below)
#   ITER        iteration to load (default: its latest_checkpoint.txt)
#   EVERY       render every Nth val case? -> FK_EVAL_CASES (default 2 = both val cases)
#   OUT         output root base name (suffixed if it exists)
#   SCALE       fk_displacement_scale to SAMPLE under, when it differs from the
#               config's.  Needed for any run that trained under the ten-episode
#               value (0.07445) rather than the config's 101-episode 0.083745 --
#               the guard below refuses to start otherwise.
#   JOINT_ACTION  1 (default) denoises the future ACTION alongside the video and FK,
#               which is the deployed configuration.  0 keeps the two-segment arm
#               (clean action) whose numbers are already recorded; the difference
#               between the two, at the same case and seed, is what the clean action
#               was worth.

set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKDIR"

REL="cosmos3_action/action_sft/action_policy_fk_singlerighthand_edge"
RUNS_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/runs/cosmos"
SOURCE_RUN_NAME="${SOURCE_RUN:-fk-singlerighthand-edge-101}"
SOURCE_RUN="$RUNS_ROOT/$SOURCE_RUN_NAME"
SRC="$SOURCE_RUN/$REL"
CONFIG="$WORKDIR/cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_singlerighthand_edge.py"
# Overridable: the episode set decides which val episodes get evaluated, and it has
# to match the source run's or the numbers describe a different dataset.
ALLOWLIST="${ALLOWLIST:-$WORKDIR/examples/singlerighthand_101_episodes.txt}"
PYTHON_BIN="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/WorldAct-cosmos3-edge-droid-sft/.venv/bin/python"
# Defaulted here rather than left to each use: this script runs under ``set -u``, and
# the tail construction below reads ``$SCALE`` directly.  That is what killed the
# second entry of the sweep -- the first entry passes SCALE=, the next passes nothing,
# and the error landed only after a full sampling run had already finished, so it cost
# a run to find instead of nothing.
SCALE="${SCALE:-}"

[[ -d "$SRC" ]] || { echo "ERROR: no such run: $SRC" >&2; exit 1; }

if [[ -n "${ITER:-}" ]]; then
    RESUME_ITER="$ITER"
else
    LATEST="$(cat "$SRC/checkpoints/latest_checkpoint.txt" 2>/dev/null || true)"
    [[ -n "$LATEST" ]] || { echo "ERROR: $SRC has no latest_checkpoint.txt; pass ITER=" >&2; exit 1; }
    RESUME_ITER="$((10#${LATEST#iter_}))"   # 10# so 000010000 is decimal, not octal
fi
CKPT="$(printf 'iter_%09d' "$RESUME_ITER")"
[[ -d "$SRC/checkpoints/$CKPT" ]] || { echo "ERROR: no checkpoint $SRC/checkpoints/$CKPT" >&2; exit 1; }

# A checkpoint comes in two shapes, and they need different loads.
#
# COMPLETE -- model/ optim/ scheduler/ trainer/ -- is resume-loaded exactly as this
# script has always done: symlink it in, write latest_checkpoint.txt, pin
# max_iter to its own iteration, and the loop's first check exits right after the
# start validation.  Nothing here changes for that case.
#
# PRUNED -- model/ alone -- is what every OLDER iteration of these runs looks like:
# the training state is deleted as a run goes, keeping the weights for inference, and
# each run's latest_checkpoint.txt points at the one that still has it.  A resume asks
# for all of CHECKPOINT_KEYS (utils/checkpoint/dcp.py:737) and a component that is not
# there fails in a way that names nothing: the reader returns no metadata and torch
# raises ``AssertionError: metadata is None``, its real message swallowed by a bare
# ``except Exception``.  So a pruned checkpoint is loaded with the absent components
# excluded instead -- see FK_SKIP_RESUME_KEYS in the recipe.
#
# What the exclusion costs is the iteration, which only the ``trainer`` component
# carries (dcp.py:920): the load reports 0, so max_iter is pinned to 0 as well.  That
# is the same trick, not a weaker one -- the trainer tests ``iteration >= max_iter``
# AFTER run_validation_on_start (trainer/__init__.py:244 then :288), so the eval runs
# and no training step does.  The eval then writes step_0000000, not step_<ITER>.
PRUNED=0
for _part in optim scheduler trainer; do
    [[ -d "$SRC/checkpoints/$CKPT/$_part" ]] || PRUNED=1
done
if [[ "$PRUNED" == 1 ]]; then
    MAX_ITER=0
    export FK_SKIP_RESUME_KEYS="optim,scheduler,trainer,dataloader"
else
    MAX_ITER="$RESUME_ITER"
    unset FK_SKIP_RESUME_KEYS
fi

# The scale the config will sample under, against the one the source run trained
# under.  Read from the two RESOLVED configs, not from a grep of the source file:
# the whole point is which value was in force, and -10's mismatch came from exactly
# that difference.  ``SCALE`` is the escape hatch for a run that trained under the
# ten-episode value -- the one case where the config's value is the wrong one, and
# where editing the config would make every future run of the *other* allowlist
# wrong.
SRC_SCALE="$(grep -o 'fk_displacement_scale: [0-9.]*' "$SRC/config.yaml" 2>/dev/null | head -1 | awk '{print $2}')"
NOW_SCALE="${SCALE:-$(grep -o 'fk_displacement_scale"\] = [0-9.]*' "$CONFIG" | head -1 | awk '{print $3}')}"
[[ -n "$SRC_SCALE" && -n "$NOW_SCALE" ]] || { echo "ERROR: could not read both scales" >&2; exit 1; }
if [[ "$SRC_SCALE" != "$NOW_SCALE" ]]; then
    echo "ERROR: scale mismatch -- every predicted displacement would be off by $NOW_SCALE/$SRC_SCALE." >&2
    echo "       $SOURCE_RUN_NAME trained under $SRC_SCALE; this eval would sample under $NOW_SCALE." >&2
    echo "       Pass SCALE=$SRC_SCALE to sample under the trained value, or set" >&2
    echo "       model.config.rectified_flow_training_config.fk_displacement_scale via EXTRA_TAIL_OVERRIDES." >&2
    exit 1
fi

BASE_OUT="${OUT:-fk-joint-eval}"
OUT_ROOT="$RUNS_ROOT/$BASE_OUT"
suffix=1
while [[ -e "$OUT_ROOT" ]]; do
    suffix=$((suffix + 1))
    OUT_ROOT="$RUNS_ROOT/$BASE_OUT-$suffix"
done
[[ "$OUT_ROOT" == "$RUNS_ROOT/$BASE_OUT" ]] || echo "(using $BASE_OUT-$suffix; $BASE_OUT exists)"

DST="$OUT_ROOT/$REL"
mkdir -p "$DST/checkpoints"
ln -s "$SRC/checkpoints/$CKPT" "$DST/checkpoints/$CKPT"
printf '%s' "$CKPT" > "$DST/checkpoints/latest_checkpoint.txt"
# Carry the source run's case selection over, so this eval scores the SAME windows
# the training run did and the two numbers are comparable.  Without it the cases
# are re-selected here, and the eval-case drift documented in fk_eval_cases is
# exactly what that costs.
if [[ -f "$SRC/fk_eval/fixed_cases.json" ]]; then
    cp "$SRC/fk_eval/fixed_cases.json" "$DST/fk_eval/fixed_cases.json" 2>/dev/null || {
        mkdir -p "$DST/fk_eval"; cp "$SRC/fk_eval/fixed_cases.json" "$DST/fk_eval/fixed_cases.json"; }
fi

JOINT_ACTION="${JOINT_ACTION:-1}"
case "$JOINT_ACTION" in
    1|true|yes|on) JOINT_ACTION_FLAG=true ;;
    0|false|no|"") JOINT_ACTION_FLAG=false ;;
    *) echo "ERROR: JOINT_ACTION must be 0 or 1, got '$JOINT_ACTION'" >&2; exit 1 ;;
esac

echo "=== FK joint-arm eval (inference only) ==="
echo "  source run : $SOURCE_RUN_NAME  @ $CKPT"
echo "  output     : $OUT_ROOT"
echo "  scale      : $NOW_SCALE  (matches the source run${SCALE:+; from SCALE=, not the config})"
if [[ "$PRUNED" == 1 ]]; then
    echo "  ckpt shape : PRUNED (model/ only) -> loads model alone via"
    echo "               FK_SKIP_RESUME_KEYS=$FK_SKIP_RESUME_KEYS"
    echo "  max_iter   : 0  -> exits after the initial validation; no training"
    echo "               (the iteration lives in the pruned trainer/ component, so the"
    echo "                load reports 0 and the eval writes step_0000000)"
else
    echo "  ckpt shape : COMPLETE (model/ optim/ scheduler/ trainer/) -> full resume"
    echo "  max_iter   : $RESUME_ITER  -> exits after the initial validation; no training"
    echo "               (eval writes $(printf 'step_%07d' "$RESUME_ITER"))"
fi
if [[ "$JOINT_ACTION_FLAG" == true ]]; then
    echo "  joint arm  : FK_EVAL_JOINT=1 + JOINT_ACTION -> video AND future action from"
    echo "               noise, FK denoised alongside (the deployed configuration)"
else
    echo "  joint arm  : FK_EVAL_JOINT=1 (video from noise, action still clean,"
    echo "               FK denoised alongside)"
fi
echo

# FK_ROLLOUT + FK_ROLLOUT_GENERATED is what makes the video LONG.  FK_EVAL_JOINT
# alone samples one window per case -- 32 steps, 2.1 s -- because that is what
# fixed_cases selects.  The rollout tiles the episode's middle seconds into
# consecutive windows and generates a video for each, and the renderer stitches
# them with the boundary frame kept once: the same construction the PointFlow
# stage clips use (4 windows x 33 frames - 3 shared = 129), each window labelled
# "independent prediction" because the seam is real and is not smoothed over.
# The scale override goes through the CLI rather than the config so that the run's
# own config.yaml keeps recording the *trained* value -- which is what the guard
# above compares against on the next eval.
_TAIL="trainer.max_iter=$MAX_ITER"
if [[ -n "$SCALE" ]]; then
    _TAIL="model.config.rectified_flow_training_config.fk_displacement_scale=$SCALE $_TAIL"
fi

SINGLERIGHTHAND_EPISODE_ALLOWLIST="$ALLOWLIST" \
FK_EVAL_JOINT=1 \
FK_EVAL_JOINT_ACTION="$JOINT_ACTION_FLAG" \
FK_ROLLOUT=1 \
FK_ROLLOUT_GENERATED=1 \
FK_ROLLOUT_JOINT_ACTION="$JOINT_ACTION_FLAG" \
FK_EVAL_FIGURES=false \
OUTPUT_ROOT="$OUT_ROOT" \
EXTRA_TAIL_OVERRIDES="${EXTRA_TAIL_OVERRIDES:-} $_TAIL" \
    bash examples/launch_sft_action_policy_fk_singlerighthand_edge.sh

echo
echo "=== artefacts ==="
# The arm names the subdirectory AND the rollout root, so derive both from the one
# switch rather than restating it: the callback writes them and this only lists them.
ARM="joint"
ROLLOUT_DIR="fk_rollout"
if [[ "$JOINT_ACTION_FLAG" == true ]]; then
    ARM="joint_action"
    ROLLOUT_DIR="fk_rollout_joint_action"
fi
find "$DST/fk_eval" -name prediction.npz -path "*/$ARM/*" 2>/dev/null | sed "s|$DST/||" || true
find "$DST/$ROLLOUT_DIR" -name rollout.npz 2>/dev/null | sed "s|$DST/||" || true
echo
# The step directory is whatever the run actually wrote, read back rather than
# assumed.  It is step_<RESUME_ITER> for a complete checkpoint and step_0000000 for a
# pruned one (the iteration lives in the component that was pruned), and printing a
# guessed name is how this script first handed out a path that never existed.
STEP_DIR="$(find "$DST/fk_eval" -maxdepth 1 -type d -name 'step_*' -printf '%f\n' 2>/dev/null | sort | tail -1)"
if [[ -z "$STEP_DIR" ]]; then
    echo "WARNING: no step_* directory under $DST/fk_eval -- the eval wrote nothing." >&2
    echo "         Check the run log for the checkpoint load: it prints" >&2
    echo "         'Resuming ckpt <path> with keys: [...]' before it samples." >&2
else
    _EXPECTED="$(printf 'step_%07d' "$MAX_ITER")"
    [[ "$STEP_DIR" == "$_EXPECTED" ]] || {
        echo "NOTE: eval wrote $STEP_DIR, not the expected $_EXPECTED." >&2
        echo "      A pruned checkpoint reports iteration 0 -- see the header above." >&2
    }
fi
if [[ -n "$STEP_DIR" ]]; then
    echo "Render it -- REAL vs GENERATED panels, GT + prediction skeletons:"
    echo "  bash tools/render_fk_joint.sh            # finds the newest eval, both ARMs"
    for case in val_00 val_01; do
        echo "  PYTHONPATH=. $PYTHON_BIN tools/render_fk_projection.py \\"
        echo "      --case-dir $DST/fk_eval/$STEP_DIR/$case/$ARM --out <renders>/$case"
    done
fi
echo
echo "CHECK THE GREEN SKELETON FIRST: if GT is not on the hand in the REAL panel,"
echo "the projection is wrong and the generated panel means nothing."
