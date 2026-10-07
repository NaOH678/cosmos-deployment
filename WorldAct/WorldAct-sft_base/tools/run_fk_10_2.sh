#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Retrain the 10-episode FK recipe with the CORRECT displacement scale.
#
#     bash tools/run_fk_10_2.sh
#
# Lands in .../runs/cosmos/fk-singlerighthand-edge-10-2 (run_fk_retrain.sh suffixes
# fk-singlerighthand-edge-10, which exists).
#
# What this is: run -10, unchanged except for the scale.  -10 trained on the
# ten-episode allowlist but read fk_displacement_scale = 0.083745 from the config
# file -- the value measured over the **101**-episode set, which by then had been
# written into the file for the upcoming 101-episode run.  Run -9 had pinned
# 0.074450 explicitly on its command line; -10 did not, so it silently picked up
# the wider set's number.
#
# 0.074450 is the pooled per-element std over THIS run's episode set
# (examples/pointflow_sandwich_10_episodes.txt, n=26,522,496).  The scale is a
# property of the selection, not a constant: the ten episodes are a strict subset
# of the 101, and the 91 added episodes carry larger hand motion, which is why the
# two differ by 12%.
#
# Why an override and not an edit to the config file: the file's 0.083745 is
# correct for the 101-episode recipe, and tools/run_fk_101.sh greps for exactly
# that literal and refuses to start if it drifts.  Changing it would fix this run
# by breaking that one.  The resolved config records the override, so the run
# documents what it actually trained under -- same as -9.
#
# Only the scale differs from -10, so -10 vs -10-2 isolates the scale's effect.
# (-9 vs -10 could not: those two also differ in the loss objective, mse vs
# charbonnier.)

set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKDIR"

SCALE=0.074450
ALLOWLIST="$WORKDIR/examples/pointflow_sandwich_10_episodes.txt"
CONFIG="$WORKDIR/cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_singlerighthand_edge.py"

[[ -f "$ALLOWLIST" ]] || { echo "ERROR: missing allowlist $ALLOWLIST" >&2; exit 1; }
[[ "$(grep -c . "$ALLOWLIST" || true)" == "10" ]] || {
    echo "ERROR: $ALLOWLIST should hold 10 episodes, found $(grep -c . "$ALLOWLIST" || true)" >&2
    echo "       $SCALE was measured on the ten-episode set; a different set needs a re-scan:" >&2
    echo "       python tools/scan_fk_displacement_scale.py --episodes <allowlist>" >&2
    exit 1; }

# The config file must still carry the 101-episode value.  If it does not, someone
# has been editing it and the two recipes' scales are no longer where this script
# thinks they are -- stop and look rather than launching under a guess.
grep -q 'fk_displacement_scale"\] = 0.083745' "$CONFIG" || {
    echo "ERROR: $CONFIG no longer sets fk_displacement_scale = 0.083745." >&2
    echo "       That is the 101-episode value this run overrides; if it moved, work" >&2
    echo "       out which recipe the file now describes before launching." >&2
    exit 1; }

# ⚠️  The trailing space is load-bearing.  run_fk_retrain.sh appends its own
# overrides with NO separator:
#     export EXTRA_TAIL_OVERRIDES="${EXTRA_TAIL_OVERRIDES:-}$OVERRIDES"
# so without it the last one fuses with trainer.max_iter into a single garbage
# token.  (It fails loudly -- an invalid fk_loss_objective -- but only at config
# validation, after the GPU is already allocated.  Hence the assert below.)
export EXTRA_TAIL_OVERRIDES="model.config.rectified_flow_training_config.fk_displacement_scale=$SCALE model.config.rectified_flow_training_config.fk_loss_objective=charbonnier "
[[ "${EXTRA_TAIL_OVERRIDES}" == *" " ]] || { echo "ERROR: EXTRA_TAIL_OVERRIDES must end with a space" >&2; exit 1; }

# 1 GPU.  ⚠️  This is NOT "everything else unchanged": max_samples_per_batch is
# PER RANK, so the effective global batch goes 256 -> 32 (8x smaller).  Run -10 vs
# -10-2 therefore differs in the scale AND the batch, and the scale's own effect
# cannot be read off the pair.  Set NPROC_PER_NODE=8 for a clean comparison.
# Which physical card it lands on is CUDA_VISIBLE_DEVICES' business, not this script's.
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

# Rollout: the middle 16 s of each of the two val episodes (8 windows x 32 frames),
# sampled and written to <run>/fk_rollout/step_*/<episode>/rollout.npz.
# tools/render_fk_rollout.py turns those into the animation and the projection onto
# the real video.  FK_ROLLOUT_SECONDS=0 takes the whole episode (~40 s, ~18 windows),
# which is ~2.5x the forwards.
# FK_ROLLOUT_EVERY counts validation PASSES, not steps: validation_iter=100 puts a
# pass every 100 steps, so 10 means a rollout every ~1000 steps (10 per 10000).
export FK_ROLLOUT="${FK_ROLLOUT:-1}"
export FK_ROLLOUT_EVERY="${FK_ROLLOUT_EVERY:-10}"
export FK_ROLLOUT_SECONDS="${FK_ROLLOUT_SECONDS:-16}"

# The base name, not the final one: run_fk_retrain.sh suffixes it because the
# directory exists, which is what produces -10-2.  Run it twice and the second
# lands in -10-3 rather than overwriting.
export RUN_NAME=fk-singlerighthand-edge-10
export SINGLERIGHTHAND_EPISODE_ALLOWLIST="$ALLOWLIST"

echo "=== FK 10-episode retrain, corrected scale ==="
echo "  scale       : $SCALE   (corrected; -10 used 0.083745)"
echo "  episodes    : 10   ($(basename "$ALLOWLIST"))"
echo "  objective   : charbonnier   (same as -10)"
echo "  max_iter    : from the recipe ($(grep -m1 'max_iter' "$WORKDIR/examples/toml/sft_config/action_policy_fk_singlerighthand_edge.toml" | tr -dc '0-9'))"
echo "  gpus        : $NPROC_PER_NODE   (global batch = $((NPROC_PER_NODE * 32)); -10 ran 8 GPUs = 256)"
echo "  eval figures: FK_EVAL_FIGURES=${FK_EVAL_FIGURES:-false}  (off: no comparison.png during training;"
echo "                draw any step later with tools/render_fk_case.py --eval-dir <run>/.../fk_eval --all --video)"
echo "  rollout     : FK_ROLLOUT=$FK_ROLLOUT, middle ${FK_ROLLOUT_SECONDS}s clips, every $FK_ROLLOUT_EVERY validation passes"
echo

exec bash tools/run_fk_retrain.sh
