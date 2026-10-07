#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# The six-checkpoint joint sweep: three checkpoints from each of the two ten-episode
# runs, each sampled with video + future action + FK all denoised together.
#
#     bash tools/run_fk_joint_sweep.sh              # what would run
#     RUN=1 bash tools/run_fk_joint_sweep.sh        # actually run it
#     ONLY=9:6000 bash tools/run_fk_joint_sweep.sh  # one entry
#
# Why a list and not six typed commands: the two runs trained under DIFFERENT
# fk_displacement_scale values (0.07445 vs 0.083745, a 12% difference), and the
# sampler multiplies by whatever the config says.  Getting that wrong scales every
# predicted displacement with no error message and no other visible symptom --
# run_fk_joint_eval.sh refuses to launch on a mismatch, which is the guard, and
# SCALE= per entry is what satisfies it.  A table puts the value next to the run it
# belongs to instead of leaving it to whoever remembers.
#
# The three checkpoints per run are start / middle / end, so the sweep answers
# "is it still improving" -- a single final checkpoint cannot distinguish a
# converged branch from one that peaked and decayed.
#
# Cost is the reason this is sequential and not parallel: each entry loads the
# model and runs 8 windows x 32 steps per episode plus ~8 forwards per fixed case.
# The GPU count is the caller's decision (NPROC_PER_NODE), and two entries sharing
# one card would just interleave.
#
# Renders are NOT produced here: they need the VAE decode and are done afterwards
# by tools/render_fk_joint.sh, per output root, so a failed render never costs a
# sampling run.

set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKDIR"

# run-name-suffix : iteration : fk_displacement_scale the run TRAINED under.
# The scale column is read from each run's own config.yaml and checked below
# rather than trusted from this table -- a typo here would be a silent 12% scaling.
ENTRIES=(
    "9:2000:0.07445"
    "9:6000:0.07445"
    "9:10000:0.07445"
    "10:3000:0.083745"
    "10:6000:0.083745"
    "10:9500:0.083745"
)
ALLOWLIST="$WORKDIR/examples/pointflow_sandwich_10_episodes.txt"
RUNS_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/runs/cosmos"

[[ -f "$ALLOWLIST" ]] || { echo "ERROR: missing allowlist $ALLOWLIST" >&2; exit 1; }

# Every entry is checked before any of them runs.  A sweep that dies at entry four
# after spending an hour on the first three is worse than one that refuses to start.
PLAN=()
for entry in "${ENTRIES[@]}"; do
    IFS=: read -r suffix iter scale <<<"$entry"
    name="fk-singlerighthand-edge-$suffix"
    [[ "$scale" == "0.07445" || "$scale" == "0.083745" ]] || {
        echo "ERROR: $entry has scale $scale, which is neither measured value" >&2; exit 1; }
    ckpt="$(printf 'iter_%09d' "$iter")"
    ckpt_dir="$RUNS_ROOT/$name/cosmos3_action/action_sft/action_policy_fk_singlerighthand_edge/checkpoints/$ckpt"
    [[ -d "$ckpt_dir" ]] || { echo "ERROR: $name has no $ckpt" >&2; exit 1; }
    # The table's scale against the run's own resolved config: the whole point of
    # the column is that it is easy to get wrong, so it is not taken on faith.
    trained="$(grep -o 'fk_displacement_scale: [0-9.]*' \
        "$RUNS_ROOT/$name/cosmos3_action/action_sft/action_policy_fk_singlerighthand_edge/config.yaml" \
        2>/dev/null | head -1 | awk '{print $2}')"
    [[ "$trained" == "$scale" ]] || {
        echo "ERROR: $name trained under $trained, the table says $scale." >&2
        echo "       Fix the table rather than the run: SCALE must be the TRAINED value." >&2
        exit 1; }
    # 0.07445 is the ten-episode value; the config carries the 101-episode one, so
    # every ten-episode entry needs the override and no other entry does.
    if [[ "$scale" == "0.07445" ]]; then
        over="SCALE=$scale"
    else
        over=""
    fi
    PLAN+=("$suffix $iter $name $over")
done

echo "=== FK joint sweep: video + action + FK, ${#PLAN[@]} checkpoints ==="
printf '  %-4s %-7s %-32s %s\n' RUN ITER SOURCE_RUN SCALE_OVERRIDE
for row in "${PLAN[@]}"; do
    # shellcheck disable=SC2086
    set -- $row
    printf '  %-4s %-7s %-32s %s\n' "$1" "$2" "$3" "${4:-<config, matches>}"
done
echo

if [[ "${RUN:-0}" != "1" ]]; then
    echo "Dry run.  Re-run with RUN=1 to execute, one entry at a time."
    echo "Each entry creates its own output root under $RUNS_ROOT (fk-joint-eval-N),"
    echo "so nothing overwrites and a re-run resumes into a fresh directory."
    exit 0
fi

for row in "${PLAN[@]}"; do
    # shellcheck disable=SC2086
    set -- $row
    suffix="$1"; iter="$2"; name="$3"; over="${4:-}"
    if [[ -n "${ONLY:-}" && "$ONLY" != "$suffix:$iter" ]]; then
        continue
    fi
    echo
    echo "===================================================================="
    echo "== $name @ iter_$(printf '%09d' "$iter")   ${over:-<no scale override>}"
    echo "===================================================================="
    # OUT is per entry: run_fk_joint_eval.sh suffixes when the name exists, and a
    # shared name would make "which directory is this checkpoint's" a question.
    # shellcheck disable=SC2086
    env SOURCE_RUN="$name" ITER="$iter" ALLOWLIST="$ALLOWLIST" \
        OUT="fk-joint-eval-$suffix-$iter" $over \
        bash tools/run_fk_joint_eval.sh
done

echo
echo "=== done ==="
echo "Render each root in turn -- the renderer's default is newest-only, and each"
echo "root holds one checkpoint's 17 s clip:"
echo "  for d in \$(ls -dt $RUNS_ROOT/fk-joint-eval-*); do echo \"== \$d\"; \\"
echo "      ROOT=\"\$d\" OUT_ROOT=\".../renders/fk_joint/\$(basename \"\$d\")\" \\"
echo "      bash tools/render_fk_joint.sh; done"
