#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Retrain the FK branch, in a fresh output root, with a disk-space check.
#
#     bash tools/run_fk_retrain.sh
#
# Why this exists rather than a one-liner: the output root is ~100 characters and
# pasting
#     OUTPUT_ROOT=/mnt/.../fk-singlerighthand-edge-5 \
#       bash examples/launch_...
# breaks the moment the terminal wraps the path -- bash then reads the tail of the
# path as a command ("edge-5: command not found").  Keeping the long literal in a
# file removes the wrap hazard entirely.
#
# The output root is suffixed rather than reused: a fresh run must not resume into
# an older one's checkpoints, and after the fk_displacement_scale change (see
# docs/fk_modality_design.md 6.8) an old checkpoint cannot be resumed onto this
# recipe at all -- the branch learned a different target scale.
#
# Override any of these from the environment:
#   RUNS_ROOT   where runs live          (default below)
#   RUN_NAME    base name for the run    (auto-suffixed if it exists)
#   RESUME_FROM run to continue from (warm start; full resume when the checkpoint has
#               optim/, otherwise weights only and the iteration counter restarts at 0)
#   MAX_ITER    absolute target step (default: the recipe's max_iter)
#   DRY_RUN=1   print the plan and exit without launching

set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKDIR"

REL="cosmos3_action/action_sft/action_policy_fk_singlerighthand_edge"
RUNS_ROOT="${RUNS_ROOT:-/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/runs/cosmos}"
RUN_NAME="${RUN_NAME:-fk-singlerighthand-edge-5}"
# The guard compares against the checkpoints this run will actually write (computed
# below from the recipe, or from MAX_ITER when resuming), not against a fixed guess:
# a fixed number is either too low to prevent the fill it exists to prevent, or too
# high to allow a resume that comfortably fits.  MIN_FREE_GB, if set, is an extra
# floor on top of that.  The mount is shared, so a pass means only that there is room
# *now* -- old checkpoints still have to be deleted as the run proceeds.
MIN_FREE_GB="${MIN_FREE_GB:-0}"

# The sibling worktree's venv, same interpreter the launcher pins.  Needed here
# for the eval-manifest seeder below, which imports torch.
SIBLING_VENV="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/WorldAct-cosmos3-edge-droid-sft/.venv"
PYTHON_BIN="${PYTHON_BIN:-$SIBLING_VENV/bin/python}"

# Suffix rather than delete: an existing run's checkpoints are the user's data, and
# this script's job is to start a new run, not to decide what to throw away.
BASE_NAME="$RUN_NAME"
suffix=1
while [[ -e "$RUNS_ROOT/$RUN_NAME" ]]; do
    suffix=$((suffix + 1))
    RUN_NAME="$BASE_NAME-$suffix"
done
OUTPUT_ROOT="$RUNS_ROOT/$RUN_NAME"
[[ "$OUTPUT_ROOT" == "$RUNS_ROOT/$BASE_NAME" ]] || echo "(using $RUN_NAME; $BASE_NAME exists)"

# The launcher itself requires CWD to be its own worktree root, and refuses
# otherwise -- it is guarding against silently importing the sibling worktree's
# cosmos_framework (see the memory note on the shared venv).
[[ "$(pwd)" == "$WORKDIR" ]] || { echo "ERROR: expected CWD $WORKDIR" >&2; exit 1; }

# What the run will actually write.  Read from the recipe rather than repeated
# here, so this cannot drift from the toml.
read -r RECIPE_MAX_ITER SAVE_ITER < <(python3 - "$WORKDIR/examples/toml/sft_config/action_policy_fk_singlerighthand_edge.toml" <<'PY'
import tomllib, sys
d = tomllib.load(open(sys.argv[1], "rb"))
print(d["trainer"]["max_iter"], d["checkpoint"]["save_iter"])
PY
)
# Optional warm start: RESUME_FROM=<run name or path> continues from that run's
# latest checkpoint.  Two kinds exist here and they need different handling -- see
# the block below, which picks by whether the checkpoint carries ``optim/``.
if [[ -n "${RESUME_FROM:-}" ]]; then
    WARM="$RESUME_FROM"
    [[ "$WARM" == */* ]] || WARM="$RUNS_ROOT/$WARM"
    [[ -f "$WARM/$REL/checkpoints/latest_checkpoint.txt" ]] || {
        echo "ERROR: RESUME_FROM=$WARM has no $REL/checkpoints/latest_checkpoint.txt" >&2; exit 1; }
    WARM_CKPT="$(cat "$WARM/$REL/checkpoints/latest_checkpoint.txt")"
    WARM_DIR="$WARM/$REL/checkpoints/$WARM_CKPT"
    [[ -d "$WARM_DIR" ]] || { echo "ERROR: $WARM_DIR does not exist" >&2; exit 1; }
    [[ -d "$WARM_DIR/model" ]] || { echo "ERROR: $WARM_DIR has no model/ subdir" >&2; exit 1; }
    # Which kind of checkpoint is this?  A complete one (model + optim + scheduler +
    # trainer) resumes normally: symlink it and let latest_checkpoint.txt drive the
    # loader, which keeps the iteration counter and the optimizer state.  A weights-only
    # one cannot go through that path -- the loader asserts on a missing ``optim``
    # ("metadata is None") -- so it goes through load_path with load_training_state=False,
    # which restarts the counter at 0.  Run -5-2's checkpoints are weights-only and run
    # -6's are complete, so both cases are real and this has to tell them apart.
    if [[ -d "$WARM_DIR/optim" ]]; then
        WARM_MODE=full
        START_ITER="$((10#${WARM_CKPT#iter_}))"
        # Not under DRY_RUN: creating the output root here would make the next real
        # launch see it as taken and silently pick a suffixed name instead.
        if [[ -z "${DRY_RUN:-}" ]]; then
            mkdir -p "$OUTPUT_ROOT/$REL/checkpoints"
            ln -sfn "$WARM_DIR" "$OUTPUT_ROOT/$REL/checkpoints/$WARM_CKPT"
            printf '%s' "$WARM_CKPT" > "$OUTPUT_ROOT/$REL/checkpoints/latest_checkpoint.txt"
        fi
        echo "  warm start  : FULL resume at $WARM_CKPT from $(basename "$WARM")  (optimizer + iteration kept)"
    else
        WARM_MODE=weights
        WARM_OVERRIDES="checkpoint.load_path=$WARM_DIR checkpoint.load_training_state=False"
        START_ITER=0
        echo "  warm start  : WEIGHTS ONLY at $WARM_CKPT from $(basename "$WARM")  (no optim/ in it; iteration restarts at 0)"
    fi
    echo
fi

if [[ -z "${MAX_ITER:-}" ]]; then
    MAX_ITER="$RECIPE_MAX_ITER"
    # A full resume past the recipe's max_iter would end on arrival; give it 2000 more.
    if [[ "${WARM_MODE:-}" == full && "$START_ITER" -ge "$RECIPE_MAX_ITER" ]]; then
        MAX_ITER=$(( START_ITER + 2000 ))
    fi
fi
# Only the checkpoints this run will actually write: a full resume starts at the
# checkpoint's iteration, a weights-only warm start restarts at 0, so START_ITER is
# what the block above resolved.
START_ITER="${START_ITER:-0}"
CHECKPOINTS=$(( (MAX_ITER - START_ITER) / SAVE_ITER + 1 ))

AVAIL_GB=$(df -BG --output=avail "$RUNS_ROOT" 2>/dev/null | tail -1 | tr -dc '0-9')
echo "=== FK retrain ==="
echo "  output root : $OUTPUT_ROOT"
NEED_GB=$(( CHECKPOINTS * 30 + 50 ))   # checkpoints plus eval/wandb scratch
(( NEED_GB < MIN_FREE_GB )) && NEED_GB="$MIN_FREE_GB"
echo "  checkpoints : $CHECKPOINTS x ~30 GB = ~$(( CHECKPOINTS * 30 )) GB (iter $START_ITER..$MAX_ITER / save_iter=$SAVE_ITER)"
echo "  runs root   : ${AVAIL_GB:-?} GB free, need >= ${NEED_GB} GB"
echo

if [[ -n "${AVAIL_GB:-}" && "$AVAIL_GB" -lt "$NEED_GB" ]]; then
    echo "ERROR: only ${AVAIL_GB} GB free on $RUNS_ROOT (shared), refuse to start a run that needs ~${NEED_GB} GB." >&2
    echo "       Free space first -- pruning old checkpoints is the usual way:" >&2
    echo "         ls -d <run>/$REL/checkpoints/iter_* | sort | head -n -4 | xargs -r rm -rf" >&2
    exit 1
fi

# The eval set is pinned by the new run's own fk_eval/fixed_cases.json, not by
# the config -- see tools/seed_fk_eval_manifest.py.  It has to happen *here*,
# after the suffix loop above has decided the output root's final name: seeding
# earlier would create the directory, the loop would then see it as taken, and
# the run would land in a suffixed root next to its own manifest.
#   SEED_EVAL_FROM=<run name or path>   copy that run's manifest, keeping its
#                                       train_00/train_01/val_01 byte-identical
#   REPLACE_VAL00_INDEX=<int>           the val window to put in val_00 instead
if [[ -n "${SEED_EVAL_FROM:-}" ]]; then
    SEED_FROM="$SEED_EVAL_FROM"
    [[ "$SEED_FROM" == */* ]] || SEED_FROM="$RUNS_ROOT/$SEED_FROM"
    SEED_ARGS=(--base "$SEED_FROM" --out "$OUTPUT_ROOT" --replace-val00-index "${REPLACE_VAL00_INDEX:?set REPLACE_VAL00_INDEX with SEED_EVAL_FROM}")
    [[ -n "${DRY_RUN:-}" ]] && SEED_ARGS+=(--dry-run)
    echo "=== seeding fk_eval manifest from $(basename "$SEED_FROM") ==="
    # The venv is the sibling worktree's; the tool needs torch for the split
    # permutation and numpy for the labels, so it cannot use the system python3.
    PYTHONPATH="$WORKDIR" "$PYTHON_BIN" "$WORKDIR/tools/seed_fk_eval_manifest.py" "${SEED_ARGS[@]}"
    echo
fi

if [[ -n "${DRY_RUN:-}" ]]; then
    echo "(DRY_RUN: not launching)"
    exit 0
fi

# A stable short pointer to the run just started.  The real path is ~100 characters
# and pasting it wraps the terminal, which breaks the command (bash then reads the
# tail of the path as a command).  With this, following the log is
#     tail -f "$(cat /tmp/fk_last_run)"/logs/*.log
# and pruning old checkpoints is equally short -- see the message below.
printf '%s\n' "$OUTPUT_ROOT" > /tmp/fk_last_run

# The A/B diagnostic resumes the PREVIOUS run's checkpoint and needs the GPU too,
# so running it and this at once would leave one of them waiting.
echo "note: this takes the GPU. If the sampler probe still needs to run, do it first:"
echo "      bash tools/run_fk_sampler_probe.sh"
echo
# There is no retention policy in the trainer, so this is the only thing keeping
# the filesystem from filling again at ~step 7000.  Printed unconditionally
# because it has to be run *during* the run, not after it fails.
echo "follow the log with:"
echo "      tail -f \"\$(cat /tmp/fk_last_run)\"/logs/*.log"
echo "prune old checkpoints (keeps the newest 4) whenever the disk gets tight:"
echo "      ls -d \"\$(cat /tmp/fk_last_run)\"/cosmos3_action/action_sft/action_policy_fk_singlerighthand_edge/checkpoints/iter_* \\"
echo "        | sort | head -n -4 | xargs -r rm -rf"
echo

OVERRIDES="trainer.max_iter=$MAX_ITER"
[[ -n "${WARM_OVERRIDES:-}" ]] && OVERRIDES="$OVERRIDES $WARM_OVERRIDES"
export EXTRA_TAIL_OVERRIDES="${EXTRA_TAIL_OVERRIDES:-}$OVERRIDES"
echo "  max_iter    : $MAX_ITER (recipe says $RECIPE_MAX_ITER)"
[[ -n "${WARM_OVERRIDES:-}" ]] && echo "  warm ckpt   : $(echo "$WARM_OVERRIDES" | cut -d' ' -f1 | cut -d= -f2)"
echo

OUTPUT_ROOT="$OUTPUT_ROOT" exec bash examples/launch_sft_action_policy_fk_singlerighthand_edge.sh
