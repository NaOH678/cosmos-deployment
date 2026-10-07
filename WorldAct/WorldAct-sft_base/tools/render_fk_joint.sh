#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Render the joint-arm eval: REAL vs GENERATED panels, GT + prediction skeletons.
#
#     bash tools/render_fk_joint.sh              # newest joint eval, both val cases
#     bash tools/render_fk_joint.sh val_01       # one case
#     ARM=joint bash tools/render_fk_joint.sh    # the two-segment arm instead
#     ROOT=<eval root> bash tools/render_fk_joint.sh   # a specific eval, not the newest
#
# Why a script and not the one-liner: the case directory is ~150 characters, and
# pasting it into a terminal wraps.  bash then reads the tail of the path as a
# command -- the run that hit this produced
#     D=/mnt/.../fk_
#     eval/step_0010000
#     -bash: eval/step_0010000: No such file or directory
# and $D ended up as ".../fk_", so the renderer looked for prediction.npz under a
# directory that never existed.  Nothing here takes a path argument for that reason.
#
# Needs a GPU: the generated panel is a VAE decode (``decode_head_view`` runs on
# cuda), so this cannot be run from the dev node.
#
# Read it as: the LEFT panel is the real recording, the RIGHT is the video the model
# generated.  Both carry the SAME two skeletons -- green = recorded GT, red =
# predicted -- so the panels differ only in the background.  Check the GREEN on the
# LEFT first: if GT is not on the hand, the projection is wrong and nothing else in
# the picture means anything.

set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKDIR"

RUNS_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/runs/cosmos"
# Renders go under shichaojian, not /tmp: they are the artefact you come back to
# look at, and /tmp is not where anything worth keeping lives.  Kept out of the run
# directory too, so pruning checkpoints does not take the videos with them.
OUT_ROOT="${OUT_ROOT:-/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/renders/fk_joint}"
PYTHON_BIN="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/WorldAct-cosmos3-edge-droid-sft/.venv/bin/python"
WANTED="${1:-}"

# Which of the two joint layouts to render.  ``joint_action`` is the deployed
# configuration (the future action denoised too) and the default; ``joint`` is the
# two-segment arm whose numbers are already recorded.  They are different
# subdirectories of the same eval, so this is a lookup key, not a preference.
ARM="${ARM:-joint_action}"
case "$ARM" in
    joint|joint_action) ;;
    *) echo "ERROR: ARM must be 'joint' or 'joint_action', got '$ARM'" >&2; exit 1 ;;
esac

# Newest joint eval that actually produced something.  The first attempt at this
# crashed before writing any case, and picking the directory blindly would report
# "no prediction.npz" instead of finding the run that worked.
#
# ``ROOT`` pins a specific eval, which is what a six-checkpoint sweep needs: the
# newest-first scan can only ever reach one root, so rendering the rest means
# naming them.
ROOT="${ROOT:-}"
if [[ -z "$ROOT" ]]; then
    while read -r candidate; do
        [[ -n "$candidate" ]] || continue
        if find "$candidate" -name prediction.npz -path "*/$ARM/*" -print -quit 2>/dev/null | grep -q .; then
            ROOT="$candidate"
            break
        fi
    done < <(ls -dt "$RUNS_ROOT"/fk-joint-eval* 2>/dev/null || true)
fi

[[ -n "$ROOT" && -d "$ROOT" ]] || {
    echo "ERROR: no fk-joint-eval* with a $ARM prediction.npz under $RUNS_ROOT${ROOT:+, and ROOT=$ROOT is not a directory}" >&2
    exit 1; }
echo "=== rendering $(basename "$ROOT")  arm=$ARM ==="

# The two held-out cases by default.  train_00/train_01 are also there and are
# worth a look later -- the model may generate a better video for a window it
# memorised, which would separate "the video branch is weak" from "it is weak on
# unseen motion" -- but each case is a VAE decode, and val is the question.
CASES=()
if [[ -n "$WANTED" ]]; then
    case "$WANTED" in
        all) while read -r name; do [[ -n "$name" ]] && CASES+=("$name"); done \
                 < <(find "$ROOT" -path "*/$ARM/prediction.npz" -printf '%h\n' 2>/dev/null | xargs -r -n1 dirname | xargs -r -n1 basename | sort -u) ;;
        *)   CASES=("$WANTED") ;;
    esac
else
    CASES=(val_00 val_01)
fi

STATUS=0
for case in "${CASES[@]}"; do
    NPZ="$(find "$ROOT" -path "*/$case/$ARM/prediction.npz" -print -quit 2>/dev/null || true)"
    if [[ -z "$NPZ" ]]; then
        echo "  $case: no $ARM prediction.npz -- skipped" >&2
        STATUS=1
        continue
    fi
    # The arm is a directory level of its own: both layouts render the same case
    # names, and one would otherwise write over the other's panels.
    echo "  $case -> $OUT_ROOT/$ARM/$case"
    PYTHONPATH=. "$PYTHON_BIN" tools/render_fk_projection.py \
        --case-dir "$(dirname "$NPZ")" --out "$OUT_ROOT/$ARM/$case"
done

# The multi-window rollout, when the eval ran one (FK_ROLLOUT=1).  This is the LONG
# artefact -- consecutive windows tiled over the middle seconds, one generated video
# each, stitched with the boundary frame kept once.  A separate tool because the npz
# carries the per-window latents, not a single case's.
#
# ``find`` over the whole eval root picks up whichever rollout roots exist; the
# render mirrors the source root's name so the two arms land in separate trees
# rather than the later one replacing the earlier.
while read -r npz; do
    [[ -n "$npz" ]] || continue
    name="$(basename "$(dirname "$npz")")"
    # .../<rollout_root>/step_X/<episode>/rollout.npz -> three levels up is the root,
    # whose name is the arm (fk_rollout / fk_rollout_joint_action).
    rollout_root="$(basename "$(dirname "$(dirname "$(dirname "$npz")")")")"
    echo "  rollout $name -> $(dirname "$OUT_ROOT")/$rollout_root/$name"
    PYTHONPATH=. "$PYTHON_BIN" tools/render_fk_rollout.py \
        --rollout "$npz" --out "$(dirname "$OUT_ROOT")/$rollout_root/$name" || STATUS=1
done < <(find "$ROOT" -name rollout.npz 2>/dev/null | sort)

echo
echo "CHECK THE GREEN SKELETON FIRST: if GT is not on the hand in the REAL (left)"
echo "panel, the projection is wrong and the generated panel means nothing."
echo "A snap at each window boundary is expected -- every window restarts from its"
echo "own GT anchor."
exit "$STATUS"
