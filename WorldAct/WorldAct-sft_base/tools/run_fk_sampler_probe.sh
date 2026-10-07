#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Ask, with numbers, why a 4-step FK sample lands near the noise.
#
# Resumes the trained run at its last checkpoint and runs the sampler diagnostic
# once, at the resume iteration, then exits.  It does NOT train: max_iter is
# pinned to the resume point so the loop breaks on its first iteration, after the
# initial validation has already produced the probe.
#
# Why resume rather than point `checkpoint.load_path` at the iter_* directory:
# `latest_checkpoint.txt` is the path these DCP checkpoints are built for, and it
# is the one already on disk.  The load_path route needs load_training_state and
# the right key set inside the DCP tree; getting that subtly wrong is exactly the
# failure this probe exists to detect, and it would be indistinguishable from a
# genuinely broken sampler.
#
# The checkpoint is SYMLINKED, not copied (30 GB), so the source run is untouched
# and the probe writes only into its own directory.
#
# Read the output at:
#   $PROBE_ROOT/cosmos3_action/action_sft/action_policy_fk_singlerighthand_edge/
#       fk_sampler_probe/step_0008000_val_00.json
# and the same table in the log.

set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKDIR"

REL="cosmos3_action/action_sft/action_policy_fk_singlerighthand_edge"
RUNS_ROOT="${RUNS_ROOT:-/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/runs/cosmos}"
# Default to the newest FK run and its latest checkpoint.  Both used to be
# required arguments, but a command carrying them is long enough that it has now
# failed twice on the way in -- once wrapped mid-path, once with the underscores
# turned into spaces, which bash then read as a command named SOURCE.  Resolving
# them here makes the invocation `bash tools/run_fk_sampler_probe.sh`, which has
# already been typed successfully several times.
#
# Accept a bare run name as well as a path.
if [[ -z "${SOURCE_RUN:-}" ]]; then
    # Newest first; probe output roots are excluded because they are this script's
    # own products, not runs to resume.
    while IFS= read -r candidate; do
        [[ -f "$candidate/$REL/checkpoints/latest_checkpoint.txt" ]] || continue
        SOURCE_RUN="$candidate"
        break
    done < <(ls -dt "$RUNS_ROOT"/fk-* 2>/dev/null | grep -v "/fk-sampler-probe")
    [[ -n "${SOURCE_RUN:-}" ]] || { echo "ERROR: no FK run with a checkpoint under $RUNS_ROOT" >&2; exit 1; }
fi
[[ "$SOURCE_RUN" == */* ]] || SOURCE_RUN="$RUNS_ROOT/$SOURCE_RUN"

if [[ -z "${RESUME_ITER:-}" ]]; then
    LATEST="$(cat "$SOURCE_RUN/$REL/checkpoints/latest_checkpoint.txt")"
    RESUME_ITER="$((10#${LATEST#iter_}))"   # 10# so 000005000 is decimal, not octal
fi
PROBE_ROOT="${PROBE_ROOT:-$RUNS_ROOT/fk-sampler-probe}"

SRC="$SOURCE_RUN/$REL"
CKPT="$(printf 'iter_%09d' "$RESUME_ITER")"

[[ -d "$SRC/checkpoints/$CKPT" ]] || { echo "ERROR: no checkpoint $SRC/checkpoints/$CKPT" >&2; exit 1; }

# A fresh output root each time: reusing one would resume into it instead of
# starting clean, and a stale fixed_cases.json there would decide which windows
# get measured.  Suffix rather than delete -- an earlier probe's report is worth
# keeping, and nothing here should remove a directory the user did not name.
BASE_PROBE_ROOT="$PROBE_ROOT"
suffix=1
while [[ -e "$PROBE_ROOT" ]]; do
    suffix=$((suffix + 1))
    PROBE_ROOT="$BASE_PROBE_ROOT-$suffix"
done
[[ "$PROBE_ROOT" == "$BASE_PROBE_ROOT" ]] || echo "(using $PROBE_ROOT; $BASE_PROBE_ROOT exists)"
# Derived AFTER the suffix loop.  Computing it earlier (as this script first did)
# put the symlinked checkpoint in the *previous* probe's directory while the run
# wrote to the suffixed one -- the run then found no latest_checkpoint.txt, fell
# back to the base checkpoint, and trained from scratch for hours instead of
# running one diagnostic.
DST="$PROBE_ROOT/$REL"
mkdir -p "$DST/checkpoints"
ln -s "$SRC/checkpoints/$CKPT" "$DST/checkpoints/$CKPT"
printf '%s' "$CKPT" > "$DST/checkpoints/latest_checkpoint.txt"

# The eval writes its fixed cases here, so they are selected fresh in this root.
[[ -f "$SRC/fk_eval/fixed_cases.json" ]] && cp "$SRC/fk_eval/fixed_cases.json" "$DST/fixed_cases_source.json" || true

echo "=== FK sampler probe ==="
echo "  source run : $SRC"
echo "  resume at  : $CKPT"
echo "  output     : $PROBE_ROOT"
echo "  max_iter   : $RESUME_ITER (exits after the initial validation; no training)"
echo

# FK_SAMPLER_PROBE registers the callback; without it an ordinary run is unaffected.
# trainer.max_iter pinned to the resume point so the loop breaks immediately.
FK_SAMPLER_PROBE=1 \
OUTPUT_ROOT="$PROBE_ROOT" \
EXTRA_TAIL_OVERRIDES="trainer.max_iter=$RESUME_ITER" \
    bash examples/launch_sft_action_policy_fk_singlerighthand_edge.sh

echo
echo "=== probe report ==="
REPORT="$(ls "$DST"/fk_sampler_probe/*.json 2>/dev/null | head -1 || true)"
if [[ -z "$REPORT" ]]; then
    echo "(no report written; check the log above)"
    exit 1
fi

# The load check, made automatic.  The probe replays the eval for whichever case
# fixed_cases returned first -- that is train_00, since it iterates ("train",
# "val") -- and the same case was already scored during training at this
# iteration.  If the two disagree, the checkpoint did not load as it did in the
# run, and every number below is about a model that never existed.
CASE="$(python3 -c "import json,sys; print(json.load(open('$REPORT'))['case_id'])")"
RECORDED="$SRC/fk_eval/step_$(printf '%07d' "$RESUME_ITER")/$CASE/metrics.json"
python3 - "$REPORT" "$RECORDED" <<'PY'
import json, sys
report, recorded_path = sys.argv[1], sys.argv[2]
r = json.load(open(report))
try:
    was = json.load(open(recorded_path))["all_ade_mm"]
except FileNotFoundError:
    print(f"  (no recorded metrics at {recorded_path}; cannot verify the load)")
    sys.exit(0)
now = r["replay"]["all_ade_mm"]
same = abs(now - was) < 0.5
print()
print(f"  LOAD CHECK  {r['case_id']} @ {r['iteration']}: replay {now:.2f}mm  vs recorded {was:.2f}mm  "
      f"-> {'MATCH' if same else 'MISMATCH'}")
print(f"  unit-noise  {r['unit_noise_mm']:.1f}mm   looks like pure noise: {r['looks_like_pure_noise']}")
if not same:
    print()
    print("  !! The replay does not reproduce the training-time eval.  Either the")
    print("     checkpoint did not load, or the sampler path changed.  Treat every")
    print("     number below as describing a different model, not this diagnosis.")
sys.exit(0 if same else 2)
PY
LOAD_OK=$?

echo
echo "=== velocity field on the true path (the actual answer) ==="
python3 - "$REPORT" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print(f"  case {r['case_id']}, sigma grid, reference |v| = {r['reference_velocity_norm_mm']:.1f}mm")
print(f"  {'sigma':>7} {'|v_model|':>11} {'|v_ref|':>10} {'ratio':>7} {'cosine':>8}")
for row in r["velocity_field_on_true_path"]:
    print(f"  {row['sigma']:>7.3f} {row['model_velocity_norm_mm']:>11.1f} "
          f"{row['reference_velocity_norm_mm']:>10.1f} {row['magnitude_ratio']:>7.3f} "
          f"{row['cosine_to_reference']:>+8.3f}")
PY

echo
echo "=== solver x step count, same case and seed ==="
echo "    (the deployed configuration is unipc/4; each other row moves one factor)"
python3 - "$REPORT" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print(f"  {'sampler':>8} {'steps':>6} {'all_ade':>10} {'final_ade':>10} {'ratio':>7}")
for row in r["sampler_sweep"]:
    print(f"  {row['sampler']:>8} {row['steps']:>6} {row['all_ade_mm']:>10.1f} "
          f"{row['final_ade_mm']:>10.1f} {row['ratio_to_zero']:>7.3f}")
print()
print("  How to read it: if unipc/16 is close to euler/16 and far from unipc/4,")
print("  the step count is the binding constraint.  If unipc/4 is close to")
print("  euler/4, the solver is.  Both improving means both matter.")
PY

echo
echo "=== what the sampler itself did, step by step ==="
python3 - "$REPORT" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
for row in r["sampler_evaluations"]:
    print(f"  sigma {row['sigma']:.4f}  |state| {row['state_norm_mm']:8.1f}mm  |v| {row['velocity_norm_mm']:8.1f}mm")
PY

exit "$LOAD_OK"
