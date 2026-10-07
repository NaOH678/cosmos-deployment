#!/usr/bin/env bash
# Animate the most-moving clusters of many windows, with the same motion-fraction
# selection training uses.
#
# Two things this script exists to get right:
#
#   * Cluster IDs are only meaningful inside the window that produced them -- PTv3
#     derives them from that window's point cloud -- so nothing here passes a fixed
#     ID list.  `--top-motion-clusters` ranks each window's own clusters instead.
#
#   * `--select-motion-fraction` is applied in `prepare_window`, i.e. BEFORE PTv3, so
#     the encoder, the clusters and the render all see the same reduced point set the
#     training data path would produce.  Different fractions therefore need their own
#     encoding directory, which is why the fraction is part of the path.
#
# `--start-frame` is a SOURCE frame index, not a window index, and episodes differ in
# length, so a window past an episode's end is skipped rather than aborting the sweep.
#
# Usage:
#   bash examples/launch_pointflow_motion_scan.sh
#   SELECT_MOTION_FRACTION=0.10 WINDOWS="0" bash examples/launch_pointflow_motion_scan.sh
#   EPISODES="episode_0013_20260731_133649" bash examples/launch_pointflow_motion_scan.sh

set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

B=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian

EPISODES_FILE="${EPISODES_FILE:-$REPO_ROOT/examples/pointflow_sandwich_10_episodes.txt}"
if [[ -z "${EPISODES:-}" ]]; then
    EPISODES="$(grep -v '^[[:space:]]*#' "$EPISODES_FILE" | grep -v '^[[:space:]]*$' | tr '\n' ' ')"
fi
# Windows are chosen as FRACTIONS of each episode's own legal range, because episodes
# differ in length (window starts run 0..1127 up to 0..1686 here) and a fixed list
# would cover a different part of each.  Set WINDOWS to force explicit source frames.
: "${WINDOW_FRACTIONS:=0 0.125 0.25 0.375 0.5 0.625 0.75 0.875 1}"
: "${WINDOWS:=}"
: "${SELECT_MOTION_FRACTION:=0.05}"
: "${STAGE:=3}"
: "${TOP_CLUSTERS:=8}"
: "${MIN_MEMBERS:=8}"
: "${DEVICE:=cuda}"
: "${DATA_ROOT:=$B/datasets/sandwich_dense_fullseq_10_0298_20260908/outputs}"
: "${CHECKPOINT:=$B/checkpoints/ptv3/sonata_small.pth}"
: "${OUTPUT_ROOT:=$REPO_ROOT/pointflow_outputs/motion_scan}"
: "${MAX_DISPLAY_POINTS:=2048}"
: "${TRAIL_STEPS:=5}"
: "${PYTHON_BIN:=$REPO_ROOT/.venv/bin/python}"

[[ -f "$CHECKPOINT" ]] || { echo "ERROR: missing Sonata checkpoint: $CHECKPOINT" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: no python at $PYTHON_BIN" >&2; exit 1; }
TAG="f${SELECT_MOTION_FRACTION}"

mkdir -p "$OUTPUT_ROOT/${TAG}_enc" "$OUTPUT_ROOT/${TAG}_motion"
echo "episodes  $(wc -w <<<"$EPISODES")  ($EPISODES_FILE)"
echo "windows   ${WINDOWS:-$WINDOW_FRACTIONS}   (source frames, or fractions of each episode's own range)"
echo "selection fastest-moving $SELECT_MOTION_FRACTION of anchor points"
echo "clusters  top $TOP_CLUSTERS with >= $MIN_MEMBERS members, stage $STAGE, $DEVICE"
echo "output    $OUTPUT_ROOT"
echo

RENDERED=()
for EPISODE in $EPISODES; do
    EPISODE_DIR="$DATA_ROOT/$EPISODE"
    if [[ ! -d "$EPISODE_DIR" ]]; then
        echo "=== $EPISODE -- SKIPPED (no such directory)"; echo
        continue
    fi
    LAST_START="$(
        "$PYTHON_BIN" - "$EPISODE_DIR" <<'PY'
import sys

import numpy as np

timestamps = np.load(f"{sys.argv[1]}/timestamps_sec.npy")
fits = timestamps + 32.0 / 15.0 <= timestamps[-1] + 1e-6
print(int(np.flatnonzero(fits)[-1]) if fits.any() else -1)
PY
    )"
    if [[ -n "$WINDOWS" ]]; then
        EP_WINDOWS="$WINDOWS"
    else
        EP_WINDOWS="$(
            "$PYTHON_BIN" - "$LAST_START" "$WINDOW_FRACTIONS" <<'PY'
import sys

last, fractions = int(sys.argv[1]), [float(v) for v in sys.argv[2].split()]
seen, out = set(), []
for fraction in fractions:
    window = max(0, min(last, int(round(fraction * last))))
    if window not in seen:
        seen.add(window)
        out.append(window)
print(" ".join(str(w) for w in out))
PY
        )"
    fi
    echo "=== $EPISODE  (window starts 0..$LAST_START; $(wc -w <<<"$EP_WINDOWS") windows)"
    for W in $EP_WINDOWS; do
        [[ "$W" =~ ^[0-9]+$ ]] || { echo "  window '$W' is not an integer source frame index" >&2; exit 1; }
        if [[ "$LAST_START" -lt 0 || "$W" -gt "$LAST_START" ]]; then
            echo "  w$W -- SKIPPED (past this episode's last legal start $LAST_START)"
            continue
        fi
        ENC_DIR="$OUTPUT_ROOT/${TAG}_enc/${EPISODE}_w$W/$EPISODE"
        MOTION_DIR="$OUTPUT_ROOT/${TAG}_motion/${EPISODE}_w$W"

        if [[ -f "$ENC_DIR/encoding_enc$STAGE.npz" ]]; then
            echo "  w$W [1/2] encoding present, skipping"
        else
            echo "  w$W [1/2] encoding (PTv3 forward)"
            "$PYTHON_BIN" -m cosmos_framework.scripts.validate_pointflow_sonata \
                --data-root "$DATA_ROOT" \
                --episodes "$EPISODE" \
                --checkpoint "$CHECKPOINT" \
                --output "$OUTPUT_ROOT/${TAG}_enc/${EPISODE}_w$W" \
                --device "$DEVICE" --start-frame "$W" --stages "$STAGE" \
                --select-motion-fraction "$SELECT_MOTION_FRACTION" \
                >"$OUTPUT_ROOT/${TAG}_enc/${EPISODE}_w$W.log" 2>&1 \
                || {
                    if grep -q "Fewer than 3 valid observed anchor points" "$OUTPUT_ROOT/${TAG}_enc/${EPISODE}_w$W.log"; then
                        # Some episodes end in frames whose point cloud is empty; that is a
                        # property of the data, not a failure of the sweep.
                        echo "    skipped (no valid anchor points at this frame)"
                    else
                        echo "    FAILED -- see $OUTPUT_ROOT/${TAG}_enc/${EPISODE}_w$W.log" >&2
                        tail -5 "$OUTPUT_ROOT/${TAG}_enc/${EPISODE}_w$W.log" >&2
                    fi
                    continue
                }
            # `--device cpu` exits 0 but writes no encoding, because PTv3 cannot run there.
            [[ -f "$ENC_DIR/encoding_enc$STAGE.npz" ]] || {
                echo "    FAILED: no encoding written. PTv3 runs on spconv and needs a GPU;" >&2
                echo "    --device cpu only produces window.npz and the geometry preview." >&2
                continue
            }
        fi

        "$PYTHON_BIN" -m cosmos_framework.scripts.visualize_pointflow_motion \
            --episode "$EPISODE_DIR" \
            --encoding-dir "$ENC_DIR" \
            --stage "$STAGE" \
            --top-motion-clusters "$TOP_CLUSTERS" --min-members "$MIN_MEMBERS" \
            --max-display-points "$MAX_DISPLAY_POINTS" --trail-steps "$TRAIL_STEPS" \
            --output "$MOTION_DIR" >"$OUTPUT_ROOT/${TAG}_motion/${EPISODE}_w$W.log" 2>&1 \
            || { echo "    render FAILED -- see $OUTPUT_ROOT/${TAG}_motion/${EPISODE}_w$W.log" >&2; tail -5 "$OUTPUT_ROOT/${TAG}_motion/${EPISODE}_w$W.log" >&2; continue; }

        echo "  w$W [2/2] $(grep -h '^motion-ranked clusters:' "$OUTPUT_ROOT/${TAG}_motion/${EPISODE}_w$W.log" | sed 's/^motion-ranked clusters: //')"
        RENDERED+=("$OUTPUT_ROOT/${TAG}_motion/${EPISODE}_w$W.log")
    done
    echo
done

echo "================================================================"
echo "per-window summary  (fraction $SELECT_MOTION_FRACTION)"
echo "================================================================"
"$PYTHON_BIN" - "${RENDERED[@]}" <<'PY'
import json
import re
import sys
from pathlib import Path

print(f"  {'window':<44}{'clusters':>9}{'members':>9}   mean motion (mm)")
print("  " + "-" * 92)
totals = []
for log in sys.argv[1:]:
    log = Path(log)
    text = log.read_text()
    ranked = re.search(r"^motion-ranked clusters: (.*)$", text, re.M)
    record = next(iter(sorted((log.parent / log.stem).glob("*.json"))), None)
    members = json.loads(record.read_text())["cluster_members"] if record else 0
    if ranked:
        motions = [int(m) for m in re.findall(r"(\d+)mm", ranked.group(1))]
        mean_motion = sum(motions) / len(motions)
        totals.append((log.name, len(motions), members, mean_motion))
        print(f"  {log.stem:<44}{len(motions):>9}{members:>9}{mean_motion:>19.1f}")
    else:
        print(f"  {log.stem:<44}{'--':>9}{'--':>9}{'--':>19}")
if totals:
    print("  " + "-" * 92)
    print(f"  {'mean over ' + str(len(totals)) + ' windows':<44}{'':>9}{'':>9}"
          f"{sum(t[3] for t in totals) / len(totals):>19.1f}")
PY

echo
echo "gifs / previews:"
find "$OUTPUT_ROOT/${TAG}_motion" -name "*_preview.jpg" 2>/dev/null | sort | sed 's/^/  /'
