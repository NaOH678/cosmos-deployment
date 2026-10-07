#!/usr/bin/env bash
# Camera-extrinsic calibration, one episode, bounded resources.
# Usage:  bash tools/run_calib.sh [episode_name]
# Everything is hardcoded here so no shell variables need to survive a paste.

set -u

EPISODE="${1:-episode_0013_20260731_133649}"
REPO=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/WorldAct-cosmos3-edge-droid-sft_mano
VENV=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/WorldAct-cosmos3-edge-droid-sft/.venv/bin/python
OUT=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/fk_calib
LOG="$OUT/calib_${EPISODE##*_}.log"

cd "$REPO" || { echo "cannot cd to $REPO"; exit 1; }
mkdir -p "$OUT"

# 4 GB hard cap: if it runs away it dies on its own instead of taking the box down.
ulimit -v 4000000
# single-threaded — the earlier runaway opened all 7 cores via workers=-1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH=''

echo "episode : $EPISODE"
echo "out     : $OUT"
echo "log     : $LOG"
echo "started : $(date '+%H:%M:%S')"
echo

MJS=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/mjlib
if [ -d "$MJS/mujoco" ]; then
    echo "mujoco  : shared storage ($MJS)"
elif [ -d /tmp/mjlib/mujoco ]; then
    echo "mujoco  : /tmp/mjlib  (local only — this node)"
else
    echo "!! mujoco not found on this node."
    echo "   Checked: $MJS   and   /tmp/mjlib"
    exit 1
fi

"$VENV" tools/calibrate_extrinsic_v2.py \
    --episode "$EPISODE" \
    --out "$OUT" \
    --n-frames 12 \
    > "$LOG" 2>&1

rc=$?
echo "finished: $(date '+%H:%M:%S')   exit=$rc"
echo
echo "=================== tail of log ==================="
tail -30 "$LOG"
