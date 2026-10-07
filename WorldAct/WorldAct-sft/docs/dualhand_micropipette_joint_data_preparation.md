# Dual-Hand Micropipette Joint Data Preparation

This document only covers deterministic data preparation. It does not add or
change a Cosmos training recipe, dataset loader, model, or deployment adapter.

## Data contract

- Raw root: `/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/raw_data/dualhand_micropipette`
- Output root: `/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/datasets/dual_cosmos/dualhand-micropipette-joint-cosmos`
- Episodes: 98
- Frames: 146,272 at 30 fps
- Cameras: `head`, `left_wrist`, `right_wrist`
- State/action width: 54
- Layout: left arm joint 7 + left hand 20 + right arm joint 7 + right hand 20
- State: measured `/observations/qpos`, all radians
- Arm action: `/diagnostics/arm_joint_command`, all radians
- Hand action: left/right hand fields from `action`, converted from degrees to radians

The three-view frame layout follows the existing Cosmos multi-view convention:
the head camera is on top, the left wrist is at bottom-left, and the right wrist
is at bottom-right. Camera aspect ratios are preserved before the final 480p
resize. Padding remains deferred to the Cosmos action transform pipeline.

## 1. Numeric conversion

LMDB mmap is unsupported on GPFS, so each episode LMDB is copied to a temporary
local directory before reading. Use the system Python that already contains
`python-lmdb`:

```bash
cd /mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/WorldAct-cosmos3-edge-droid-sft

/usr/bin/python3 tools/prepare_dualhand_joint_raw.py \
  --raw-root /mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/raw_data/dualhand_micropipette \
  --output-root /mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/datasets/dual_cosmos/dualhand-micropipette-joint-cosmos \
  --expected-episodes 98 \
  --overwrite
```

This writes `manifest.json` and 98 `episodes/<source-episode>.npz` files. Each
NPZ contains float32 `state` and `action` arrays with shape `[num_frames, 54]`.

## 2. Three-view video cache

Activate the Cosmos environment and expose the FFmpeg/NPP libraries used by
TorchCodec:

```bash
cd /mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/WorldAct-cosmos3-edge-droid-sft
source .venv/bin/activate

PYTHON_BASE_PREFIX="$(python -c 'import sys; print(sys.base_prefix)')"
NPP_LIB="$(python -c 'import nvidia.npp, pathlib; print(pathlib.Path(nvidia.npp.__path__[0]) / "lib")')"
export PATH="$PATH:$PYTHON_BASE_PREFIX/bin"
export LD_LIBRARY_PATH="$PYTHON_BASE_PREFIX/lib:$NPP_LIB:${LD_LIBRARY_PATH:-}"

python tools/prepare_dualhand_joint_video_cache.py \
  --raw-root /mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/raw_data/dualhand_micropipette \
  --cache-root /mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/datasets/dual_cosmos/dualhand-micropipette-joint-cosmos \
  --resolution 480 \
  --workers 8 \
  --torch-threads 2
```

The command is resumable: complete `.npy` episode caches are validated from
their headers and exact byte sizes, then reused unless `--overwrite` is passed.
Frames are written sequentially because this GPFS mount does not support
writable mmap. `video_manifest.json` is written only after every episode
succeeds.

## 3. Verify all output

```bash
/usr/bin/python3 tools/verify_dualhand_joint_cosmos.py \
  --dataset-root /mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/datasets/dual_cosmos/dualhand-micropipette-joint-cosmos \
  --expected-episodes 98 \
  --expected-total-frames 146272 \
  --require-video-cache
```

Expected output ends with a `[PASS]` line. This verifies all 98 numeric files
and all 98 video-cache headers and exact byte sizes without importing the
training pipeline or requiring mmap support from the current mount.
