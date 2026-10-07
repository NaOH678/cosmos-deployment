# Single-Right-Hand Edge-DROID Post-Training

This recipe fine-tunes the local Cosmos3-Edge-Policy-DROID checkpoint on the
single-right-hand sandwich episodes. It runs directly with `torchrun`; it does
not use Slurm, Apptainer, or network downloads.

## Fixed data contract

- Source data: 30 fps raw episodes with `head.mp4`, `right_wrist.mp4`, and LMDB.
- Training cadence: 15 fps.
- Sliding stride: one source frame (1/30 second), matching the official
  `sample_stride=1`; observations inside each window are sampled every two
  source frames to produce 15 fps.
- Each sample: 33 observation frames and 32 target action frames.
- Full dataset: 101 episodes, 106,731 source frames, and 100,267 valid windows
  before the deterministic train/validation episode split.
- State/action width: 27.
- State: right EEF xyz + quaternion xyzw + 20 current hand joints in radians.
- Action: right EEF xyz + quaternion xyzw + 20 target hand joints in radians.
- Model action shape before padding: `[33, 27]` when `use_state=true`: one
  initial state followed by 32 target actions.
- Deployment must convert the final 20 output channels from radians back to
  degrees before sending them to the existing hand controller.

The old LeRobot v2 dataset is not used by this recipe because its
`observation.state[:7]` contains arm joint positions while its action contains
EEF pose. The raw data provides the correct current EEF state.

## Paths

```bash
REPO=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/WorldAct-cosmos3-edge-droid-sft
RAW=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/raw_data/singlerighthand_sandwich_100
CACHE=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/datasets/singlerighthand-sandwich-100-cosmos-cache
HF_MODEL=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/models/cosmos3-edge-droid
DCP=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/models/cosmos3-edge-droid-dcp
```

## 1. Activate the reused environment

```bash
cd "$REPO"
source .venv/bin/activate
```

The environment uses Python 3.13 and PyTorch 2.10.0+cu128. The `.venv` symlink
is intentionally shared with the previous Cosmos worktree, but its editable
`cosmos-framework` package points at this worktree.

## 2. Prepare the small numeric cache

This has already been generated. Rebuild it only after the raw data changes:

```bash
/usr/bin/python3 tools/prepare_singlerighthand_raw.py \
  --raw-root "$RAW" \
  --output-root "$CACHE" \
  --overwrite
```

The cache is about 23 MB and contains only state/action NPZ files plus a
manifest. Videos remain in the raw episode directories.

## 3. Prepare the deterministic video frame cache

With online image augmentation disabled, dual-camera decode, view composition,
and aspect-preserving resize are deterministic. Cache the resized content once
per source frame instead of repeating them for every 33-frame sliding window:

```bash
PYTHON_BASE_PREFIX="$(python -c 'import sys; print(sys.base_prefix)')"
NPP_LIB="$(python -c 'import nvidia.npp, pathlib; print(pathlib.Path(nvidia.npp.__path__[0]) / "lib")')"
export PATH="$PATH:$PYTHON_BASE_PREFIX/bin"
export LD_LIBRARY_PATH="$PYTHON_BASE_PREFIX/lib:$NPP_LIB:${LD_LIBRARY_PATH:-}"

python tools/prepare_singlerighthand_video_cache.py \
  --raw-root "$RAW" \
  --cache-root "$CACHE" \
  --resolution 480 \
  --workers 8 \
  --torch-threads 2
```

The uncompressed `uint8` cache is approximately 116 GiB. Each episode is one
read-only `.npy` memmap, so ranks and dataloader workers share the Linux page
cache instead of each loading a private copy. The generator writes
`video_manifest.json` only after every episode is complete. The training
launcher automatically enables the cache when that manifest exists.

## 4. Convert Edge-DROID to DCP

Run this once in a training shell with enough host RAM. The current CPU control
container has only 15 GiB total RAM and is not suitable for loading the 7.57 GB
checkpoint plus conversion overhead.

```bash
python -m cosmos_framework.scripts.convert_model_to_dcp \
  --checkpoint-path "$HF_MODEL" \
  --output-path "$DCP"
```

Completion check:

```bash
test -f "$DCP/model/.metadata"
```

## 5. Validate the resolved training config

```bash
export SINGLERIGHTHAND_RAW_ROOT="$RAW"
export SINGLERIGHTHAND_CACHE_ROOT="$CACHE"
export EDGE_DROID_MODEL_PATH="$HF_MODEL"
export WAN_VAE_PATH="$HF_MODEL/vae/Wan2.2_VAE.pth"
export BASE_CHECKPOINT_PATH="$DCP"
export IMAGINAIRE_OUTPUT_ROOT=/tmp/cosmos3-edge-droid-sft-dryrun

python -m cosmos_framework.scripts.train \
  --sft-toml=examples/toml/sft_config/action_policy_singlerighthand_edge.toml \
  --dryrun
```

## 6. Run a two-step GPU smoke test

Set `NPROC_PER_NODE` to the number of visible GPUs. The example below assumes
eight GPUs and reduces CPU workers and batch size for the smoke test.

```bash
export NPROC_PER_NODE=8
export EXTRA_TAIL_OVERRIDES="trainer.max_iter=2 checkpoint.save_iter=2 \
dataloader_train.max_samples_per_batch=2 \
dataloader_train.dataloader.batch_size=1 \
dataloader_train.dataloader.num_workers=2"

bash examples/launch_sft_action_policy_singlerighthand_edge.sh
```

## 7. Start the full run

```bash
unset EXTRA_TAIL_OVERRIDES
export NPROC_PER_NODE=8
bash examples/launch_sft_action_policy_singlerighthand_edge.sh
```

Outputs default to:

```text
/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/runs/cosmos/singlerighthand-edge-droid
```

The current eight-GPU configuration uses `lr=2e-5`, 32 samples per rank,
gradient accumulation 1, and an effective global batch of 256. CPU and GPU
image augmentation are disabled by default for the main training run. Keep the
official temporal data contract fixed when tuning throughput settings.

### GPU video augmentation

The single-right-hand launcher disables online GPU augmentation by default with
`COSMOS_GPU_VIDEO_AUGMENTATION=false`. Set it to `true` only for a controlled
A/B run. When enabled, it applies a temporally consistent 95% random
crop/rescale and color jitter after each batch reaches its rank's GPU, before
video normalization and Wan VAE encoding.
