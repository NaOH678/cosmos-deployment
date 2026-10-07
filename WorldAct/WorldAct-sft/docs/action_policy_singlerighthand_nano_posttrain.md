# Single-Right-Hand Nano-Policy-DROID Post-Training

This recipe fine-tunes the 16B `nvidia/Cosmos3-Nano-Policy-DROID` checkpoint on
the same single-right-hand sandwich dataset and deterministic video cache used
by the Edge recipe. It is isolated from the 4B run: experiment name, TOML, DCP
checkpoint, launcher, and output root are all different.

## Model isolation

| Variant | Experiment | Base DCP | Default output root |
| --- | --- | --- | --- |
| Edge / 4B | `action_policy_singlerighthand_edge` | `cosmos3-edge-droid-dcp` | `singlerighthand-edge-droid` |
| Nano / 16B | `action_policy_singlerighthand_nano` | `cosmos3-nano-policy-droid-dcp` | `singlerighthand-nano-policy-droid` |

The 4B training checkpoint cannot be resumed into the 16B architecture. The
Nano run starts at iteration zero from the converted Nano-Policy-DROID weights.
The existing dataset cache is shared read-only.

## 1. Wait for the HF download to finish

```bash
HF_MODEL=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/models/cosmos3-nano-policy-droid

python - <<'PY'
import json
from pathlib import Path

root = Path("/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/models/cosmos3-nano-policy-droid")
indexes = [root / "model.safetensors.index.json", root / "transformer/diffusion_pytorch_model.safetensors.index.json"]
missing = []
for index in indexes:
    data = json.loads(index.read_text())
    missing.extend(str(root / name) for name in sorted(set(data["weight_map"].values())) if not (root / name).is_file())
if missing:
    raise SystemExit("Missing weight files:\n" + "\n".join(missing))
print("Nano-Policy-DROID download is complete")
PY
```

## 2. Convert the 16B checkpoint to a separate DCP

The HF policy snapshot carries a Diffusers-format VAE, but model construction
during conversion requires the original Wan tokenizer `.pth`. Reuse the local
copy already used by the Edge run; this does not modify either model directory.

```bash
cd /mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/WorldAct-cosmos3-edge-droid-sft
source .venv/bin/activate

HF_MODEL=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/models/cosmos3-nano-policy-droid
DCP=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/models/cosmos3-nano-policy-droid-dcp
WAN_VAE=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/models/cosmos3-edge-droid/vae/Wan2.2_VAE.pth

HF_HUB_OFFLINE=1 python -m cosmos_framework.scripts.convert_model_to_dcp \
  --checkpoint-path "$HF_MODEL" \
  --output-path "$DCP" \
  --video-vae-path "$WAN_VAE"

test -f "$DCP/model/.metadata"
```

## 3. Validate the Nano config

```bash
export SINGLERIGHTHAND_RAW_ROOT=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/raw_data/singlerighthand_sandwich_100
export SINGLERIGHTHAND_CACHE_ROOT=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/datasets/singlerighthand-sandwich-100-cosmos-cache
export NANO_DROID_MODEL_PATH="$HF_MODEL"
export WAN_VAE_PATH="$WAN_VAE"
export BASE_CHECKPOINT_PATH="$DCP"
export IMAGINAIRE_OUTPUT_ROOT=/tmp/cosmos3-nano-policy-droid-sft-dryrun

HF_HUB_OFFLINE=1 python -m cosmos_framework.scripts.train \
  --sft-toml=examples/toml/sft_config/action_policy_singlerighthand_nano.toml \
  --dryrun
```

## 4. Launch training

The Nano TOML uses 8-way FSDP sharding, BF16, full activation checkpointing,
no gradient accumulation, CPU/GPU image augmentation off, and W&B offline.

Default 10,000-step run:

```bash
HF_HUB_OFFLINE=1 \
NPROC_PER_NODE=8 \
MASTER_PORT=50140 \
SINGLERIGHTHAND_MODEL_VARIANT=nano \
SINGLERIGHTHAND_USE_VIDEO_CACHE=true \
COSMOS_GPU_VIDEO_AUGMENTATION=false \
bash examples/launch_sft_action_policy_singlerighthand.sh
```

For a 50,000-step run saving every 2,500 steps, the scheduler cycle must be
extended together with `trainer.max_iter`:

```bash
HF_HUB_OFFLINE=1 \
NPROC_PER_NODE=8 \
MASTER_PORT=50140 \
SINGLERIGHTHAND_MODEL_VARIANT=nano \
SINGLERIGHTHAND_USE_VIDEO_CACHE=true \
COSMOS_GPU_VIDEO_AUGMENTATION=false \
EXTRA_TAIL_OVERRIDES="trainer.max_iter=50000 checkpoint.save_iter=2500 scheduler.cycle_lengths=[50000] scheduler.f_max=[1.0] scheduler.f_min=[0.0] scheduler.f_start=[0.0] scheduler.warm_up_steps=[100]" \
bash examples/launch_sft_action_policy_singlerighthand.sh
```

To select the unchanged 4B path, set `SINGLERIGHTHAND_MODEL_VARIANT=edge` or
continue invoking `launch_sft_action_policy_singlerighthand_edge.sh` directly.
