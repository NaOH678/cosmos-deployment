#!/usr/bin/env bash
# Pointflow per-point SFT on all 101 sandwich episodes (9.24 labeled delivery),
# pinned for the inspur A800 cluster (/data/shichaojian layout). Wraps
# examples/launch_sft_action_policy_singlerighthand_edge.sh so the launch line
# stays short and long paste lines cannot get truncated.
#
#   OUTPUT_ROOT=/data/shichaojian/runs/<new-dir> bash examples/launch_pointflow_sandwich101.sh
#
# SenseCore injects the torchrun topology (SENSECORE_* / MASTER_ADDR / MASTER_PORT);
# FSDP shards within a node and replicates across nodes (HSDP), derived here from
# the injected device/node counts, so the same command works for 1 or N nodes.
#
# Env overrides: OUTPUT_ROOT (required, fresh per run) and every export below.
set -euo pipefail

ROOT="${COSMOS_REPO_ROOT:-/mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow}"

# Cluster layout (shared venv lives in the base checkout). LD_LIBRARY_PATH must be
# EMPTY for this stack (docs/setup.md#pytorch-import-issue) — force it here so the
# launch line stays a single command.
export LD_LIBRARY_PATH=""
export PYTHON_BIN="${PYTHON_BIN:-/mnt/afs/WorldAct-cosmos3-edge-droid-sft/.venv/bin/python}"
export EDGE_DROID_MODEL_PATH="${EDGE_DROID_MODEL_PATH:-/data/shichaojian/models/cosmos3-edge-droid}"
export BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-/data/shichaojian/models/cosmos3-edge-droid-dcp}"
export POINTFLOW_SONATA_CHECKPOINT="${POINTFLOW_SONATA_CHECKPOINT:-/data/shichaojian/checkpoints/ptv3/sonata_small.pth}"
export SINGLERIGHTHAND_RAW_ROOT="${SINGLERIGHTHAND_RAW_ROOT:-/data/shichaojian/raw_data/singlerighthand_sandwich_100}"
export SINGLERIGHTHAND_CACHE_ROOT="${SINGLERIGHTHAND_CACHE_ROOT:-/data/shichaojian/datasets/singlerighthand-sandwich-100-cosmos-cache}"
export POINTFLOW_MANIFEST="${POINTFLOW_MANIFEST:-$ROOT/pointflow_outputs/sandwich_924_20260928/manifest.json}"
export POINTFLOW_EPISODE_ALLOWLIST="${POINTFLOW_EPISODE_ALLOWLIST:-$ROOT/examples/pointflow_sandwich_all_101_episodes.txt}"

# Pointflow recipe: per-point tokens, stratified semantic selection (hand/object/
# surface = 40/45/15 of 500 points), phantom-drift guard, precomputed window cache.
export POINTFLOW_TOKEN_MODE="${POINTFLOW_TOKEN_MODE:-per_point}"
export POINTFLOW_SELECT_TOP_N="${POINTFLOW_SELECT_TOP_N:-500}"
export POINTFLOW_SELECT_REGION_QUOTAS="${POINTFLOW_SELECT_REGION_QUOTAS:-2:0.40,3:0.45,4:0.15}"
export POINTFLOW_MIN_VOXEL_MEMBERS="${POINTFLOW_MIN_VOXEL_MEMBERS:-3}"
export POINTFLOW_SELECT_MIN_VALID_STEPS="${POINTFLOW_SELECT_MIN_VALID_STEPS:-16}"
export POINTFLOW_SELECT_PHANTOM_GUARD="${POINTFLOW_SELECT_PHANTOM_GUARD:-true}"
export POINTFLOW_WINDOW_CACHE_ROOT="${POINTFLOW_WINDOW_CACHE_ROOT:-$SINGLERIGHTHAND_CACHE_ROOT/pointflow_windows}"
export POINTFLOW_SUPERVISE_CLUSTER_N="${POINTFLOW_SUPERVISE_CLUSTER_N:-0}"
export POINTFLOW_SELECT_MOTION_FRACTION="${POINTFLOW_SELECT_MOTION_FRACTION:-0}"

# Per-frame displacement scale (off by default): a per-frame (per-channel)
# vector overriding the scalar POINTFLOW_DISPLACEMENT_SCALE.  Prefer the file
# form: point FRAME_SCALES_FILE at the <stem>_frame_scales_env.json written by
# tools/scan_pointflow_selection.py; the comma-string form (32 or 96 values)
# remains for ad-hoc overrides.  Either way the target parameterisation
# changes, so train from scratch.
export POINTFLOW_DISPLACEMENT_FRAME_SCALES_FILE="${POINTFLOW_DISPLACEMENT_FRAME_SCALES_FILE:-}"
export POINTFLOW_DISPLACEMENT_FRAME_SCALES="${POINTFLOW_DISPLACEMENT_FRAME_SCALES:-}"

# Fuse per-point level-0 geometry + relative XYZ with noisy motion before pooling.
# Changes motion_encoder input width; use a fresh OUTPUT_ROOT when switching modes.
export POINTFLOW_GEOMETRY_MOTION_FUSION="${POINTFLOW_GEOMETRY_MOTION_FUSION:-false}"

# Decode-side upgrades (off by default): multi-resolution Sonata skip features
# ("1,2" adds level-1/2 per-point features) and per-point transformer blocks
# between broadcast and the velocity head.
export POINTFLOW_DECODE_SKIP_LEVELS="${POINTFLOW_DECODE_SKIP_LEVELS:-}"
export POINTFLOW_DECODE_POINT_BLOCKS="${POINTFLOW_DECODE_POINT_BLOCKS:-0}"
export POINTFLOW_DECODE_POINT_DIM="${POINTFLOW_DECODE_POINT_DIM:-256}"
export POINTFLOW_DECODE_POINT_HEADS="${POINTFLOW_DECODE_POINT_HEADS:-4}"

# Eval: deployment-shaped joint rollout (video/action/point dream canvas).
export POINTFLOW_EVAL_JOINT="${POINTFLOW_EVAL_JOINT:-true}"

# Runtime: flash2 varlen (validated, ~2.2x step time) + fragmentation guard.
export COSMOS_FLASH2_VARLEN="${COSMOS_FLASH2_VARLEN:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}}"

# Per-node log file: with NNODES>1 every node's torchrun tees to the same
# $OUTPUT_ROOT/logs/<file>, and the two streams carry independent file offsets
# so they overwrite (not append) each other — the rank-0 startup section (model
# build, scale loading) gets clobbered. Give each node its own log. NODE_RANK is
# not injected yet on Kubeflow-style pods, so mirror the hostname fallback from
# _sft_launcher_common.sh.
_NODE_RANK="${SENSECORE_PYTORCH_NODE_RANK:-${NODE_RANK:-}}"
if [[ -z "$_NODE_RANK" && "${HOSTNAME:-}" =~ ^(.+)-(master|worker)-([0-9]+)$ ]]; then
    if [[ "${BASH_REMATCH[2]}" == "master" ]]; then _NODE_RANK=0; else _NODE_RANK=$((BASH_REMATCH[3] + 1)); fi
fi
export LOG_FILENAME="${LOG_FILENAME:-action_policy_singlerighthand_edge_sft_node${_NODE_RANK:-0}.log}"

# HSDP: shard within the node (NVLink), replicate across nodes. The launcher
# enforces shard x replicate == world size.
_GPUS_PER_NODE="$NPROC_PER_NODE"
_NODES="${SENSECORE_PYTORCH_NNODES:-${NNODES:-1}}"
export EXTRA_TAIL_OVERRIDES="model.config.parallelism.data_parallel_shard_degree=${_GPUS_PER_NODE} model.config.parallelism.data_parallel_replicate_degree=${_NODES} dataloader_train.max_samples_per_batch=16 trainer.run_validation_on_start=true model.config.rectified_flow_training_config.pointflow_displacement_scale=0.0528 ${EXTRA_TAIL_OVERRIDES:-}"

echo "OUTPUT_ROOT: ${OUTPUT_ROOT:?set OUTPUT_ROOT to a fresh directory}"
echo "manifest:    $POINTFLOW_MANIFEST"
echo "allowlist:   $POINTFLOW_EPISODE_ALLOWLIST"
echo "topology:    shard=$_GPUS_PER_NODE replicate=$_NODES"
echo "log:         $LOG_FILENAME"
echo "overrides:   $EXTRA_TAIL_OVERRIDES"
# Surface the injected topology env so multi-node misconfiguration is visible in
# the job console (scheduled jobs inject SENSECORE_* / MASTER_* / WORLD_SIZE ...;
# interactive pods inject nothing).
env | grep -E '^(SENSECORE|MASTER_|NNODES|NODE_RANK|WORLD_SIZE|GROUP_RANK|RANK|PET_|TORCH_|LOCAL_RANK)' | sort | sed 's/^/[topology-env] /' || true

exec bash "$ROOT/examples/launch_sft_action_policy_singlerighthand_edge.sh"
