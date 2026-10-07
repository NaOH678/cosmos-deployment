#!/usr/bin/env bash
# Pointflow per-point SFT on the 101 labeled dropper episodes.
# Wraps examples/launch_sft_action_policy_singlerighthand_edge.sh with the
# pointflow env pinned so long paste lines cannot get truncated.
#
# Prerequisite: the window-latent cache must exist first:
#   bash examples/cache_dropper_window_latents.sh
#
#   bash examples/launch_pointflow_dropper101.sh
#
# Env overrides: OUTPUT_ROOT, NPROC_PER_NODE, and every POINTFLOW_* below.
#
# Differences from the sandwich labeled29 run:
#   * arm_action_space="joint" (read from the cache manifest, no toml change)
#   * pointflow_displacement_scale=0.0482, measured on dropper top-300 with
#     tools/scan_pointflow_selection.py (dropper_101_20260923/scale_scan_top300.json);
#     the recipe default 0.0740 is sandwich-labeled29-specific.
#   * phantom-drift profile is inverted vs sandwich: 23.4% of selected points,
#     concentrated on hand tracks (L2 30.2%), see
#     pointflow_outputs/dropper_101_20260923/phantom_drift_scan.json.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export WANDB_MODE="${WANDB_MODE:-offline}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/runs/cosmos/pointflow_dropper101_300_scale00482_20260923}"
export SINGLERIGHTHAND_RAW_ROOT="${SINGLERIGHTHAND_RAW_ROOT:-/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/raw_data/singlerighthand_dropper_100}"
export SINGLERIGHTHAND_CACHE_ROOT="${SINGLERIGHTHAND_CACHE_ROOT:-/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/datasets/singlerighthand-dropper-100-cosmos-cache}"
export POINTFLOW_MANIFEST="${POINTFLOW_MANIFEST:-$ROOT/pointflow_outputs/dropper_101_20260923/dropper_manifest.json}"
export SINGLERIGHTHAND_EPISODE_ALLOWLIST="${SINGLERIGHTHAND_EPISODE_ALLOWLIST:-${POINTFLOW_EPISODE_ALLOWLIST:-$ROOT/examples/pointflow_dropper_all_101_episodes.txt}}"
export POINTFLOW_TOKEN_MODE="${POINTFLOW_TOKEN_MODE:-per_point}"
export POINTFLOW_SELECT_TOP_N="${POINTFLOW_SELECT_TOP_N:-300}"
export POINTFLOW_MIN_VOXEL_MEMBERS="${POINTFLOW_MIN_VOXEL_MEMBERS:-3}"
export POINTFLOW_SELECT_MIN_VALID_STEPS="${POINTFLOW_SELECT_MIN_VALID_STEPS:-16}"
export POINTFLOW_SUPERVISE_CLUSTER_N="${POINTFLOW_SUPERVISE_CLUSTER_N:-0}"
export POINTFLOW_SELECT_MOTION_FRACTION="${POINTFLOW_SELECT_MOTION_FRACTION:-0}"
export EXTRA_TAIL_OVERRIDES="${EXTRA_TAIL_OVERRIDES:-model.config.rectified_flow_training_config.pointflow_displacement_scale=0.0482}"

echo "OUTPUT_ROOT: $OUTPUT_ROOT"
echo "raw root:    $SINGLERIGHTHAND_RAW_ROOT"
echo "cache root:  $SINGLERIGHTHAND_CACHE_ROOT"
echo "manifest:    $POINTFLOW_MANIFEST"
echo "allowlist:   $SINGLERIGHTHAND_EPISODE_ALLOWLIST"
echo "overrides:   $EXTRA_TAIL_OVERRIDES"
echo "nproc:       $NPROC_PER_NODE"

exec bash "$ROOT/examples/launch_sft_action_policy_singlerighthand_edge.sh"
