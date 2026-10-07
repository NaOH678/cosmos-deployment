#!/usr/bin/env bash
# Pointflow per-point SFT on the 29 labeled sandwich episodes.
# Wraps examples/launch_sft_action_policy_singlerighthand_edge.sh with the
# pointflow env pinned so long paste lines cannot get truncated.
#
#   bash examples/launch_pointflow_labeled29_sandwich.sh
#
# Env overrides: OUTPUT_ROOT, NPROC_PER_NODE, and every POINTFLOW_* below.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export WANDB_MODE="${WANDB_MODE:-offline}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/runs/cosmos/pointflow_labeled29_300_mvs16_20260922}"
export POINTFLOW_MANIFEST="${POINTFLOW_MANIFEST:-$ROOT/pointflow_outputs/manifest_sandwich_labeled_20260921.json}"
export SINGLERIGHTHAND_EPISODE_ALLOWLIST="${SINGLERIGHTHAND_EPISODE_ALLOWLIST:-${POINTFLOW_EPISODE_ALLOWLIST:-$ROOT/examples/pointflow_sandwich_labeled_29_episodes.txt}}"
export POINTFLOW_TOKEN_MODE="${POINTFLOW_TOKEN_MODE:-per_point}"
export POINTFLOW_SELECT_TOP_N="${POINTFLOW_SELECT_TOP_N:-300}"
export POINTFLOW_MIN_VOXEL_MEMBERS="${POINTFLOW_MIN_VOXEL_MEMBERS:-3}"
export POINTFLOW_SELECT_MIN_VALID_STEPS="${POINTFLOW_SELECT_MIN_VALID_STEPS:-16}"
export POINTFLOW_SUPERVISE_CLUSTER_N="${POINTFLOW_SUPERVISE_CLUSTER_N:-0}"
export POINTFLOW_SELECT_MOTION_FRACTION="${POINTFLOW_SELECT_MOTION_FRACTION:-0}"

echo "OUTPUT_ROOT: $OUTPUT_ROOT"
echo "manifest:    $POINTFLOW_MANIFEST"
echo "allowlist:   $SINGLERIGHTHAND_EPISODE_ALLOWLIST"
echo "nproc:       $NPROC_PER_NODE"

exec bash "$ROOT/examples/launch_sft_action_policy_singlerighthand_edge.sh"
