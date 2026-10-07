#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export NPROC_PER_NODE=1
export POINTFLOW_RUN_VALIDATION="${POINTFLOW_RUN_VALIDATION:-true}"
export POINTFLOW_RUN_VALIDATION_ON_START="${POINTFLOW_RUN_VALIDATION_ON_START:-true}"
export POINTFLOW_EVAL_EVERY="${POINTFLOW_EVAL_EVERY:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export MAX_ITER="${MAX_ITER:-100}"
export SINGLERIGHTHAND_EPISODE_ALLOWLIST="${SINGLERIGHTHAND_EPISODE_ALLOWLIST:-${POINTFLOW_EPISODE_ALLOWLIST:-$REPO_ROOT/examples/pointflow_sandwich_10_episodes.txt}}"
export EXTRA_TAIL_OVERRIDES="trainer.max_iter=${MAX_ITER}"
exec bash examples/launch_sft_action_policy_singlerighthand_edge.sh "$@"
