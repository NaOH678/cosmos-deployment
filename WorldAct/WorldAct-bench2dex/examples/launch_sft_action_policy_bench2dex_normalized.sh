#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${BENCH2DEX_ACTION_STATS_PATH:?Set training-only action_stats.json}"
: "${OUTPUT_ROOT:?Set a new normalized-run output directory}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
result=$(python tools/prepare_bench2dex_normalized_run.py --stats "$BENCH2DEX_ACTION_STATS_PATH" --output-root "$OUTPUT_ROOT")
export BENCH2DEX_ACTION_STATS_PATH="$(head -n 1 <<< "$result")"
export BENCH2DEX_ACTION_STATS_SHA256="$(tail -n 1 <<< "$result")"
bash examples/launch_sft_action_policy_bench2dex_edge.sh -- \
  trainer.max_iter=20000 scheduler.cycle_lengths='[20000]' trainer.grad_accum_iter=1 \
  trainer.seed=42 trainer.logging_iter=10 checkpoint.save_iter=500 \
  dataloader_train.max_samples_per_batch=16 dataloader_train.dataloader.batch_size=16 \
  dataloader_train.dataloader.num_workers=4 dataloader_train.dataloader.prefetch_factor=2 \
  trainer.run_validation=false trainer.run_validation_on_start=false "$@"
