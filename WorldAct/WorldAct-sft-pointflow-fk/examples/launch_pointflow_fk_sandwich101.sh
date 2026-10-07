#!/usr/bin/env bash
# Four modalities on the current cluster baseline. Local RoPE is opt-in.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export FK_ENCODER_CHECKPOINT=1
export POINTFLOW_FK_UNIFIED_EVAL="${POINTFLOW_FK_UNIFIED_EVAL:-true}"
export POINTFLOW_EVAL_JOINT="${POINTFLOW_EVAL_JOINT:-true}"
export FK_PROJECT_ANCHORS="${FK_PROJECT_ANCHORS:-true}"
export POINTFLOW_FK_LOCAL_ROPE="${POINTFLOW_FK_LOCAL_ROPE:-false}"
export POINTFLOW_FK_ATTN_IMPL="${POINTFLOW_FK_ATTN_IMPL:-partition}"
export POINTFLOW_SONATA_STAGE="${POINTFLOW_SONATA_STAGE:-1}"
export POINTFLOW_CLUSTER_TOKEN_CAP="${POINTFLOW_CLUSTER_TOKEN_CAP:-128}"
export POINTFLOW_DECODE_SKIP_LEVELS="${POINTFLOW_DECODE_SKIP_LEVELS:-1,2}"
export POINTFLOW_DECODE_POINT_BLOCKS="${POINTFLOW_DECODE_POINT_BLOCKS:-4}"
export POINTFLOW_GEOMETRY_MOTION_FUSION="${POINTFLOW_GEOMETRY_MOTION_FUSION:-false}"
export POINTFLOW_MANIFEST="${POINTFLOW_MANIFEST:-$ROOT/pointflow_outputs/sandwich_924_20260928/manifest.json}"
export POINTFLOW_EPISODE_ALLOWLIST="${POINTFLOW_EPISODE_ALLOWLIST:-$ROOT/examples/pointflow_sandwich_all_101_episodes.txt}"
export SINGLERIGHTHAND_EPISODE_ALLOWLIST="$POINTFLOW_EPISODE_ALLOWLIST"
export POINTFLOW_DISPLACEMENT_SCALE="${POINTFLOW_DISPLACEMENT_SCALE:-0.0528}"
export FK_DISPLACEMENT_SCALE="${FK_DISPLACEMENT_SCALE:-0.083745}"
export COSMOS_FLASH2_VARLEN="${COSMOS_FLASH2_VARLEN:-1}"
export I4_ATTN_BACKENDS="${I4_ATTN_BACKENDS:-flash2}"
export POINTFLOW_REFERENCE_ATTENTION=false
export NPROC_PER_NODE="${NPROC_PER_NODE:-${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}}"
_NODES="${NNODES:-${SENSECORE_PYTORCH_NNODES:-1}}"
# Compile Transformer blocks with dynamic shapes; keep packing outside the
# compiled region. Explicit caller overrides are appended last.
export EXTRA_TAIL_OVERRIDES="model.config.parallelism.data_parallel_shard_degree=$NPROC_PER_NODE model.config.parallelism.data_parallel_replicate_degree=$_NODES model.config.rectified_flow_training_config.independent_pointflow_schedule=false model.config.rectified_flow_training_config.independent_fk_schedule=false model.config.compile.enabled=true model.config.compile.compiled_region=language model.config.compile.compile_dynamic=true ${EXTRA_TAIL_OVERRIDES:-}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT to a fresh directory}"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf 'repo=%s\nlocal_rope=%s\nlocal_attention=%s\nproject_fk=%s\noutput=%s\noverrides=%s\n' \
        "$ROOT" "$POINTFLOW_FK_LOCAL_ROPE" "$POINTFLOW_FK_ATTN_IMPL" "$FK_PROJECT_ANCHORS" "$OUTPUT_ROOT" "$EXTRA_TAIL_OVERRIDES"
    exit 0
fi
exec bash "$ROOT/examples/launch_sft_action_policy_fk_point_singlerighthand_edge.sh" "$@"
