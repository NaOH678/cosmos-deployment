#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="/mnt/afs/WorldAct-cosmos3-edge-droid-sft/.venv/bin:$PATH"
export LD_LIBRARY_PATH='' PYTHONPATH="$ROOT" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export SIM_POINTFK_BUNDLE="${SIM_POINTFK_BUNDLE:-/data/shichaojian/sim_data/bench2dex_task21_pointfk10_rgb3_v2}"
export SIM_POINTFK_SELECTION_ROOT="$SIM_POINTFK_BUNDLE/datasets/bench2dex-task21-cosmos-cache/pointflow_fk_handguided1024"
export EDGE_DROID_MODEL_PATH=/data/shichaojian/models/cosmos3-edge-droid
export BASE_CHECKPOINT_PATH=/data/shichaojian/models/cosmos3-edge-droid-dcp
export WAN_VAE_PATH="$EDGE_DROID_MODEL_PATH/vae/Wan2.2_VAE.pth"
export POINTFLOW_SONATA_CHECKPOINT=/data/shichaojian/checkpoints/ptv3/sonata_small.pth
export POINTFLOW_TOKEN_MODE=cluster POINTFLOW_SONATA_STAGE="${POINTFLOW_SONATA_STAGE:-2}" POINTFLOW_CLUSTER_TOKEN_CAP=1024
export POINTFLOW_DECODE_SKIP_LEVELS=1,2 POINTFLOW_DECODE_POINT_BLOCKS=4 POINTFLOW_GEOMETRY_MOTION_FUSION=false
export POINTFLOW_FPS=20 POINTFLOW_STEPS=32 POINTFLOW_STEPS_PER_TOKEN=4
export FK_ENCODER_CHECKPOINT=1 FK_KEYPOINTS=42 FK_FPS=20 FK_STEPS=32 FK_STEPS_PER_TOKEN=4
export POINTFLOW_FK_LOCAL_ROPE=true POINTFLOW_FK_ATTN_IMPL=partition COSMOS_FLASH2_VARLEN=1 I4_ATTN_BACKENDS=flash2
export POINTFLOW_FK_UNIFIED_EVAL=true POINTFLOW_EVAL_JOINT=true POINTFLOW_REFERENCE_ATTENTION=false
export TORCHINDUCTOR_MIX_ORDER_REDUCTION=0 TORCHINDUCTOR_PERSISTENT_REDUCTIONS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True COSMOS_CPU_AFFINITY_MODE=partition
export COSMOS_GPU_VIDEO_AUGMENTATION=false FK_ROLLOUT=false
readarray -t SCALES < <(python -c 'import json,os; d=json.load(open(os.environ["SIM_POINTFK_SELECTION_ROOT"]+"/manifest.json"))["train_only_scales"]; print(d["pointflow"]); print(d["fk"])')
export POINTFLOW_DISPLACEMENT_SCALE="${SCALES[0]}" FK_DISPLACEMENT_SCALE="${SCALES[1]}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/data/shichaojian/runs/sim_v2_handguided1024_stage2_selective16_8gpu_20261005}"
export IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT"
export NPROC_PER_NODE=8 NNODES=1
export MASTER_ADDR=127.0.0.1 NODE_RANK=0
export MASTER_PORT="${MASTER_PORT:-29644}"
if [[ "${1:-}" == --background ]]; then
    mkdir -p "$OUTPUT_ROOT/logs"
    nohup bash "$ROOT/examples/launch_sim_pointfk_v2.sh" > "$OUTPUT_ROOT/logs/launcher.log" 2>&1 < /dev/null &
    echo "$!" > "$OUTPUT_ROOT/train.pid"
    echo "Started PID $(cat "$OUTPUT_ROOT/train.pid"); logs: $OUTPUT_ROOT/logs/launcher.log"
    exit 0
fi
TOML_FILE=examples/toml/sft_config/action_policy_sim_pointfk_edge.toml
DATASET_PATH="$SIM_POINTFK_BUNDLE"
TAIL_OVERRIDES=("model.config.rectified_flow_training_config.independent_pointflow_schedule=false"
 "model.config.rectified_flow_training_config.independent_fk_schedule=false" "trainer.seed=42"
 "trainer.run_validation_on_start=true" ${EXTRA_TAIL_OVERRIDES:-})
source examples/_sft_launcher_common.sh
