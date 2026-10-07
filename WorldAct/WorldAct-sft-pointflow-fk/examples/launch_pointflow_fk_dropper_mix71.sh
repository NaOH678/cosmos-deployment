#!/usr/bin/env bash
# Dropper DAgger mix: E6b architecture/training recipe, dataset-specific scales.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SINGLERIGHTHAND_RAW_ROOT=/data/shichaojian/raw_data/dropper_dagger_mix
export SINGLERIGHTHAND_CACHE_ROOT=/data/shichaojian/datasets/dropper-dagger-mix-cosmos-cache
export SINGLERIGHTHAND_VAE_WINDOW_LATENT_ROOT="$SINGLERIGHTHAND_CACHE_ROOT/vae_window_latents"
export FK_ANNOTATION_ROOT=/data/shichaojian/raw_data/dropper_dagger_mix_fk21
export POINTFLOW_MANIFEST="$ROOT/pointflow_outputs/dagger_mix_71_20261001/manifest.json"
export POINTFLOW_EPISODE_ALLOWLIST="$ROOT/examples/pointflow_dropper_dagger_mix_71_episodes.txt"
export POINTFLOW_WINDOW_CACHE_ROOT="$SINGLERIGHTHAND_CACHE_ROOT/pointflow_windows"
export POINTFLOW_DISPLACEMENT_SCALE=0.037493
export FK_DISPLACEMENT_SCALE=0.043357
export POINTFLOW_DISPLACEMENT_FRAME_SCALES=''
export POINTFLOW_DISPLACEMENT_FRAME_SCALES_FILE=''
export POINTFLOW_TOKEN_MODE=cluster
export POINTFLOW_SONATA_STAGE=1
export POINTFLOW_CLUSTER_TOKEN_CAP=128
export POINTFLOW_DECODE_SKIP_LEVELS=1,2
export POINTFLOW_DECODE_POINT_BLOCKS=4
export POINTFLOW_GEOMETRY_MOTION_FUSION=false
export POINTFLOW_FK_LOCAL_ROPE="${POINTFLOW_FK_LOCAL_ROPE:-true}"
export POINTFLOW_FK_ATTN_IMPL=partition
export POINTFLOW_EVAL_JOINT=true
export EXTRA_TAIL_OVERRIDES="model.config.activation_checkpointing.mode=selective model.config.activation_checkpointing.save_ops_regex=[_flash_attn.*forward] model.config.activation_checkpointing.save_mm_shapes=[[2048,2048],[2048,1024]] trainer.callbacks.pointflow_eval.val_episodes=7 trainer.callbacks.pointflow_eval.val_windows=1 trainer.seed=42 trainer.grad_accum_iter=1 trainer.run_validation=true trainer.run_validation_on_start=true trainer.validation_iter=500 dataloader_train.max_samples_per_batch=16 dataloader_train.dataloader.in_order=true dataloader_train.dataloader.datasets.singlerighthand.dataset.split_val_ratio=0.1 dataloader_val.dataloader.datasets.singlerighthand.dataset.split_val_ratio=0.1 ${EXTRA_TAIL_OVERRIDES:-}"
exec bash "$ROOT/examples/launch_pointflow_fk_sandwich101.sh" "$@"
