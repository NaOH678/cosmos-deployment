#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Dropper-DAgger-mix variant of tools/run_fk_point_101.sh — same four-modality
# training (video + action + FK + PointFlow), same launcher chain, only the
# dataset and the two data-dependent scales differ. Everything is baked in
# here; the command line carries topology and output only:
#
#   OUTPUT_ROOT=/data/shichaojian/runs/cosmos/fk-point-dagger-mix-<date> NNODES=2 \
#     bash /mnt/afs/WorldAct-cosmos3-edge-droid-sft_base/tools/run_fk_point_dagger_mix.sh
#
# DRY_RUN=1 / RESUME=1 / MAX_ITER / PER_RANK_BATCH work exactly as in the
# sandwich wrapper.
#
# The mix is the two DAgger batches, combined as plain episode dirs under one
# raw root: dropper_dagger (51 ep, 2026-08-30) + dropper_dagger_new (20 ep,
# 2026-09-03) = 71. Random mixing needs no code: the dataset indexes every
# valid window of every allowlisted episode into one flat list and the
# dataflow shuffles window blocks across all of it
# (singlerighthand_raw_dataset.py get_shuffle_blocks), so the two batches
# interleave at window granularity on every step.
#
# The two scales are properties of the data selection, measured on the full
# 71-episode mix (do not reuse the dropper-101 or sandwich values):
#   FK        0.043357  (tools/scan_fk_displacement_scale.py
#                        --root .../dropper_dagger_mix_fk21, 71 ep, 137.6M elements)
#   PointFlow 0.037493  (pooled per-element std over
#                        pointflow_outputs/scale_scan_dagger_mix_strat500_20261001.json;
#                        same protocol that gives 0.038031 for dropper-101)
#
# NOTE: like dropper-101, the mix cosmos cache stores actions in JOINT space
# (7 arm + 20 hand joints), not sandwich's EEF space -- do NOT resume a
# sandwich-trained checkpoint onto this data.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export ALLOWLIST="${ALLOWLIST:-$ROOT/examples/dropper_dagger_mix_71_episodes.txt}"
export SINGLERIGHTHAND_RAW_ROOT="${SINGLERIGHTHAND_RAW_ROOT:-/data/shichaojian/raw_data/dropper_dagger_mix}"
export SINGLERIGHTHAND_CACHE_ROOT="${SINGLERIGHTHAND_CACHE_ROOT:-/data/shichaojian/datasets/dropper-dagger-mix-cosmos-cache}"
export FK_ANNOTATION_ROOT="${FK_ANNOTATION_ROOT:-/data/shichaojian/raw_data/dropper_dagger_mix_fk21}"
export FK_CAMERA_PROFILE="dagger"
export FK_DISPLACEMENT_SCALE="${FK_DISPLACEMENT_SCALE:-0.043357}"
export POINTFLOW_MANIFEST="${POINTFLOW_MANIFEST:-/mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow/pointflow_outputs/dagger_mix_71_20261001/manifest.json}"
export POINTFLOW_DISPLACEMENT_SCALE="${POINTFLOW_DISPLACEMENT_SCALE:-0.037493}"
# per_point is the mainline tokenisation (dropper/sandwich 16-GPU runs all used
# it); the model-side default is cluster, so bake it in rather than rely on the
# command line.
export POINTFLOW_TOKEN_MODE="${POINTFLOW_TOKEN_MODE:-per_point}"

echo ">>> dagger-mix dataset: 71 episodes (51 dagger + 20 dagger_new), fk_camera=$FK_CAMERA_PROFILE, fk_scale=$FK_DISPLACEMENT_SCALE, pointflow_scale=$POINTFLOW_DISPLACEMENT_SCALE, token_mode=$POINTFLOW_TOKEN_MODE"
exec bash "$ROOT/tools/run_fk_point_101.sh" "$@"
