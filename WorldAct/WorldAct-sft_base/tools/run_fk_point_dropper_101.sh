#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Dropper variant of tools/run_fk_point_101.sh — same four-modality training
# (video + action + FK + PointFlow), same launcher chain, only the dataset and
# the two data-dependent scales differ. Everything is baked in here; the command
# line carries topology and output only:
#
#   OUTPUT_ROOT=/data/shichaojian/runs/cosmos/fk-point-dropper-<date> NNODES=2 \
#     bash /mnt/afs/WorldAct-cosmos3-edge-droid-sft_base/tools/run_fk_point_dropper_101.sh
#
# DRY_RUN=1 / RESUME=1 / MAX_ITER / PER_RANK_BATCH work exactly as in the
# sandwich wrapper.
#
# The two scales are properties of the data selection, measured on the full
# dropper 101-episode set (do not reuse the sandwich values):
#   FK        0.040438  (tools/scan_fk_displacement_scale.py --root .../dropper_fk21)
#   PointFlow 0.03803   (tools/scan_pointflow_selection.py, strat500+guard, w256;
#                        scan dump: pointflow_outputs/scale_scan_dropper_strat500_20261001.json)
#
# NOTE: the dropper cosmos cache stores actions in JOINT space (7 arm + 20 hand
# joints), where sandwich used EEF (end-effector pose + 20 hand joints). Same 27
# dims, different semantics, same domain_id -- do NOT resume a sandwich-trained
# checkpoint onto this data.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export ALLOWLIST="${ALLOWLIST:-$ROOT/examples/singlerighthand_dropper_101_episodes.txt}"
export SINGLERIGHTHAND_RAW_ROOT="${SINGLERIGHTHAND_RAW_ROOT:-/data/shichaojian/raw_data/singlerighthand_dropper_100}"
export SINGLERIGHTHAND_CACHE_ROOT="${SINGLERIGHTHAND_CACHE_ROOT:-/data/shichaojian/datasets/singlerighthand-dropper-100-cosmos-cache}"
export FK_ANNOTATION_ROOT="${FK_ANNOTATION_ROOT:-/data/shichaojian/raw_data/dropper_fk21}"
export FK_DISPLACEMENT_SCALE="${FK_DISPLACEMENT_SCALE:-0.040438}"
export POINTFLOW_MANIFEST="${POINTFLOW_MANIFEST:-/mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow/pointflow_outputs/dropper_924_20260928/manifest.json}"
export POINTFLOW_DISPLACEMENT_SCALE="${POINTFLOW_DISPLACEMENT_SCALE:-0.03803}"

echo ">>> dropper dataset: 101 episodes, fk_scale=$FK_DISPLACEMENT_SCALE, pointflow_scale=$POINTFLOW_DISPLACEMENT_SCALE"
exec bash "$ROOT/tools/run_fk_point_101.sh" "$@"
