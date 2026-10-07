#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Dev-machine adapter for launch_sft_action_policy_singlerighthand_edge.sh.
#
# The original launcher is byte-identical to the cluster version; this wrapper
# only fixes up the ENVIRONMENT it runs in, then execs it:
#   1. Prepends this worktree's own .venv/bin to PATH. Ambient `python` on the
#      dev box is a Conda base with no torch, and /usr/local/bin/torchrun is not
#      this environment — the original launcher probes `python` for its prefix
#      and calls `torchrun` from PATH, so both must resolve to the venv.
#   2. Clears LD_LIBRARY_PATH (docs/setup.md gotcha: a stray one breaks the
#      torch._C import). The original launcher re-adds the prefixes it needs.
#   3. Points the data / model / output variables at the dev box layout under
#      /data/shichaojian (the compiled-in defaults are cluster paths that do
#      not exist here).
#
# Everything else — recipe, batch size, callbacks, checkpoint cadence, log
# format — is exactly what the native launcher does. All of its env knobs pass
# straight through: NPROC_PER_NODE, NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT,
# EXTRA_TAIL_OVERRIDES, and any of the paths below when set explicitly.
#
# Single node (8 GPUs):
#   bash examples/launch_sft_action_policy_singlerighthand_edge_dev.sh
# Two nodes (2 x 8 GPUs), node 0:
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=<node0-host> bash examples/launch_sft_action_policy_singlerighthand_edge_dev.sh
# Two nodes, node 1:
#   NNODES=2 NODE_RANK=1 MASTER_ADDR=<node0-host> bash examples/launch_sft_action_policy_singlerighthand_edge_dev.sh
# See docs/native_baseline_dev_multinode.md for the full multi-node runbook.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VENV_BIN="$REPO_ROOT/.venv/bin"
[[ -x "$VENV_BIN/torchrun" ]] || { echo "ERROR: $VENV_BIN/torchrun not found; build the worktree venv first (docs/setup.md)" >&2; exit 1; }
export PATH="$VENV_BIN:$PATH"
export LD_LIBRARY_PATH=''

: "${SINGLERIGHTHAND_RAW_ROOT:=/data/shichaojian/raw_data/singlerighthand_sandwich_100}"
: "${SINGLERIGHTHAND_CACHE_ROOT:=/data/shichaojian/datasets/singlerighthand-sandwich-100-cosmos-cache}"
: "${EDGE_DROID_MODEL_PATH:=/data/shichaojian/models/cosmos3-edge-droid}"
: "${BASE_CHECKPOINT_PATH:=/data/shichaojian/models/cosmos3-edge-droid-dcp}"
: "${OUTPUT_ROOT:=/data/shichaojian/runs/cosmos/singlerighthand-edge-droid}"
export SINGLERIGHTHAND_RAW_ROOT SINGLERIGHTHAND_CACHE_ROOT EDGE_DROID_MODEL_PATH
export BASE_CHECKPOINT_PATH OUTPUT_ROOT

# Per-window VAE latent cache (ported from the _base worktree: dataset +
# omni_mot_model consumer). The dataset hands the model each window's
# pre-encoded Wan2.2 latent, so training skips the frozen-VAE encode over every
# pixel frame of every window on every step. Auto-detection mirrors the video
# frame cache: unset -> enable when the cache manifest exists; explicitly empty
# -> off; explicit path -> used as-is. The dataset validates the manifest's
# fps/chunk_length/sample_stride against its own, so a stale cache is refused
# rather than misread.
if [[ -z "${SINGLERIGHTHAND_VAE_WINDOW_LATENT_ROOT+x}" ]]; then
    SINGLERIGHTHAND_VAE_WINDOW_LATENT_ROOT=""
    [[ -f "$SINGLERIGHTHAND_CACHE_ROOT/vae_window_latents/window_manifest.json" ]] \
        && SINGLERIGHTHAND_VAE_WINDOW_LATENT_ROOT="$SINGLERIGHTHAND_CACHE_ROOT/vae_window_latents"
fi
# Legacy whole-episode cache: off by default (not equivalent to a fresh encode;
# see SingleRightHandRawDataset._read_window_latent).
: "${SINGLERIGHTHAND_VAE_LATENT_ROOT:=}"
export SINGLERIGHTHAND_VAE_WINDOW_LATENT_ROOT SINGLERIGHTHAND_VAE_LATENT_ROOT

# Memory fit for A800-80GB: the recipe's native 32 samples/rank OOMs on this
# box (73 GiB already allocated when the first backward runs at 480p, smoke
# verified 2026-09-30). 16/rank fits and also matches the per-rank batch of
# the _base comparison runs. Prepended so an explicit user value in
# EXTRA_TAIL_OVERRIDES still wins (the recipe applies the LAST occurrence).
: "${PYTORCH_ALLOC_CONF:=expandable_segments:True}"
export PYTORCH_ALLOC_CONF

# Attention backend: native code bans flash2 for varlen packs, which on sm80
# (A800) leaves natten as the only varlen backend (~13.3 s/step, smoke measured
# 2026-09-30). This worktree carries the same COSMOS_FLASH2_VARLEN escape hatch
# as _base (model/attention/flash2/checks.py); enabling it selects flash2
# (~1.85x faster on attention-heavy packs). Set COSMOS_FLASH2_VARLEN=0 to go
# back to strictly-native natten behavior.
: "${COSMOS_FLASH2_VARLEN:=1}"
export COSMOS_FLASH2_VARLEN
case " ${EXTRA_TAIL_OVERRIDES:-} " in
    *" dataloader_train.max_samples_per_batch="*) ;;
    *) EXTRA_TAIL_OVERRIDES="dataloader_train.max_samples_per_batch=16 ${EXTRA_TAIL_OVERRIDES:-}" ;;
esac
export EXTRA_TAIL_OVERRIDES

# Multi-node topology fallback, mirroring _base's _sft_launcher_common.sh so a
# SINGLE identical command works on every node: platform-injected topology env
# is honored as defaults, and with NNODES set but NODE_RANK unset the rank is
# derived from the hostname -- Kubeflow pods (<job>-master-0 -> rank 0 and the
# rendezvous host; <job>-worker-N -> rank N+1, MASTER_ADDR=<job>-master-0) or
# dev machines (dev-<id>-N -> rank N, MASTER_ADDR=dev-<id>-0). NODE_RANK must
# stay UNSET for the fallback to fire -- defaulting it to 0 makes every node
# believe it is rank 0 and they wait for each other at rendezvous forever.
: "${NNODES:=${SENSECORE_PYTORCH_NNODES:-}}"
: "${NODE_RANK:=${SENSECORE_PYTORCH_NODE_RANK:-}}"
if [[ -n "${NNODES:-}" && -z "${NODE_RANK:-}" ]]; then
    if [[ "${HOSTNAME:-}" =~ ^(.+)-(master|worker)-([0-9]+)$ ]]; then
        if [[ "${BASH_REMATCH[2]}" == "master" ]]; then
            NODE_RANK=0
            : "${MASTER_ADDR:=$HOSTNAME}"
        else
            NODE_RANK=$((BASH_REMATCH[3] + 1))
            : "${MASTER_ADDR:=${BASH_REMATCH[1]}-master-0}"
        fi
    elif [[ "${HOSTNAME:-}" =~ ^(.+)-([0-9]+)$ ]]; then
        NODE_RANK="${BASH_REMATCH[2]}"
        : "${MASTER_ADDR:=${BASH_REMATCH[1]}-0}"
    fi
    if [[ -n "${NODE_RANK:-}" ]]; then
        echo ">>> topology fallback: HOSTNAME=$HOSTNAME -> NODE_RANK=$NODE_RANK MASTER_ADDR=$MASTER_ADDR"
    fi
fi
export NNODES NODE_RANK
[[ -n "${MASTER_ADDR:-}" ]] && export MASTER_ADDR

# Benign noise on this box: the original launcher probes `import nvidia.npp`
# (a cluster-only wheel) and prints a ModuleNotFoundError traceback before
# continuing with an empty NPP path. It is not a failure.
exec bash "$REPO_ROOT/examples/launch_sft_action_policy_singlerighthand_edge.sh" "$@"
