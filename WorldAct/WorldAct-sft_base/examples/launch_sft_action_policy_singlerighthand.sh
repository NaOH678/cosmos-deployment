#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_VARIANT="${SINGLERIGHTHAND_MODEL_VARIANT:-edge}"

case "${MODEL_VARIANT,,}" in
    edge|4b)
        exec bash "$SCRIPT_DIR/launch_sft_action_policy_singlerighthand_edge.sh"
        ;;
    nano|16b)
        exec bash "$SCRIPT_DIR/launch_sft_action_policy_singlerighthand_nano.sh"
        ;;
    *)
        echo "ERROR: SINGLERIGHTHAND_MODEL_VARIANT must be edge/4b or nano/16b; got: $MODEL_VARIANT" >&2
        exit 2
        ;;
esac
