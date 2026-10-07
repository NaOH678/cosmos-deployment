# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Protocol-v2 serving support for Cosmos robot action policies."""

from cosmos_framework.inference.robot_policy.adapters import ModelAdapter, create_model_adapter
from cosmos_framework.inference.robot_policy.config import RobotPolicyConfig, load_robot_policy_config

__all__ = ["ModelAdapter", "RobotPolicyConfig", "create_model_adapter", "load_robot_policy_config"]
