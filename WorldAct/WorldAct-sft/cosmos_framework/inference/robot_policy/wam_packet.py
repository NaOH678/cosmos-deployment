"""CPU WAM packet builder sharing native camera transforms and prompt formatter.

The expanded temporal view is used only for prompt duration metadata. Only the
single real frame is returned; model-side future-frame construction is unchanged.
"""

import json

import numpy as np
import torch

from cosmos_framework.data.generator.action.domain_utils import get_domain_id
from cosmos_framework.data.generator.action.json_formatter import ActionPromptJsonFormatter
from cosmos_framework.data.generator.action.transforms import (
    build_sequence_plan_from_mode,
    find_closest_target_size,
    reflection_pad_to_target,
)
from cosmos_framework.inference.robot_policy.adapters import _compose_right_wrist_over_head


def build_first_frame_packet(config, images, state):
    if config.model.native_action_dim != 27 or config.model.native_chunk_size != 32:
        raise ValueError("This builder requires 27D actions and 32-step WAM chunks")
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (27,) or not np.isfinite(state).all():
        raise ValueError("Expected finite 27D WAM state")
    frames = config.model.native_chunk_size + 1
    composed = _compose_right_wrist_over_head(images["head"], images["right_wrist"])
    target_w, target_h = find_closest_target_size(composed.shape[1], composed.shape[2], config.model.resolution)
    padded = {"video": composed.unsqueeze(1)}
    reflection_pad_to_target(padded, ["video"], True, target_w, target_h)
    first_frame = padded["video"][:, 0].contiguous()
    action = torch.zeros((frames, config.model.max_action_dim), dtype=torch.float32)
    action[0, :27] = torch.from_numpy(state)
    plan = build_sequence_plan_from_mode(mode="wam", video_length=frames, action_length=frames, has_text=True)
    if list(plan.condition_frame_indexes_action) != [0] or plan.action_start_frame_offset != 0:
        raise ValueError("This packet builder requires first-state-only WAM conditioning")
    prompt_data = {
        "ai_caption": config.model.task,
        "video": first_frame.unsqueeze(1).expand(-1, frames, -1, -1),
        "action": action,
        "conditioning_fps": torch.tensor(config.deployment.action_rate_hz),
        "image_size": padded["image_size"],
        "mode": "wam",
        "viewpoint": "concat_view",
        "additional_view_description": "The upper view is from the right wrist-mounted camera. The lower view is from the head-mounted third-person camera.",
    }
    formatted = ActionPromptJsonFormatter(caption_key="ai_caption")(prompt_data)["ai_caption"]
    prompt = json.dumps(formatted) if isinstance(formatted, dict) else str(formatted)
    return {
        "metadata": {
            "schema": "wam-native-v1",
            "prompt": prompt,
            "domain_id": int(get_domain_id(config.model.domain_name)),
            "fps": config.deployment.action_rate_hz,
            "raw_action_dim": 27,
            "num_frames": frames,
            "num_inference_steps": config.model.num_steps,
            "guidance_scale": config.model.guidance,
            "flow_shift": config.model.shift,
            "seed": config.model.seed,
            "action_condition_indexes": [0],
            "action_start_frame_offset": 0,
        },
        "first_frame": first_frame.numpy(),
        "action": action.numpy(),
        "image_size": padded["image_size"].numpy(),
    }
