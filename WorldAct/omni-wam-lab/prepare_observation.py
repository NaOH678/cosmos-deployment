"""Build an offline WAM packet using native camera transforms and prompt formatter."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from cosmos_framework.inference.robot_policy.config import load_robot_policy_config
from cosmos_framework.inference.robot_policy.adapters import (
    _compose_right_wrist_over_head,
)
from cosmos_framework.data.generator.action.domain_utils import get_domain_id
from cosmos_framework.data.generator.action.json_formatter import (
    ActionPromptJsonFormatter,
)
from cosmos_framework.data.generator.action.transforms import (
    build_sequence_plan_from_mode,
    find_closest_target_size,
    reflection_pad_to_target,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--observation", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    c = load_robot_policy_config(a.config)
    with np.load(a.observation, allow_pickle=False) as z:
        state = z["state"].astype(np.float32)
        composed = _compose_right_wrist_over_head(z["head"], z["right_wrist"])
    if state.shape != (27,) or not np.isfinite(state).all():
        raise ValueError("Expected finite 27D state")
    t = c.model.native_chunk_size + 1
    w, h = find_closest_target_size(
        composed.shape[1], composed.shape[2], c.model.resolution
    )
    padded = {"video": composed.unsqueeze(1)}
    reflection_pad_to_target(padded, ["video"], True, w, h)
    video = torch.zeros((3, t, h, w), dtype=padded["video"].dtype)
    video[:, :1] = padded["video"]
    action = torch.zeros((t, c.model.max_action_dim), dtype=torch.float32)
    action[0, :27] = torch.from_numpy(state)
    plan = build_sequence_plan_from_mode(
        mode="wam", video_length=t, action_length=t, has_text=True
    )
    prompt_data = {
        "ai_caption": c.model.task,
        "video": video,
        "action": action,
        "conditioning_fps": torch.tensor(c.deployment.action_rate_hz),
        "image_size": padded["image_size"],
        "mode": "wam",
        "viewpoint": "concat_view",
        "additional_view_description": "The upper view is from the right wrist-mounted camera. The lower view is from the head-mounted third-person camera.",
    }
    prompt = ActionPromptJsonFormatter(caption_key="ai_caption")(prompt_data)[
        "ai_caption"
    ]
    if not isinstance(prompt, str):
        prompt = json.dumps(prompt)
    metadata = {
        "schema": "wam-native-v1",
        "source": str(a.observation.resolve()),
        "prompt": prompt,
        "domain_id": get_domain_id(c.model.domain_name),
        "fps": c.deployment.action_rate_hz,
        "raw_action_dim": 27,
        "num_frames": t,
        "num_inference_steps": c.model.num_steps,
        "guidance_scale": c.model.guidance,
        "flow_shift": c.model.shift,
        "seed": c.model.seed,
        "action_condition_indexes": list(plan.condition_frame_indexes_action),
        "action_start_frame_offset": int(plan.action_start_frame_offset),
    }
    if metadata["action_condition_indexes"] != [0]:
        raise ValueError("Missing state condition")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.output,
        first_frame=video[:, 0].numpy(),
        action=action.numpy(),
        image_size=padded["image_size"].numpy(),
        metadata=np.array(json.dumps(metadata)),
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
