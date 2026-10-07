"""Native 52D joint actions through WorldAct's action processing contract."""
import json
from pathlib import Path
import numpy as np
import torch

EMBODIMENT = 'bench2dex_wuji'
ACTION_DIM = 52


def action_normalizer(path, joint_names, scale_floor_rad=.05):
    from cosmos_framework.data.generator.action.action_processing import ActionAffineNormalization
    report = json.loads(Path(path).read_text())
    if report['joint_names'] != list(joint_names):
        raise ValueError('Normalization joint order differs from the recorded action order')
    if not report['source'].startswith('training episodes only'):
        raise ValueError('Expected training-only action statistics')
    low, high = (np.asarray(report['action'][k], np.float32) for k in ['q01', 'q99'])
    if low.shape != (ACTION_DIM,) or high.shape != (ACTION_DIM,):
        raise ValueError('Expected 52D action statistics')
    if not np.isfinite([low, high]).all() or np.any(high < low) or scale_floor_rad <= 0:
        raise ValueError('Invalid action normalization statistics')
    # Invertible, with no clipping of rare but valid joint targets. Constant
    # channels have a nonzero radians-scale floor instead of exploding values.
    return ActionAffineNormalization(offset=torch.from_numpy((low+high)/2),
        scale=torch.from_numpy(np.maximum((high-low)/2, scale_floor_rad)))


def prepare_action_sample(sample, normalization_path, max_action_dim=64):
    """Action stage only; video/text/sequence-plan transforms remain downstream."""
    from cosmos_framework.data.generator.action.action_processing import ActionProcessor
    from cosmos_framework.data.generator.action.domain_utils import get_domain_id, get_action_dim
    if get_action_dim(EMBODIMENT) != ACTION_DIM:
        raise ValueError('WorldAct has an incompatible Bench2Dex action domain')
    action = sample['action']
    if action.ndim != 2 or action.shape[-1] != ACTION_DIM or not torch.isfinite(action).all():
        raise ValueError('Expected finite [T,52] native action values')
    if sample['action_type'] != 'absolute_joint_position_radians':
        raise ValueError('Expected absolute joint targets in radians')
    fields = dict(sample, domain_id=torch.tensor(get_domain_id(EMBODIMENT), dtype=torch.long),
                  embodiment=EMBODIMENT)
    return ActionProcessor(max_action_dim=max_action_dim, action_channel_masking=True).preprocess_action(
        fields, action, action_normalizer=action_normalizer(normalization_path, sample['joint_names']))
