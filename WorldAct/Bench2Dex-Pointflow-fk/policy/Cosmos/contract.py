"""Explicit training contract: no guessed joint order or DROID gripper conversion."""
import json
from pathlib import Path

import numpy as np

CAMERAS = ('cam_overhead', 'cam_wrist_left', 'cam_wrist_right')


def load_contract(path):
    contract = json.loads(Path(path).read_text())
    names = contract['joint_names']
    if len(names) != 52 or len(set(names)) != 52:
        raise ValueError('This UR5+Wuji adapter requires 52 unique training joint names')
    if contract['action_representation'] != 'absolute_joint_position_radians':
        raise ValueError('Expected absolute joint targets in radians')
    if contract['view_composition'] != 'head_top__left_wrist_bottom_left__right_wrist_bottom_right':
        raise ValueError('Training video composition differs from this adapter')
    if contract['fps'] != 20 or contract['action_chunk_size'] != 32:
        raise ValueError('This simulator adapter uses 20 Hz and 32 future action steps')
    if not isinstance(contract['resolution'], (str, int)) or not contract['domain_name']:
        raise ValueError('Training resolution and action domain must be explicit')
    if not isinstance(contract['additional_view_description'], str):
        raise ValueError('Copy the exact additional_view_description from training')
    normalization = contract['normalization']
    if normalization['kind'] == 'affine':
        offset = np.asarray(normalization['offset'], np.float32)
        scale = np.asarray(normalization['scale'], np.float32)
        if offset.shape != (52,) or scale.shape != (52,) or not np.isfinite([offset, scale]).all() or np.any(scale <= 0):
            raise ValueError('Affine normalization requires 52 finite offsets and positive scales')
    elif normalization['kind'] != 'none':
        raise ValueError('Export training normalization as affine offset/scale or explicit none')
    return contract


def prepare_observation(observation, contract):
    names = list(observation['joint_names'])
    training_names = contract['joint_names']
    if len(names) != 52 or len(set(names)) != 52 or set(names) != set(training_names):
        raise ValueError('Simulator joint names differ from the training contract')
    state = np.asarray(observation['joint_action']['vector'], dtype=np.float32)
    if state.shape != (52,) or not np.isfinite(state).all():
        raise ValueError('Expected finite current state [52]')
    available = observation.get('available_camera_ids')
    if available is not None and not set(CAMERAS).issubset(available):
        raise ValueError('Missing required live camera; refusing the simulator black-frame fallback')
    images = []
    for camera in CAMERAS:
        image = np.asarray(observation['observation'][camera]['rgb'])
        if image.shape != (480, 640, 3) or image.dtype != np.uint8:
            raise ValueError(f'{camera}: expected original RGB uint8 [480,640,3]')
        images.append(image)
    prompt = observation.get('language')
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('A task instruction is required')
    state_order = [names.index(name) for name in training_names]
    output_order = [training_names.index(name) for name in names]
    return dict(images=images, state=state[state_order], prompt=prompt, output_order=output_order)


def runtime_actions(prediction, output_order, execute_steps):
    """Model output is already unpadded and denormalized by Cosmos."""
    prediction = np.asarray(prediction, np.float32)
    if prediction.shape != (33, 52) or not np.isfinite(prediction).all():
        raise ValueError(f'Expected state+32 raw 52D actions, got {prediction.shape}')
    if not 1 <= execute_steps <= 32:
        raise ValueError('execute_steps must be between 1 and 32')
    if sorted(output_order) != list(range(52)):
        raise ValueError('Invalid output joint permutation')
    return np.ascontiguousarray(prediction[1:1 + execute_steps, output_order])
