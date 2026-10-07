import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from policy.Cosmos.contract import CAMERAS, load_contract, prepare_observation, runtime_actions
from policy.Cosmos.deploy_policy import raw_sample, load_compositor, LocalSession


def contract():
    return dict(joint_names=[f'joint_{i}' for i in range(52)], action_representation='absolute_joint_position_radians',
        view_composition='head_top__left_wrist_bottom_left__right_wrist_bottom_right', fps=20,
        action_chunk_size=32, resolution='480', domain_name='bench2dex_wuji', normalization=dict(kind='none'),
        additional_view_description='Head on top; left wrist bottom-left; right wrist bottom-right.')


def observation():
    return dict(joint_names=list(reversed(contract()['joint_names'])),
        joint_action=dict(vector=np.arange(52, dtype=np.float32)), language='load the condiment box',
        available_camera_ids=list(CAMERAS), observation={camera: dict(rgb=np.full((480, 640, 3), i*70, np.uint8))
            for i, camera in enumerate(CAMERAS)})


def test_joint_order_and_no_gripper_flip():
    prepared = prepare_observation(observation(), contract())
    np.testing.assert_array_equal(prepared['state'], np.arange(51, -1, -1))
    predicted = np.tile(np.arange(52, dtype=np.float32), (33, 1))
    predicted[0] = -999
    action = runtime_actions(predicted, prepared['output_order'], 4)
    assert action.shape == (4, 52)
    np.testing.assert_array_equal(action[0], np.arange(51, -1, -1))
    assert action[0, -1] == 0  # DROID's 1-gripper conversion must never apply.


def test_missing_live_camera_rejected():
    obs = observation(); obs['available_camera_ids'].remove('cam_overhead')
    with pytest.raises(ValueError, match='Missing required live camera'):
        prepare_observation(obs, contract())


def test_wrong_joint_set_rejected():
    obs = observation(); obs['joint_names'][0] = 'wrong_joint'
    with pytest.raises(ValueError, match='joint names'):
        prepare_observation(obs, contract())


def test_native_compositor_and_current_only_input():
    worldact = Path(os.environ.get('WORLDACT_ROOT', Path(__file__).resolve().parents[2] / 'WorldAct-pointflow-fk'))
    if not worldact.is_dir():
        pytest.skip('Set WORLDACT_ROOT to the external WorldAct checkout for integration tests')
    compose = load_compositor(worldact)
    sample = raw_sample(prepare_observation(observation(), contract()), contract(), compose, 28)
    assert sample['video'].shape == (3, 33, 720, 640)
    assert torch.all(sample['video'][:, 0, :480] == 0)
    assert torch.all(sample['video'][:, 0, 480:, :320] == 70)
    assert torch.all(sample['video'][:, 0, 480:, 320:] == 140)
    assert torch.all(sample['video'][:, 1:] == 0)
    assert torch.all(sample['action'][1:] == 0)
    assert not {'pointflow', 'fk'} & sample.keys()


def test_contract_rejects_bad_normalization(tmp_path):
    value = contract(); value['normalization'] = dict(kind='affine', offset=[0]*52, scale=[0]*52)
    path = tmp_path/'contract.json'; path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='positive scales'):
        load_contract(path)


@pytest.mark.parametrize('shape', [(32, 52), (33, 54), (33, 64)])
def test_wrong_output_shape_is_not_silently_sliced(shape):
    with pytest.raises(ValueError, match=r'state\+32'):
        runtime_actions(np.zeros(shape), list(range(52)), 4)


def test_session_seed_forwarded(monkeypatch):
    class Backend:
        def reset(self, seed): self.seed = seed
        def get_action(self, obs):
            assert obs['language'] == 'test instruction'
            return np.zeros((4, 52), np.float32)
    backend = Backend()
    monkeypatch.setattr('policy.Cosmos.deploy_policy.get_model', lambda config: backend)
    session = LocalSession({})
    session.reset('test instruction', seed=123)
    assert backend.seed == 123
    assert session.get_action_chunk({}).shape == (4, 52)


def test_native_action_denormalization_exactly_once(monkeypatch):
    worldact = Path(os.environ.get('WORLDACT_ROOT', Path(__file__).resolve().parents[2] / 'WorldAct-pointflow-fk'))
    if not worldact.is_dir():
        pytest.skip('Set WORLDACT_ROOT to the external WorldAct checkout for integration tests')
    monkeypatch.syspath_prepend(str(worldact))
    from cosmos_framework.data.generator.action.action_processing import ActionAffineNormalization, ActionProcessor
    raw = torch.arange(33 * 52, dtype=torch.float32).reshape(33, 52) / 1000
    normalizer = ActionAffineNormalization(offset=torch.linspace(-1, 1, 52), scale=torch.linspace(.1, 2, 52))
    processor = ActionProcessor(max_action_dim=64, action_channel_masking=True)
    sample = processor.preprocess_action({}, raw, action_normalizer=normalizer)
    # This is the same inverse used by generate_samples_from_batch before returning actions.
    decoded = ActionProcessor.postprocess_action(sample['action'], sample['action_processing_record'])
    actions = runtime_actions(decoded.numpy(), list(range(52)), 4)
    np.testing.assert_allclose(actions, raw[1:5].numpy(), rtol=0, atol=3e-7)
