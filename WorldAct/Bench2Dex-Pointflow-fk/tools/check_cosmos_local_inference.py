"""Check real RGB/qpos input without weights, or measure a real local model request."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import h5py
import numpy as np

from policy.Cosmos.contract import CAMERAS, load_contract, prepare_observation
from policy.Cosmos.deploy_policy import load_compositor, raw_sample, get_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--hdf5', type=Path, required=True)
    parser.add_argument('--frame', type=int, default=1)
    parser.add_argument('--run-model', action='store_true', help='Load actual weights and execute one CUDA request')
    parser.add_argument('--output', type=Path, default=Path('outputs/cosmos_local/input_check.json'))
    args = parser.parse_args()
    from script.config_utils import load_config_with_overrides
    config = load_config_with_overrides(args.config)
    contract = load_contract(config['contract_path'])
    with h5py.File(args.hdf5) as source:
        names = [value.decode() for value in source['robot/joint_names'][:]]
        observation = dict(joint_names=names, available_camera_ids=list(CAMERAS),
            joint_action=dict(vector=source['robot/qpos'][args.frame]),
            language=source['meta/instruction'][()].decode(), observation={})
        for camera in CAMERAS:
            bgr = cv2.imdecode(source[f'cameras/{camera}/rgb'][args.frame], cv2.IMREAD_COLOR)
            observation['observation'][camera] = dict(rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    prepared = prepare_observation(observation, contract)
    if config.get('backend') == 'training_bundle':
        import runpy
        import torch
        official = runpy.run_path(str(Path(config['worldact_root']) / 'cosmos_framework/utils/bench2dex_contract.py'))
        def compose(*images):
            rgb = {camera: value[0].permute(1, 2, 0).numpy() for camera, value in zip(CAMERAS, images)}
            return torch.from_numpy(official['compose_rgb'](rgb))[None]
    else:
        compose = load_compositor(config['worldact_root'])
    sample = raw_sample(prepared, contract, compose, domain_id=0)  # domain lookup belongs to actual model run
    output = dict(status='input contract passed', video_shape=list(sample['video'].shape),
        action_shape=list(sample['action'].shape), future_video_zero=bool((sample['video'][:, 1:] == 0).all()),
        future_action_zero=bool((sample['action'][1:] == 0).all()), model_executed=False)
    if args.run_model:
        model = get_model(config)
        model.reset(int(config.get('seed', 0)))
        prediction = model.get_action(observation)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output.with_suffix('.actions.npy'), prediction)
        output.update(status='actual model inference passed', model_executed=True,
                      returned_action_shape=list(prediction.shape), profile=str(model.profile_path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
