"""One observation -> one action chunk. No robot actuation is performed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from policy.Cosmos.contract import CAMERAS, load_contract, prepare_observation
from script.policy_rpc import RemotePolicyClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observation', type=Path, required=True,
                        help='NPZ: qpos [52], joint_names [52] Unicode, and three camera RGB arrays')
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=9000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.observation, allow_pickle=False) as data:
        obs = dict(joint_names=data['joint_names'].tolist(),
                   joint_action={'vector': data['qpos']},
                   available_camera_ids=list(CAMERAS),
                   language=json.loads(args.manifest.read_text())['task_text'],
                   observation={camera: {'rgb': data[camera]} for camera in CAMERAS})
    prepare_observation(obs, load_contract(args.contract))
    client = RemotePolicyClient(args.host, args.port, timeout_s=600)
    try:
        client.reset_model(args.seed)  # Once per episode, not once per control step.
        actions = np.asarray(client.get_action(obs), dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 52 or not 1 <= len(actions) <= 32 or not np.isfinite(actions).all():
            raise ValueError(f'Invalid action chunk: {actions.shape}')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('wb') as stream:
            np.save(stream, actions)
        print(f'Saved {actions.shape} absolute joint targets in input joint_names order: {args.output}')
    finally:
        client.close()


if __name__ == '__main__':
    main()
