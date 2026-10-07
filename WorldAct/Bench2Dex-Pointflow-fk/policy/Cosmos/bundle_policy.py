"""Use the supplied training source's Bench2Dex batch construction and loader."""
import json
import os
from pathlib import Path
import sys
import time
import typing

from policy.Cosmos.contract import load_contract, prepare_observation


class BundlePolicy:
    def __init__(self, config):
        if config.get('fixture_only'):
            raise ValueError('Cannot run an input fixture as a trained policy')
        self.contract = load_contract(config['contract_path'])
        os.environ['COSMOS_TRAINING'] = '0'
        os.environ['COSMOS_FLASH2_VARLEN'] = '1'
        # Python 3.11 lacks the typing-only decorator used by the source.
        if not hasattr(typing, 'override'):
            from typing_extensions import override
            typing.override = override
        sys.path.insert(0, str(Path(config['worldact_root']).resolve()))
        from cosmos_framework.inference.common.init import init_script
        init_script(training=False)
        from cosmos_framework.inference.robot_policy.bench2dex import Bench2DexPolicy
        options = json.loads(Path(config['bundle_deployment']).read_text())
        options.update(checkpoint_path=config['checkpoint_path'],
                       config_file=config['training_config_path'],
                       execution_horizon=int(config.get('execute_steps', 4)),
                       num_steps=int(config.get('num_steps', 4)),
                       guidance=float(config.get('guidance', 3)),
                       output_dir=config['output_dir'])
        self.backend = Bench2DexPolicy(options)
        self.profile_path = Path(config['output_dir']) / 'local_inference.jsonl'
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)

    def reset(self, seed=None):
        with self.backend._lock:
            if seed is not None:
                self.backend.seed = int(seed)
            self.backend.reset_model()

    def get_action(self, observation):
        import torch
        prepare_observation(observation, self.contract)
        observation = dict(observation)
        observation['joint_action'] = dict(observation['joint_action'],
                                          qpos=observation['joint_action']['vector'])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        actions = self.backend.get_action(observation)
        torch.cuda.synchronize()
        profile = dict(seconds=time.perf_counter()-started, returned_actions=len(actions),
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                       cuda_free_bytes=torch.cuda.mem_get_info()[0])
        with self.profile_path.open('a') as stream:
            stream.write(json.dumps(profile)+'\n')
        print('[cosmos-local] '+json.dumps(profile), flush=True)
        return actions
