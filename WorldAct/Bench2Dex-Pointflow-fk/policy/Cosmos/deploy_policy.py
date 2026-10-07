"""Local Cosmos model process, compatible with Bench2Dex's existing loopback RPC."""
import ast
import json
from pathlib import Path
import sys
import time

import numpy as np

from policy.Cosmos.contract import load_contract, prepare_observation, runtime_actions


def load_compositor(worldact):
    import torch
    import torch.nn.functional as F
    import torchvision.transforms.functional as transforms_F
    source = Path(worldact) / 'tools/prepare_dualhand_joint_video_cache.py'
    tree = ast.parse(source.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {'_resize_to_width', 'compose_dualhand_views'}]
    if len(functions) != 2:
        raise ValueError(f'Missing official dual-hand compositor in {source}')
    namespace = dict(torch=torch, F=F, transforms_F=transforms_F)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['compose_dualhand_views']


def raw_sample(prepared, contract, compose, domain_id):
    import torch
    images = [torch.from_numpy(image.copy()).permute(2, 0, 1)[None] for image in prepared['images']]
    image = compose(*images)[0]
    if image.shape != (3, 720, 640):
        raise ValueError(f'Unexpected compositor shape {image.shape}')
    video = torch.zeros((3, 33, 720, 640), dtype=torch.uint8)
    video[:, 0] = image
    action = torch.zeros((33, 52), dtype=torch.float32)
    action[0] = torch.from_numpy(prepared['state'])
    return dict(ai_caption=prepared['prompt'], video=video, action=action,
        conditioning_fps=torch.tensor(contract['fps']), mode='wam',
        domain_id=torch.tensor(domain_id, dtype=torch.long), viewpoint='concat_view',
        additional_view_description=contract['additional_view_description'])


class CosmosPolicy:
    def __init__(self, config):
        if config.get('fixture_only'):
            raise ValueError('Input-only test fixture is not a baseline training contract')
        for key in ['contract_path', 'worldact_root', 'checkpoint_path']:
            if not config.get(key):
                raise ValueError(f'Supply {key} from your trained baseline artifacts')
        self.contract = load_contract(config['contract_path'])
        self.execute_steps = int(config.get('execute_steps', 4))
        if not 1 <= self.execute_steps <= 32:
            raise ValueError('execute_steps must be 1..32')
        worldact = Path(config['worldact_root']).expanduser().resolve()
        checkpoint = Path(config['checkpoint_path']).expanduser().resolve()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f'Local checkpoint directory missing: {checkpoint}')
        if not (worldact / 'cosmos_framework').is_dir():
            raise FileNotFoundError(worldact)
        sys.path.insert(0, str(worldact))
        self.compose = load_compositor(worldact)
        # The original SFT service owns runtime initialization and checkpoint loading.
        from cosmos_framework.scripts.action_policy_server_robolab import (
            RobolabPolicyService, RobolabServerArgs, _build_data_batch_from_sample)
        from cosmos_framework.inference.args import OmniSetupOverrides
        from cosmos_framework.inference.common.init import init_output_dir
        from cosmos_framework.scripts.action_policy_server_utils import disable_runtime_ema_for_frozen_config
        from cosmos_framework.data.generator.action.domain_utils import get_domain_id
        from cosmos_framework.data.generator.action.action_processing import ActionAffineNormalization
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for real Cosmos inference')
        self.domain_id = get_domain_id(self.contract['domain_name'])
        contract = self.contract
        explicit_training_config = config.get('training_config_path')
        if explicit_training_config:
            explicit_training_config = str(Path(explicit_training_config).expanduser().resolve())
            if not Path(explicit_training_config).is_file():
                raise FileNotFoundError(explicit_training_config)

        class Bench2DexService(RobolabPolicyService):
            def _build_setup_args(service, args):
                if not explicit_training_config:
                    return super()._build_setup_args(args)
                overrides = dict(checkpoint_path=args.checkpoint_path,
                    config_file=explicit_training_config, output_dir=args.output_dir,
                    sampler=args.sampler)
                if args.experiment is not None:
                    overrides['experiment'] = args.experiment
                if args.experiment_overrides:
                    overrides['experiment_overrides'] = list(args.experiment_overrides)
                setup = OmniSetupOverrides.model_validate(overrides).build_setup()
                init_output_dir(setup.output_dir)
                return disable_runtime_ema_for_frozen_config(setup)

            def _build_transform(service, training_config, args):
                if explicit_training_config:
                    training_config = service.setup_args.load_config()
                if training_config is None:
                    raise ValueError('Training transform config is required; refusing DROID defaults')
                dataset = training_config.dataloader_train.dataloaders.action_data.dataloader.dataset
                if len(dataset.list_of_datasets) != 1:
                    raise ValueError('Expected one training action dataset; select the correct transform explicitly')
                entry = dataset.list_of_datasets[0]
                for field, expected in [('fps', contract['fps']), ('chunk_length', contract['action_chunk_size'])]:
                    found = getattr(entry.dataset, field, None)
                    if found is not None and float(found) != expected:
                        raise ValueError(f'Training {field}={found} differs from contract {expected}')
                resolution = entry.resolution if entry.resolution is not None else dataset.resolution
                if resolution is not None and str(resolution) != str(contract['resolution']):
                    raise ValueError('Training resolution differs from inference contract')
                return super()._build_transform(training_config, args)

        args = RobolabServerArgs(checkpoint_path=str(checkpoint),
            allow_dcp_checkpoint=bool(config.get('allow_dcp_checkpoint', False)),
            experiment=config.get('experiment'), experiment_overrides=config.get('experiment_overrides', []),
            domain_name=contract['domain_name'], decode_video=False,
            output_dir=Path(config.get('output_dir', 'outputs/cosmos_local/model')),
            sampler=config.get('sampler', 'unipc'), seed=int(config.get('seed', 0)),
            guidance=float(config.get('guidance', 3.0)), num_steps=int(config.get('num_steps', 4)),
            shift=float(config.get('shift', 5.0)), resolution=str(contract['resolution']),
            conditioning_fps=contract['fps'], action_chunk_size=32, action_dim=52,
            image_height=720, image_width=640, action_space='joint_pos', use_state=True, history_length=1)
        self.service = Bench2DexService(args)
        self.batch_from_sample = _build_data_batch_from_sample
        normalization = contract['normalization']
        self.normalizer = None if normalization['kind'] == 'none' else ActionAffineNormalization(
            offset=torch.tensor(normalization['offset'], dtype=torch.float32),
            scale=torch.tensor(normalization['scale'], dtype=torch.float32))
        self.profile_path = Path(args.output_dir) / 'local_inference.jsonl'
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)

    def reset(self, seed=None):
        if seed is not None:
            with self.service._lock:
                self.service._rng = np.random.default_rng(int(seed))

    def get_action(self, observation):
        import torch
        prepared = prepare_observation(observation, self.contract)
        with self.service._lock, torch.inference_mode():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            sample = raw_sample(prepared, self.contract, self.compose, self.domain_id)
            sample = self.service._transform(sample, self.service.cfg.resolution,
                                             action_normalizer=self.normalizer)
            if isinstance(sample.get('ai_caption'), dict):
                sample['ai_caption'] = json.dumps(sample['ai_caption'])
            batch = self.batch_from_sample(sample)
            seed = self.service._next_seed()
            outputs = self.service.model.generate_samples_from_batch(batch,
                guidance=self.service.cfg.guidance, seed=[seed], num_steps=self.service.cfg.num_steps,
                shift=self.service.cfg.shift)
            # Official generate_samples_from_batch already applies ActionProcessor.postprocess_action.
            prediction = outputs['action'][0].detach().float().cpu().numpy()
            actions = runtime_actions(prediction, prepared['output_order'], self.execute_steps)
            torch.cuda.synchronize()
            profile = dict(seconds=time.perf_counter()-started, seed=seed, returned_actions=len(actions),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                cuda_free_bytes=torch.cuda.mem_get_info()[0])
            with self.profile_path.open('a') as stream:
                stream.write(json.dumps(profile) + '\n')
            print('[cosmos-local] ' + json.dumps(profile), flush=True)
            return actions


def get_model(config):
    if config.get('backend') == 'training_bundle':
        from policy.Cosmos.bundle_policy import BundlePolicy
        return BundlePolicy(config)
    return CosmosPolicy(config)


class LocalSession:
    def __init__(self, config):
        self.model = get_model(config)
        self.instruction = None

    def reset(self, instruction=None, *, seed=None):
        self.instruction = instruction
        self.model.reset(seed)

    def get_action_chunk(self, observation, instruction=None):
        observation = dict(observation)
        if instruction or self.instruction:
            observation['language'] = instruction or self.instruction
        return self.model.get_action(observation)

    def update_after_action(self, observation, instruction=None):
        return None

    def close(self):
        return None


def create_remote_session(config):
    # "remote" is the framework hook name; both processes run on 127.0.0.1.
    return LocalSession(config)
