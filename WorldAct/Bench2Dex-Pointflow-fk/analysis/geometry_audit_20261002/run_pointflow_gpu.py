"""Single-GPU EFEP diagnostic. Saves raw outputs; never fits to FK or true depth."""
import argparse
import os
from pathlib import Path
import sys
import json
import time

ROOT = Path('/data/shichaojian/Track4World_portable/Track4World')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TRACK4WORLD_DA3_MODEL', '/data/shichaojian/checkpoints/DA3NESTED-GIANT-LARGE-1.1')
sys.path[:0] = [str(ROOT), str(ROOT.parent)]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-backbone', action='store_true',
                        help='Explicit ablation; historical export does not record this switch.')
    args = parser.parse_args()
    import cv2
    import numpy as np
    import torch
    from types import SimpleNamespace
    from demo import load_model, forward_video3d_pair
    torch.manual_seed(17)
    np.random.seed(17)
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    assert torch.cuda.is_available(), 'A CUDA device must be available.'
    assert torch.cuda.device_count() == 1, 'Select one device with CUDA_VISIBLE_DEVICES.'
    base = Path(__file__).resolve().parent / 'pointflow_validation'
    config = json.loads((ROOT/'track4world/config/eval/v1.json').read_text())
    options = SimpleNamespace(coordinate='world_depthanythingv3',
        ckpt_init='/data/shichaojian/checkpoints/track4world_da3.pth',
        use_original_backbone=args.original_backbone, metric_scale=True, Ts=-1, inference_iters=4)
    tag = 'gpu_original_backbone' if args.original_backbone else 'gpu_track4world_backbone'
    model = load_model(options, config)
    for clip in ['start', 'motion']:
        dest = base/clip/tag
        if (dest/'complete.json').exists():
            print('Already complete:', dest, flush=True)
            continue
        dest.mkdir(exist_ok=True)
        files = sorted((base/clip/'rgb').glob('*.png'))
        assert len(files) == 64
        images = [cv2.cvtColor(cv2.resize(cv2.imread(str(p)), (640,448),
                  interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB) for p in files]
        rgbs = torch.from_numpy(np.stack(images)).permute(0,3,1,2).unsqueeze(0).float().cuda()
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        with torch.inference_mode():
            result, views = forward_video3d_pair(rgbs, model, options)
        shapes = {}
        for key, value in result.items():
            if key == 'rgbs':
                continue
            if torch.is_tensor(value):
                arr = value.detach().cpu().float().numpy()
                np.save(dest/f'{key}.npy', arr)
                shapes[key] = list(arr.shape)
        # Preserve backbone scale/focal diagnostics without imposing true calibration.
        diagnostics = {}
        for key in ['_da3_focal', '_metric_scale']:
            value = getattr(model, key, None)
            if torch.is_tensor(value):
                value = value.detach().cpu().float().numpy().tolist()
            if value is not None:
                diagnostics[key] = value
        geometry = np.load(base/clip/'geometry.npz')
        np.save(dest/'frame_indices.npy', geometry['frame_indices'][list(views)])
        report = dict(clip=clip, seed=17, options=vars(options), config=config,
            shapes=shapes, diagnostics=diagnostics, seconds=time.time()-started,
            peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9,
            gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__,
            note='Raw 64-frame EFEP output; no SAM filtering or FK calibration. Historical backbone switch remains unverified.')
        (dest/'complete.json').write_text(json.dumps(report,indent=2))
        print('COMPLETE', dest, flush=True)
        del result, rgbs
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
