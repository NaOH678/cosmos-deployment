"""Finalize ten converted episodes, split membership, loader QA and preview videos."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import h5py
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('outputs/sim_pointflow/batch10_dense_oracle'))
    p.add_argument('--worldact', type=Path, default=Path('../WorldAct-pointflow-fk'))
    p.add_argument('--count', type=int, default=10)
    a = p.parse_args()
    sys.path.insert(0, str(a.worldact.resolve()))
    from utils.bench2dex_training import Bench2DexWindowDataset
    from cosmos_framework.data.pointflow_dataset import collate_pointflow_windows, pointflow_sample
    from cosmos_framework.data.pointflow_window import PointFlowTiming
    reports = [json.loads((a.root/f'episode_{i:06d}'/'episode.json').read_text()) for i in range(a.count)]
    if any(r['status'] != 'converted_and_worldact_windows_checked' for r in reports):
        raise ValueError('Not all episodes are complete')
    assert all(r['fps'] == 20 and r['window_steps'] == 32 for r in reports)
    windows = [w for r in reports for w in r['windows']]
    dynamic = all(r.get('replenish_queries_each_frame', False) for r in reports)
    manifest = dict(schema='bench2dex_rgbd_oracle_mask_windows_v1', fps=20, window_steps=32,
        stride=32, uses_gt_query_mask=True, uses_gt_xyz=False, conditioning_uses_future_frames=False,
        split_method='episodes 000000..000007 train; 000008..000009 validation; no within-episode splitting',
        episodes=[{k: r[k] for k in ['episode', 'split', 'source', 'rgbd', 'joint_names', 'window_count', 'source_frames']} for r in reports], windows=windows)
    manifest['source_download_manifest'] = str(Path('outputs/sim_pointflow/data/task21_10episodes_manifest.json').resolve())
    manifest['dynamic_birth_tracks'] = dynamic
    manifest['training_anchor_semantics'] = 'Current-frame visible points only; later birth tracks preserved in raw arrays, excluded from clean anchor'
    path = a.root/'manifest.json'
    path.write_text(json.dumps(manifest, indent=2)+'\n')
    checks = []
    for split in ['train', 'validation']:
        dataset = Bench2DexWindowDataset(path, split)
        # Every window already passed preparation; now check a fresh loader read
        # at start, middle and end of every episode from the final split manifest.
        for episode in {r['episode'] for r in dataset.rows}:
            index = [i for i, row in enumerate(dataset.rows) if row['episode'] == episode]
            for i in sorted({index[0], index[len(index)//2], index[-1]}):
                sample = dataset[i]
                batch = collate_pointflow_windows([sample['pointflow']])
                assert tuple(sample['video'].shape) == (3, 33, 480, 640)
                assert tuple(sample['action'].shape) == (33, 52)
                assert sample['raw_frame_ids'][-1]-sample['raw_frame_ids'][0] == 32
                checks.append(dict(episode=episode, start=int(sample['raw_frame_ids'][0]), split=split,
                    points=int(batch['inputs']['anchor_xyz'].shape[0]), passed=True))
                del sample, batch
    actions, states = [], []
    for r in reports:
        if r['split'] != 'train':
            continue
        with np.load(a.root/r['episode']/'state_action.npz') as data:
            names = data['joint_names']
            np.testing.assert_array_equal(names, np.array(reports[0]['joint_names']))
            valid = data['action_valid']
            actions.append(data['action'][valid]); states.append(data['state'][valid])
    norm = dict(source='training episodes only; absolute radians; statistics, no clipping applied', joint_names=reports[0]['joint_names'])
    for name, arrays in [('action', actions), ('state', states)]:
        data = np.concatenate(arrays)
        norm[name] = dict(count=len(data), mean=data.mean(0).tolist(), std=data.std(0).tolist(),
                          q01=np.quantile(data, .01, axis=0).tolist(), q99=np.quantile(data, .99, axis=0).tolist())
    (a.root/'normalization_train_only.json').write_text(json.dumps(norm, indent=2)+'\n')
    valid = np.array([w['future_xyz_valid_fraction'] for w in windows])
    summary = dict(status='10 episodes converted; every window packed by WorldAct; sampled final loader checks passed',
        episodes=len(reports), windows=len(windows), original_frames=sum(r['source_frames'] for r in reports),
        train_episodes=8, validation_episodes=2,
        train_windows=sum(w['split'] == 'train' for w in windows), validation_windows=sum(w['split'] == 'validation' for w in windows),
        future_xyz_valid_fraction=dict(min=float(valid.min()), median=float(np.median(valid)), mean=float(valid.mean())),
        low_valid_windows=[w for w in windows if w['future_xyz_valid_fraction'] < .5],
        loader_checks=checks, per_episode=[dict(episode=r['episode'], split=r['split'], windows=r['window_count'],
            mean_xyz_valid_fraction=float(np.mean([w['future_xyz_valid_fraction'] for w in r['windows']]))) for r in reports],
        limitations=['Estimated visibility, not perfect occlusion handling.', 'Conversion/loader QA is not model training or benchmark success.',
            'FK21 mapping and 52D model-domain integration remain separate; raw joint states and actions are preserved.'])
    (a.root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    if dynamic:
        summary['dense_oracle'] = dict(grid_step=reports[0]['conversion_config']['grid_step'],
            training_points_min=min(w['training_points'] for w in windows),
            training_points_max=max(w['training_points'] for w in windows),
            training_points_mean=float(np.mean([w['training_points'] for w in windows])),
            total_later_births=sum(w['later_births'] for w in windows),
            total_later_hand_births=sum(w['later_hand_births'] for w in windows),
            birth_check='Every window: all pre-birth positions zero and invalid; training point IDs have birth frame zero')
        summary['limitations'].append('Later births are stored as dynamic tracks; the fixed-anchor training interface only consumes points visible at its window start.')
        (a.root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    fields = ['episode', 'split', 'source_frame_start', 'source_frame_end', 'hand_queries', 'object_queries',
              'training_points', 'initial_queries', 'later_births', 'later_hand_births', 'future_xyz_valid_fraction', 'last_frame_valid_fraction', 'path']
    with (a.root/'windows.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(windows)
    source_files = [Path(p) for p in ['tools/convert_sim_episode_windows.py', 'utils/sim_query_mask.py',
                    'utils/sim_surface_gt.py', 'utils/rgbd_pointflow.py', 'utils/bench2dex_training.py', 'utils/sim_dynamic_queries.py']]
    provenance = dict(data_manifest=json.loads(Path(manifest['source_download_manifest']).read_text()),
        assets_manifest=json.loads(Path('../dex2bench_dataset/task21_download_manifest.json').read_text()),
        implementation_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        checkpoint_sha256=reports[0]['checkpoint_sha256'])
    (a.root/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    spec = importlib.util.spec_from_file_location('documented_renderer', a.worldact/'tools/visualize_pointflow_selection.py')
    renderer = importlib.util.module_from_spec(spec); spec.loader.exec_module(renderer)
    preview = a.root/'previews'; preview.mkdir(exist_ok=True)
    clips = []
    timing = PointFlowTiming(fps=20, steps=32, steps_per_token=4)
    for report in reports:
        record = report['windows'][len(report['windows'])//2]
        window_path = Path(record['path'])
        with np.load(window_path/'worldact_window.npz') as z:
            win = {k: z[k] for k in z.files}
        sample = pointflow_sample(win, report['episode'], record['source_frame_start'], 0, timing)
        args = SimpleNamespace(output=preview, select_top_n=0, phantom_guard=False, trail_steps=5)
        clip = renderer.render_window(sample, window_path, record['source_frame_start'], args, timing)
        # Label cuts between episodes while preserving the documented rendering.
        target = preview/f'{report["episode"]}_preview.mp4'
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(clip), '-vf',
            f'drawtext=text={report["episode"]}:x=8:y=h-26:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.6',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(target)], check=True)
        clips.append(target.resolve())
    concat = preview/'clips.txt'
    concat.write_text(''.join(f"file '{str(c)}'\n" for c in clips))
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', str(concat),
                    '-c', 'copy', '-movflags', '+faststart', str(a.root/'batch10_preview.mp4')], check=True)
    (a.root/'progress.json').write_text(json.dumps(dict(status='complete', current=None,
        count=len(reports), finished=[dict(episode=r['episode'], windows=r['window_count'], split=r['split']) for r in reports]), indent=2)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ['loader_checks', 'low_valid_windows', 'per_episode']}, indent=2))


if __name__ == '__main__':
    main()
