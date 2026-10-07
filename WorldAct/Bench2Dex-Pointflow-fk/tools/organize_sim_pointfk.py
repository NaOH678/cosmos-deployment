"""Arrange the portable simulation export following pointflow-fk's data pipeline."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import h5py
import numpy as np


def write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('exports/bench2dex_task21_dense10_v1'))
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / 'code/worldact'))
    from cosmos_framework.data.pointflow_window_cache import write_episode_archive, write_manifest
    manifest = json.loads((root / 'manifest.json').read_text())
    cache = root / 'datasets/bench2dex-task21-cosmos-cache'
    for name in ['episodes', 'video_frames', 'pointflow_windows']:
        (cache / name).mkdir(parents=True, exist_ok=True)
    config = dict(fps=20, chunk_length=32, sample_stride=32, max_points=16384, voxel_size=.02,
        seed=0, select_motion_fraction=0., select_top_n=0, min_voxel_members=0, supervise_cluster_n=0,
        select_regions=[], select_region_quotas=[], select_min_valid_steps=0, select_phantom_guard=False,
        phantom_guard_disp_mm=30., phantom_guard_uv_px=2.)
    write_manifest(cache / 'pointflow_windows', config)
    cache_rows, video_rows, pf_rows, window_rows = [], [], [], []
    pf_sum, pf_sq, pf_count = np.zeros((32, 3)), np.zeros((32, 3)), np.zeros((32, 3), dtype=np.int64)
    fk_sum, fk_sq, fk_count = 0., 0., 0
    for episode in manifest['episodes']:
        name = episode['episode']
        raw = root / 'raw_data/bench2dex_task21' / name
        (raw / 'videos').mkdir(parents=True, exist_ok=True)
        (raw / 'annotations').mkdir(exist_ok=True)
        with np.load(root / 'fk21_urdf' / f'{name}.npz') as fk:
            annotation = {key: fk[key] for key in fk.files}
        annotation['positions'] = annotation['positions_camera']
        annotation['side_is_observed'] = np.array([True, True])
        np.savez_compressed(raw / 'annotations/wuji_fk21.npz', **annotation)
        with h5py.File(root / episode['rgbd']) as source:
            ids = source['time/source_frame_id'][:]
            np.testing.assert_array_equal(ids, np.arange(len(ids)))
            names = [s.decode() for s in source['robot/joint_names'][:]]
            action_names = [s.decode() for s in source['action/action_names'][:]]
            order = [action_names.index(n) for n in names]
            task = source['meta/instruction'][()].decode()
            timestamps = source['time/sim_step'][:] * float(source['meta/physics_dt'][()])
            np.savez_compressed(cache / 'episodes' / f'{name}.npz', state=source['robot/qpos'][:],
                action=source['action/commanded'][:][:, order], action_valid=source['action/action_valid'][:],
                source_frame_ids=ids, timestamps_sec=timestamps, joint_names=np.array(names))
            frames = np.lib.format.open_memmap(cache / 'video_frames' / f'{name}.npy', mode='w+',
                                               dtype=np.uint8, shape=(len(ids), 3, 480, 640))
            encoder = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '640x480', '-r', '20', '-i', '-',
                '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart', str(raw / 'videos/head.mp4')], stdin=subprocess.PIPE)
            try:
                for row in range(len(ids)):
                    bgr = cv2.imdecode(source['cameras/cam_overhead/rgb'][row], cv2.IMREAD_COLOR)
                    frames[row] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
                    encoder.stdin.write(bgr.tobytes())
            finally:
                encoder.stdin.close()
                code = encoder.wait()
            assert code == 0
            frames.flush()
            del frames
        shutil.move(str(root / episode['rgbd']), raw / 'observations.hdf5')
        episode['rgbd'] = str((raw / 'observations.hdf5').relative_to(root))
        cache_rows.append(dict(name=name, num_frames=len(ids), source_fps=20., task_text=task,
                               split=episode['split'], joint_names=names))
        video_rows.append(dict(name=name, path=f'video_frames/{name}.npy', shape=[len(ids), 3, 480, 640],
                               image_size=[480, 640, 480, 640]))
        windows = {}
        offsets = []
        for row in [r for r in manifest['windows'] if r['episode'] == name]:
            path = root / row['path']
            with np.load(path / 'worldact_window.npz') as archive:
                window = {key: archive[key] for key in archive.files}
            start = row['source_frame_start']
            windows[start] = window
            offsets.extend((window['timestamps_sec'] - window['raw_frame_ids'] / 20).tolist())
            if row['split'] == 'train':
                displacement = window['target_displacement'].astype(np.float64)
                valid = window['target_valid'][..., None]
                pf_sum += np.where(valid, displacement, 0).sum(axis=1)
                pf_sq += np.where(valid, displacement ** 2, 0).sum(axis=1)
                pf_count += np.broadcast_to(valid.sum(axis=1), (32, 3))
                fk_xyz = annotation['positions_camera'][window['raw_frame_ids']].astype(np.float64)
                fk_d = fk_xyz[1:] - fk_xyz[:1]
                fk_sum += fk_d.sum(); fk_sq += (fk_d ** 2).sum(); fk_count += fk_d.size
            dest = root / 'pf_out/bench2dex_task21/labeled' / name / 'windows' / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), dest)
            row['path'] = str(dest.relative_to(root))
            # The canonical native archive now owns the prepared representation.
            (dest / 'worldact_window.npz').unlink()
            original_report = Path('outputs/sim_pointflow/batch10_dense_oracle') / name / 'windows' / path.name / 'report.json'
            report = json.loads(original_report.read_text())
            report.update(video=str(Path('../../../../../../raw_data/bench2dex_task21') / name / 'videos/head.mp4'),
                          video_path_relative_to='report directory', track_id_scope='this 33-frame window')
            for old in ['input', 'output', 'checkpoint']:
                report.pop(old, None)
            write(dest / 'report.json', report)
            window_rows.append(dict(episode=name, split=row['split'], start_frame=start,
                                    frame_ids=window['raw_frame_ids'].tolist(), raw_path=row['path']))
        assert np.ptp(offsets) < 1e-4
        write_episode_archive(cache / 'pointflow_windows' / f'{name}.npz', windows)
        pf_rows.append(dict(name=name, pointflow_source=dict(
            path=f'../../pf_out/bench2dex_task21/labeled/{name}', uv_to_video=[[1, 0, 0], [0, 1, 0]],
            video_size_wh=[640, 480], timestamp_offset_sec=float(np.mean(offsets)))))
        write(root / 'pf_out/bench2dex_task21/labeled' / name / 'report.json', dict(
            layout='window_local_flat_labeled', track_id_scope='independent within each window',
            training_source='canonical PointFlow window cache; exact starts in window_index.json',
            uses_gt_query_mask=True, conditioning_uses_future_frames=False,
            future_labels_use_future_frames=True))
        print(f'pointflow-fk layout: {name}, {len(windows)} windows, {len(ids)} RGB frames', flush=True)
    write(cache / 'manifest.json', dict(schema_version=2, raw_root='../../raw_data/bench2dex_task21',
        task_text=cache_rows[0]['task_text'], arm_action_space='joint', embodiment='bench2dex_wuji',
        state_layout='52 joints in joint_names order, radians', action_layout='52 absolute joint targets, radians',
        episodes=cache_rows))
    write(cache / 'video_manifest.json', dict(schema_version=1, resolution='native_640x480', dtype='uint8',
        layout='TCHW', padding='deferred_to_ActionTransformPipeline', episodes=video_rows))
    write(cache / 'window_index.json', dict(schema_version=1, fps=20, chunk_length=32,
        enumeration='explicit starts: first valid action, stride32, final overlapping tail; no arbitrary starts', windows=window_rows))
    write(root / 'pointflow_outputs/bench2dex_task21/manifest.json', dict(schema_version=1, episodes=pf_rows))
    for split in ('train', 'validation', 'all'):
        names = [row['name'] for row in cache_rows if split == 'all' or row['split'] == split]
        path = root / 'examples' / f'pointflow_bench2dex_task21_{split}_episodes.txt'
        path.parent.mkdir(exist_ok=True)
        path.write_text('\n'.join(names) + '\n')
    scales = np.sqrt(np.maximum(pf_sq / pf_count - (pf_sum / pf_count) ** 2, 0))
    assert (scales > 0).all()
    scalar = float(np.sqrt(pf_sq.sum() / pf_count.sum() - (pf_sum.sum() / pf_count.sum()) ** 2))
    fk_scale = float(np.sqrt(fk_sq / fk_count - (fk_sum / fk_count) ** 2))
    write(root / 'pointflow_outputs/bench2dex_task21/train_frame_scales_env.json', dict(steps=32,
        channels=3, order='frame-major x,y,z', stat='std', selection='dense current anchor, every training window',
        scales=scales.reshape(-1).tolist()))
    write(root / 'pointflow_outputs/bench2dex_task21/train_scale_audit.json', dict(
        windows=177, validation_used=False, pointflow_std_metres=scalar, fk_std_metres=fk_scale,
        pointflow_per_frame_channel_counts=pf_count.tolist(), selection_config=config))
    write(root / 'manifest.json', manifest)
    for split in ['train', 'validation']:
        write(root / f'{split}_windows.json', [row for row in window_rows if row['split'] == split])
    shutil.copy2('utils/sim_pointfk_dataset.py', root / 'code/bench2dex/utils/sim_pointfk_dataset.py')
    shutil.copy2('tools/organize_sim_pointfk.py', root / 'organize_pointfk_source.py')
    print(f'Native cache complete; PointFlow std={scalar}, FK std={fk_scale}', flush=True)


if __name__ == '__main__':
    main()
