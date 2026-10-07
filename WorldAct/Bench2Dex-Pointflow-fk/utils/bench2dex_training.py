"""First-window overfit sample for the Bench2Dex RGB-D oracle-mask experiment.

Offline trajectories supply future labels only. Clean anchors are independently
recomputed from current mask queries and current depth, never tracker confidence.
"""
import json
from pathlib import Path
import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.rgbd_pointflow import lift_depth


class Bench2DexFirstWindow(Dataset):
    """A single explicit overfit window; not a held-out evaluation dataset."""
    def __init__(self, rgbd, pointflow, queries, steps=32, max_points=500, seed=0):
        self.rgbd, self.pointflow, self.queries = Path(rgbd), Path(pointflow), Path(queries)
        self.steps, self.max_points, self.seed = steps, max_points, seed
        self.query_report = json.loads(self.queries.with_suffix('.json').read_text())
        self.report = json.loads((self.pointflow/'report.json').read_text())
        if Path(self.query_report['input']).resolve() != self.rgbd.resolve():
            raise ValueError('Query/source mismatch')
        if not self.report.get('uses_gt_query_mask'):
            raise ValueError('This adapter explicitly requires an oracle-mask export')
        self.source_ids = np.load(self.pointflow/'frame_indices.npy')[:steps+1]
        if len(self.source_ids) != steps+1 or not np.all(np.diff(self.source_ids) == 1):
            raise ValueError('Incomplete consecutive training window')
        if self.source_ids[0] != self.query_report['source_frame_id']:
            raise ValueError('Only the query birth window has an independent current anchor')

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index != 0:
            raise IndexError('Only the first window is currently prepared for overfit')
        from cosmos_framework.data.pointflow_window import prepare_window, PointFlowTiming
        from cosmos_framework.data.pointflow_dataset import pointflow_sample
        with np.load(self.queries) as q:
            query_uv = q['query_uv']
        with h5py.File(self.rgbd) as f:
            ids = f['time/source_frame_id'][:]
            rows = np.searchsorted(ids, self.source_ids)
            np.testing.assert_array_equal(ids[rows], self.source_ids)
            cam = f[f"cameras/{self.query_report['camera']}"]
            K = cam['intrinsic'][:]
            _, anchor_valid, anchor_xyz = lift_depth(query_uv, cam['depth'][rows[0]], K)
            fps = float(f['meta/fps'][()])
            seconds = f['time/sim_step'][rows]*float(f['meta/physics_dt'][()])
            np.testing.assert_allclose(np.diff(seconds), 1/fps, atol=1e-8)
            bgr = [cv2.imdecode(cam['rgb'][row], cv2.IMREAD_COLOR) for row in rows]
            anchor = dict(position=anchor_xyz, uv_px=query_uv, valid=anchor_valid,
                          source_frame_id=int(self.source_ids[0]))
            timing = PointFlowTiming(fps=fps, steps=self.steps, steps_per_token=4)
            window = prepare_window(self.pointflow, start_frame=int(self.source_ids[0]),
                max_points=self.max_points, seed=self.seed, timing=timing,
                causal_anchor=anchor, anchor_frame_bgr=bgr[0], select_top_n=0,
                min_voxel_members=0, select_min_valid_steps=0, select_phantom_guard=False)
            point = pointflow_sample(window, 'task21_episode000000', int(self.source_ids[0]), self.seed, timing)
            point['metadata'].update(geometry_source='current_RGBD_anchor_and_offline_RGB_tracking_targets',
                uses_gt_query_mask=True, uses_gt_xyz=False, conditioning_uses_future_frames=False,
                future_labels_use_future_frames=True, uv_to_video=np.array([[1, 0, 0], [0, 1, 0]], np.float32),
                video_size_wh=np.array([bgr[0].shape[1], bgr[0].shape[0]]))
            state_names = [x.decode() for x in f['robot/joint_names'][:]]
            action_names = [x.decode() for x in f['action/action_names'][:]]
            if len(set(state_names)) != 52 or set(state_names) != set(action_names):
                raise ValueError('Expected matching 52D state/action joint names')
            order = [action_names.index(name) for name in state_names]
            if not f['action/action_valid'][rows[:-1]].all():
                raise ValueError('Invalid action rows in the overfit window')
            state = f['robot/qpos'][rows[0]].astype(np.float32)
            commands = f['action/commanded'][rows[:-1]][:, order].astype(np.float32)
            actions = np.concatenate([state[None], commands])
            if not np.isfinite(actions).all():
                raise ValueError('Nonfinite state/action')
            video = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in bgr])
            return dict(ai_caption=f['meta/instruction'][()].decode(),
                video=torch.from_numpy(video).permute(3, 0, 1, 2).contiguous(),
                action=torch.from_numpy(actions), conditioning_fps=torch.tensor(int(fps)),
                mode='wam', viewpoint='cam_overhead', additional_view_description='Fixed overhead RGB-D camera.',
                pointflow=point, episode_name='task21_episode000000', raw_frame_ids=self.source_ids.copy(),
                action_frame_ids=self.source_ids[:-1].copy(), timestamps_sec=seconds-seconds[0],
                joint_names=state_names, action_type='absolute_joint_position_radians',
                split_purpose='single-window overfit; not validation or benchmark evidence')


class Bench2DexWindowDataset(Dataset):
    """Read converted current-anchor windows with episode-level split membership."""
    def __init__(self, manifest, split='train', fk_root=None):
        if split not in {'train', 'validation'}:
            raise ValueError(split)
        self.root = Path(manifest).resolve().parent
        document = json.loads(Path(manifest).read_text())
        if document.get('schema') != 'bench2dex_rgbd_oracle_mask_windows_v1':
            raise ValueError('Unsupported manifest')
        train = {r['episode'] for r in document['windows'] if r['split'] == 'train'}
        validation = {r['episode'] for r in document['windows'] if r['split'] == 'validation'}
        if train & validation:
            raise ValueError('Episode leakage across splits')
        self.rows = [r for r in document['windows'] if r['split'] == split]
        self.episodes = {e['episode']: e for e in document['episodes']}
        self.fps, self.steps, self.split = document['fps'], document['window_steps'], split
        fk_path = fk_root if fk_root is not None else document.get('fk_root')
        self.fk_root = self.resolve_path(fk_path) if fk_path is not None else None
        if not self.rows:
            raise ValueError(f'Empty {split} split')

    def __len__(self):
        return len(self.rows)

    def resolve_path(self, value):
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def __getitem__(self, index):
        from cosmos_framework.data.pointflow_window import PointFlowTiming
        from cosmos_framework.data.pointflow_dataset import pointflow_sample
        row = self.rows[index]; path = self.resolve_path(row['path']); episode = self.episodes[row['episode']]
        with np.load(path/'worldact_window.npz') as archive:
            window = {k: archive[k] for k in archive.files}
        timing = PointFlowTiming(fps=self.fps, steps=self.steps, steps_per_token=4)
        point = pointflow_sample(window, row['episode'], row['source_frame_start'], 0, timing)
        point['metadata'].update(geometry_source='current_RGBD_anchor_offline_RGB_tracking_targets',
            source_path=str(path), uses_gt_query_mask=True, conditioning_uses_future_frames=False,
            uv_to_video=np.array([[1, 0, 0], [0, 1, 0]], np.float32), video_size_wh=window['image_size_wh'].copy())
        with np.load(path/'action.npz') as action:
            action_values = np.concatenate([action['state'][None], action['action']])
            action_ids = action['action_frame_ids'].copy()
            np.testing.assert_array_equal(action['source_frame_ids'], window['raw_frame_ids'])
        with h5py.File(self.resolve_path(episode['rgbd'])) as f:
            ids = f['time/source_frame_id'][:]
            source_rows = np.searchsorted(ids, window['raw_frame_ids'])
            np.testing.assert_array_equal(ids[source_rows], window['raw_frame_ids'])
            video = np.stack([cv2.cvtColor(cv2.imdecode(f['cameras/cam_overhead/rgb'][r], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for r in source_rows])
            caption = f['meta/instruction'][()].decode()
        sample = dict(video=torch.from_numpy(video).permute(3, 0, 1, 2).contiguous(),
            action=torch.from_numpy(action_values.astype(np.float32)), ai_caption=caption,
            pointflow=point, conditioning_fps=torch.tensor(int(self.fps)), mode='wam', viewpoint='cam_overhead',
            additional_view_description='Fixed overhead RGB-D camera.', episode_name=row['episode'],
            raw_frame_ids=window['raw_frame_ids'], action_frame_ids=action_ids,
            timestamps_sec=window['timestamps_sec'], joint_names=episode['joint_names'],
            action_type='absolute_joint_position_radians', split=self.split)
        if self.fk_root is not None:
            from utils.sim_fk21 import fk_window_sample
            with np.load(self.fk_root/f'{row["episode"]}.npz') as archive:
                sample['fk'] = fk_window_sample(archive, window['raw_frame_ids'], row['episode'], self.fps)
        return sample
