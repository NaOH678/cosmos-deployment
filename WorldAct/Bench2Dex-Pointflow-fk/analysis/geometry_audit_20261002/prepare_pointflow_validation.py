"""CPU-only preparation; does not import torch or run inference.

Exports lossless RGB, all-link FK and +/-1 frame hypotheses. No fitted alignment.
"""
from pathlib import Path
import ast
import hashlib
import json
import xml.etree.ElementTree as ET
import cv2
import h5py
import numpy as np

OUT = Path(__file__).resolve().parent / 'pointflow_validation'
OUT.mkdir(exist_ok=True)
SOURCE = Path('/tmp/bench2dex_replay21_ep0.hdf5')
URDF = Path('/tmp/bench2dex_wuji.urdf')
# Reuse audited pure kinematics functions without executing the USD audit.
tree = ast.parse((OUT.parent / 'audit_geometry.py').read_text())
scope = {'np': np}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in {'rot', 'rpy', 'tf', 'fk'}:
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'audited_fk', 'exec'), scope)
tf, rpy = scope['tf'], scope['rpy']
joints = {}
for j in ET.parse(URDF).getroot().findall('joint'):
    o, a = j.find('origin'), j.find('axis')
    xyz = np.fromstring(o.get('xyz', '0 0 0'), sep=' ') if o is not None else np.zeros(3)
    angles = np.fromstring(o.get('rpy', '0 0 0'), sep=' ') if o is not None else np.zeros(3)
    joints[j.get('name')] = dict(parent=j.find('parent').get('link'), child=j.find('child').get('link'),
        origin=tf(rpy(angles), xyz), axis=np.fromstring(a.get('xyz'), sep=' ') if a is not None else None,
        type=j.get('type'))
scope['uj'] = joints
fk = scope['fk']
C = np.array([[0., -1, 0], [0, 0, -1], [1, 0, 0]])
Tbase = tf(rpy([0, 0, np.pi / 2]), [.5, -.43, .75])
with h5py.File(SOURCE) as f:
    q = f['robot/qpos'][:]
    names = [x.decode() for x in f['robot/joint_names'][:]]
    nodes = [fk(dict(zip(names, row))) for row in q]
    links = sorted(nodes[0])
    xyz = np.array([[n[k][:3, 3] for k in links] for n in nodes])
    world = xyz @ Tbase[:3, :3].T + Tbase[:3, 3]
    hand_ids = [i for i, k in enumerate(links) if 'finger' in k or 'palm' in k]
    # Select a contiguous 64-frame window by total hand travel, plus a start control.
    speed = np.linalg.norm(np.diff(world[:, hand_ids], axis=0), axis=-1).mean(axis=1)
    start = int(np.argmax(np.convolve(speed, np.ones(63), mode='valid')))
    fps = float(f['meta/effective_fps'][()])
    manifest = {'source': str(SOURCE), 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'urdf_sha256': hashlib.sha256(URDF.read_bytes()).hexdigest(), 'fps': fps,
        'inference_performed': False, 'gpu_used': False,
        'historical_settings': {'mode': '3d_efep', 'coordinate': 'world_depthanythingv3',
            'metric_scale': True, 'infer_width': 640, 'infer_height': 448,
            'inference_call': 'model.infer_pair(force_projection=True)'},
        'pending': ['Confirm original backbone switch and exporter settings before inference.',
            'RGB versus qpos lag is unresolved; offsets are hypotheses, not corrections.',
            'No PointFlow geometry accuracy is measured by this preparation.'], 'clips': []}
    for tag, begin in [('start', 0), ('motion', start)]:
        ids = np.arange(begin, begin + 64)
        d = OUT / tag
        d.mkdir(exist_ok=True)
        image_dir = d / 'rgb'
        image_dir.mkdir(exist_ok=True)
        cam = f['cameras/cam_overhead']
        E = cam['extrinsic_world_from_cam'][ids]
        K = cam['intrinsic'][:]
        candidates, candidate_ids = [], []
        for lag in [-1, 0, 1]:
            qi = ids + lag
            valid = (qi >= 0) & (qi < len(q))
            w = world[np.clip(qi, 0, len(q)-1)]
            p = np.einsum('tni,tij->tnj', w-E[:, None, :3, 3], E[:, :3, :3]) @ C.T
            p[~valid] = np.nan
            candidates.append(p)
            candidate_ids.append(np.where(valid, qi, -1))
        candidates = np.array(candidates)
        panels = []
        for local, raw in enumerate(ids):
            bgr = cv2.imdecode(cam['rgb'][raw], cv2.IMREAD_COLOR)
            assert bgr is not None and bgr.shape == (480, 640, 3)
            assert cv2.imwrite(str(image_dir / f'{local:06d}.png'), bgr)
            if local in [0, 21, 42, 63]:
                row = []
                for li, lag in enumerate([-1, 0, 1]):
                    im = bgr.copy()
                    for side, color in [('left', (255,180,0)), ('right', (0,220,255))]:
                        for finger in range(1,6):
                            chain = [side+'_palm_link']+[f'{side}_finger{finger}_link{k}' for k in range(1,5)]+[f'{side}_finger{finger}_tip_link']
                            p = candidates[li, local, [links.index(k) for k in chain]]
                            uv = p @ K.T
                            if not np.isfinite(uv).all() or (uv[:,2] <= 0).any():
                                continue
                            uv = np.round(uv[:,:2]/uv[:,2:]).astype(np.int32)
                            for a,b in zip(uv[:-1],uv[1:]):
                                cv2.line(im, tuple(a), tuple(b), color, 1, cv2.LINE_AA)
                    cv2.putText(im, f'RGB {raw} | qpos {raw+lag}', (8,22), cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
                    row.append(im)
                panels.append(np.concatenate(row,axis=1))
        np.savez_compressed(d/'geometry.npz', frame_indices=ids, qpos=q[ids], joint_names=names,
            link_names=links, fk_world_m=world[ids], fk_optical_candidates_m=candidates,
            candidate_qpos_indices=candidate_ids, qpos_minus_rgb_offsets=[-1,0,1],
            intrinsic_original=K, extrinsic_world_from_camera_body=E,
            optical_from_camera_body=C, T_world_from_base=Tbase)
        assert cv2.imwrite(str(d/'lag_hypotheses.jpg'), np.concatenate(panels,axis=0))
        manifest['clips'].append({'name':tag, 'first_frame':int(ids[0]), 'last_frame':int(ids[-1]),
            'frames':len(ids), 'camera':'cam_overhead', 'rgb_format':'lossless PNG; original 640x480',
            'camera_extrinsic_max_change':float(np.max(np.abs(E-E[0]))),
            'mean_hand_link_travel_m':float(speed[begin:begin+63].sum())})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))
