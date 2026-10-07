#!/usr/bin/env python3
"""Compare FK-21, PointFlow (DA3) hand points and the D435 depth hand, in camera metres.

Companion to ``Bench2Dex/analysis/sandwich_real_audit/visualize_like_benchmark.py``:
that figure plots the FK **hand surface** against PointFlow; this one plots the FK
**skeleton** (the 21 joint centres, wired by the kinematic edges) and, with ``--d435``,
a third cloud from the head camera's own stereo depth -- so the same hand region can be
read from three independent sources instead of two.

WHERE EACH CLOUD COMES FROM, AND WHY THEY ARE COMPARABLE
    PointFlow  ``pf_out/9.24/sandwich/efep_seg_v2/<episode>/`` -- the raw per-frame
               observations.  ``frame_offsets`` slices ``obs_pos`` for one frame; the
               hand is ``obs_label == 2`` (``mesh_check.py:41`` uses the same filter),
               kept only where ``obs_valid & obs_unique``.  DA3 camera frame, metres.
    FK         ``raw_data/sandwich_fk21/<episode>/annotations/wuji_fk21.npz``, asserted
               to be ``Link_Base`` / ``metre``, put into the camera frame by the
               generated ``cosmos_framework/data/fk_camera_extrinsic.py``:
                   p_cam = R @ p_base + t
    D435       ``<episode>/auxiliary_camera/depth.lmdb``, z16 PNG, back-projected with
               the factory depth intrinsics and moved to the colour frame by the
               factory ``extrinsics_to_color`` -- the same three lines as
               ``tools/calibrate_extrinsic_v2.py:243-249``.

    No fitting, no scale search anywhere: the FK extrinsic is the verified one (URDF md5
    checked against the file it was generated from) and the depth path uses only factory
    calibration, so a gap in the picture is a real disagreement.

WHY THE D435 CLOUD IS SAMPLED AT THE POINTFLOW PIXELS
    "Replace DA3's depth with the D435's" only means something if the pixel support is
    held fixed: the D435 value is read at exactly the colour-image pixels PointFlow
    called hand, so any 3D difference is the depth source and not a different choice of
    which pixels are the hand.  Sampling instead from an FK-centred disc would let an FK
    error pick the pixels as well, and the two effects cannot be told apart afterwards.

WHY FRAMES ARE RESTRICTED WITH --d435
    The head depth is recorded at 6 fps against the 30 fps video, so only about a fifth of
    the frames have one.  The spacing is nominally every 5th frame but it drifts (mostly
    5, occasionally 4 or 6), so which frames qualify is read from the lmdb key set, not
    from a modulus -- on episode_0013, 239 of 1192 frames have head depth and their
    residues mod 5 are not all equal.  A frame without depth is refused, not snapped to a
    neighbour: 5 frames is 1/6 s and the hand moves centimetres in that time, so a
    substituted depth frame would compare two different hand poses and read as a depth
    disagreement.

Usage:
    python tools/render_fk_vs_pointflow.py                       # FK vs PointFlow only
    python tools/render_fk_vs_pointflow.py --d435
"""

from __future__ import annotations

import argparse
import hashlib
import runpy
import shutil
import sys
from pathlib import Path

import numpy as np

DATA = Path('/data/shichaojian')
PF_ROOT = DATA / 'pf_out/9.24/sandwich/efep_seg_v2'
FK_ROOT = DATA / 'raw_data/sandwich_fk21'
RAW_ROOT = DATA / 'raw_data/singlerighthand_sandwich_100'
URDF = DATA / 'wuji-mjlab/marvin_wuji_d435_description/urdf/marvin_wuji_d435_complete.urdf'
EXTRINSIC_PY = Path(__file__).resolve().parent.parent / 'cosmos_framework/data/fk_camera_extrinsic.py'
OUT_DEFAULT = DATA / 'renders/fk_vs_pointflow'
DEPTH_SCRATCH = Path('/tmp/fk_vs_pointflow_depth')

HAND_LABEL = 2          # mesh_check.py:41 — label 2 is the hand, 3/4 are the objects

# The exporter's canvas is 640x448 (cv2.resize of the 640x480 colour frame), and
# obs_uv lives in that canvas; the depth is a native 640x480 image.  Everything below
# works in native colour pixels, so obs_uv is mapped on the way in.
CANVAS_W, CANVAS_H = 640, 448

PF_ORANGE = '#ff7f0e'   # the benchmark figure's two colours, kept so panels compare
FK_BLUE = '#1f77b4'
D435_GREEN = '#2ca02c'

# Kinematic edges, mirrored from tools/verify_fk_camera_projection.py:74-76.
EDGES = [(0, 1), (0, 5), (0, 9), (0, 13), (0, 17)]
for _base in (1, 5, 9, 13, 17):
    EDGES += [(_base, _base + 1), (_base + 1, _base + 2), (_base + 2, _base + 3)]


# ------------------------------------------------------------------ camera / FK setup
def base_to_camera():
    """The verified Link_Base -> D435 colour transform, with its URDF fingerprint checked.

    Executed rather than imported: the generated module lives under ``cosmos_framework``,
    whose package import pulls in the training stack, and the file is dependency-free
    by design.
    """
    ext = runpy.run_path(str(EXTRINSIC_PY))
    digest = hashlib.md5(URDF.read_bytes()).hexdigest()
    if digest != ext['URDF_MD5']:
        raise SystemExit(f'URDF changed under the extrinsic: {digest} != {ext["URDF_MD5"]}\n'
                         f'Regenerate with tools/export_fk_camera_extrinsic.py before '
                         f'trusting any picture this script draws.')
    return ext['R'], ext['t']


def load_fk(episode, R, t):
    """(right-hand joint centres in camera metres [T,21,3], keypoint names)."""
    path = FK_ROOT / episode / 'annotations/wuji_fk21.npz'
    with np.load(path) as n:
        frame, units = str(n['coordinate_frame']), str(n['units'])
        if frame != 'Link_Base' or units != 'metre':
            raise SystemExit(f'{path}: expected Link_Base/metre, got {frame}/{units}')
        side = list(map(str, n['side_names'])).index('right')
        if not bool(n['side_is_observed'][side]):
            raise SystemExit(f'{path}: right hand not observed in this episode')
        return n['positions'][:, side] @ R.T + t, [str(x) for x in n['keypoint_names']]


def load_pointflow(episode):
    """Memmapped observations; obs_pos alone is ~800 MB, so never read whole."""
    d = PF_ROOT / episode
    if not d.is_dir():
        raise SystemExit(f'no PointFlow episode at {d}')
    return {k: np.load(d / f'{k}.npy', mmap_mode='r')
            for k in ('frame_offsets', 'obs_pos', 'obs_uv', 'obs_label', 'obs_valid', 'obs_unique')}


def hand_observations(obs, frame):
    """PointFlow hand points and their canvas uv for one frame."""
    s, e = (int(x) for x in obs['frame_offsets'][frame:frame + 2])
    keep = ((np.asarray(obs['obs_label'][s:e]) == HAND_LABEL)
            & np.asarray(obs['obs_valid'][s:e])
            & np.asarray(obs['obs_unique'][s:e]))
    return (np.asarray(obs['obs_pos'][s:e][keep], dtype=np.float64),
            np.asarray(obs['obs_uv'][s:e][keep], dtype=np.float64))


def canvas_uv_to_colour(uv):
    """640x448 canvas pixels -> native 640x480 colour pixels (the resize was a squash)."""
    u = (uv[:, 0] + .5) * 640 / CANVAS_W - .5
    v = (uv[:, 1] + .5) * 480 / CANVAS_H - .5
    return u, v


def colour_intrinsics(episode):
    """Factory head colour intrinsics of this episode, with the serial fallback audit.py uses."""
    import json

    def head(ep):
        p = RAW_ROOT / ep / 'auxiliary_camera/metadata.json'
        return json.loads(p.read_text())['capture_metadata']['cameras']['head']

    h = head(episode)
    ci = (h.get('streams') or {}).get('color', {}).get('intrinsics')
    if ci:
        return ci
    # Some episodes were recorded without their own intrinsics block; the serial is what
    # says two recordings used the same physical camera.
    for d in sorted(RAW_ROOT.glob('episode_*')):
        try:
            h2 = head(d.name)
        except (FileNotFoundError, KeyError):
            continue
        ci2 = (h2.get('streams') or {}).get('color', {}).get('intrinsics')
        if ci2 and h2.get('serial_number') == h.get('serial_number'):
            return ci2
    raise SystemExit(f'no head colour intrinsics for {episode}')


# ------------------------------------------------------------------------- D435 depth
def depth_metadata(episode):
    import json
    d = json.loads((RAW_ROOT / episode / 'auxiliary_camera/metadata.json').read_text())
    st = d['capture_metadata']['cameras']['head']['streams']['depth']
    i, x = st['intrinsics'], st['extrinsics_to_color']
    R = np.asarray(x['rotation'], float).reshape(3, 3)
    t = np.asarray(x['translation'], float)
    return i, R, t, float(st['depth_scale_m'])


def local_depth_dir(episode, scratch=DEPTH_SCRATCH):
    """The lmdb copied to local disk: GPFS cannot serve lmdb's mmap ('No such device')."""
    src = RAW_ROOT / episode / 'auxiliary_camera/depth.lmdb'
    dst = scratch / episode / 'depth.lmdb'
    if dst.exists() and any(dst.iterdir()):
        return dst
    dst.mkdir(parents=True, exist_ok=True)
    print(f'copying {src} -> {dst}  (GPFS cannot mmap lmdb)')
    for item in src.iterdir():
        shutil.copy2(item, dst / item.name)
    return dst


def depth_keys(depth_dir):
    import lmdb
    env = lmdb.open(str(depth_dir), readonly=True, lock=False, max_readers=4)
    with env.begin() as txn:
        out = sorted(int(k.decode().split('/')[-1])
                     for k in txn.cursor().iternext(keys=True, values=False)
                     if k.decode().startswith('depth/head/'))
    env.close()
    return out


def fetch_depth(depth_dir, key):
    import lmdb
    env = lmdb.open(str(depth_dir), readonly=True, lock=False, max_readers=4)
    with env.begin() as txn:
        blob = txn.get(f'depth/head/{key:06d}'.encode())
    env.close()
    return blob


def depth_cloud(blob, meta):
    """(points in the colour frame, colour-frame uv, metres) from one z16 PNG.

    The three lines that matter are the same as calibrate_extrinsic_v2.py:243-249 --
    factory depth intrinsics to back-project, factory extrinsics_to_color to move frame.
    Kept identical on purpose: this script must not be a second, slightly different,
    depth path.
    """
    import cv2
    i, R, t, scale = meta
    d = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_UNCHANGED).astype(np.float32) * scale
    h, w = d.shape
    jj, ii = np.meshgrid(np.arange(w), np.arange(h))
    xyz = np.stack([(jj - i['ppx']) / i['fx'] * d, (ii - i['ppy']) / i['fy'] * d, d], -1)
    pts = xyz.reshape(-1, 3) @ R.T + t
    pts[d.reshape(-1) <= 0.05] = np.nan          # 0 and near-0 are "no measurement"
    return pts


def front_cluster(z, gap_m=0.05):
    """(lo, hi) of the nearest connected run of depths, or (nan, nan) if empty.

    The D435 returns the first surface along each ray, and the hand is held in front of
    the table, so the nearest run of depths in the hand region is the hand.  The split
    matters: at frame 900 of episode_0013 the hand region is bimodal -- 36% of pixels at
    0.65-0.75 m (the hand, where FK puts it) and 64% at 0.90-1.20 m (the table) -- because
    the PointFlow tracks that define the region have drifted off the hand by then.  A
    single gap in the sorted depths separates them without reference to FK or to the
    drifted tracks.
    """
    z = np.sort(np.asarray(z, float))
    if z.size == 0:
        return np.nan, np.nan
    brk = np.nonzero(np.diff(z) > gap_m)[0]
    return float(z[0]), float(z[brk[0]] if brk.size else z[-1])


def hand_region_mask(u_col, v_col, hand_uv, radius_px=6):
    """Which depth points land on the pixels PointFlow called hand.

    The mask is built in the colour image and dilated, because the two cameras sample
    different grids and a depth point rarely lands exactly on a hand pixel: without the
    dilation the D435 cloud would be mostly holes, and with too much of it the cloud
    would spill onto the table and read as a depth error of its own.
    """
    import cv2
    hu, hv = canvas_uv_to_colour(hand_uv)
    m = np.zeros((480, 640), np.uint8)
    inside = (hu >= 0) & (hu < 640) & (hv >= 0) & (hv < 480)
    m[np.rint(hv[inside]).astype(int), np.rint(hu[inside]).astype(int)] = 1
    if radius_px > 0:
        k = np.ones((2 * radius_px + 1, 2 * radius_px + 1), np.uint8)
        m = cv2.dilate(m, k)
    # NaN (a depth that measured nothing) must be masked before the cast: np.rint(nan)
    # .astype(int64) is undefined and lands anywhere, including valid pixel indices.
    finite = np.isfinite(u_col) & np.isfinite(v_col)
    uu = np.zeros(len(u_col), np.int64)
    vv = np.zeros(len(v_col), np.int64)
    uu[finite] = np.rint(u_col[finite]).astype(np.int64)
    vv[finite] = np.rint(v_col[finite]).astype(np.int64)
    ok = finite & (uu >= 0) & (uu < 640) & (vv >= 0) & (vv < 480)
    sel = np.zeros(len(uu), bool)
    sel[ok] = m[vv[ok], uu[ok]] > 0
    return sel


def d435_hand_points(episode, frame, hand_uv, K, depth_dir, meta, radius_px=6,
                     gate='front', gate_gap_m=0.05, fk_z=None, fk_band_m=0.06):
    """The D435 hand cloud in the colour frame, plus a record of what the gate did.

    ``gate`` decides which depths inside the pixel region count as hand:

    ``front``  the nearest connected run (see front_cluster) -- a property of the depth
               alone, so it stays honest when the PointFlow region is contaminated
    ``fk``     within ``fk_band_m`` of the FK skeleton's depth span -- uses the joint
               encoders to say where the hand is; expect |D435 - FK| to shrink by
               construction, so read the DA3 column, not that one
    ``none``   everything in the pixel region, contamination included

    The returned dict carries the masked count, the front cluster and the kept count for
    every frame, because a gate that silently changes the population is how a comparison
    turns into a self-fulfilling one.
    """
    blob = fetch_depth(depth_dir, frame)
    if blob is None:
        raise SystemExit(f'no depth blob for frame {frame}')
    pts = depth_cloud(blob, meta)
    hom = pts @ K.T
    with np.errstate(divide='ignore', invalid='ignore'):
        u, v = hom[:, 0] / hom[:, 2], hom[:, 1] / hom[:, 2]
    in_region = np.isfinite(pts).all(1) & (hom[:, 2] > 0) & hand_region_mask(u, v, hand_uv, radius_px)

    z = pts[in_region][:, 2]
    lo, hi = front_cluster(z, gate_gap_m)
    info = dict(masked=int(in_region.sum()), front_lo=lo, front_hi=hi, gate=gate)
    if gate == 'none' or not np.isfinite(lo):
        return pts[in_region], info
    if gate == 'front':
        keep = in_region & (pts[:, 2] >= lo - 1e-9) & (pts[:, 2] <= hi + 1e-9)
    elif gate == 'fk':
        if fk_z is None:
            raise SystemExit('gate=fk needs the FK skeleton depths')
        keep = (in_region & (pts[:, 2] >= fk_z.min() - fk_band_m)
                & (pts[:, 2] <= fk_z.max() + fk_band_m))
    else:
        raise SystemExit(f'unknown gate {gate!r}')
    info['kept'] = int(keep.sum())
    return pts[keep], info


def colour_K(ci):
    """Native 640x480 colour intrinsics.  The metadata spells the centre ppx/ppy."""
    return np.array([[ci['fx'], 0, ci['ppx']],
                     [0, ci['fy'], ci['ppy']],
                     [0, 0, 1.0]])


# --------------------------------------------------------------------------- plotting
def plot(frames, ep, skeleton, clouds, out, subsample, seed, with_d435, html, calibrated=None, corrected=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    # Only the clouds -- the skeleton is drawn separately as bones + joints.  These
    # labels must stay in the same order as the per-frame dict is built in main().
    series = [(PF_ORANGE, 'PointFlow hand (DA3)')]
    if with_d435:
        series.append((D435_GREEN, 'D435 depth hand'))
    # One shared box for every panel, or the frames become impossible to compare: a
    # per-frame box silently rescales the axis and makes a moving hand look still.
    allpts = [skeleton[f] for f in frames] + [c for f in frames for c in clouds[f].values()]
    if calibrated is not None:
        allpts += [c for f in frames for c in calibrated[f].values()]
    if corrected is not None:
        allpts += [c for f in frames for c in corrected[f].values()]
    allpts = np.concatenate([a for a in allpts if len(a)])
    lo, hi = allpts.min(0) - 0.03, allpts.max(0) + 0.03

    fig = plt.figure(figsize=(6.6 * len(frames), 19 if corrected is not None else (13 if calibrated is not None else 7.8)), constrained_layout=True)
    fig.suptitle(f'{ep}  |  camera metres, no fitting  |  FK extrinsic = verified base->camera, '
                 f'D435 = factory depth intrinsics + extrinsics_to_color', fontsize=13)
    collections = [("Original", clouds)] + ([("B: real RGB intrinsics", calibrated)] if calibrated is not None else [])
    if corrected is not None:
        collections.append(("B + own-chunk scale restoration", corrected))
    for row, (variant, collection) in enumerate(collections):
        rng = np.random.default_rng(seed)  # identical sampled point indices per variant
        for col, f in enumerate(frames):
            ax = fig.add_subplot(len(collections), len(frames), row * len(frames) + col + 1, projection='3d')
            full = collection[f]
            for (color, label), pts in zip(series, full.values()):
                if not len(pts):
                    continue
                draw = pts[rng.choice(len(pts), subsample, replace=False)] if len(pts) > subsample else pts
                ax.scatter(draw[:, 0], draw[:, 1], draw[:, 2], s=3, c=color,
                           alpha=.40 if 'DA3' in label else .75,
                           depthshade=False, label=f'{label} ({len(pts):,})')
            s = skeleton[f]
            for a, b in EDGES:
                ax.plot(*np.stack([s[a], s[b]], 1), c=FK_BLUE, lw=2.2, alpha=.9,
                        solid_capstyle='round')
            ax.scatter(s[:, 0], s[:, 1], s[:, 2], s=30, c=FK_BLUE, depthshade=False,
                       edgecolors='white', linewidths=.7)
            ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]),
                   xlabel='Camera X (m)', ylabel='Camera Y (m)', zlabel='Camera Z (m)')
            ax.set_box_aspect(hi - lo)
            ax.view_init(elev=22, azim=-62)              # the benchmark figure's viewpoint
            pts_full = next(iter(full.values()))
            gap = np.linalg.norm(s[:, None] - pts_full[None], axis=2).min(1) * 1000
            ax.set_title(f'{variant} | frame {f} | median {np.median(gap):.1f} mm (p95 {np.percentile(gap, 95):.1f})', fontsize=11)
            ax.legend(loc='upper right', fontsize=9)
    png = out / f'{ep}_fk_vs_pointflow{"_d435" if with_d435 else ""}.png'
    if calibrated is not None:
        fig.suptitle(f'{ep} | DA3 depth, original selected points, fixed FK, no fitting | top: original; bottom: B\nB = cached-depth calibration using rerun Metric scale (not a full tracking rerun)', fontsize=14)
        png = out / f'{ep}_fk_vs_pointflow_B.png'
    if corrected is not None:
        png = out / f'{ep}_fk_vs_pointflow_three_rows.png'
        fig.suptitle(f'{ep} | original / B with legacy scale / B with own-chunk scale\nFixed original point support and FK; rerun-derived scale reconstruction; no fitting; motion not corrected', fontsize=14)
    fig.savefig(png, dpi=150, bbox_inches='tight')
    print(f'wrote {png}')
    if html:
        write_html(frames, ep, skeleton, clouds, lo, hi, series, out, with_d435)
    return png


def write_html(frames, ep, skeleton, clouds, lo, hi, series, out, with_d435):
    """A rotatable version -- one 3D view always hides whatever is behind it."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print('  (plotly not installed here; skipping the html — the Track4World venv has it)')
        return
    fig = go.Figure()
    for f in frames:
        for (color, label), pts in zip(series, clouds[f].values()):
            if not len(pts):
                continue
            marker = dict(size=1.5, color=color, opacity=.35)
            fig.add_trace(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                                       mode='markers', marker=marker, name=f'{label} f{f}'))
        s = skeleton[f]
        fig.add_trace(go.Scatter3d(x=s[:, 0], y=s[:, 1], z=s[:, 2], mode='markers',
                                   marker=dict(size=4, color=FK_BLUE), name=f'FK-21 f{f}'))
    fig.update_layout(title=f'{ep}: FK-21 vs PointFlow(DA3)'
                            f'{" vs D435 depth" if with_d435 else ""}',
                      scene=dict(xaxis_title='Camera X (m)', yaxis_title='Camera Y (m)',
                                 zaxis_title='Camera Z (m)', aspectmode='data',
                                 xaxis_range=[lo[0], hi[0]], yaxis_range=[lo[1], hi[1]],
                                 zaxis_range=[lo[2], hi[2]]))
    path = out / f'{ep}_fk_vs_pointflow{"_d435" if with_d435 else ""}.html'
    fig.write_html(path, include_plotlyjs=True)
    print(f'wrote {path}')


# ------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--episode', default='episode_0013_20260731_133649')
    ap.add_argument('--frames', default=None,
                    help='comma-separated frame indices.  With --d435 every frame must '
                         'have head depth (~1 frame in 5, exact set read from the lmdb); '
                         'the default switches to 1,301,600,900, which were checked to '
                         'have depth on episode_0013')
    ap.add_argument('--chunk-scale-report', type=Path, required=True)
    ap.add_argument('--b-scale-report', type=Path, help='Validated original-model Metric scaling report; freezes existing observations')
    ap.add_argument('--d435', action='store_true', help='add the D435 depth hand cloud')
    ap.add_argument('--depth-radius-px', type=int, default=6,
                    help='dilation of the hand mask in the colour image (see hand_region_mask)')
    ap.add_argument('--gate', choices=('front', 'fk', 'none'), default='front',
                    help='which depths inside the pixel region count as hand: the nearest '
                         'connected run (default, depth-only and so unbiased), within a band '
                         'of the FK depth span, or nothing.  See d435_hand_points.')
    ap.add_argument('--gate-gap-m', type=float, default=0.05,
                    help='gap that ends the front cluster')
    ap.add_argument('--fk-band-m', type=float, default=0.06,
                    help='half-width of the FK gate (only with --gate fk)')
    ap.add_argument('--subsample', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=Path, default=OUT_DEFAULT)
    ap.add_argument('--scratch', type=Path, default=DEPTH_SCRATCH)
    ap.add_argument('--html', action='store_true')
    ap.add_argument('--dump', type=Path, default=None)
    args = ap.parse_args()

    frames = [int(x) for x in (args.frames or ('1,301,600,900' if args.d435 else '0,650')).split(',') if x.strip()]

    R, t = base_to_camera()
    skeleton, names = load_fk(args.episode, R, t)
    obs = load_pointflow(args.episode)
    T = len(skeleton)
    if bad := [f for f in frames if not 0 <= f < T]:
        raise SystemExit(f'frame(s) {bad} outside 0..{T - 1} for {args.episode}')

    K = colour_K(colour_intrinsics(args.episode))
    if args.d435:
        depth_dir = local_depth_dir(args.episode, args.scratch)
        meta = depth_metadata(args.episode)
        keys = set(depth_keys(depth_dir))
        missing = [f for f in frames if f not in keys]
        if missing:
            near = sorted(keys)
            hint = [min(near, key=lambda k: abs(k - f)) for f in missing]
            raise SystemExit(
                f'no head depth at frame(s) {missing}: the head depth is recorded at 6 fps\n'
                f'against a 30 fps video.  The spacing is nominally every 5th frame but it\n'
                f'drifts -- mostly 5, occasionally 4 or 6 -- so the test is membership in the\n'
                f'recorded key set, not a modulus; {len(keys)} of {T} frames have one.\n'
                f'nearest recorded frames: {hint}\n'
                f'Refusing to snap to a neighbour on purpose -- 5 frames is 1/6 s and the\n'
                f'hand moves centimetres in that time, so the two clouds would be of\n'
                f'different poses and read as a depth disagreement.')
        print(f'head depth: {len(keys)} frames in {args.episode}, '
              f'max key {max(keys)} (video has {T})')

    corrected = {}
    correction_metrics = {}
    import json

    from track4world_scale_fix import correct_cached_depth
    chunk_report = json.loads(args.chunk_scale_report.read_text())
    if chunk_report["episode"] != args.episode:
        raise SystemExit("Chunk report episode mismatch")
    last_chunk = max(chunk_report["chunks"].values(), key=lambda row: row["start"])
    last_scale = last_chunk["norms"]["estimated"]
    if not args.b_scale_report:
        raise SystemExit("--b-scale-report is required for three-row comparison")
    calibrated = None
    b_metrics = {}
    if args.b_scale_report:
        import json
        if args.d435:
            raise SystemExit('B uses DA3 depth; --d435 is a separate experiment')
        b_report = json.loads(args.b_scale_report.read_text())
        if b_report['episode'] != args.episode or b_report['focal_relative_error'] >= .002:
            raise SystemExit('Scale report does not validate this episode')
        b_ratio = float(b_report['ratio_B_over_original'])
        original_K = np.load(PF_ROOT / args.episode / 'intrinsics.npy')
        calibrated = {}
    clouds, dumped, gate_note = {}, {}, {}
    for f in frames:
        pf, uv = hand_observations(obs, f)
        row = {PF_ORANGE: pf}
        if calibrated is not None:
            # Verify obs_pos really is the final projected camera point map,
            # despite the export metadata's world_depthanythingv3 mode name.
            k = original_K[f]
            uv_reprojected = np.column_stack((
                (pf[:, 0] / pf[:, 2] * k[0, 0] + k[0, 2]) * CANVAS_W - .5,
                (pf[:, 1] / pf[:, 2] * k[1, 1] + k[1, 2]) * CANVAS_H - .5))
            pixel_error = np.max(np.abs(uv_reprojected - uv))
            if pixel_error > .5:
                raise SystemExit(f'Cached point/UV inconsistency {pixel_error} pixels at {f}')
            u, v = canvas_uv_to_colour(uv)
            z = pf[:, 2] * b_ratio
            b = np.column_stack(((u - K[0, 2]) / K[0, 0] * z,
                                 (v - K[1, 2]) / K[1, 1] * z, z))
            calibrated[f] = {PF_ORANGE: b}
            stats = {}
            for name, points in [('original', pf), ('B', b)]:
                from scipy.spatial import cKDTree
                d, nearest = cKDTree(points).query(skeleton[f])
                residual = points[nearest] - skeleton[f]
                stats[name] = dict(median_mm=float(np.median(d)*1000),
                                   p95_mm=float(np.percentile(d,95)*1000),
                                   median_abs_xyz_mm=(np.median(abs(residual),axis=0)*1000).tolist(),
                                   median_xy_mm=float(np.median(np.linalg.norm(residual[:,:2],axis=1))*1000))
            b_metrics[f] = dict(points=len(pf), projection_max_error_px=float(pixel_error), **stats)
            chunks = [r for r in chunk_report['chunks'].values() if r['start'] <= f < r['end']]
            if len(chunks) != 1:
                raise SystemExit(f'Missing or overlapping chunk scale for frame {f}')
            chunk = chunks[0]
            z_fixed = correct_cached_depth(pf[:, 2], chunk['norms']['estimated'], last_scale, chunk['metric_ratio'])
            fixed = np.column_stack(((u-K[0,2])/K[0,0]*z_fixed, (v-K[1,2])/K[1,1]*z_fixed, z_fixed))
            corrected[f] = {PF_ORANGE: fixed}
            distances, nearest = cKDTree(fixed).query(skeleton[f])
            residual = fixed[nearest] - skeleton[f]
            correction_metrics[f] = dict(chunk=[chunk['start'],chunk['end']],
                depth_multiplier_from_original=(chunk['norms']['estimated']+1e-6)/last_scale*chunk['metric_ratio'],
                median_mm=float(np.median(distances)*1000),p95_mm=float(np.percentile(distances,95)*1000),
                median_abs_xyz_mm=(np.median(abs(residual),axis=0)*1000).tolist())

        if args.d435:
            d4, info = d435_hand_points(args.episode, f, uv, K, depth_dir, meta,
                                        args.depth_radius_px, args.gate, args.gate_gap_m,
                                        skeleton[f][:, 2], args.fk_band_m)
            row[D435_GREEN] = d4
            gate_note[f] = info
        clouds[f] = row
        dumped[f] = row
        line = f'frame {f:5d}: PointFlow {len(pf):7,}'
        if args.d435:
            i = gate_note[f]
            line += (f'   D435 {len(d4):6,} of {i["masked"]:6,} masked'
                     f'   front cluster {i["front_lo"]:.3f}-{i["front_hi"]:.3f} m')
            # Both comparisons are run joint -> cloud, the same direction, so the two
            # numbers can be read against each other: "is the FK skeleton buried in this
            # source's hand".  A cloud -> joint direction would answer a different
            # question and the two must not be printed side by side as if comparable.
            if len(d4):
                to_d435 = np.linalg.norm(skeleton[f][:, None] - d4[None], axis=2).min(1) * 1000
                line += f'   FK->D435 median {np.median(to_d435):6.1f} mm'
            if len(pf):
                to_da3 = np.linalg.norm(skeleton[f][:, None] - pf[None], axis=2).min(1) * 1000
                line += f'   FK->DA3 median {np.median(to_da3):6.1f} mm'
        print(line)

    args.out.mkdir(parents=True, exist_ok=True)
    plot(frames, args.episode, skeleton, clouds, args.out,
         args.subsample, args.seed, args.d435, args.html, calibrated=calibrated, corrected=corrected)
    if calibrated is not None:
        (args.out / "comparison_metrics.json").write_text(json.dumps(dict(scale_report=b_report, chunk_report=chunk_report, frames=b_metrics, corrected=correction_metrics), indent=2) + "\n")
        np.savez_compressed(args.out / "comparison_points.npz", **{f"original_{f}": clouds[f][PF_ORANGE] for f in frames}, **{f"B_{f}": calibrated[f][PF_ORANGE] for f in frames}, **{f"fk_{f}": skeleton[f] for f in frames}, **{f"corrected_{f}": corrected[f][PF_ORANGE] for f in frames})
    if args.dump:
        np.savez_compressed(
            args.dump,
            **{f'skeleton_{f}': skeleton[f] for f in frames},
            **{f'pointflow_{f}': dumped[f][PF_ORANGE] for f in frames},
            **({f'd435_{f}': dumped[f][D435_GREEN] for f in frames} if args.d435 else {}),
            keypoint_names=np.array(names))
        print(f'wrote {args.dump}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
