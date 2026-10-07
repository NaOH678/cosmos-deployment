#!/usr/bin/env python3
"""Animate the FK-21 / PointFlow(DA3) / D435 comparison from render_fk_vs_pointflow.py.

Same three clouds and the same verified extrinsics as the still figure -- this only adds
time, so that "is the DA3 depth far off, or just off at a few frames" can be answered by
watching it move instead of by picking frames and hoping they are representative.

TWO THINGS MAKE THE VIDEO READABLE, AND BOTH ARE LOAD-BEARING

  * The axis box is shared by every frame, computed once from all of them.  A per-frame
    box rescales silently and makes a moving hand look still -- the one failure mode a
    video is supposed to expose, not hide.

  * Frames are restricted to those with head depth, so all three clouds are present in
    every frame.  Plotting FK and DA3 on frames without depth would change the population
    mid-video and any apparent jump in the numbers would be the cast, not the geometry.
    This costs nothing in time: consecutive depth frames are 5 video frames apart, so
    playing them at 6 fps is real time with no gaps.

The per-frame numbers in the title are the same two as the still figure, both in the same
direction (joint -> nearest cloud point) so they can be read against each other as the
video plays.

Usage:
    python tools/render_fk_vs_pointflow_video.py --frames 0-1191 --stride 1
    python tools/render_fk_vs_pointflow_video.py --frames 0-600 --gate none
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_fk_vs_pointflow as core  # noqa: E402

OUT_DEFAULT = core.DATA / 'renders/fk_vs_pointflow'


def parse_range(spec: str) -> list[int]:
    """`start-end` or a comma list -> the requested indices, before any filtering."""
    if '-' in spec:
        lo, hi = (int(x) for x in spec.split('-', 1))
        return list(range(lo, hi + 1))
    return [int(x) for x in spec.split(',') if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--episode', default='episode_0013_20260731_133649')
    ap.add_argument('--frames', default='0-1191',
                    help='range start-end or comma list, intersected with the frames that '
                         'have head depth; how many were dropped is printed')
    ap.add_argument('--stride', type=int, default=1, help='keep every Nth depth frame')
    ap.add_argument('--fps', type=float, default=6.0,
                    help='6 is real time: depth frames are 5 video frames (1/6 s) apart')
    ap.add_argument('--gate', choices=('front', 'fk', 'none'), default='front')
    ap.add_argument('--gate-gap-m', type=float, default=0.05)
    ap.add_argument('--fk-band-m', type=float, default=0.06)
    ap.add_argument('--depth-radius-px', type=int, default=6)
    ap.add_argument('--subsample', type=int, default=2500)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--size', type=int, default=900, help='panel size in pixels')
    ap.add_argument('--out', type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()

    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    R, t = core.base_to_camera()
    skeleton, _ = core.load_fk(args.episode, R, t)
    obs = core.load_pointflow(args.episode)
    K = core.colour_K(core.colour_intrinsics(args.episode))
    meta = core.depth_metadata(args.episode)
    depth_dir = core.local_depth_dir(args.episode)
    keys = core.depth_keys(depth_dir)
    key_set = set(keys)

    wanted = parse_range(args.frames)
    T = len(skeleton)
    off = [f for f in wanted if not 0 <= f < T]
    if off:
        raise SystemExit(f'{len(off)} requested frame(s) outside 0..{T - 1}, e.g. {off[:5]}')
    # Intersect with the depth frames rather than demanding every frame have one.  This
    # loses nothing: consecutive depth frames are 5 video frames apart, so playing them at
    # 6 fps reproduces real time with no gaps.  Only frames that have no depth at all are
    # dropped, and how many is reported, so a range that is mostly empty cannot pass as a
    # full-episode video.
    frames = [f for f in wanted if f in key_set][::args.stride]
    dropped = len(wanted) - len([f for f in wanted if f in key_set])
    if not frames:
        raise SystemExit(
            f'none of the {len(wanted)} requested frames has head depth.  The head depth\n'
            f'covers {len(keys)} of {T} frames (6 fps against a 30 fps video) spanning\n'
            f'{min(keys)}..{max(keys)}; try --frames {min(keys)}-{max(keys)}.')
    if dropped:
        print(f'{dropped} of {len(wanted)} requested frames have no head depth and were '
              f'dropped (the head depth covers {len(keys)} of {T} frames)')
    print(f'{len(frames)} frames with head depth, from {frames[0]} to {frames[-1]}, '
          f'at {args.fps:g} fps -> {len(frames) / args.fps:.1f} s')

    # -- clouds for every frame first: the shared box needs them, and the render loop
    # -- should not be doing analysis between frames.
    data, gaps = {}, []
    for i, f in enumerate(frames):
        pf, uv = core.hand_observations(obs, f)
        d4, info = core.d435_hand_points(args.episode, f, uv, K, depth_dir, meta,
                                         args.depth_radius_px, args.gate, args.gate_gap_m,
                                         skeleton[f][:, 2], args.fk_band_m)
        data[f] = (pf, d4, skeleton[f])
        gaps.append((
            np.median(np.linalg.norm(skeleton[f][:, None] - d4[None], axis=2).min(1)) * 1000
            if len(d4) else np.nan,
            np.median(np.linalg.norm(skeleton[f][:, None] - pf[None], axis=2).min(1)) * 1000
            if len(pf) else np.nan))
        if i % 50 == 0:
            print(f'  loaded {i + 1}/{len(frames)}', flush=True)

    # -- one box for the whole video, from a decimated sample of every cloud
    samp = []
    for f in frames[::max(1, len(frames) // 24)]:
        pf, d4, sk = data[f]
        samp += [sk, pf[::max(1, len(pf) // 400)]] if len(pf) else [sk]
        if len(d4):
            samp.append(d4)
    allpts = np.concatenate([a for a in samp if len(a)])
    lo, hi = allpts.min(0) - 0.03, allpts.max(0) + 0.03
    print(f'shared box: X {lo[0]:.3f}..{hi[0]:.3f}  Y {lo[1]:.3f}..{hi[1]:.3f}  '
          f'Z {lo[2]:.3f}..{hi[2]:.3f} m')

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f'{args.episode}_fk_vs_pointflow_d435.mp4'
    dpi = 100
    fig = plt.figure(figsize=(args.size / dpi, args.size / dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), args.fps,
                             (args.size, args.size))
    if not writer.isOpened():
        raise SystemExit(f'VideoWriter failed to open {path}')
    rng = np.random.default_rng(args.seed)

    for i, f in enumerate(frames):
        pf, d4, sk = data[f]
        ax.clear()
        for pts, color, label in ((pf, core.PF_ORANGE, 'PointFlow (DA3)'),
                                  (d4, core.D435_GREEN, 'D435 depth')):
            if not len(pts):
                continue
            draw = pts[rng.choice(len(pts), args.subsample, replace=False)] \
                if len(pts) > args.subsample else pts
            ax.scatter(draw[:, 0], draw[:, 1], draw[:, 2], s=3, c=color,
                       alpha=.40 if 'DA3' in label else .75, depthshade=False, label=label)
        for a, b in core.EDGES:
            ax.plot(*np.stack([sk[a], sk[b]], 1), c=core.FK_BLUE, lw=2.2, alpha=.9,
                    solid_capstyle='round')
        ax.scatter(sk[:, 0], sk[:, 1], sk[:, 2], s=30, c=core.FK_BLUE, depthshade=False,
                   edgecolors='white', linewidths=.7, label='FK-21')
        g4, g3 = gaps[i]
        ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]),
               xlabel='Camera X (m)', ylabel='Camera Y (m)', zlabel='Camera Z (m)')
        ax.set_box_aspect(hi - lo)
        ax.view_init(elev=22, azim=-62)
        ax.set_title(f'frame {f}   FK->D435 {g4:5.1f} mm   FK->DA3 {g3:5.1f} mm'
                     f'   [{len(d4)} / {len(pf)} pts]', fontsize=11)
        ax.legend(loc='upper right', fontsize=8)
        fig.canvas.draw()
        writer.write(cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[:, :, :3],
                                  cv2.COLOR_RGB2BGR))
        if i % 25 == 0:
            print(f'  rendered {i + 1}/{len(frames)}', flush=True)

    writer.release()
    plt.close(fig)
    print(f'wrote {path}  ({len(frames)} frames, {len(frames) / args.fps:.1f} s)')

    ok = ~np.isnan([g[1] for g in gaps])
    print(f'FK->DA3 over the video: median {np.nanmedian([g[1] for g in gaps]):.1f} mm, '
          f'p10 {np.nanpercentile([g[1] for g in gaps], 10):.1f}, '
          f'p90 {np.nanpercentile([g[1] for g in gaps], 90):.1f}, '
          f'worst {np.nanmax([g[1] for g in gaps]):.1f} mm  (over {ok.sum()} frames)')
    print(f'FK->D435 over the video: median {np.nanmedian([g[0] for g in gaps]):.1f} mm, '
          f'p10 {np.nanpercentile([g[0] for g in gaps], 10):.1f}, '
          f'p90 {np.nanpercentile([g[0] for g in gaps], 90):.1f}, '
          f'worst {np.nanmax([g[0] for g in gaps]):.1f} mm')
    np.savez_compressed(args.out / f'{args.episode}_gaps.npz',
                        frames=np.array(frames), gaps=np.array(gaps))
    return 0


if __name__ == '__main__':
    sys.exit(main())
