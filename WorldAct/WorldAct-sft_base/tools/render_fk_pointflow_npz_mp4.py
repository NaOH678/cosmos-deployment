"""Render saved original/B/FK snapshots as a fixed-view comparison MP4.

Each snapshot is held for a specified duration; no geometry interpolation.
"""
import argparse
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seconds-per-frame', type=float, default=3)
    parser.add_argument('--fps', type=int, default=24)
    args = parser.parse_args()
    if args.fps <= 0 or args.seconds_per_frame <= 0:
        parser.error('fps and seconds-per-frame must be positive')
    output = args.output or args.input.with_suffix('.mp4')
    output.parent.mkdir(parents=True, exist_ok=True)
    with np.load(args.input) as data:
        snapshots = {k: data[k] for k in data.files}
    frames = sorted(int(k.removeprefix('fk_')) for k in snapshots if k.startswith('fk_'))
    if not frames:
        raise ValueError('No fk_<frame> arrays found')
    all_points = np.concatenate([snapshots[f'{kind}_{f}'] for f in frames for kind in ('original', 'B', 'fk')])
    lo, hi = all_points.min(0) - .03, all_points.max(0) + .03
    edges = [(0, j) for j in (1, 5, 9, 13, 17)]
    edges += [(j, j + 1) for start in (1, 5, 9, 13, 17) for j in range(start, start + 3)]
    width, height = 1600, 800
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', f'{width}x{height}', '-r', str(args.fps), '-i', '-', '-an',
               '-c:v', 'libx264', '-crf', '18', '-preset', 'medium', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(output)]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        try:
            for frame in frames:
                fig = plt.figure(figsize=(16, 8), dpi=100)
                fig.suptitle(f'FK vs PointFlow | source frame {frame} | camera coordinates (metres)', fontsize=18)
                fig.text(.5, .035, 'Four saved snapshots; 3D geometry is not interpolated. B: cached-depth calibration, fixed original point support.', ha='center', fontsize=11)
                fk = snapshots[f'fk_{frame}']
                for column, (kind, title) in enumerate([('original', 'Original'), ('B', 'B: real RGB intrinsics')]):
                    ax = fig.add_subplot(1, 2, column + 1, projection='3d')
                    cloud = snapshots[f'{kind}_{frame}']
                    distances = cKDTree(cloud).query(fk)[0] * 1000
                    ax.scatter(*cloud.T, s=2, c='#ff7f0e', alpha=.4, depthshade=False, label=f'PointFlow ({len(cloud):,})')
                    for a, b in edges:
                        ax.plot(*fk[[a, b]].T, color='#1f77b4', linewidth=2)
                    ax.scatter(*fk.T, s=28, c='#1f77b4', edgecolors='white', linewidths=.5, depthshade=False, label='FK-21')
                    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]), xlabel='Camera X (m)', ylabel='Camera Y (m)', zlabel='Camera Z (m)')
                    ax.set_box_aspect(hi - lo)
                    ax.view_init(elev=22, azim=-62)
                    ax.set_title(f'{title}\nJoint-to-surface median {np.median(distances):.1f} mm | p95 {np.percentile(distances, 95):.1f} mm', fontsize=13)
                    ax.legend(loc='upper right', fontsize=9)
                fig.subplots_adjust(left=.03, right=.96, bottom=.12, top=.86, wspace=.08)
                fig.canvas.draw()
                rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy().tobytes()
                assert len(rgb) == width * height * 3
                for _ in range(max(1, round(args.seconds_per_frame * args.fps))):
                    encoder.stdin.write(rgb)
                plt.close(fig)
                print(f'Rendered frame {frame}', flush=True)
        finally:
            encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError('ffmpeg failed')
    print(output)


if __name__ == '__main__':
    main()
