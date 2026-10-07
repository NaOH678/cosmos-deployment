#!/usr/bin/env python3
"""Render a labeled Track4World episode to MP4 with the official Viser look.

Same scene recipe as pf_out's build_efep_labeled_review.py (label palette,
video-colour/label tint mix, <=5 cm chain-linked track segments) and the same
render path as Track4World's record_3d_efep_chunks.py (real WebGL client +
ClientHandle.get_render), except the browser client is automated with
playwright headless chromium and frames are encoded with cv2 instead of
ffmpeg, so the whole thing runs unattended on a headless box.

Run with the Track4World venv (has viser + cv2 + playwright):

    $T4W/.venv/bin/python tools/render_efep_labeled_mp4.py \
        --data-dir /path/to/pf_out/<export>/efep_labeled/<episode> \
        --output out.mp4 [--time-stride 4] [--cloud-stride 1]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

DEFAULT_COLORS = {1: (255, 40, 180), 2: (20, 160, 255), 3: (255, 160, 20), 4: (235, 170, 0)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="efep_labeled/<episode> directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time-stride", type=int, default=4)
    parser.add_argument("--cloud-stride", type=int, default=1)
    parser.add_argument("--track-limit-per-label", type=int, default=700)
    parser.add_argument("--label-color-mix", type=float, default=0.75)
    parser.add_argument("--point-size", type=float, default=0.0022)
    parser.add_argument("--line-width", type=float, default=2.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=7.5)
    parser.add_argument("--port", type=int, default=8941)
    parser.add_argument("--client-timeout", type=float, default=60.0)
    args = parser.parse_args()
    if args.time_stride < 1 or args.cloud_stride < 1:
        parser.error("strides must be positive")

    d = args.data_dir
    report = json.loads((d / "report.json").read_text())
    off = np.load(d / "frame_offsets.npy")
    tid = np.load(d / "obs_track.npy")
    pos = np.load(d / "obs_pos.npy")
    valid = np.load(d / "obs_valid.npy")
    rgb = np.load(d / "obs_rgb.npy")
    tlabel = np.load(d / "track_label.npy")
    times = np.load(d / "timestamps_sec.npy")
    T, n = len(off) - 1, len(tlabel)

    palette = np.zeros((256, 3), np.uint8)
    for k, c in DEFAULT_COLORS.items():
        palette[k] = c
    mix = float(np.clip(args.label_color_mix, 0.0, 1.0))
    obs_label = np.load(d / "obs_label.npy") if (d / "obs_label.npy").exists() else None
    cloud_ok = valid & np.load(d / "obs_unique.npy") if (d / "obs_unique.npy").exists() else valid

    # Per label, draw the tracks with the most valid observations (their recipe).
    nvalid = np.bincount(tid[valid], minlength=n)
    ids = []
    for k in np.unique(tlabel):
        group = np.nonzero((tlabel == k) & (nvalid > 0))[0]
        ids.extend(group[np.argsort(-nvalid[group], kind="stable")][: args.track_limit_per_label])
    ids = np.asarray(ids, np.int64)
    slot = np.full(n, -1, np.int32)
    slot[ids] = np.arange(len(ids))
    if obs_label is None:
        obs_label = tlabel[tid]
    # Per-frame slice pass: no obs_frame/repeat giant temporaries on small-RAM boxes.
    tpos = np.full((T, len(ids), 3), np.nan, np.float32)
    tval = np.zeros((T, len(ids)), bool)
    for t in range(T):
        s, e = off[t], off[t + 1]
        sl = slot[tid[s:e]]
        m = sl >= 0
        tpos[t, sl[m]] = pos[s:e][m]
        tval[t, sl[m]] = valid[s:e][m]
    del tid, slot
    step = np.full((T, len(ids)), np.inf, np.float32)
    step[1:] = np.linalg.norm(tpos[1:] - tpos[:-1], axis=-1)
    link = np.zeros_like(tval)
    link[1:] = tval[1:] & tval[:-1] & (step[1:] <= 0.05)

    def cloud(t):
        s, e = off[t], off[t + 1]
        v = cloud_ok[s:e]
        p = pos[s:e][v][:: args.cloud_stride]
        base = rgb[s:e][v][:: args.cloud_stride].astype(np.float32)
        tint = palette[obs_label[s:e][v][:: args.cloud_stride]].astype(np.float32)
        return p, (mix * tint + (1.0 - mix) * base).astype(np.uint8)

    import viser

    server = viser.ViserServer(host="127.0.0.1", port=args.port)
    p0, c0 = cloud(0)
    cloud_node = server.scene.add_point_cloud(
        "/efep_labeled", points=p0, colors=c0, point_size=args.point_size, point_shape="rounded"
    )
    line = server.scene.add_line_segments(
        "/tracks",
        points=np.zeros((0, 2, 3), np.float32),
        colors=np.zeros((0, 2, 3), np.uint8),
        line_width=args.line_width,
    )

    # Camera fit from the first frame's valid cloud (their formula).
    first = off[0], off[1]
    p = pos[first[0] : first[1]][valid[first[0] : first[1]]]
    if len(p) < 10:
        p = pos[valid]
    bounds = np.quantile(p, [0.02, 0.98], axis=0)
    center = bounds.mean(0)
    radius = max(np.linalg.norm(bounds[1] - bounds[0]) / 2, 0.05)
    cam_pos = center + np.array([0, 0, -2.0 * radius])

    # Automated WebGL client: headless chromium pointed at the Viser server.
    from playwright.sync_api import sync_playwright

    pw = sync_playwright().start()
    browser = pw.chromium.launch(args=["--use-gl=swiftshader", "--disable-gpu-sandbox"])
    page = browser.new_page(viewport={"width": args.width, "height": args.height})
    page.goto(f"http://127.0.0.1:{args.port}", wait_until="load")
    deadline = time.monotonic() + args.client_timeout
    while not server.get_clients():
        if time.monotonic() >= deadline:
            browser.close()
            pw.stop()
            raise TimeoutError("No Viser browser client connected in time")
        time.sleep(0.1)
    client = list(server.get_clients().values())[-1]
    # Apply the camera only after the client sent its initial state (their ordering).
    time.sleep(1.0)
    client.camera.position = np.asarray(cam_pos, dtype=np.float64)
    client.camera.look_at = np.asarray(center, dtype=np.float64)
    client.camera.up_direction = np.array([0.0, -1.0, 0.0])
    client.flush()
    time.sleep(1.0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
    if not writer.isOpened():
        browser.close()
        pw.stop()
        raise RuntimeError(f"Cannot open MP4 writer: {args.output}")

    keys = list(range(0, T, args.time_stride))
    try:
        for i, t in enumerate(keys):
            cloud_node.points, cloud_node.colors = cloud(t)
            if i:
                prev = keys[i - 1]
                chain_ok = link[prev + 1 : t + 1].all(0) if t > prev else np.zeros(len(ids), bool)
                take = np.nonzero(chain_ok & tval[prev] & tval[t])[0]
                if len(take):
                    line.points = np.stack([tpos[prev, take], tpos[t, take]], 1)
                    col = palette[tlabel[ids[take]]]
                    line.colors = np.stack([col, col], 1)
                else:
                    line.points, line.colors = np.zeros((0, 2, 3), np.float32), np.zeros((0, 2, 3), np.uint8)
            server.flush()
            image = client.get_render(height=args.height, width=args.width, transport_format="jpeg")
            # get_render returns RGB; cv2.VideoWriter expects BGR (contiguous).
            image = np.ascontiguousarray(image[:, :, :3][:, :, ::-1])
            cv2.rectangle(image, (12, 10), (360, 42), (255, 255, 255), -1)
            cv2.putText(
                image,
                f"{d.name} | frame {t}/{T - 1} | +{times[t] - times[0]:.2f}s",
                (20, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (25, 25, 25),
                1,
                cv2.LINE_AA,
            )
            writer.write(image)
            if i % 25 == 0 or i == len(keys) - 1:
                print(f"RENDER {i + 1}/{len(keys)} (frame {t})", flush=True)
    finally:
        writer.release()
        browser.close()
        pw.stop()
        server.stop()
    print("MP4", args.output, args.output.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
