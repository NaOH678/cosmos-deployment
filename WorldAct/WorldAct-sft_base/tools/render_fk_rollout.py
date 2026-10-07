#!/usr/bin/env python3
"""Render a whole-episode FK rollout: the skeleton animation, and the same skeletons on the real video.

Reads the ``rollout.npz`` that ``cosmos_framework.callbacks.fk_rollout`` writes, and
produces two artefacts:

* ``rollout.mp4`` -- the 1x2 orthographic layout ``fk_visualize.write_video`` uses
  (x/y and x/z, solid GT, dashed prediction, one colour per finger), extended from
  one window's 32 steps to the whole episode.  Because the callback tiles the
  episode with one-window-spaced windows, consecutive frames come from consecutive
  windows with no overlap: every frame is predicted by exactly one window.
* ``projection.mp4`` -- the GT and predicted skeletons drawn on the real head-camera
  video, the image-plane view ``tools/render_fk_projection.py`` produces for a single
  case, run here across the episode.

Both read ABSOLUTE camera-frame metres, which is what the callback stores.  A
window's displacement is relative to its own anchor, so a rollout would otherwise
stack ~18 different origins on top of each other.

The one visible seam this cannot remove: each window starts from its own **ground
truth** anchor, so at every boundary the skeleton snaps back to the recorded pose.
That is not a rendering bug and should not be smoothed away -- it is the honest
consequence of each window being independently conditioned.

    PYTHONPATH=. python tools/render_fk_rollout.py --rollout <dir>/rollout.npz   # --out defaults under renders/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from render_fk_projection import draw_skeletons, in_frame  # noqa: E402
from verify_fk_camera_projection import (  # noqa: E402
    IMG_H,
    IMG_W,
    project,
    read_frame,
)

GT_COLOR = (80, 220, 80)  # BGR green, same as render_fk_projection
PRED_COLOR = (60, 60, 240)  # BGR red


def scale_v(uv, factor):
    """The same skeleton, drawn where a *cropped* canvas actually put it.

    Only for the generated panel -- see :func:`generated_v_scale`.
    """
    out = np.array(uv, dtype=np.float64, copy=True)
    out[:, 1] *= factor
    return out


def generated_v_scale(latent_h, latent_w):
    """How much taller the decoded generated head view is than the real one.

    The model is not fed the whole composed canvas.  The cache stores it padded to a
    patch multiple (736 px -> a 46-row latent), but the packer hands the model only
    the top patch-aligned part of the *content* -- 704 px, a 44-row latent.  That was
    measured, not inferred: the rollout's conditioning frame equals the cache window
    cropped to its top 44 rows with a mean absolute difference of exactly 0.000000.

    ``decode_head_view`` then resizes whatever head block survives -- 308..704 = 396
    rows -- up to 480, where the true mapping is 408 -> 480.  So the generated panel
    is stretched by 408/396 = 1.0303, and a skeleton projected with the real
    intrinsics lands progressively high, ~14 px at the bottom of the frame.

    Multiplying v by this factor draws it where the stretched content actually is.
    The real-video panel is untouched: it never went through the model.
    """
    from render_fk_projection import COMPOSED_H, COMPOSED_W, canvas_crop

    canvas_h, canvas_w = int(latent_h) * 16, int(latent_w) * 16
    # Both geometries through the same derivation the decoder uses, so this cannot
    # drift from the crop it is correcting.  ``s`` puts the composite on the same
    # width as the decode, which is what makes the two comparable.
    s = canvas_w / COMPOSED_W
    true_h, _, true_top = canvas_crop((round(COMPOSED_H * s), canvas_w))
    dec_h, _, dec_top = canvas_crop((canvas_h, canvas_w))
    head_rows = dec_h - dec_top
    if head_rows <= 0:
        raise ValueError(f"latent {latent_h}x{latent_w} leaves no head rows below y={dec_top}")
    return (true_h - true_top) / head_rows


def load_rollout(path):
    """The npz -> floats, bools and a string.  ``allow_pickle`` for the episode name."""
    data = np.load(path, allow_pickle=True)
    record = dict(
        prediction=np.asarray(data["prediction"], np.float64),
        target=np.asarray(data["target"], np.float64),
        valid=np.asarray(data["valid"], bool),
        frame_ids=np.asarray(data["frame_ids"], np.int64),
        episode=str(data["episode"]),
    )
    # Present only for the joint arm.  ``vision`` is [n_windows, C, T, H, W] when the
    # rollout generated video and the 0-d None placeholder when it did not, so the
    # rank is what separates them -- ``is not None`` is true for both.
    vision = data["vision"] if "vision" in data.files else None
    record["vision"] = None if vision is None or getattr(vision, "ndim", 0) < 5 else np.asarray(vision)
    record["window_frames"] = 0
    if record["vision"] is not None:
        windows = record["vision"].shape[0]
        frames = record["prediction"].shape[0]
        if frames % windows:
            raise ValueError(f"{frames} predicted frames do not divide {windows} vision windows")
        # DERIVED from the two shapes, not read from the file's ``window_frames``:
        # that field was written as the index stride (64) instead of the steps per
        # window (32), and the first render died on
        # "8 vision windows x 64 steps != 256 predicted frames".  The shapes already
        # determine it, and a derived value cannot go stale.
        record["window_frames"] = frames // windows
    if record["prediction"].shape != record["target"].shape:
        raise ValueError(f"prediction {record['prediction'].shape} vs target {record['target'].shape}")
    if record["frame_ids"].size != record["prediction"].shape[0]:
        raise ValueError(
            f"{record['frame_ids'].size} frame ids for {record['prediction'].shape[0]} frames -- "
            "one per predicted frame is the contract"
        )
    return record


def render_animation(record, output, *, fps=15.0, stride=1):
    """The 1x2 layout, animated across the whole episode.

    Axis limits are fixed to the episode's own extent rather than autoscaled per
    frame: with ``datalim`` the skeleton appears to drift whenever the view rescales,
    which is indistinguishable from the model being wrong.
    """
    import matplotlib

    matplotlib.use("Agg")
    import imageio
    import matplotlib.pyplot as plt

    from cosmos_framework.callbacks.fk_visualize import EDGES, FINGER_COLORS, finger_of

    target, prediction, valid = record["target"], record["prediction"], record["valid"]
    horizon = target.shape[0]
    steps = range(0, horizon, stride)

    def limits(ix, iy, values, mask):
        picked = values[:, :, [ix, iy]][mask]
        if picked.size == 0:
            return (-1.0, 1.0), (-1.0, 1.0)
        lo, hi = picked.min(axis=0), picked.max(axis=0)
        pad = np.maximum(hi - lo, 1e-3) * 0.12
        return (lo[0] - pad[0], hi[0] + pad[0]), (lo[1] - pad[1], hi[1] + pad[1])

    both, mask = np.concatenate([target, prediction]), np.concatenate([valid, valid])
    panels = {}
    for name, ix, iy in (("xy", 0, 1), ("xz", 0, 2)):
        panels[name] = limits(ix, iy, both, mask)

    fig = plt.figure(figsize=(11.0, 4.6))
    # NOT macro_block_size=1.  libx264 with yuv420p requires even dimensions, and
    # tight_layout lands the canvas on an odd height (459 at this figsize), which
    # makes ffmpeg fail while writing a 0-byte file and no traceback.  Letting
    # imageio pad to its default multiple is the whole point of the parameter.
    writer = imageio.get_writer(output, fps=fps)
    try:
        for step in steps:
            fig.clf()
            gt, pred = target[step], prediction[step]
            for column, (name, ix, iy, ylabel) in enumerate(
                (("xy", 0, 1, "y (down, m)"), ("xz", 0, 2, "z (depth, m)"))
            ):
                axes = fig.add_subplot(1, 2, column + 1)
                for a, b in EDGES:
                    colour = FINGER_COLORS[finger_of(a)]
                    axes.plot(gt[[a, b], ix], gt[[a, b], iy], color=colour, lw=2.0, alpha=0.9)
                    axes.plot(pred[[a, b], ix], pred[[a, b], iy], color=colour, lw=1.4, ls="--", alpha=0.9)
                axes.scatter(gt[:, ix], gt[:, iy], s=8, c="k", zorder=3)
                axes.scatter(pred[:, ix], pred[:, iy], s=8, c="w", edgecolors="k", zorder=3)
                axes.set_xlim(*panels[name][0])
                axes.set_ylim(*panels[name][1])
                axes.set_aspect("equal", adjustable="box")
                axes.grid(alpha=0.25)
                axes.set_xlabel("x (right, m)", fontsize=9)
                axes.set_ylabel(ylabel, fontsize=9)
                axes.tick_params(labelsize=7)
            err = float(np.linalg.norm(prediction[step] - target[step], axis=-1)[valid[step]].mean() * 1000.0)
            fig.suptitle(
                f"{record['episode']}  frame {int(record['frame_ids'][step])} "
                f"({step + 1}/{horizon})   step error {err:.1f} mm   solid GT / dashed prediction",
                fontsize=10,
            )
            fig.tight_layout()
            fig.canvas.draw()
            writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    finally:
        writer.close()
        plt.close(fig)
    return output


def decode_windows(latents, vae_path, *, device="cuda"):
    """``[W,C,T,H,W]`` latents -> ``[W, T_pix, 480, 640, 3]`` uint8, one clip per window.

    Each window decodes to ``steps + 1`` frames -- the anchor plus the steps -- so the
    frame for local step ``t`` is index ``t + 1``, the same offset the single-window
    renderer uses.  Decoding crops the composed canvas back to the head view, which is
    what the 640x480 intrinsics apply to.
    """
    from render_fk_projection import decode_head_view

    return np.stack([decode_head_view(z, vae_path, device=device) for z in latents])


def render_projection(record, output, *, scale=2, fps=15.0, stride=1, vae=None):
    """GT and predicted skeletons on the real head video, across the episode.

    Reads the raw ``head.mp4`` rather than any generated video: this panel answers
    "is the skeleton on the hand", and putting it on a generated video would mix that
    with "is the video right".  The generated-video comparison lives in
    ``tools/render_fk_projection.py``, which needs the saved latents.
    """
    import cv2

    target, prediction, valid = record["target"], record["prediction"], record["valid"]
    episode = record["episode"]
    generated = None
    v_factor = 1.0  # the real panel is never rescaled; only the generated one is
    if record["vision"] is not None:
        if vae is None:
            raise ValueError("this rollout carries generated video; pass --vae")
        generated = decode_windows(record["vision"], vae)
        print(f"  decoded {generated.shape[0]} windows x {generated.shape[1]} frames")
        v_factor = generated_v_scale(record["vision"].shape[-2], record["vision"].shape[-1])
        if abs(v_factor - 1.0) > 1e-6:
            print(
                f"  generated panel: the model saw a {int(record['vision'].shape[-2]) * 16}px canvas "
                f"(cropped), so the head view arrives stretched; skeleton v corrected by {v_factor:.4f}"
            )

    panels = 2 if generated is not None else 1
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (IMG_W * scale * panels, IMG_H * scale)
    )
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed to open {output}")
    try:
        for step in range(0, target.shape[0], stride):
            fid = int(record["frame_ids"][step])
            frame = read_frame(episode, fid)
            guv, gfront = project(target[step])
            puv, pfront = project(prediction[step])
            series = [
                (GT_COLOR, guv, in_frame(guv, gfront) & valid[step]),
                (PRED_COLOR, puv, in_frame(puv, pfront) & valid[step]),
            ]
            row = [
                draw_skeletons(
                    frame, series,
                    label=f"REAL   {episode[-18:]}  frame {fid}  ({step + 1}/{target.shape[0]})"
                          f"   green=GT  red=pred",
                    scale=scale,
                )
            ]
            if generated is not None:
                window, local = divmod(step, record["window_frames"])
                # Each window generated its clip from its OWN anchor, from its own
                # noise, so this panel is an independent prediction and not a
                # continuation of the previous one.  The label says so, as the
                # PointFlow stage clips do -- the seam is a property of windowed
                # generation, not a rendering fault, and hiding it would make the
                # clip read as one continuous rollout when it is not.
                row.append(
                    draw_skeletons(
                        generated[window][local + 1],
                        [(c, scale_v(u, v_factor), m) for c, u, m in series],
                        label=f"GENERATED (i2v)  window {window + 1}/{generated.shape[0]}"
                              f" | independent prediction   green=GT  red=pred",
                        scale=scale,
                    )
                )
            writer.write(np.concatenate(row, axis=1))
    finally:
        writer.release()
    return output


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rollout", required=True, help="a rollout.npz written by FKRolloutCallback")
    # Under shichaojian, not /tmp: this is the artefact you come back to look at.
    ap.add_argument(
        "--out",
        default="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/renders/fk_rollout",
    )
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--stride", type=int, default=1, help="render every Nth frame")
    ap.add_argument("--scale", type=int, default=2, help="projection panel upscale")
    ap.add_argument("--only", choices=("animation", "projection"), default=None)
    # Needed only when the rollout generated video (the joint arm); the FK-only
    # rollout has no latent to decode and never loads the VAE.
    ap.add_argument(
        "--vae",
        default="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/"
        "models/cosmos3-edge-droid/vae/Wan2.2_VAE.pth",
    )
    args = ap.parse_args()

    record = load_rollout(args.rollout)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    horizon = record["target"].shape[0]
    print(f"{record['episode']}: {horizon} predicted frames, {len(np.unique(record['frame_ids']))} unique")

    if args.only != "projection":
        path = render_animation(record, out_dir / "rollout.mp4", fps=args.fps, stride=args.stride)
        print(f"wrote {path}")
    if args.only != "animation":
        path = render_projection(
            record, out_dir / "projection.mp4",
            scale=args.scale, fps=args.fps, stride=args.stride, vae=args.vae,
        )
        print(f"wrote {path}")

    print("\nCHECK THE GREEN SKELETON FIRST: if GT is not on the hand in projection.mp4,")
    print("the projection is wrong and the red series means nothing. A snap at each")
    print("window boundary is expected -- every window restarts from its own GT anchor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
