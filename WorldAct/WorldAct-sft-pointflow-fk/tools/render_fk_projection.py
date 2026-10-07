#!/usr/bin/env python3
"""Draw the predicted FK in the image plane, on the real video and the generated one.

Why this exists: the FK eval scores the hand in 3D (`fk_visualize.render_case`), which
answers "is the trajectory right" but not "is it on the hand".  Overlaying the
skeleton on the pixels answers both at once, and drawing it on the *generated* video
as well separates the two ways the joint arm can fail -- FK cannot track, versus FK is
faithfully tracking a video that is itself wrong.

The projection is not fitted.  ``fk_camera_extrinsic.base_to_camera`` is numerically
identical to ``roll_about_z(180) @ base_to_head_camera(URDF)`` (verified to 4e-13), so
training's frame, ``prediction.npz``'s frame and the tools' projection frame are one
frame.  ``anchor + prediction[t]`` is already camera-frame metres, and
``verify_fk_camera_projection.project`` takes exactly that.

Two frame facts make the rest trivial:

* ``raw_frame_ids[t] = raw_start + 2*t`` are absolute 30 Hz indices into
  ``<raw>/<episode>/videos/head.mp4``, so step ``t``'s skeleton belongs on frame
  ``frame_ids[t + 1]`` -- no stride arithmetic.
* the GT row is the same code path with ``anchor + target[t]``.  **Look at that row
  first**: if the GT skeleton is not on the hand in the real video, the projection is
  wrong and every other panel is meaningless.  It is the built-in check.

The generated video is a LATENT (``prediction.npz["vision"]``), decoded here rather
than during training.  It is decoded at the *composed* canvas geometry and then
cropped back to the head view, because the dataset stacks wrist over head and
reflection-pads to a fixed target size -- the 640x480 intrinsics do not apply to the
canvas, only to the head sub-image after the crop.  ``canvas_crop`` derives the
split from the decoded size, so it follows the checkpoint's resolution.

Caveats worth keeping in mind:

* the extrinsic is a single constant, valid only while the head camera is rigid on
  ``Link_Base``.  If the head joint moves mid-episode every later frame is wrong, with
  no error message.
* this overlay is a diagnostic, not a claim about what the model was given.  The
  modality was deliberately built without uv (see docs/fk_modality_design.md); the
  transform used here is the visually-verified one, used for rendering only.

    PYTHONPATH=. python tools/render_fk_projection.py --case-dir <case>   # --out defaults under renders/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from verify_fk_camera_projection import (  # noqa: E402
    EDGES,
    IMG_H,
    IMG_W,
    project,
    read_frame,
)

RAW_ROOT = "/data/shichaojian/raw_data/singlerighthand_sandwich_100"
VAE_DEFAULT = "/data/shichaojian/models/cosmos3-edge-droid/vae/Wan2.2_VAE.pth"

# The composed canvas BEFORE the model's resize: wrist stacked over head, both at
# width 640.  wrist (848x480) resized to width 640 is 362 rows; head is 640x480, so
# 480 rows.  842 x 640.
#
# These are the only canvas numbers that are fixed.  The crop constants (content
# height, head offset) are DERIVED from the decoded frame size below, because they
# are a function of the target the model was trained at -- and hardcoding them went
# wrong immediately: the first version carried 716/308 from an earlier derivation,
# the joint arm decoded to 704 rows, and the render died on "decoded (704, 544) is
# smaller than the content 716x544".  Deriving cannot drift from the decode.
COMPOSED_H, COMPOSED_W = 362 + 480, 640
WRIST_ROWS = 362  # the head block starts here in the composite

# Two series, two colours.  Per-finger colours are the repo's convention for a single
# skeleton, but with GT and prediction on the same frame they stop being readable.
GT_COLOR = (80, 220, 80)  # BGR, green
PRED_COLOR = (60, 60, 240)  # BGR, red

_VAE_CACHE = {}


def load_vae(path: str):
    """The vision tokenizer alone -- no model, no DCP.

    ``decode`` is not decorated with ``no_grad`` on this interface (only the
    ``OmniMoTModel.decode`` wrapper is), so the caller wraps it.
    """
    if path not in _VAE_CACHE:
        from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import Wan2pt2VAEInterface

        _VAE_CACHE[path] = Wan2pt2VAEInterface(vae_path=path, causal=True)
    return _VAE_CACHE[path]


def canvas_crop(decoded_hw):
    """``(content_h, content_w, head_top)`` for a decoded canvas of ``decoded_hw``.

    The model's transform resizes the 842x640 composite aspect-preserving to fit the
    padded target and reflection-pads the remainder, so the content is the fitted
    rectangle and the head is its lower 480/842.  Reading the split off the decoded
    size means it follows whatever target the checkpoint was trained at, instead of
    restating a number that has to be kept in sync by hand.
    """
    height, width = int(decoded_hw[0]), int(decoded_hw[1])
    # Scale from the WIDTH alone.  Taking the min with the height-derived scale is
    # wrong here because the height is the axis the model crops: it is fed only the
    # top 44 of the canvas's 46 latent rows (704 of 736 px, measured -- the rollout's
    # conditioning frame equals the cache window cut to its top 44 rows exactly), so
    # height/COMPOSED_H reads 0.836 where the true scale is 0.85.  That put head_top
    # at 303 instead of 308 and mixed five wrist rows into the head view.  The width
    # is never cropped, so it is the axis that still measures the scale.
    scale = width / COMPOSED_W
    content_h = min(height, round(COMPOSED_H * scale))
    content_w = min(width, round(COMPOSED_W * scale))
    head_top = round(WRIST_ROWS * scale)
    if head_top >= content_h:
        raise ValueError(f"decoded {decoded_hw} leaves no head rows (head_top={head_top})")
    return content_h, content_w, head_top


def decode_head_view(latent, vae_path, *, device="cuda"):
    """``[C,T,H,W]`` latent -> ``[T,480,640,3]`` uint8 of the head camera view.

    The latent covers the *padded composed canvas*, so the decode output must be
    cropped twice: the reflection pad off the bottom/right, then the wrist block off
    the top to leave the head image, which is then resized to the head camera's own
    640x480 so the published intrinsics apply to it unchanged.
    """
    import cv2
    import torch

    tokenizer = load_vae(vae_path)
    z = torch.as_tensor(latent).unsqueeze(0).to(device=device, dtype=torch.float32)  # [1,C,T,H,W]
    with torch.inference_mode():
        video = tokenizer.decode(z)[0]  # [3,T,H',W'] in ~[-1,1]
    frames = ((video.clamp(-1.0, 1.0) + 1.0) / 2.0).clamp(0.0, 1.0) * 255.0
    frames = frames.permute(1, 2, 3, 0).round().to(torch.uint8).cpu().numpy()  # [T,H',W',3]
    content_h, content_w, head_top = canvas_crop(frames.shape[1:3])
    content = frames[:, :content_h, :content_w]
    head = content[:, head_top:, :]
    head = np.stack([cv2.resize(f, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR) for f in head])
    # RGB -> BGR before returning.  The tokenizer decodes to RGB (channel 0 = red),
    # but every consumer of this function draws with cv2 and writes with
    # cv2.VideoWriter, both of which are BGR.  Without the swap the generated panel
    # comes out with red and blue exchanged: yellow-toned food and wood read as blue,
    # and a blue background reads as yellow.  The real-video panel is unaffected
    # because read_frame goes through OpenCV and is BGR from the start -- which is
    # exactly why the two panels disagree rather than both being wrong.
    return np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in head])


def draw_skeletons(frame, series, *, label, scale=2):
    """``series`` is a list of ``(color, uv, valid)``, drawn in order (last on top)."""
    import cv2

    canvas = cv2.resize(frame, (IMG_W * scale, IMG_H * scale), interpolation=cv2.INTER_LINEAR)
    for color, uv, valid in series:
        for a, b in EDGES:
            if not (valid[a] and valid[b]):
                continue
            pa = tuple(np.round(uv[a] * scale).astype(int))
            pb = tuple(np.round(uv[b] * scale).astype(int))
            cv2.line(canvas, pa, pb, (0, 0, 0), 6 * scale)
            cv2.line(canvas, pa, pb, color, 2 * scale)
        for k in range(len(uv)):
            if not valid[k]:
                continue
            p = tuple(np.round(uv[k] * scale).astype(int))
            cv2.circle(canvas, p, 4 * scale, (0, 0, 0), -1)
            cv2.circle(canvas, p, 3 * scale, color, -1)
    bar = 26 * scale
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], bar), (0, 0, 0), -1)
    cv2.putText(
        canvas, label, (6 * scale, 19 * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (255, 255, 255), 1, cv2.LINE_AA
    )
    return canvas


def stored_latent(data, *, no_generated=False):
    """The saved generated-video latent, or ``None`` when this case has none.

    The clean-video arm does **not** omit the key: ``fk_visualize.write_case``
    always writes it, with ``None`` as the value, and ``np.savez`` stores that as a
    0-d object array.  So ``data["vision"] is not None`` is *True* for a case with
    no generated video, and the placeholder would be handed to ``[0]`` as if it
    were ``[N,C,T,H,W]``.  The rank is what separates them -- a real latent is 4-d
    (or 5-d with the sample axis), the placeholder is 0-d.
    """
    if no_generated or "vision" not in data.files:
        return None
    stored = data["vision"]
    if stored is None or getattr(stored, "ndim", 0) == 0:
        return None
    value = np.asarray(stored)
    # Strip leading singleton axes down to [C,T,H,W].  How many there are depends on
    # how the callback stacked: ``_record`` stacks one entry per vision item, and a
    # packed item already carries its own batch axis, so the joint arm's real shape
    # is [N, 1, C, T, H, W] -- not the [N, C, T, H, W] this assumed.  The decoder
    # wants [C,T,H,W], and anything left over would reach it as a 5- or 6-d tensor.
    while value.ndim > 4 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 4:
        raise ValueError(f"expected a [C,T,H,W] latent, got {np.asarray(stored).shape}")
    return value


def contact_sheet(stills):
    """Lay still panels two-up, padding a half row rather than raising on it.

    An odd panel count is the *normal* case, not an edge case: the default
    ``--stills`` names index 32, which a 32-step horizon never reaches, leaving
    three panels.  Without the pad the last row is half as wide as the others and
    ``np.concatenate`` refuses to stack them.
    """
    rows = []
    for i in range(0, len(stills), 2):
        pair = list(stills[i : i + 2])
        if len(pair) == 1:
            pair.append(np.zeros_like(pair[0]))
        rows.append(np.concatenate(pair, axis=1))
    return np.concatenate(rows, axis=0)


def in_frame(uv, front):
    return front & (uv[:, 0] >= 0) & (uv[:, 0] < IMG_W) & (uv[:, 1] >= 0) & (uv[:, 1] < IMG_H)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case-dir", required=True, help="a case directory holding prediction.npz")
    ap.add_argument("--raw-root", default=RAW_ROOT)
    ap.add_argument("--vae", default=VAE_DEFAULT)
    # Under shichaojian, not /tmp: this is the artefact you come back to look at.
    ap.add_argument(
        "--out",
        default="/data/shichaojian/renders/fk_projection",
    )
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--fps", type=float, default=15.0, help="playback rate; source frames are strided by 2")
    ap.add_argument("--stills", default="1,9,17,32", help="step indices for the contact sheet")
    # No --content-h/--head-top: the crop is derived from the decoded frame size
    # (canvas_crop), so there is nothing to keep in sync with the checkpoint's
    # resolution and nothing to pass wrong.
    ap.add_argument("--no-generated", action="store_true", help="skip the generated-video panel")
    args = ap.parse_args()

    import cv2

    case_dir = Path(args.case_dir)
    data = np.load(case_dir / "prediction.npz", allow_pickle=True)
    prediction, target = data["prediction"], data["target"]
    anchor = data["anchor"]
    valid = np.asarray(data["valid"], bool)
    frame_ids = np.asarray(data["frame_ids"], np.int64)
    episode = str(data["episode"])
    steps = prediction.shape[0]
    if frame_ids.size != steps + 1:
        raise ValueError(f"expected {steps + 1} frame ids, got {frame_ids.size}")

    # Camera-frame metres, absolute.  `project` takes camera frame already -- do NOT
    # route this through render_fk_overlay.project_episode, which applies base->camera
    # a second time and shifts everything by the camera's base offset.
    gt_cam = anchor[None] + target
    pred_cam = anchor[None] + prediction
    gt_uv = [project(gt_cam[t]) for t in range(steps)]
    pred_uv = [project(pred_cam[t]) for t in range(steps)]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    latent = stored_latent(data, no_generated=args.no_generated)
    generated = None
    if latent is not None:
        generated = decode_head_view(latent, args.vae)
        # A window observes ``steps + 1`` frames -- the anchor plus the steps -- and
        # the causal VAE decodes one pixel frame per observed frame
        # (1 + (T_lat-1)*4 = 33 for T_lat=9).  So the generated frame for step t is
        # index t+1, exactly the offset the real video already uses via
        # frame_ids[t+1]; index t would be the anchor, one frame early.
        if generated.shape[0] != steps + 1:
            raise ValueError(f"decoded {generated.shape[0]} frames for {steps} steps (expected {steps + 1})")

    stills_wanted = {int(x) for x in args.stills.split(",") if x.strip()}
    writer = cv2.VideoWriter(
        str(out_dir / "fk_projection.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (IMG_W * args.scale * (2 if generated is not None else 1), IMG_H * args.scale),
    )
    if not writer.isOpened():
        raise RuntimeError("VideoWriter failed to open")

    stills = []
    for t in range(steps):
        fid = int(frame_ids[t + 1])
        frame = read_frame(episode, fid)
        guv, gfront = gt_uv[t]
        puv, pfront = pred_uv[t]
        gval, pval = in_frame(guv, gfront) & valid[t], in_frame(puv, pfront) & valid[t]
        gt_series = (GT_COLOR, guv, gval)
        pred_series = (PRED_COLOR, puv, pval)

        panels = [
            draw_skeletons(
                frame,
                [gt_series, pred_series],
                label=f"REAL   {episode[-18:]}  step {t + 1}/{steps}  frame {fid}   green=GT  red=pred",
                scale=args.scale,
            )
        ]
        if generated is not None:
            panels.append(
                draw_skeletons(
                    generated[t + 1],
                    [gt_series, pred_series],
                    label=f"GENERATED (i2v)   step {t + 1}/{steps}   green=GT  red=pred",
                    scale=args.scale,
                )
            )
        canvas = np.concatenate(panels, axis=1)
        writer.write(canvas)
        if t in stills_wanted:
            stills.append(canvas)
    writer.release()

    if stills:
        cv2.imwrite(str(out_dir / "fk_projection_stills.png"), contact_sheet(stills))

    print(f"wrote {out_dir / 'fk_projection.mp4'}")
    if stills:
        print(f"wrote {out_dir / 'fk_projection_stills.png'}  ({len(stills)} steps)")
    if generated is None:
        print("note: no generated-video latent in this case (clean-video arm, or --no-generated)")
    print("\nCHECK THE GREEN SKELETON FIRST: if it is not on the hand in the REAL panel,")
    print("the projection is wrong and the rest of the picture means nothing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
