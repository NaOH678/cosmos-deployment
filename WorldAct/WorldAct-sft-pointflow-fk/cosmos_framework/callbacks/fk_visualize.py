"""Rendering and metrics for the FK eval: 21 keypoints, camera-frame metres.

Deliberately self-contained -- no source video, no uv, no canvas.  v1 does not
resolve where FK lands in the image (that mapping is exactly what the design
skips; see docs/fk_modality_design.md §1.4a), so overlaying on the video would
need the very transform the modality was built to avoid, and a picture that
relied on it could look right while being wrong.

⚠️ That paragraph conflated two different things and is worth correcting in place.
What the design skips is feeding *uv to the model* -- as a token feature or a
position encoding -- because it is derivable from the 3D position and would let the
model shortcut learning 3D.  The 3D->camera transform itself is not skipped: it is
how the labels are made (``fk_source.py`` uses ``base_to_camera``, coordinate
``camera_d435_real``), and it is fully determined -- ``fk_camera_extrinsic`` equals
``roll_about_z(180) @ base_to_head_camera(URDF)`` to 4e-13.  So a *rendering* in the
image plane is legitimate; it is a diagnostic about the output, not an input the
model was given.  That rendering lives in ``tools/render_fk_projection.py``, kept out
of this module so the eval's own figures stay free of the transform.

What is left is still the readable thing: the hand's skeleton as the model
believes it moves, against the recorded skeleton, in the camera frame the model
actually predicts in, plus the error against the future step.
"""

import json
from pathlib import Path

import numpy as np

# Mirrors tools/verify_fk_camera_projection.py so a skeleton drawn here and one
# drawn there are the same object.
KEYPOINT_NAMES = [
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
]
FINGER_NAMES = ("wrist", "thumb", "index", "middle", "ring", "pinky")
FINGER_COLORS = ("#7f7f7f", "#d62728", "#ff7f0e", "#bcbd22", "#2ca02c", "#1f77b4")
EDGES = [(0, 1), (0, 5), (0, 9), (0, 13), (0, 17)]
for base in (1, 5, 9, 13, 17):
    EDGES += [(base, base + 1), (base + 1, base + 2), (base + 2, base + 3)]
KEYPOINTS = len(KEYPOINT_NAMES)


def finger_of(index: int) -> int:
    return 0 if index == 0 else (index - 1) // 4 + 1


def trajectory_metrics(prediction, target, valid=None):
    """Per-step mm error over the keys the loss actually supervises.

    ``all_ade_mm`` is the number the design's verdict is stated in, and it is
    only meaningful next to ``zero_all_ade_mm`` -- a stationary trajectory that
    predicts no displacement at all.  The masked MSE the trainer reports cannot
    answer this: even a prediction of zero scores 1.0 there, because the
    unit-variance noise dominates.
    """
    prediction, target = np.asarray(prediction, float), np.asarray(target, float)
    if prediction.shape != target.shape:
        raise ValueError(f"prediction {prediction.shape} and target {target.shape} differ")
    horizon = prediction.shape[0]
    if valid is None:
        valid = np.ones(prediction.shape[:2], bool)
    valid = np.asarray(valid, bool)
    error = np.linalg.norm(prediction - target, axis=-1) * 1000.0  # [H,N] mm
    per_step = np.array([error[h][valid[h]].mean() if valid[h].any() else np.nan for h in range(horizon)])
    present = valid.any(axis=1)
    final = error[horizon - 1][valid[horizon - 1]].mean() if valid[horizon - 1].any() else np.nan
    return {
        "all_ade_mm": float(np.nanmean(per_step)),
        "final_ade_mm": float(final),
        "per_step_ade_mm": per_step.tolist(),
        "supervised_steps": int(present.sum()),
    }


def per_keypoint_error(prediction, target, valid=None):
    prediction, target = np.asarray(prediction, float), np.asarray(target, float)
    if valid is None:
        valid = np.ones(prediction.shape[:2], bool)
    error = np.linalg.norm(prediction - target, axis=-1) * 1000.0  # [H,N] mm
    valid = np.asarray(valid, bool)
    out = {}
    for k, name in enumerate(KEYPOINT_NAMES):
        column = error[:, k][valid[:, k]]
        out[f"{name}_ade_mm"] = float(column.mean()) if column.size else float("nan")
    return out


def render_case(case, output_dir, *, title=None):
    """Write ``comparison.png`` and ``error_curve.png`` for one eval case.

    ``case`` holds ``target``/``prediction`` as ``[H,N,3]`` camera-frame metres
    and ``anchor`` as ``[N,3]``; the anchor is drawn too, because a skeleton that
    is offset from the start is a different failure from one that moves wrongly.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = np.asarray(case["target"], float)
    prediction = np.asarray(case["prediction"], float)
    anchor = np.asarray(case["anchor"], float)
    horizon = target.shape[0]
    steps = [0, horizon // 4, horizon // 2, horizon - 1]

    fig = plt.figure(figsize=(4.2 * len(steps), 8.2))
    for column, step in enumerate(steps):
        gt = anchor + target[step]
        pred = anchor + prediction[step]
        # Two orthographic views of the same camera-frame points.  Depth (z) is
        # what the RGB projection hides, so it gets its own panel.
        for row, (ix, iy, ylabel) in enumerate(((0, 1, "y (down, m)"), (0, 2, "z (depth, m)"))):
            axes = fig.add_subplot(2, len(steps), row * len(steps) + column + 1)
            for edge in EDGES:
                a, b = edge
                colour = FINGER_COLORS[finger_of(a)]
                axes.plot(gt[[a, b], ix], gt[[a, b], iy], color=colour, lw=2.0, alpha=0.9)
                axes.plot(pred[[a, b], ix], pred[[a, b], iy], color=colour, lw=1.4, ls="--", alpha=0.9)
            axes.scatter(gt[:, ix], gt[:, iy], s=8, c="k", zorder=3)
            axes.scatter(pred[:, ix], pred[:, iy], s=8, c="w", edgecolors="k", zorder=3)
            axes.set_aspect("equal", adjustable="datalim")
            axes.grid(alpha=0.25)
            if row == 0:
                axes.set_title(f"step {step + 1}/{horizon}", fontsize=10)
            if column == 0:
                axes.set_ylabel(ylabel, fontsize=9)
            axes.tick_params(labelsize=7)
    fig.suptitle(
        f"{title or case.get('case_id', 'fk')}   solid = GT, dashed = prediction   "
        f"ADE {case['metrics']['all_ade_mm']:.2f} mm  (zero {case['metrics']['zero_all_ade_mm']:.2f} mm)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "comparison.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(figsize=(6.4, 3.6))
    axes.plot(range(1, horizon + 1), case["metrics"]["per_step_ade_mm"], marker="o", ms=3, label="ADE (mm)")
    axes.axhline(case["metrics"]["zero_all_ade_mm"], color="k", ls=":", lw=1.2, label="zero-motion baseline")
    axes.set_xlabel("future step")
    axes.set_ylabel("mm")
    axes.grid(alpha=0.25)
    axes.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "error_curve.png", dpi=110)
    plt.close(fig)

    return {"comparison.png": output_dir / "comparison.png", "error_curve.png": output_dir / "error_curve.png"}


def write_video(case, output_dir, *, fps=8, size=(6.4, 4.8)):
    """``comparison.mp4``: the skeleton stepping through the horizon.

    The triptych in ``comparison.png`` shows four snapshots; drift that develops
    between them, or a limb that swings the wrong way for one step and recovers,
    only shows up in motion.  Same two orthographic views, animated.
    """
    import imageio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = np.asarray(case["target"], float)
    prediction = np.asarray(case["prediction"], float)
    anchor = np.asarray(case["anchor"], float)
    horizon = target.shape[0]
    path = output_dir / "comparison.mp4"

    fig = plt.figure(figsize=size)
    writer = imageio.get_writer(path, fps=fps, macro_block_size=1)
    try:
        for step in range(horizon):
            fig.clf()
            gt, pred = anchor + target[step], anchor + prediction[step]
            for column, (ix, iy, ylabel) in enumerate(((0, 1, "y (down, m)"), (0, 2, "z (depth, m)"))):
                axes = fig.add_subplot(1, 2, column + 1)
                for a, b in EDGES:
                    colour = FINGER_COLORS[finger_of(a)]
                    axes.plot(gt[[a, b], ix], gt[[a, b], iy], color=colour, lw=2.0, alpha=0.9)
                    axes.plot(pred[[a, b], ix], pred[[a, b], iy], color=colour, lw=1.4, ls="--", alpha=0.9)
                axes.scatter(gt[:, ix], gt[:, iy], s=8, c="k", zorder=3)
                axes.scatter(pred[:, ix], pred[:, iy], s=8, c="w", edgecolors="k", zorder=3)
                axes.set_aspect("equal", adjustable="datalim")
                axes.grid(alpha=0.25)
                axes.set_ylabel(ylabel, fontsize=9)
                axes.tick_params(labelsize=7)
            err = float(np.linalg.norm(prediction[step] - target[step], axis=-1).mean() * 1000.0)
            fig.suptitle(
                f"{case.get('case_id', 'fk')}  step {step + 1}/{horizon}   "
                f"step error {err:.1f} mm   solid GT / dashed prediction",
                fontsize=10,
            )
            fig.tight_layout()
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            writer.append_data(frame)
    finally:
        writer.close()
        plt.close(fig)
    return path


def write_case(case, output_dir, *, title=None, draw=True):
    """``prediction.npz`` + ``metrics.json``, and optionally the two figures.

    ``draw=False`` writes only the data.  The figures are matplotlib at ~0.3 s a
    frame, and ``write_video`` renders 32 of them per case on *every* eval -- 128
    figures per validation pass, which is real time against a single-GPU step and
    pure cost for a step nobody will look at.  ``tools/render_fk_case.py`` makes the
    identical figures offline from the npz, so deferring them loses nothing and can
    be done for any past step.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "prediction.npz",
        prediction=np.asarray(case["prediction"], np.float32),
        target=np.asarray(case["target"], np.float32),
        anchor=np.asarray(case["anchor"], np.float32),
        valid=np.asarray(case["valid"], bool) if case.get("valid") is not None else None,
        keypoint_names=np.asarray(KEYPOINT_NAMES),
        frame_ids=np.asarray(case.get("frame_ids", []), np.int64),
        episode=np.asarray(case.get("episode", "")),
        units=np.asarray("metre"),
        coordinate=np.asarray("camera_d435_real"),
        # The joint arm's generated video, as a LATENT -- `[N,C,T,H,W]` float16, one
        # entry per sample.  Written here rather than decoded because a VAE pass per
        # eval is pure cost when nothing scores it; tools/render_fk_projection.py
        # decodes it offline.  Absent (None) for the clean-video arm, which has no
        # generated video to show.  This key list is explicit, so a field added to
        # _record's dict but not here is silently dropped.
        vision=np.asarray(case["vision"], np.float16) if case.get("vision") is not None else None,
    )
    (output_dir / "metrics.json").write_text(json.dumps(case["metrics"], indent=2, sort_keys=True) + "\n")
    if not draw:
        return {}
    return render_case(case, output_dir, title=title)
