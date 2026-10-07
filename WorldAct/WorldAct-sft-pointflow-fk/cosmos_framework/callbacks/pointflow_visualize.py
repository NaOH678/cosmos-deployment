"""PointFlow panels adapted from LingBot wan_va/validation/visualize.py.

Cosmos supplies tracker UV and per-frame camera projections explicitly.
"""

import base64
import html
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image

# H.264 output parameters.  The W&B media viewer downloads a logged video whole
# before it plays, so the previous GIFs (~2.7 MB per case, ~9.5 MB stitched,
# and base64-inlined in the stage viewer HTML) never finished loading.  libx264
# is ~8x smaller for identical frames and streams progressively.
VIDEO_CRF = 28


def write_h264(frames_rgb, path, fps, crf=VIDEO_CRF):
    """Encode RGB frames to H.264 mp4 through imageio's bundled ffmpeg."""
    path = Path(path)
    if path.suffix.lower() != ".mp4":
        raise ValueError(f"Expected an .mp4 destination, got {path.name}")
    frames = [np.asarray(frame, dtype=np.uint8) for frame in frames_rgb]
    if not frames:
        raise ValueError("No frames to encode")
    imageio.mimsave(
        str(path),
        frames,
        fps=float(fps),
        codec="libx264",
        macro_block_size=1,
        ffmpeg_params=["-crf", str(crf), "-preset", "veryfast", "-pix_fmt", "yuv420p"],
    )
    return str(path)


def read_clip(path):
    """Return ``(frames_rgb, fps)`` for a rendered clip, GIF or H.264 mp4."""
    path = Path(path)
    if path.suffix.lower() == ".gif":
        with Image.open(path) as clip:
            duration = int(clip.info.get("duration", 67))
            frames = []
            for index in range(clip.n_frames):
                clip.seek(index)
                frames.append(np.asarray(clip.convert("RGB")).copy())
        if not frames:
            raise ValueError(f"No frames decoded from {path}")
        return frames, 1000.0 / max(duration, 1)
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise FileNotFoundError(path)
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    return frames, (fps if fps > 0 else 15.0)


def point_video_grid_diagnostic(canvas, uv_to_video, anchor_uv, *, pixel_stride=32, max_points=600):
    """Draw where the point tokens land on the video the model actually receives.

    Everywhere else the visualisations read the tracker's own ``uv_px`` and
    re-project the 3D labels, so they stay correct even when the point tokens are
    handed image coordinates in the wrong canvas.  This panel instead replays the
    model's own arithmetic -- the composed tracker->video affine applied to the
    anchor UV, then divided by the patch stride -- so a canvas mismatch shows up
    as points off the grid instead of staying invisible.

    ``canvas`` is the ``[C,T,H,W]`` (or ``[T,C,H,W]``/``HxWx3``) video tensor the
    batch carries; only its first frame is drawn.
    """
    frame = np.asarray(canvas)
    if frame.ndim == 4:  # [C,T,H,W] as the dataloader emits it
        frame = frame[:, 0]
        frame = np.transpose(frame, (1, 2, 0))
    if frame.dtype != np.uint8:
        frame = np.clip(frame * (255.0 if frame.max() <= 1.5 else 1.0), 0, 255).astype(np.uint8)
    height, width = frame.shape[:2]
    panel = frame.copy()

    grid_w, grid_h = width // pixel_stride, height // pixel_stride
    grid_px = np.array([grid_w, grid_h]) * pixel_stride
    cv2.rectangle(panel, (0, 0), (int(grid_px[0]) - 1, int(grid_px[1]) - 1), (255, 255, 255), 1)

    affine = np.vstack([np.asarray(uv_to_video, dtype=np.float64), [0, 0, 1]])
    uv = np.asarray(anchor_uv, dtype=np.float64)
    projected = np.c_[uv, np.ones(len(uv))] @ affine.T
    patch = (projected[:, :2] - (pixel_stride - 1) / 2.0) / pixel_stride
    inside = (patch[:, 0] >= 0) & (patch[:, 0] < grid_w) & (patch[:, 1] >= 0) & (patch[:, 1] < grid_h)

    order = np.linspace(0, len(uv) - 1, min(max_points, len(uv)), dtype=int) if len(uv) else np.empty(0, int)
    for index in order:
        x, y = patch[index] * pixel_stride + (pixel_stride - 1) / 2.0
        colour = (60, 220, 60) if inside[index] else (60, 60, 235)
        # Off-canvas points are clamped to the border rather than dropped: they are
        # exactly the failure this panel exists to show, so they must stay visible
        # instead of silently vanishing off the edge.
        point = (int(round(np.clip(x, 0, width - 1))), int(round(np.clip(y, 0, height - 1))))
        cv2.circle(panel, point, 1, colour, -1)

    fraction = float(inside.mean()) if len(uv) else float("nan")
    cv2.rectangle(panel, (0, 0), (width, 26), (0, 0, 0), -1)
    cv2.putText(
        panel,
        f"point tokens in video grid {grid_w}x{grid_h}: {fraction:.1%} green=inside red=outside",
        (5, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
    )
    return panel, dict(position_grid_inside_fraction=fraction, position_grid_wh=[grid_w, grid_h])


def trajectory_metrics(pred, gt, valid, moving):
    """Unweighted physical errors; no t=0 dilution and no loss-mask visibility."""
    if not np.isfinite(pred).all():
        raise FloatingPointError("non-finite predicted flow")
    error = np.linalg.norm(pred - gt, axis=-1)
    result = {}
    for label, mask in [("all", valid), ("moving", valid & moving[None]), ("static", valid & ~moving[None])]:
        future = mask[1:]
        result[f"{label}_count"] = int(future.sum())
        if future.any():
            result[f"{label}_ade_mm"] = float(error[1:][future].mean() * 1000)
        if mask[-1].any():
            result[f"{label}_fde_mm"] = float(error[-1][mask[-1]].mean() * 1000)
    result["valid_fraction"] = float(valid.mean())
    result["initial_flow_mm"] = float(np.linalg.norm(pred[0], axis=-1).mean() * 1000)
    return result


def render_case(record, video_path, output_dir, *, max_points=96, width=320, error_max_mm=50.0, make_video=True):
    """Return image/video paths and projection diagnostics for one fixed window."""
    if max_points < 1 or width < 64 or error_max_mm <= 0:
        raise ValueError("invalid PointFlow visualization settings")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    xyz0, valid = record["xyz0"], record["valid"].astype(bool)
    gt, pred = xyz0[None] + record["flow_gt"], xyz0[None] + record["flow"]
    frames = record["frames_bgr"]
    height, width = frames[0].shape[:2]
    source_fps = float(record["fps"])
    gt_uv, pred_uv = record["gt_uv"], record["pred_uv"]
    gt_xyz_for_draw, pred_xyz_for_draw = gt, pred
    projection_diagnostics = dict(projection_mode="tracker_uv_and_per_frame_intrinsics")
    visible = valid & np.isfinite(pred_uv).all(-1) & (pred_xyz_for_draw[..., 2] > 0.05)
    inside = visible & (pred_uv[..., 0] >= 0) & (pred_uv[..., 0] < width)
    inside &= (pred_uv[..., 1] >= 0) & (pred_uv[..., 1] < height)
    # Selection depends only on GT. Preserve stationary points as a diagnostic.
    groups = [np.flatnonzero(record["moving"] & valid.any(0)), np.flatnonzero(~record["moving"] & valid.any(0))]
    budget = max_points // 4
    static_count = min(budget, len(groups[1]))
    moving_count = min(max_points - static_count, len(groups[0]))
    static_count = min(max_points - moving_count, len(groups[1]))
    ids = (
        np.concatenate(
            [g[np.linspace(0, len(g) - 1, n, dtype=int)] for g, n in zip(groups, (moving_count, static_count)) if n]
        )
        if (moving_count + static_count)
        else np.array([], dtype=int)
    )
    green, purple = (50, 230, 50), (230, 70, 220)
    # Like sonata_motion: small context points and trails only for a few
    # moving points. Use the same GT-selected IDs in both prediction panels.
    moving_ids = ids[record["moving"][ids]]
    trail_ids = (
        set(moving_ids[np.linspace(0, len(moving_ids) - 1, min(16, len(moving_ids)), dtype=int)])
        if len(moving_ids)
        else set()
    )

    def draw(frame, uv, xyz, step, color):
        for q in ids:
            for t in range(max(1, step - 5), step + 1) if q in trail_ids else ():
                if not (valid[t - 1 : t + 1, q].all() and (xyz[t - 1 : t + 1, q, 2] > 0.05).all()):
                    continue
                points = uv[t - 1 : t + 1, q]
                if not np.isfinite(points).all():
                    continue
                # Clamp before conversion to avoid overflow for divergent predictions.
                a, b = np.rint(np.clip(points, -100000, 100000)).astype(int)
                accepted, a, b = cv2.clipLine((0, 0, width, height), tuple(a), tuple(b))
                if accepted:
                    cv2.line(frame, a, b, color, 1, cv2.LINE_AA)
            point = uv[step, q]
            if (
                valid[step, q]
                and xyz[step, q, 2] > 0.05
                and np.isfinite(point).all()
                and (point >= 0).all()
                and point[0] < width
                and point[1] < height
            ):
                cv2.circle(frame, tuple(np.rint(point).astype(int)), 1, color, -1)

    def triptych(step):
        panels = [frames[step].copy() for _ in range(3)]
        draw(panels[0], gt_uv, gt_xyz_for_draw, step, green)
        draw(panels[1], pred_uv, pred_xyz_for_draw, step, purple)
        draw(panels[2], gt_uv, gt_xyz_for_draw, step, green)
        draw(panels[2], pred_uv, pred_xyz_for_draw, step, purple)
        if "fk_pred_uv" in record:
            from cosmos_framework.callbacks.fk_visualize import EDGES

            for panel, kinds in zip(panels, (("gt",), ("pred",), ("gt", "pred")), strict=True):
                for kind in kinds:
                    uv = record[f"fk_{kind}_uv"][step]
                    mask = record[f"fk_{kind}_valid"][step]
                    color = (0, 220, 255) if kind == "gt" else (255, 180, 0)
                    hand_edges = [
                        (a + offset, b + offset)
                        for offset in range(0, len(uv), 21)
                        for a, b in EDGES
                        if b + offset < len(uv)
                    ]
                    for a, b in hand_edges:
                        if mask[a] and mask[b]:
                            xy = np.rint(np.clip(uv[[a, b]], -100000, 100000)).astype(int)
                            accepted, left, right = cv2.clipLine((0, 0, width, height), tuple(xy[0]), tuple(xy[1]))
                            if accepted:
                                cv2.line(panel, left, right, color, 1, cv2.LINE_AA)
        titles = ("GT | tracker UV", "Pred | " + str(record.get("prediction_kind", "estimate")), "GT + Pred")
        for panel, title in zip(panels, titles):
            cv2.rectangle(panel, (0, 0), (width, 24), (0, 0, 0), -1)
            cv2.putText(panel, f"{title} | +{step}", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return np.concatenate(panels, axis=1)

    comparison = np.concatenate([triptych(int(t)) for t in np.linspace(0, len(gt) - 1, 4)], axis=0)
    comparison_path = output_dir / "comparison.png"
    Image.fromarray(cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB)).save(comparison_path)
    errors = np.linalg.norm(record["flow"] - record["flow_gt"], axis=-1) * 1000
    count = valid[1:].sum(0)
    mean_error = np.where(valid[1:], errors[1:], 0).sum(0) / np.maximum(count, 1)
    colors = cv2.applyColorMap(np.uint8(np.clip(mean_error / error_max_mm, 0, 1) * 255), cv2.COLORMAP_TURBO).reshape(
        -1, 3
    )
    error_image = frames[0].copy()
    positions = gt_uv[0]
    for q in np.flatnonzero(count):
        p = positions[q]
        if np.isfinite(p).all() and (p >= 0).all() and p[0] < width and p[1] < height:
            cv2.circle(error_image, tuple(np.rint(p).astype(int)), 1, tuple(int(v) for v in colors[q]), -1)
    cv2.rectangle(error_image, (0, 0), (width, 24), (0, 0, 0), -1)
    cv2.putText(
        error_image, f"ADE: blue=0 red>={error_max_mm:g}mm", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
    )
    error_path = output_dir / "error_map.png"
    Image.fromarray(cv2.cvtColor(error_image, cv2.COLOR_BGR2RGB)).save(error_path)
    paths = dict(comparison=str(comparison_path), error_map=str(error_path))
    if make_video:
        video_path_out = output_dir / "comparison.mp4"
        video_frames = [cv2.cvtColor(triptych(t), cv2.COLOR_BGR2RGB) for t in range(len(gt))]
        paths["video"] = write_h264(video_frames, video_path_out, source_fps)
    diagnostics = dict(
        **projection_diagnostics,
        pred_outside_fraction=float((valid & ~inside).sum() / max(1, valid.sum())),
        displayed_points=len(ids),
    )
    return paths, diagnostics


def stitch_windows(paths, output_path):
    """Join rendered windows, keeping each boundary once; never join point identities."""
    frames, fps = [], None
    for window, path in enumerate(paths):
        clip_frames, clip_fps = read_clip(path)
        if fps is None:
            fps = clip_fps
        for index, frame in enumerate(clip_frames):
            if window and index == 0:
                continue
            frame = frame.copy()
            cv2.rectangle(frame, (0, 24), (frame.shape[1], 43), (0, 0, 0), -1)
            cv2.putText(
                frame,
                f"Window {window + 1}/{len(paths)} | independent prediction",
                (5, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )
            frames.append(frame)
    if not frames:
        raise ValueError("No frames to stitch")
    return write_h264(frames, output_path, fps)


def stage_viewer(cases, output_path):
    """Self-contained stage viewer; no remote assets or web server.

    Exactly three entries (early/middle/late) get the position slider.  More
    entries — one per (stage, kind), e.g. conditional + joint rollout + dream
    canvas — render as a grid with one row per ``kind`` and one column per
    stage so all canvases of a stage play side by side.
    """
    if [case["stage"] for case in cases] == ["early", "middle", "late"]:
        page = _stage_slider_page(cases)
    else:
        page = _stage_grid_page(cases)
    path = Path(output_path)
    path.write_text(page)
    path.with_suffix(".json").write_text(json.dumps(cases, indent=2) + "\n")
    return str(path)


def _embedded_media(video_path, label):
    video = Path(video_path)
    image = base64.b64encode(video.read_bytes()).decode("ascii")
    # H.264 streams in the browser; a base64 GIF of the same frames did not.
    if video.suffix.lower() == ".mp4":
        return f'<video autoplay loop muted playsinline src="data:video/mp4;base64,{image}" style="width:100%"></video>'
    return f'<img alt="{html.escape(label)}" src="data:image/gif;base64,{image}" style="width:100%">'


def _stage_slider_page(cases):
    panels = []
    for i, case in enumerate(cases):
        label = html.escape(case["label"])
        hidden = " hidden" if i else ""
        panels.append(
            f'<section id="stage-{i}"{hidden}><p>{label}</p>{_embedded_media(case["video"], case["label"])}</section>'
        )
    return (
        """<!doctype html><meta charset="utf-8"><title>Validation stages</title>
<style>body{font:14px sans-serif;margin:12px;background:#171717;color:#eee}
input{width:100%}.labels{display:flex;justify-content:space-between}section[hidden]{display:none}</style>
<h3>Validation: early / middle / late</h3>
<p>Independent windows | GT / Prediction / Overlay | playback 1x | 32-frame prediction windows; window count shown per stage</p>
"""
        + "\n".join(panels)
        + """
<input aria-label="Validation stage" type="range" min="0" max="2" step="1" value="0" id="stage-slider">
<div class="labels"><span>Early / 前段</span><span>Middle / 中段</span><span>Late / 后段</span></div>
<script>document.getElementById('stage-slider').addEventListener('input',function(){
for(let i=0;i<3;i++){document.getElementById('stage-'+i).hidden=(i!==Number(this.value));}});</script>
"""
    )


def _stage_grid_page(cases):
    stage_order = [stage for stage in ("early", "middle", "late") if any(case["stage"] == stage for case in cases)]
    kinds = []
    for case in cases:
        kind = case.get("kind", "conditional")
        if kind not in kinds:
            kinds.append(kind)
    cells = {(case["stage"], case.get("kind", "conditional")): case for case in cases}
    header = "".join(f"<th>{stage}</th>" for stage in stage_order)
    rows = []
    for kind in kinds:
        tds = []
        for stage in stage_order:
            case = cells.get((stage, kind))
            if case is None:
                tds.append("<td></td>")
                continue
            tds.append(f"<td><p>{html.escape(case['label'])}</p>{_embedded_media(case['video'], case['label'])}</td>")
        rows.append(f'<tr><th class="row-label">{html.escape(kind)}</th>{"".join(tds)}</tr>')
    return (
        """<!doctype html><meta charset="utf-8"><title>Validation stages</title>
<style>body{font:14px sans-serif;margin:12px;background:#171717;color:#eee}
table{border-collapse:collapse;width:100%}td,th{border:1px solid #333;vertical-align:top;padding:6px}
.row-label{writing-mode:vertical-rl;text-orientation:middle;white-space:nowrap;width:1em}</style>
<h3>Validation: early / middle / late, one row per canvas</h3>
<p>Independent windows | GT / Prediction / Overlay | playback 1x | 32-frame prediction windows; window count shown per stage</p>
"""
        + f"<table><tr><th></th>{header}</tr>{''.join(rows)}</table>"
    )
