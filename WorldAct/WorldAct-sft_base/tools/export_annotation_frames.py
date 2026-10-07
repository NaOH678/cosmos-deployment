#!/usr/bin/env python3
"""Export clean frames for manual hand annotation.

Purpose: every measurement so far has used FK to say where the hand *should* be,
so FK's own camera extrinsic is both the suspect and the ruler.  Hand-clicked
landmarks are independent of it, which is the only way to get a non-circular
number for the residual offset.

The exported frames deliberately carry **no FK overlay** — seeing FK's guess
would bias the annotator.  Only a light coordinate grid is drawn, to make the
clicked pixel coordinates easy to read off.

Frame selection uses FK only for *choosing* frames (hand nearly still, whole hand
inside the image, fingers spread enough that the tips are separable).  That does
not bias the annotation itself.

Usage:
    python tools/export_annotation_frames.py --episode episode_0013_20260731_133649 \
        --out /path/to/annotate --n 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian")
RAW = ROOT / "raw_data"

TIPS = [4, 8, 12, 16, 20]          # thumb..pinky tip indices in FK-21
KEYPOINT_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

COLOR_FX, COLOR_FY = 605.5706176757812, 604.4129638671875
COLOR_CX, COLOR_CY = 324.4994812011719, 238.25637817382812

MINOR = 25         # px between thin grid lines
MAJOR = 100        # px between thick, labelled lines
SCALE = 2          # exported at 2x so clicking is easier


def draw_grid(img):
    """Two-level grid in ORIGINAL pixel coordinates (before scaling).

    Minor lines every 25 px, major lines every 100 px with the coordinate printed
    at every major intersection, so a value can be read off without counting.
    """
    import cv2

    h, w = img.shape[:2]
    for x in range(0, w, MINOR):
        major = (x % MAJOR == 0)
        cv2.line(img, (x, 0), (x, h), (70, 70, 70) if major else (48, 48, 48), 1)
    for y in range(0, h, MINOR):
        major = (y % MAJOR == 0)
        cv2.line(img, (0, y), (w, y), (70, 70, 70) if major else (48, 48, 48), 1)

    for y in range(0, h, MAJOR):
        for x in range(0, w, MAJOR):
            cv2.putText(img, f"{x},{y}", (x + 3, y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (215, 215, 215), 1)
    for y in range(0, h, MAJOR):                    # left/right edge rulers
        cv2.putText(img, str(y), (3, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (150, 220, 255), 1)
    for x in range(0, w, MAJOR):                    # top/bottom edge rulers
        cv2.putText(img, str(x), (x + 3, h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (150, 220, 255), 1)

    # principal point, as a coordinate landmark
    cv2.drawMarker(img, (int(COLOR_CX), int(COLOR_CY)), (0, 180, 255), cv2.MARKER_CROSS, 16, 1)
    cv2.putText(img, f"principal point ({COLOR_CX:.1f},{COLOR_CY:.1f})",
                (int(COLOR_CX) + 12, int(COLOR_CY) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 180, 255), 1)
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="singlerighthand_sandwich_100")
    ap.add_argument("--episode", default="episode_0013_20260731_133649")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-bend-deg", type=float, default=40.0,
                    help="max angle between consecutive finger bones; the hand must be open\n"
                         "enough that the fingertips are not hidden behind the palm")
    args = ap.parse_args()

    import cv2

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from verify_fk_camera_projection import base_to_head_camera, roll_about_z, URDF

    fk_path = RAW / "sandwich_fk21" / args.episode / "annotations" / "wuji_fk21.npz"
    if not fk_path.is_file():
        print(f"FATAL: no FK annotation at {fk_path}", flush=True)
        return 1
    video = RAW / args.dataset / args.episode / "videos" / "head.mp4"
    if not video.is_file():
        print(f"FATAL: no video at {video}", flush=True)
        return 1

    fk = np.load(fk_path, allow_pickle=True)
    P = fk["positions"][:, 1]                       # (T, 21, 3) right hand, Link_Base
    T = P.shape[0]

    R, t = base_to_head_camera(URDF)
    M = roll_about_z(180.0)
    R, t = M @ R, M @ t

    c = P.mean(1)
    disp = np.r_[np.zeros(5), np.linalg.norm(c[5:] - c[:-5], axis=1)] * 100

    cam = P @ R.T + t
    with np.errstate(divide="ignore", invalid="ignore"):
        u = COLOR_FX * cam[..., 0] / cam[..., 2] + COLOR_CX
        v = COLOR_FY * cam[..., 1] / cam[..., 2] + COLOR_CY
    inside = (u > 40) & (u < 600) & (v > 40) & (v < 440) & (cam[..., 2] > 0.2)
    n_inside = inside.sum(1)

    spread = np.array([np.linalg.norm(P[f, TIPS][:, None] - P[f, TIPS][None], axis=-1).max()
                       for f in range(T)])

    # Finger bend, per finger: the angle between consecutive bone directions.  A
    # closed fist hides its own fingertips behind the palm, so a frame where the
    # fingers are curled cannot be annotated at all — this is the filter that
    # matters most, and the grasping episodes fail it completely.
    def bend_of(finger: int):
        b = 1 + 4 * finger          # mcp index; +1 pip, +2 dip, +3 tip
        seg = [P[:, b + 1] - P[:, b], P[:, b + 2] - P[:, b + 1], P[:, b + 3] - P[:, b + 2]]
        seg = [s / (np.linalg.norm(s, axis=1, keepdims=True) + 1e-9) for s in seg]
        return np.maximum(
            np.degrees(np.arccos(np.clip((seg[0] * seg[1]).sum(1), -1, 1))),
            np.degrees(np.arccos(np.clip((seg[1] * seg[2]).sum(1), -1, 1))),
        )

    worst_bend = np.stack([bend_of(f) for f in range(5)]).max(0)

    def select(bend_max, disp_max, n_ins, spread_min):
        return np.nonzero(
            (worst_bend < bend_max) & (disp < disp_max) & (n_inside >= n_ins) & (spread > spread_min)
        )[0]

    # Hand motion is deliberately NOT required: the annotator looks at a single
    # still, so a fast-moving hand annotates just as well.  Motion only matters
    # for depth-based measurement, which is a separate step.
    cand = select(args.max_bend_deg, 0.8, 21, 0.06)
    for bend_extra, disp_max, n_ins, spread_min in (
        (args.max_bend_deg, 1.5, 21, 0.05),
        (args.max_bend_deg, 3.0, 21, 0.04),
        (args.max_bend_deg, 99.0, 21, 0.04),      # motion ignored entirely
        (args.max_bend_deg + 10, 99.0, 21, 0.03),
        (args.max_bend_deg + 20, 99.0, 20, 0.02),
    ):
        if len(cand) >= args.n:
            break
        cand = select(bend_extra, disp_max, n_ins, spread_min)
    if len(cand) == 0:
        print("FATAL: no frame passed the filters", flush=True)
        return 1

    # Greedy: best score first, skipping anything too close to an existing pick, so
    # the choices spread over the episode instead of clustering where candidates
    # happen to be dense.
    score = spread - 0.02 * disp
    order = cand[np.argsort(-score[cand])]
    min_gap = max(1, T // (2 * args.n))
    picks: list[int] = []
    for f in order:
        if all(abs(int(f) - p) >= min_gap for p in picks):
            picks.append(int(f))
        if len(picks) >= args.n:
            break
    # No fallback fill: adjacent frames add nothing for annotation, and stuffing
    # them in would defeat the separation this loop exists to enforce.  Fewer,
    # well-separated frames are more useful than a cluster of near-duplicates.
    picks = sorted(picks)

    args.out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    print(f"{args.episode}: {T} frames, {len(cand)} candidates, exporting {len(picks)}", flush=True)
    for f in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        okr, img = cap.read()
        if not okr:
            continue
        draw_grid(img)
        big = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
        path = args.out / f"frame_{f:04d}.png"
        cv2.imwrite(str(path), big)
        print(f"  f{f:5d}  hand speed {disp[f]:.2f} cm/1/6s  spread {spread[f]*100:.1f} cm  -> {path.name}", flush=True)
    cap.release()

    # CSV template for the clicks
    cols = ["frame"] + [f"{n}_u" for n in KEYPOINT_NAMES] + [f"{n}_v" for n in KEYPOINT_NAMES]
    csv = args.out / "landmarks.csv"
    with open(csv, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for f in picks:
            fh.write(",".join([str(f)] + [""] * (len(cols) - 1)) + "\n")

    readme = args.out / "README.txt"
    readme.write_text(
        f"人工标注：{args.episode}\n"
        f"{'=' * 64}\n\n"
        "目的：拿到一个【不依赖 FK】的手部像素位置真值，用来量 FK 投影的残余偏移。\n"
        "     图上刻意没有画 FK —— 画了会被它的猜测带偏。\n\n"
        f"图的规格：放大 {SCALE}x，但坐标一律用【原图】的 (0-639 横向 / 0-479 纵向)。\n"
        f"网格：细线每 {MINOR} px，粗线每 {MAJOR} px。每个粗线交点都印了「x,y」坐标，\n"
        "      边缘还有刻度，可以直接读数，不用数格子。\n"
        f"橙色十字 = 相机主点 (cx={COLOR_CX:.1f}, cy={COLOR_CY:.1f})。\n\n"
        "── 最少只需要标 6 个点 ──────────────────────────────────────\n\n"
        "              ┌─────────────────────────────┐\n"
        "              │          手背 / 手心          │\n"
        "              │                             │\n"
        "   拇指 ──►   ╱                             │\n"
        "   thumb_tip ●                             │\n"
        "              ╲                            │\n"
        "               │   ● index_tip   (食指)     │\n"
        "               │   ● middle_tip  (中指)     │\n"
        "               │   ● ring_tip    (无名指)    │\n"
        "               │   ● pinky_tip   (小指)     │\n"
        "              └──────────●─────────────────┘\n"
        "                        wrist\n\n"
        "  wrist      = 手腕。手套袖口与手掌交界那条褶皱的中心。\n"
        "  *_tip      = 各手指指尖。标在手套手指的最末端、正中间。\n\n"
        "  其余 15 列（cmc/mcp/pip/dip 这些指关节）留空即可 —— 6 个点就够用。\n\n"
        "── 怎么填 ──────────────────────────────────────────────────\n\n"
        "  landmarks.csv：每帧一行，填 u（列号，0=最左）和 v（行号，0=最上）。\n"
        "  看不清的点就留空，不要猜。\n\n"
        f"帧列表：{picks}\n",
        encoding="utf-8",
    )
    print(f"\nwrote {len(picks)} frames + landmarks.csv + README.txt to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
