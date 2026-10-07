"""Audit PointFlow with its own DA3 intrinsics and FK with its existing projector."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from cosmos_framework.data.fk_camera_extrinsic import base_to_camera
from tools.verify_fk_camera_projection import EDGES, IMG_H, IMG_W, project


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--cache-root",
        type=Path,
        default=Path("/data/shichaojian/datasets/singlerighthand-sandwich-100-cosmos-cache/pointflow_windows"),
    )
    p.add_argument("--fk-root", type=Path, default=Path("/data/shichaojian/raw_data/sandwich_fk21"))
    p.add_argument("--raw-root", type=Path, default=Path("/data/shichaojian/raw_data/singlerighthand_sandwich_100"))
    p.add_argument("--geometry-root", type=Path, default=Path("/data/shichaojian/pf_out/9.24/sandwich/efep_seg_v61"))
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    rows = []
    images = []
    point_errors = []
    for path in sorted(a.cache_root.glob("episode_*.npz"))[:8]:
        with np.load(a.fk_root / path.stem / "annotations/wuji_fk21.npz") as f:
            positions = f["positions"][:, 1]
        source_frames = np.load(a.geometry_root / path.stem / "frame_indices.npy")
        intrinsics = np.load(a.geometry_root / path.stem / "intrinsics.npy")
        with np.load(path) as data:
            windows = sorted(int(k.split("/")[0]) for k in data.files if k.endswith("/anchor_xyz"))
            for i in [0, len(windows) // 2, len(windows) - 1]:
                w = str(windows[i])
                frame = int(data[w + "/raw_frame_ids"][0])
                xyz = data[w + "/anchor_xyz"]
                uv = data[w + "/anchor_uv"]
                wh = data[w + "/image_size_wh"]
                cam = base_to_camera(positions[frame])
                fk_uv, front = project(cam)
                # Check the old projector against actual recorded intrinsics.
                meta = json.loads((a.raw_root / path.stem / "auxiliary_camera/metadata.json").read_text())
                recorded = meta["capture_metadata"]["cameras"]["head"]["streams"]["color"]["intrinsics"]
                projected = np.stack(
                    [
                        recorded["fx"] * cam[:, 0] / cam[:, 2] + recorded["ppx"],
                        recorded["fy"] * cam[:, 1] / cam[:, 2] + recorded["ppy"],
                    ],
                    -1,
                )
                old_error = float(abs(projected - fk_uv).max())
                # Use the recorded DA3 intrinsic for this exact source frame.
                # Normalized image coordinates refer to pixel centers:
                # pixel_uv = normalized_uv * image_size - 0.5.
                matches = np.flatnonzero(source_frames == frame)
                if len(matches) != 1:
                    raise ValueError(f"Expected one intrinsic row for source frame {frame}")
                intrinsic_row = int(matches[0])
                k = intrinsics[intrinsic_row]
                if not np.isfinite(xyz).all() or np.any(xyz[:, 2] <= 0):
                    raise ValueError("PointFlow anchor must have finite positive depth")
                q = xyz @ k.T
                da_uv = q[:, :2] / q[:, 2:] * wh - 0.5
                da_error = np.linalg.norm(da_uv - uv, axis=1)
                point_errors.append(da_error)
                # Draw XYZ reprojected through DA3, rather than the stored UV.
                pf_head = (da_uv + 0.5) * [IMG_W, IMG_H] / wh - 0.5
                rows.append(
                    dict(
                        episode=path.stem,
                        window=int(w),
                        frame=frame,
                        existing_projector_vs_recorded_K_max_px=old_error,
                        point_da3_reprojection_px=np.quantile(da_error, [0.5, 0.95, 1]).tolist(),
                        point_count=len(xyz),
                        intrinsic_row=intrinsic_row,
                        intrinsics_normalized=k.tolist(),
                        fk_depth_m=np.quantile(cam[:, 2], [0, 0.5, 1]).tolist(),
                        point_depth_m=np.quantile(xyz[:, 2], [0, 0.5, 1]).tolist(),
                    )
                )
                if i == 0:
                    cap = cv2.VideoCapture(str(a.raw_root / path.stem / "videos/head.mp4"))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                    ok, image = cap.read()
                    cap.release()
                    if not ok:
                        raise RuntimeError("Cannot read source frame")
                    for xy in pf_head:
                        cv2.circle(image, tuple(np.round(xy).astype(int)), 1, (80, 230, 80), -1)
                    for j, l in EDGES:
                        if front[j] and front[l]:
                            cv2.line(
                                image,
                                tuple(np.round(fk_uv[j]).astype(int)),
                                tuple(np.round(fk_uv[l]).astype(int)),
                                (50, 50, 255),
                                2,
                            )
                    cv2.putText(
                        image,
                        f"{path.stem[:12]} frame {frame} | green PF via DA3, red FK",
                        (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
                    images.append(image)
    result = dict(
        scope="24 anchors, first 8 sorted episodes; not a calibrated 3D correspondence test",
        point_count=sum(len(errors) for errors in point_errors),
        point_da3_reprojection_quantiles=[0.5, 0.95, 0.99, 1.0],
        point_da3_reprojection_px=np.quantile(np.concatenate(point_errors), [0.5, 0.95, 0.99, 1.0]).tolist(),
        windows=rows,
        note="PointFlow XYZ uses its own DA3 K; FK uses the existing MANO projector. No cross-application of intrinsics.",
    )
    (a.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    cv2.imwrite(
        str(a.output / "anchor_overlays.jpg"),
        np.concatenate([np.concatenate(images[i : i + 2], 1) for i in range(0, len(images), 2)], 0),
    )
    print(
        json.dumps(
            dict(
                windows=len(rows),
                projector_max_error=max(r["existing_projector_vs_recorded_K_max_px"] for r in rows),
                point_count=result["point_count"],
                point_da3_reprojection_px=result["point_da3_reprojection_px"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
