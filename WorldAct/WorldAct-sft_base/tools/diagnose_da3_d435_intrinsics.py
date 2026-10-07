"""Controlled DA3 diagnostic: predicted K vs calibrated RGB K in metric scaling + XYZ.

Uses the same raw predictions for both variants; never fits to FK or reads sensor
 depth. This four-frame DA3 rerun is NOT a reproduction of the full Track4World
export. Run with the Track4World vendored depth_anything_3 on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from addict import Dict
from omegaconf import OmegaConf
from safetensors.torch import load_file
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_fk_vs_pointcloud import read_frame

from cosmos_framework.data.fk_camera_extrinsic import base_to_camera


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", default="episode_0013_20260731_133649")
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 300, 650, 1000])
    parser.add_argument("--checkpoint", default="/data/shichaojian/checkpoints/DA3NESTED-GIANT-LARGE-1.1")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.render_only:
        report = json.loads((out / "report.json").read_text())
        with np.load(out / "comparison.npz") as saved:
            arrays = {k: saved[k] for k in saved.files}
        render_comparison(out, report, arrays)
        return
    raw = Path("/data/shichaojian/raw_data/singlerighthand_sandwich_100") / args.episode
    metadata_path = raw / "auxiliary_camera/metadata.json"
    intr = json.loads(metadata_path.read_text())["capture_metadata"]["cameras"]["head"]["streams"]["color"][
        "intrinsics"
    ]
    assert np.allclose(intr["coeffs"], 0), "Nonzero lens distortion needs explicit undistortion"
    k_rgb = np.array([[intr["fx"], 0, intr["ppx"]], [0, intr["fy"], intr["ppy"]], [0, 0, 1]], dtype=np.float32)
    # Track4World first resizes to 640x448, then floors to a patch-14 grid.
    width, height = 630, 448
    k_true = k_rgb.copy()
    # Pixel centers: p_out + .5 = scale * (p_in + .5).
    sx, sy = width / intr["width"], height / intr["height"]
    k_true[0, 0] *= sx
    k_true[1, 1] *= sy
    k_true[0, 2] = (k_rgb[0, 2] + 0.5) * sx - 0.5
    k_true[1, 2] = (k_rgb[1, 2] + 0.5) * sy - 0.5
    labels_path = Path("/data/shichaojian/pf_out/9.24/sandwich/sam2_masks") / args.episode / "labels.npy"
    fk_path = Path("/data/shichaojian/raw_data/sandwich_fk21") / args.episode / "annotations/wuji_fk21.npz"
    with np.load(fk_path) as fk_file:
        assert str(fk_file["units"]) in ("m", "meter", "metre", "meters", "metres"), fk_file["units"]
        assert str(fk_file["side_names"][1]).lower() == "right", fk_file["side_names"]
        fk = base_to_camera(fk_file["positions"][args.frames, 1])
    cap = cv2.VideoCapture(str(raw / "videos/head.mp4"))
    images, masks = [], []
    for frame in args.frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, bgr = cap.read()
        assert ok, frame
        assert bgr.shape[:2] == (intr["height"], intr["width"])
        images.append(cv2.cvtColor(cv2.resize(bgr, (640, 448)), cv2.COLOR_BGR2RGB))
        mask = read_frame(str(labels_path), frame) == 2
        masks.append(cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool))
    cap.release()
    print("Loading DA3 checkpoint (strict)", flush=True)
    from depth_anything_3.cfg import create_object

    config = json.loads((Path(args.checkpoint) / "config.json").read_text())
    # Gaussian splatting is not used by DA3 depth inference (infer_gs=False).
    # Omit only these optional modules and their weights, keeping strict loading
    # for every depth/camera parameter without requiring rendering dependencies.
    config["config"]["anyview"].pop("gs_head", None)
    config["config"]["anyview"].pop("gs_adapter", None)
    model = create_object(OmegaConf.create(config["config"]))
    state = load_file(str(Path(args.checkpoint) / "model.safetensors"))
    # safetensors deduplicates shared LayerNorm parameters. Restore only proven
    # aliases of the exact same parameter object before strict validation.
    aliases = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(param), []).append("model." + name)
    for names in aliases.values():
        saved = next((name for name in names if name in state), None)
        if saved is not None:
            for name in names:
                state.setdefault(name, state[saved])
    model.load_state_dict(
        {
            k.removeprefix("model."): v
            for k, v in state.items()
            if not k.startswith(("model.da3.gs_head.", "model.da3.gs_adapter."))
        },
        strict=True,
    )
    del state
    model = model.eval().cuda()
    torch.manual_seed(args.seed)
    x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float().cuda() / 255
    mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    x = F.interpolate((x - mean) / std, (height, width), mode="bilinear", align_corners=False, antialias=True)[None]
    print("Running shared AnyView and Metric predictions on four frames", flush=True)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        prediction = model.da3(
            x, export_feat_layers=[], infer_gs=False, use_ray_pose=False, ref_view_strategy="saddle_balanced"
        )
        metric = model.da3_metric(x)
    # Clone only fields used in postprocessing, avoiding large backbone features.
    pred_raw = Dict({k: prediction[k].float().clone() for k in ("depth", "depth_conf", "intrinsics", "extrinsics")})
    metric_raw = Dict({k: metric[k].float().clone() for k in ("depth", "sky")})
    del prediction, metric
    arrays = {"frames": np.array(args.frames), "fk_camera": fk, "K_rgb": k_rgb, "K_true": k_true}
    results = {}
    for variant in ("estimated", "calibrated_B"):
        pred = Dict({k: v.clone() for k, v in pred_raw.items()})
        met = Dict({k: v.clone() for k, v in metric_raw.items()})
        if variant == "calibrated_B":
            pred.intrinsics = torch.as_tensor(k_true, device="cuda")[None, None].expand_as(pred.intrinsics).clone()
        # Reuse the same RNG for sampled confidence / sky quantiles.
        torch.manual_seed(args.seed)
        with torch.inference_mode():
            model._apply_metric_scaling(pred, met)
            model._apply_depth_alignment(pred, met)
            model._handle_sky_regions(pred, met)
        depth = pred.depth[0].cpu().numpy()
        ks = pred.intrinsics[0].cpu().numpy()
        arrays[f"{variant}_depth"] = depth
        arrays[f"{variant}_K"] = ks
        stats = []
        for i, frame in enumerate(args.frames):
            yy, xx = np.where(masks[i] & np.isfinite(depth[i]) & (depth[i] > 0))
            z = depth[i, yy, xx]
            rays = np.stack([xx, yy, np.ones_like(xx)], axis=1) @ np.linalg.inv(ks[i]).T
            cloud = rays * z[:, None]
            assert len(cloud) > 100
            reproj = cloud @ ks[i].T
            assert np.max(np.abs(reproj[:, :2] / reproj[:, 2:] - np.stack([xx, yy], axis=1))) < 1e-3
            distances = cKDTree(cloud).query(fk[i])[0] * 1000
            arrays[f"{variant}_cloud_{frame}"] = cloud
            stats.append(
                dict(
                    frame=frame,
                    points=len(cloud),
                    joint_to_surface_median_mm=float(np.median(distances)),
                    joint_to_surface_p95_mm=float(np.percentile(distances, 95)),
                    hand_depth_median_m=float(np.median(z)),
                )
            )
        results[variant] = dict(alignment_scale=pred.scale_factor, K=ks.tolist(), frames=stats)
    ratio = results["calibrated_B"]["alignment_scale"] / results["estimated"]["alignment_scale"]
    report = dict(
        episode=args.episode,
        seed=args.seed,
        frames=args.frames,
        depth_source="DA3 only; no D435 depth",
        rgb_intrinsics=intr,
        metadata_path=str(metadata_path),
        checkpoint=args.checkpoint,
        network_size=[width, height],
        context="Joint inference of selected four frames, not full-video Track4World reproduction",
        method="Shared unconditioned AnyView and Metric predictions; override K only at Metric focal scaling and XYZ unprojection; no FK fitting",
        pixel_center_convention="K principal point resized as (c+0.5)*scale-0.5",
        scale_ratio_B_over_estimated=ratio,
        results=results,
        caveat="Joint-to-nearest-visible-surface distance is not joint ground-truth error; FK extrinsics and occlusion contribute.",
    )
    np.savez_compressed(out / "comparison.npz", **arrays)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    render_comparison(out, report, arrays)


def render_comparison(out, report, arrays):
    fk = arrays["fk_camera"]
    results = report["results"]
    ratio = report["scale_ratio_B_over_estimated"]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig = plt.figure(figsize=(6 * len(report["frames"]), 11))
    edges = [(0, start) for start in (1, 5, 9, 13, 17)] + [
        (j, j + 1) for start in (1, 5, 9, 13, 17) for j in range(start, start + 3)
    ]
    for i, frame in enumerate(report["frames"]):
        combined = np.concatenate([fk[i]] + [arrays[f"{v}_cloud_{frame}"] for v in results])
        # Zoom limits only: metrics and saved clouds retain every valid point.
        lower = np.minimum(np.percentile(combined, 1, axis=0), fk[i].min(0))
        upper = np.maximum(np.percentile(combined, 99, axis=0), fk[i].max(0))
        center = (lower + upper) / 2
        radius = max(upper - lower) * 0.55
        for row, variant in enumerate(results):
            ax = fig.add_subplot(2, len(report["frames"]), row * len(report["frames"]) + i + 1, projection="3d")
            cloud = arrays[f"{variant}_cloud_{frame}"]
            cloud = cloud[np.all((cloud >= center - radius) & (cloud <= center + radius), axis=1)]
            ax.scatter(
                *cloud[:: max(1, len(cloud) // 5000)].T, s=1, color="tab:orange", alpha=0.4, label="DA3 hand surface"
            )
            ax.scatter(*fk[i].T, s=12, color="tab:blue", label="FK-21")
            for a, b in edges:
                ax.plot(*fk[i, [a, b]].T, color="tab:blue", linewidth=1)
            stat = results[variant]["frames"][i]
            ax.set_title(
                f"{variant} | frame {frame}\nJoint-to-surface median {stat['joint_to_surface_median_mm']:.1f} mm; p95 {stat['joint_to_surface_p95_mm']:.1f} mm"
            )
            ax.set(
                xlim=(center[0] - radius, center[0] + radius),
                ylim=(center[1] - radius, center[1] + radius),
                zlim=(center[2] - radius, center[2] + radius),
                xlabel="Camera X (m)",
                ylabel="Camera Y (m)",
                zlabel="Camera Z (m)",
            )
            ax.set_box_aspect((1, 1, 1))
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.set_major_locator(MaxNLocator(5))
            ax.view_init(elev=22, azim=-65)
            if i == 0:
                ax.legend(fontsize=8)
    fig.suptitle(
        f"{report['episode']} | DA3 predicted K vs D435 RGB K in Metric scaling + XYZ\nSame raw predictions, no FK fitting; B depth scale / baseline = {ratio:.4f}; selected-frame rerun; display zoomed to 1-99% cloud bounds, all FK joints included",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
