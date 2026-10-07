"""Real-window Sonata smoke test; no Cosmos joint training is performed."""

import argparse
import json
import time
from pathlib import Path
from tempfile import TemporaryFile

import numpy as np
import torch

from cosmos_framework.data.pointflow_window import prepare_window


def synchronize(device: str) -> None:
    """Block until pending kernels finish; a no-op without a GPU."""
    if device == "cuda":
        torch.cuda.synchronize()


def overlay(window, output, mapping=None, seed=0):
    import cv2

    canvas = cv2.cvtColor(window["anchor_rgb"], cv2.COLOR_RGB2BGR)
    uv = np.rint(window["anchor_uv"]).astype(int)
    rng = np.random.default_rng(seed)
    if mapping is None:
        colors = np.tile([0, 255, 0], (len(uv), 1))
    else:
        colors = rng.integers(40, 256, size=(int(mapping.max()) + 1, 3))[mapping]
    for i in np.linspace(0, len(uv) - 1, min(len(uv), 4096), dtype=int):
        cv2.circle(canvas, tuple(uv[i]), 1, tuple(int(c) for c in colors[i]), -1)
    if not cv2.imwrite(str(output), canvas):
        raise OSError(f"Failed to write {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--episodes", nargs="+", default=["episode_0013_20260731_133649"])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-points", type=int, default=8192)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--stages", nargs="+", type=int, choices=range(5), default=[4])
    parser.add_argument(
        "--select-motion-fraction",
        type=float,
        default=0.0,
        help="keep only the fastest-moving fraction of anchor points (ground-truth ranked; 0 = full cloud)",
    )
    args = parser.parse_args()
    if args.device == "cpu" and args.backward:
        parser.error("CPU mode prepares data only; --backward requires CUDA")
    if len(set(args.episodes)) != len(args.episodes):
        parser.error("Episode names must be unique")
    try:
        args.output.mkdir(parents=True, exist_ok=True)
        with TemporaryFile(dir=args.output):
            pass
    except OSError as error:
        parser.error(
            f"Cannot write output directory {args.output}: {error}. "
            "Set OUTPUT_DIR=/path/to/writable/directory in the launcher, or pass --output."
        )
    import torch

    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No visible GPU. Run on H200, or use --device cpu for data preparation only.")
    windows = []
    reports = []
    for episode in args.episodes:
        start = time.perf_counter()
        if not 0.0 <= args.select_motion_fraction < 1.0:
            parser.error("--select-motion-fraction must be in [0, 1)")
        window = prepare_window(
            args.data_root / episode,
            args.start_frame,
            args.max_points,
            args.voxel_size,
            args.seed,
            select_motion_fraction=args.select_motion_fraction,
        )
        directory = args.output / episode
        directory.mkdir(exist_ok=True)
        np.savez_compressed(directory / "window.npz", **{k: v for k, v in window.items() if k != "anchor_rgb"})
        overlay(window, directory / "anchor_points.png")
        reports.append(
            {
                "episode": episode,
                "original_points": len(window["point_ids"]),
                "voxel_points": len(window["coord"]),
                "frame_ids": window["raw_frame_ids"].tolist(),
                "label_valid_fraction": float(window["target_valid"].mean()),
                "projection_median_px": float(np.median(window["projection_error_px"])),
                "projection_p95_px": float(np.percentile(window["projection_error_px"], 95)),
                "prepare_seconds": time.perf_counter() - start,
            }
        )
        if args.device == "cpu":
            # Exact spatial grouping for the fixed stride-2 checkpoint, not neural features.
            reports[-1]["geometry_preview"] = {}
            for stage in sorted(set(args.stages)):
                _, voxel_mapping = np.unique(window["grid_coord"] // (2**stage), axis=0, return_inverse=True)
                raw_mapping = voxel_mapping[window["original_to_voxel"]]
                overlay(window, directory / f"geometry_preview_enc{stage}.png", raw_mapping, args.seed)
                np.savez_compressed(
                    directory / f"geometry_preview_enc{stage}.npz",
                    point_ids=window["point_ids"],
                    original_to_cluster=raw_mapping,
                )
                reports[-1]["geometry_preview"][f"enc{stage}"] = {"clusters": int(raw_mapping.max()) + 1}
        windows.append(window)
        print(json.dumps(reports[-1]), flush=True)

    result = {
        "seed": args.seed,
        "fps": 15,
        "future_steps": 32,
        "samples": reports,
        "gpu_forward": "NOT_RUN",
        "gpu_backward": "NOT_RUN",
    }
    # CUDA-only, and not for want of a `.to(device)` here: PointTransformerV3 runs on
    # spconv, whose `get_current_stream` calls `torch.cuda.current_stream()`
    # unconditionally, so the sparse convolutions cannot execute without a GPU.
    # `--device cpu` therefore produces the window and the geometry preview but skips
    # this section, leaving `encoding_enc*.npz` unwritten -- and without that file
    # `visualize_pointflow_motion.py` has nothing to render.
    if args.device == "cuda":
        from cosmos_framework.auxiliary import sonata

        device = args.device
        model = sonata.load(str(args.checkpoint)).to(device).eval()
        result["strict_load"] = "PASS"
        sizes = [len(w["coord"]) for w in windows]
        batch = {
            k: torch.from_numpy(np.concatenate([w[k] for w in windows])).to(device)
            for k in ("coord", "feat", "grid_coord")
        }
        batch["offset"] = torch.tensor(np.cumsum(sizes), device=device, dtype=torch.long)
        synchronize(device)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.set_grad_enabled(args.backward):
            encoded = model(batch)
            synchronize(device)
            result["forward_seconds"] = time.perf_counter() - start
            levels = [encoded]
            while "pooling_parent" in levels[-1]:
                levels.append(levels[-1].pooling_parent)
            levels.reverse()
            if len(levels) != 5:
                raise RuntimeError("Expected five Sonata encoder stages")
            for stage in sorted(set(args.stages)):
                stage_output = levels[stage]
                if not torch.isfinite(stage_output.feat).all():
                    raise RuntimeError("Nonfinite Sonata output")
                mapping = torch.arange(len(stage_output.feat), device=device)
                level = stage_output
                while "pooling_parent" in level:
                    mapping = mapping[level.pooling_inverse]
                    level = level.pooling_parent
                if len(mapping) != sum(sizes):
                    raise RuntimeError("Pooling inverse does not cover all input voxels")
                cursor = 0
                for sample, (window, size, report) in enumerate(zip(windows, sizes, reports)):
                    raw_mapping = mapping[cursor : cursor + size][
                        torch.from_numpy(window["original_to_voxel"]).to(device)
                    ]
                    if not torch.all(stage_output.batch[raw_mapping] == sample):
                        raise RuntimeError("Cross-sample cluster mapping detected")
                    clusters, local_mapping = torch.unique(raw_mapping, sorted=True, return_inverse=True)
                    local_mapping = local_mapping.cpu().numpy()
                    count = np.bincount(local_mapping)
                    centers = {}
                    for key in ("anchor_xyz", "anchor_uv"):
                        sums = np.zeros((len(count), window[key].shape[1]), dtype=np.float64)
                        np.add.at(sums, local_mapping, window[key])
                        centers[key] = sums / count[:, None]
                    directory = args.output / report["episode"]
                    np.savez_compressed(
                        directory / f"encoding_enc{stage}.npz",
                        point_ids=window["point_ids"],
                        original_to_cluster=local_mapping,
                        cluster_counts=count,
                        cluster_xyz=centers["anchor_xyz"],
                        cluster_uv=centers["anchor_uv"],
                        cluster_features=stage_output.feat[clusters].detach().float().cpu().numpy(),
                    )
                    overlay(window, directory / f"point_clusters_enc{stage}.png", local_mapping, args.seed)
                    report.setdefault("stages", {})[f"enc{stage}"] = {
                        "clusters": len(count),
                        "feature_dim": stage_output.feat.shape[1],
                    }
                    cursor += size
            result["gpu_forward"] = "PASS"
            if args.backward:
                start = time.perf_counter()
                # A random linear probe checks gradients; this is not a PointFlow training loss.
                loss = sum(
                    (levels[stage].feat.float() * torch.randn_like(levels[stage].feat.float())).mean()
                    for stage in sorted(set(args.stages))
                )
                loss.backward()
                synchronize(device)
                result["backward_seconds"] = time.perf_counter() - start
                grad = model.embedding.stem.linear.weight.grad
                if grad is None or not torch.isfinite(grad).all() or grad.abs().sum() == 0:
                    raise RuntimeError("Missing/nonfinite/zero input-layer gradient")
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise RuntimeError("Nonfinite model gradient")
                result["gpu_backward"] = "PASS"
        if device == "cuda":
            result["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
            result["gpu"] = torch.cuda.get_device_name()
        else:
            result["device"] = "cpu"
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
