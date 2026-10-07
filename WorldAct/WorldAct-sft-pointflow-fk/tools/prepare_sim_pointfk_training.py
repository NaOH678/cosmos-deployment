"""Materialize the approved current-anchor/FK selection and train-only scales."""

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.data.pointflow_anchor_selection import hand_guided_fps
from tools.cache_sim_pointfk_window_latents import digest


def prepare_episode(root, out, episode):
    torch.set_num_threads(2)
    sys.path.append(str(root / "code/bench2dex"))
    from utils.sim_pointfk_dataset import SimPointFKDataset

    split = "train" if int(episode.split("_")[-1]) < 8 else "validation"
    ds = SimPointFKDataset(root, split)
    rows, moments = [], {"pointflow": [0, 0.0, 0.0], "fk": [0, 0.0, 0.0]}
    for index, row in enumerate(ds.rows):
        if row["episode"] != episode:
            continue
        sample = ds[index]
        point = sample["pointflow"]
        inputs = point["inputs"]
        keep, detail = hand_guided_fps(inputs["anchor_xyz"], sample["fk"]["inputs"]["anchor_xyz"], 1024, 0.05)
        for key in ["point_ids", "anchor_xyz", "anchor_uv", "normal", "color"]:
            inputs[key] = inputs[key][keep].copy()
        xyz = inputs["anchor_xyz"]
        shift = (xyz.min(0) + xyz.max(0)) / 2
        shift[2] = xyz[:, 2].min()
        centered = xyz - shift
        grid = np.floor(centered / 0.02).astype(np.int64)
        grid -= grid.min(0)
        _, representatives, inverse = np.unique(grid, axis=0, return_index=True, return_inverse=True)
        features = np.concatenate([centered, inputs["color"].astype(np.float32) / 255, inputs["normal"]], axis=1)
        inputs.update(
            coord_shift=shift,
            coord=centered[representatives],
            feat=features[representatives],
            grid_coord=grid[representatives],
            original_to_voxel=inverse,
            voxel_representatives=representatives,
        )
        for key in ["displacement", "valid"]:
            point["targets"][key] = point["targets"][key][:, keep].copy()
        raw = root / row["raw_path"]
        point["metadata"].update(
            source_path=str(raw),
            geometry_source="current anchor depth + current FK guided FPS",
            preview_video=str(root / "raw_data/bench2dex_task21" / episode / "videos/head.mp4"),
            tracker_size_wh=[640, 480],
        )
        sample["fk"]["metadata"]["intrinsics_px"] = ds._fk["intrinsics"].copy()
        sample["fk"]["metadata"]["image_size_wh"] = np.array([640, 480])
        if split == "train":
            for key in moments:
                data = sample[key]
                x = data["targets"]["displacement"][data["targets"]["valid"]].astype(np.float64)
                n, s, q = moments[key]
                moments[key] = [n + x.size, s + float(x.sum()), q + float(np.square(x).sum())]
        # RGB and latent stay in their existing read-only caches.
        sample.pop("video")
        start = row["start_frame"]
        path = out / f"{episode}_{start:06d}.pt"
        torch.save(sample, path.with_suffix(".pt.tmp"))
        os.replace(path.with_suffix(".pt.tmp"), path)
        rows.append(
            dict(
                episode=episode,
                start_frame=start,
                frame_ids=row["frame_ids"],
                split=split,
                path=path.name,
                point_ids=inputs["point_ids"].tolist(),
                selection=detail,
                sha256=digest(path),
            )
        )
    print(episode, len(rows), flush=True)
    return rows, moments


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    index_path = args.bundle / "datasets/bench2dex-task21-cosmos-cache/window_index.json"
    index = json.loads(index_path.read_text())["windows"]
    names = sorted({r["episode"] for r in index})
    records, moments = [], {"pointflow": [0, 0.0, 0.0], "fk": [0, 0.0, 0.0]}
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(prepare_episode, args.bundle, args.output, n) for n in names]
        for f in futures:
            rows, stats = f.result()
            records.extend(rows)
            for k in moments:
                moments[k] = [a + b for a, b in zip(moments[k], stats[k])]
    scales = {k: math.sqrt(q / n - (s / n) ** 2) for k, (n, s, q) in moments.items()}
    document = dict(
        schema="sim_pointfk_handguided1024_v1",
        seed=42,
        points=1024,
        radius_metres=0.05,
        global_points=512,
        per_hand_points=256,
        voxel_size=0.02,
        selection_uses_future=False,
        uses_gt_query_mask=True,
        window_index_sha256=digest(index_path),
        train_only_scales=scales,
        windows=sorted(records, key=lambda r: (r["episode"], r["start_frame"])),
    )
    (args.output / "manifest.json").write_text(json.dumps(document, indent=2) + "\n")
    print("COMPLETE", scales, flush=True)


if __name__ == "__main__":
    main()
