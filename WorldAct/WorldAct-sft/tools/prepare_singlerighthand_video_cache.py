#!/usr/bin/env python3
"""Precompute deterministic single-right-hand video frames for action SFT."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as transforms_F

from cosmos_framework.data.generator.action.datasets.singlerighthand_raw_dataset import (
    SingleRightHandRawDataset,
    _TorchCodecFrameReader,
)
from cosmos_framework.data.generator.action.transforms import find_closest_target_size


@dataclass(frozen=True)
class _EpisodeJob:
    raw_root: str
    cache_root: str
    name: str
    num_frames: int
    resolution: str
    chunk_frames: int
    torch_threads: int
    overwrite: bool


def resize_to_resolution_content(video: torch.Tensor, resolution: str | int) -> tuple[torch.Tensor, list[int]]:
    """Resize ``[T,C,H,W]`` exactly as VideoResize, but leave off deterministic padding."""
    height, width = video.shape[-2:]
    target_w, target_h = find_closest_target_size(height, width, resolution)
    scaling_ratio = min(target_w / width, target_h / height, 1.0)
    content_h = int(scaling_ratio * height + 0.5)
    content_w = int(scaling_ratio * width + 0.5)
    if (content_h, content_w) != (height, width):
        video = transforms_F.resize(
            video,
            [content_h, content_w],
            interpolation=transforms_F.InterpolationMode.BICUBIC,
            antialias=True,
        )
    return video, [target_h, target_w, content_h, content_w]


def _existing_cache_row(job: _EpisodeJob, output_path: Path) -> dict | None:
    if job.overwrite or not output_path.is_file():
        return None
    array = np.load(output_path, mmap_mode="r", allow_pickle=False)
    try:
        if array.dtype != np.uint8 or array.ndim != 4 or array.shape[:2] != (job.num_frames, 3):
            raise ValueError(
                f"Invalid existing cache {output_path}: expected uint8 [{job.num_frames},3,H,W], "
                f"got {array.dtype} {array.shape}"
            )
        content_h, content_w = (int(value) for value in array.shape[-2:])
        target_w, target_h = find_closest_target_size(content_h, content_w, job.resolution)
        return {
            "name": job.name,
            "path": str(output_path.relative_to(job.cache_root)),
            "shape": list(array.shape),
            "image_size": [target_h, target_w, content_h, content_w],
        }
    finally:
        mmap = getattr(array, "_mmap", None)
        if mmap is not None:
            mmap.close()


def _prepare_episode(job: _EpisodeJob) -> dict:
    torch.set_num_threads(job.torch_threads)
    raw_root = Path(job.raw_root)
    cache_root = Path(job.cache_root)
    output_path = cache_root / "video_frames" / f"{job.name}.npy"
    existing = _existing_cache_row(job, output_path)
    if existing is not None:
        return existing

    video_dir = raw_root / job.name / "videos"
    head_reader = _TorchCodecFrameReader(video_dir / "head.mp4")
    wrist_reader = _TorchCodecFrameReader(video_dir / "right_wrist.mp4")
    temp_path = output_path.with_suffix(".npy.tmp")
    temp_path.unlink(missing_ok=True)
    try:
        first_index = np.asarray([0], dtype=np.int64)
        first_composed = SingleRightHandRawDataset._compose_views(
            head_reader.get_frames(first_index),
            wrist_reader.get_frames(first_index),
        )
        first_resized, image_size = resize_to_resolution_content(first_composed, job.resolution)
        content_h, content_w = first_resized.shape[-2:]
        shape = (job.num_frames, 3, content_h, content_w)
        output = np.lib.format.open_memmap(temp_path, mode="w+", dtype=np.uint8, shape=shape)
        try:
            for start in range(0, job.num_frames, job.chunk_frames):
                end = min(start + job.chunk_frames, job.num_frames)
                indices = np.arange(start, end, dtype=np.int64)
                composed = SingleRightHandRawDataset._compose_views(
                    head_reader.get_frames(indices),
                    wrist_reader.get_frames(indices),
                )
                resized, current_image_size = resize_to_resolution_content(composed, job.resolution)
                if current_image_size != image_size or tuple(resized.shape[1:]) != shape[1:]:
                    raise ValueError(
                        f"Episode {job.name}: video geometry changed at frames {start}:{end}: "
                        f"{current_image_size} vs {image_size}"
                    )
                output[start:end] = resized.numpy()
            output.flush()
        finally:
            del output
        os.replace(temp_path, output_path)
    finally:
        head_reader.close()
        wrist_reader.close()

    return {
        "name": job.name,
        "path": str(output_path.relative_to(cache_root)),
        "shape": list(shape),
        "image_size": image_size,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--resolution", default="480")
    parser.add_argument("--chunk-frames", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.chunk_frames < 1 or args.workers < 1 or args.torch_threads < 1:
        raise ValueError("chunk-frames, workers, and torch-threads must all be >= 1")

    raw_root = args.raw_root.expanduser().resolve()
    cache_root = args.cache_root.expanduser().resolve()
    state_manifest_path = cache_root / "manifest.json"
    state_manifest = json.loads(state_manifest_path.read_text())
    if int(state_manifest.get("schema_version", 0)) != 2:
        raise ValueError(f"Unsupported state cache schema in {state_manifest_path}")

    output_dir = cache_root / "video_frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        _EpisodeJob(
            raw_root=str(raw_root),
            cache_root=str(cache_root),
            name=str(row["name"]),
            num_frames=int(row["num_frames"]),
            resolution=str(args.resolution),
            chunk_frames=args.chunk_frames,
            torch_threads=args.torch_threads,
            overwrite=args.overwrite,
        )
        for row in state_manifest["episodes"]
    ]

    rows_by_name: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_prepare_episode, job): job for job in jobs}
        for completed, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            row = future.result()
            rows_by_name[job.name] = row
            gib = np.prod(row["shape"], dtype=np.int64) / 1024**3
            print(f"[{completed:03d}/{len(jobs):03d}] {job.name}: {row['shape']} ({gib:.2f} GiB)", flush=True)

    manifest = {
        "schema_version": 1,
        "resolution": str(args.resolution),
        "dtype": "uint8",
        "layout": "TCHW",
        "padding": "deferred_to_ActionTransformPipeline",
        "episodes": [rows_by_name[job.name] for job in jobs],
    }
    manifest_path = cache_root / "video_manifest.json"
    temp_manifest = manifest_path.with_suffix(".json.tmp")
    temp_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    temp_manifest.replace(manifest_path)
    total_gib = sum(np.prod(row["shape"], dtype=np.int64) for row in manifest["episodes"]) / 1024**3
    print(f"Wrote {manifest_path} ({total_gib:.2f} GiB)")


if __name__ == "__main__":
    main()
