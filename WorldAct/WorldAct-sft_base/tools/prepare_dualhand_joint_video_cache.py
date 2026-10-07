#!/usr/bin/env python3
"""Precompute deterministic three-view video frames for dual-hand Cosmos data."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as transforms_F

from cosmos_framework.data.generator.action.datasets.singlerighthand_raw_dataset import _TorchCodecFrameReader
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


def _write_npy_header(file: BinaryIO, shape: tuple[int, ...]) -> None:
    header = {
        "descr": np.lib.format.dtype_to_descr(np.dtype(np.uint8)),
        "fortran_order": False,
        "shape": shape,
    }
    np.lib.format.write_array_header_2_0(file, header)


def _write_array_bytes(file: BinaryIO, array: np.ndarray) -> None:
    array = np.ascontiguousarray(array, dtype=np.uint8)
    written = file.write(memoryview(array).cast("B"))
    if written != array.nbytes:
        raise OSError(f"Short write: wrote {written} of {array.nbytes} bytes")


def _read_npy_metadata(path: Path) -> tuple[tuple[int, ...], np.dtype]:
    with path.open("rb") as file:
        version = np.lib.format.read_magic(file)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(file)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(file)
        else:
            raise ValueError(f"Unsupported NPY version {version} in {path}")
        data_offset = file.tell()
    if fortran_order:
        raise ValueError(f"Fortran-order NPY cache is not supported: {path}")
    expected_size = data_offset + int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"Truncated NPY cache {path}: expected {expected_size} bytes, got {actual_size}")
    return tuple(int(value) for value in shape), np.dtype(dtype)


def _resize_to_width(video: torch.Tensor, target_width: int) -> torch.Tensor:
    height, width = video.shape[-2:]
    target_height = max(1, round(height * target_width / width))
    if (height, width) == (target_height, target_width):
        return video
    return transforms_F.resize(
        video,
        [target_height, target_width],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )


def compose_dualhand_views(head: torch.Tensor, left_wrist: torch.Tensor, right_wrist: torch.Tensor) -> torch.Tensor:
    """Place head on top and aspect-preserved left/right wrist views on the bottom."""
    if head.ndim != 4 or left_wrist.ndim != 4 or right_wrist.ndim != 4:
        raise ValueError("All camera tensors must have shape [T,C,H,W]")
    if head.shape[:2] != left_wrist.shape[:2] or head.shape[:2] != right_wrist.shape[:2]:
        raise ValueError(
            f"Camera batch/channel mismatch: {head.shape}, {left_wrist.shape}, {right_wrist.shape}"
        )

    target_width = min(int(head.shape[-1]), int(left_wrist.shape[-1] + right_wrist.shape[-1]))
    target_width -= target_width % 2
    if target_width < 2:
        raise ValueError(f"Invalid composed target width {target_width}")
    half_width = target_width // 2
    head = _resize_to_width(head, target_width)
    left_wrist = _resize_to_width(left_wrist, half_width)
    right_wrist = _resize_to_width(right_wrist, half_width)

    wrist_height = max(int(left_wrist.shape[-2]), int(right_wrist.shape[-2]))
    if left_wrist.shape[-2] < wrist_height:
        left_wrist = F.pad(left_wrist, (0, 0, 0, wrist_height - left_wrist.shape[-2]))
    if right_wrist.shape[-2] < wrist_height:
        right_wrist = F.pad(right_wrist, (0, 0, 0, wrist_height - right_wrist.shape[-2]))
    bottom = torch.cat([left_wrist, right_wrist], dim=-1)
    return torch.cat([head, bottom], dim=-2)


def resize_to_resolution_content(video: torch.Tensor, resolution: str | int) -> tuple[torch.Tensor, list[int]]:
    """Resize exactly as VideoResize while deferring deterministic reflection padding."""
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
    shape, dtype = _read_npy_metadata(output_path)
    if dtype != np.uint8 or len(shape) != 4 or shape[:2] != (job.num_frames, 3):
        raise ValueError(
            f"Invalid existing cache {output_path}: expected uint8 [{job.num_frames},3,H,W], "
            f"got {dtype} {shape}"
        )
    content_h, content_w = shape[-2:]
    target_w, target_h = find_closest_target_size(content_h, content_w, job.resolution)
    return {
        "name": job.name,
        "path": str(output_path.relative_to(job.cache_root)),
        "shape": list(shape),
        "image_size": [target_h, target_w, content_h, content_w],
    }


def _prepare_episode(job: _EpisodeJob) -> dict:
    torch.set_num_threads(job.torch_threads)
    raw_root = Path(job.raw_root)
    cache_root = Path(job.cache_root)
    output_path = cache_root / "video_frames" / f"{job.name}.npy"
    existing = _existing_cache_row(job, output_path)
    if existing is not None:
        return existing

    video_dir = raw_root / job.name / "videos"
    readers = {
        "head": _TorchCodecFrameReader(video_dir / "head.mp4"),
        "left_wrist": _TorchCodecFrameReader(video_dir / "left_wrist.mp4"),
        "right_wrist": _TorchCodecFrameReader(video_dir / "right_wrist.mp4"),
    }
    temp_path = output_path.with_suffix(".npy.tmp")
    temp_path.unlink(missing_ok=True)
    try:
        first_index = np.asarray([0], dtype=np.int64)
        first_composed = compose_dualhand_views(
            readers["head"].get_frames(first_index),
            readers["left_wrist"].get_frames(first_index),
            readers["right_wrist"].get_frames(first_index),
        )
        first_resized, image_size = resize_to_resolution_content(first_composed, job.resolution)
        content_h, content_w = first_resized.shape[-2:]
        shape = (job.num_frames, 3, content_h, content_w)
        with temp_path.open("wb") as output:
            _write_npy_header(output, shape)
            for start in range(0, job.num_frames, job.chunk_frames):
                end = min(start + job.chunk_frames, job.num_frames)
                indices = np.arange(start, end, dtype=np.int64)
                composed = compose_dualhand_views(
                    readers["head"].get_frames(indices),
                    readers["left_wrist"].get_frames(indices),
                    readers["right_wrist"].get_frames(indices),
                )
                resized, current_image_size = resize_to_resolution_content(composed, job.resolution)
                if current_image_size != image_size or tuple(resized.shape[1:]) != shape[1:]:
                    raise ValueError(
                        f"Episode {job.name}: video geometry changed at frames {start}:{end}: "
                        f"{current_image_size} vs {image_size}"
                    )
                _write_array_bytes(output, resized.numpy())
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp_path, output_path)
    finally:
        for reader in readers.values():
            reader.close()

    return {
        "name": job.name,
        "path": str(output_path.relative_to(cache_root)),
        "shape": list(shape),
        "image_size": image_size,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    if state_manifest.get("schema") != "cosmos_dualhand_joint_v1":
        raise ValueError(f"Unsupported dual-hand state cache schema in {state_manifest_path}")
    if Path(state_manifest["raw_root"]).resolve() != raw_root:
        raise ValueError(
            f"Raw root mismatch: manifest={state_manifest['raw_root']}, command={raw_root}"
        )

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
        "schema": "cosmos_dualhand_joint_video_v1",
        "schema_version": 1,
        "resolution": str(args.resolution),
        "dtype": "uint8",
        "layout": "TCHW",
        "source_cameras": ["head", "left_wrist", "right_wrist"],
        "composition": "head_top__left_wrist_bottom_left__right_wrist_bottom_right",
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
