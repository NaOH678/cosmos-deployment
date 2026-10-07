#!/usr/bin/env python3
"""Encode each training window's own frames, so a cached run matches a cache-less run.

The model's own path (`OmniMoTModel._encode_vision_x0_tokens`) encodes one sample's
``[C, T, H, W]`` window and nothing else. The older cache stored a whole-episode
encode at source FPS and resampled 9 latent frames out of it; measured against a
fresh online encode, that differed by about as much as shifting the window one
frame, and it is the conditioning latent -- the only video input inference actually
uses -- that changes identity (a 4-frame source block instead of the current frame).

This tool stores, per window, exactly what the online path computes:

    video_frames/<episode>.npy   <- the same frames the dataset hands the model
      -> reflection_pad_to_target          (the real NVIDIA function: reflect pad)
      -> uint8 -> float32 / 127.5 - 1.0    (the model's normalization)
      -> Wan2pt2VAEInterface.encode        (the model's encoder)

Storage is bfloat16 bit patterns in a uint16 numpy array -- the VAE computes in
bfloat16 and the model casts the result to bfloat16 anyway, so this is lossless and
halves the file. The manifest records ``storage`` and the reader asserts it.

    python tools/cache_window_vae_latents.py \
        --cache-root .../singlerighthand-sandwich-100-cosmos-cache \
        --vae-path .../models/cosmos3-edge-droid/vae/Wan2.2_VAE.pth \
        --episode-allowlist examples/pointflow_sandwich_10_episodes.txt \
        --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.data.generator.action.transforms import find_closest_target_size, reflection_pad_to_target

STORAGE = "bfloat16_bits_in_uint16"
MANIFEST_NAME = "window_manifest.json"


@dataclass(frozen=True)
class _Job:
    name: str
    num_frames: int
    source_fps: float
    video_path: str
    output_path: str
    fps: float
    chunk_length: int
    sample_stride: int
    resolution: str
    vae_path: str
    overwrite: bool
    device: torch.device
    content_hw: tuple[int, int] | None = None
    batch_size: int = 8
    compile: bool = False


def _npy_shape_dtype(path: Path):
    """Shape and dtype without mmap -- GPFS rejects memory maps (OSError 19)."""
    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"Unsupported NPY version {version}: {path}")
    return shape, dtype


def window_plan(num_frames, source_fps, fps, chunk_length, sample_stride):
    """The dataset's own window enumeration (singlerighthand_raw_dataset.py:233-234).

    Returns ``(valid_windows, source_stride)``; index ``w`` here is exactly the
    dataset's ``window_offset``, which is what the cache is keyed on.
    """
    ratio = source_fps / fps
    source_stride = int(round(ratio))
    if source_stride < 1 or not np.isclose(ratio, source_stride, atol=1e-6):
        raise ValueError(f"source_fps={source_fps} is not an integer multiple of fps={fps}")
    last_start = num_frames - 1 - chunk_length * source_stride
    return (last_start // sample_stride + 1 if last_start >= 0 else 0), source_stride


# Process-level singletons keyed by device: each worker process loads the VAE
# once and reuses it across episodes. Jobs are pinned to one device per worker
# (see main), so in practice each process holds exactly one entry; the keyed
# dict just keeps the loader correct if that ever changes.
_TOKENIZERS: dict[str, object] = {}
_COMPILED: set[str] = set()


def _tokenizer(job: "_Job"):
    key = str(job.device)
    if key not in _TOKENIZERS:
        from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import Wan2pt2VAEInterface

        # The interface loads onto the current device, so pin it explicitly.
        with torch.cuda.device(job.device):
            _TOKENIZERS[key] = Wan2pt2VAEInterface(
                vae_path=job.vae_path,
                causal=True,
                encode_chunk_frames={"256": 68, "480": 24, "720": 8, "768": 8},
                encode_exact_durations=[job.chunk_length + 1],
                spatial_compression_factor=16,
                temporal_compression_factor=4,
            )
    tokenizer = _TOKENIZERS[key]
    if job.compile and key not in _COMPILED:
        # JIT-compile the whole encode: the eager path reaches only ~2% of peak
        # FLOPS (small-channel memory-bound convs), fusion has real headroom.
        # dynamic=False recompiles per batch size (full batch + one remainder).
        tokenizer.model.encode = torch.compile(tokenizer.model.encode, dynamic=False)
        _COMPILED.add(key)
    return tokenizer


def _prepare_episode(job: _Job) -> dict:
    # Every worker must pin its own GPU: a bare .cuda() in a subprocess lands on
    # device 0, so N workers would pile onto one card instead of scaling.
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.cuda.set_device(job.device)
    out_path = Path(job.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    windows, source_stride = window_plan(job.num_frames, job.source_fps, job.fps, job.chunk_length, job.sample_stride)
    if windows <= 0:
        raise ValueError(f"{job.name}: no complete window for chunk_length={job.chunk_length}")

    if not job.overwrite and out_path.is_file():
        full_shape, dtype = _npy_shape_dtype(out_path)
        if dtype != np.uint16 or len(full_shape) != 5 or full_shape[0] != windows:
            raise ValueError(f"{job.name}: stale window latent cache {full_shape} {dtype}; rerun with --overwrite")
        return {
            "name": job.name,
            "path": str(out_path.name),
            "windows": windows,
            "shape": list(full_shape[1:]),
            "storage": STORAGE,
            "content_hw": list(job.content_hw) if job.content_hw else None,
            "target_hw": [full_shape[-2] * 16, full_shape[-1] * 16],
            "source_stride": source_stride,
        }

    frames = np.load(job.video_path, allow_pickle=False)
    if frames.ndim != 4 or frames.dtype != np.uint8 or frames.shape[0] < job.num_frames:
        raise ValueError(f"{job.name}: unexpected video cache {frames.shape} {frames.dtype}")

    temp = out_path.with_suffix(".npy.tmp")
    temp.unlink(missing_ok=True)

    tokenizer = _tokenizer(job)

    offset = source_stride * np.arange(job.chunk_length + 1, dtype=np.int64)
    first = torch.from_numpy(frames[offset]).permute(1, 0, 2, 3)  # [C,T,H,W]
    height, width = first.shape[-2:]
    target_w, target_h = find_closest_target_size(height, width, job.resolution)
    target_hw = (target_h, target_w)
    content_hw = (height, width)

    def prepare_batch(start):
        end = min(start + job.batch_size, windows)
        tensors = []
        for w in range(start, end):
            raw = frames[w * job.sample_stride + offset]  # [T,3,H,W] uint8
            window = torch.from_numpy(raw).permute(1, 0, 2, 3)  # [C,T,H,W]
            tensors.append(reflection_pad_to_target({"video": window}, ["video"], True, target_w, target_h)["video"])
        return start, torch.stack(tensors)

    shape = None
    output = None
    try:
        # Double-buffer: the CPU pads/stacks batch N+1 while the GPU encodes N,
        # otherwise the GPU idles between batches (util oscillates 100% <-> 0).
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(prepare_batch, 0)
            for start in range(0, windows, job.batch_size):
                batch_start, stacked = future.result()
                if start + job.batch_size < windows:
                    future = pool.submit(prepare_batch, start + job.batch_size)
                # One batched encode instead of one call per window: a single
                # 33-frame window is launch/overhead-bound.
                x = stacked.to(job.device).float() / 127.5 - 1.0
                with torch.inference_mode():
                    latents = tokenizer.encode(x.to(torch.bfloat16)).cpu()  # [B,z,T',H',W']
                if shape is None:
                    shape = tuple(latents.shape[1:])
                    output = np.lib.format.open_memmap(temp, mode="w+", dtype=np.uint16, shape=(windows, *shape))
                for i, latent in enumerate(latents):
                    if tuple(latent.shape) != shape:
                        raise ValueError(
                            f"{job.name}: window {batch_start + i} latent {tuple(latent.shape)} != {shape}"
                        )
                    output[batch_start + i] = latent.view(torch.uint16).numpy()
        output.flush()
    finally:
        if output is not None:
            del output

    os.replace(temp, out_path)
    return {
        "name": job.name,
        "path": str(out_path.name),
        "windows": windows,
        "shape": list(shape),
        "storage": STORAGE,
        "content_hw": list(content_hw),
        "target_hw": list(target_hw),
        "source_stride": source_stride,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-root", type=Path, required=True, help="holds manifest.json and video_frames/")
    p.add_argument("--vae-path", type=Path, required=True)
    p.add_argument("--episode-allowlist", type=Path, required=True)
    p.add_argument("--output-root", type=Path, default=None, help="default: <cache-root>/vae_window_latents")
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--chunk-length", type=int, default=32)
    p.add_argument("--sample-stride", type=int, default=1)
    p.add_argument("--resolution", default="480")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="windows per VAE encode call; one window at a time is launch/overhead-bound",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the VAE encode. MEASURED BROKEN 2026-09-21: Dynamo mistraces the "
        "causal feat_cache mutation, outputs differ wildly (bit diff 48458) and it is slower. "
        "Kept only for future re-evaluation.",
    )
    p.add_argument(
        "--devices",
        default=None,
        help="comma-separated CUDA indices, one per worker (e.g. 0,1,2,3); default: all visible",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.workers < 1 or args.chunk_length < 1 or args.sample_stride < 1:
        raise ValueError("workers, chunk-length and sample-stride must all be >= 1")

    cache_root = args.cache_root.expanduser().resolve()
    output_root = (args.output_root or (cache_root / "vae_window_latents")).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    state = json.loads((cache_root / "manifest.json").read_text())
    if int(state.get("schema_version", 0)) != 2:
        raise ValueError(f"Unsupported state cache schema in {cache_root / 'manifest.json'}")
    by_name = {str(row["name"]): row for row in state["episodes"]}

    names = [
        line.strip()
        for line in args.episode_allowlist.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    missing = [n for n in names if n not in by_name]
    if missing:
        raise ValueError(f"allowlist names not in the state manifest: {missing[:3]}")

    raw_devices = args.devices.split(",") if args.devices else [str(i) for i in range(torch.cuda.device_count())]
    # set_device/.to need a real device: "0" alone is an invalid device string.
    devices = [torch.device("cuda", int(d)) if d.strip().isdigit() else torch.device(d) for d in raw_devices]
    if not devices:
        raise SystemExit("no CUDA device visible")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.workers > len(devices):
        raise SystemExit(
            f"--workers {args.workers} exceeds the {len(devices)} visible device(s); one worker per GPU is the point"
        )
    used_devices = devices[: args.workers]
    print(f"{len(names)} episodes over {args.workers} workers on devices {used_devices}", flush=True)
    video_manifest_path = cache_root / "video_manifest.json"
    video_rows = (
        {str(r["name"]): r for r in json.loads(video_manifest_path.read_text())["episodes"]}
        if video_manifest_path.is_file()
        else {}
    )
    jobs = []
    for index, name in enumerate(names):
        row = by_name[name]
        image_size = video_rows.get(name, {}).get("image_size")
        jobs.append(
            _Job(
                name=name,
                num_frames=int(row["num_frames"]),
                source_fps=float(row["source_fps"]),
                video_path=str(cache_root / "video_frames" / f"{name}.npy"),
                output_path=str(output_root / f"{name}.npy"),
                fps=args.fps,
                chunk_length=args.chunk_length,
                sample_stride=args.sample_stride,
                resolution=args.resolution,
                vae_path=str(args.vae_path),
                overwrite=args.overwrite,
                device=used_devices[index % len(used_devices)],
                content_hw=(int(image_size[2]), int(image_size[3])) if image_size else None,
                batch_size=args.batch_size,
                compile=args.compile,
            )
        )

    started = time.time()
    rows_by_name: dict[str, dict] = {}
    # One single-worker executor per device: a shared pool hands any job to any
    # worker, so a worker process drifts across GPUs and every card ends up
    # hosting one model copy PER WORKER (8 workers -> 8x50 GiB -> OOM). Pinning
    # jobs to a per-device worker keeps exactly one process (and one model) on
    # each card.
    executors = [ProcessPoolExecutor(max_workers=1) for _ in used_devices]
    try:
        futures = {}
        owners = {}
        for job in jobs:
            index = used_devices.index(job.device)
            future = executors[index].submit(_prepare_episode, job)
            futures[future] = job
            owners[future] = executors[index]
        pending = set(futures)
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                job = futures[future]
                try:
                    row = future.result()
                except BrokenProcessPool:
                    # The worker was killed outright (typically the RAM OOM-killer;
                    # a GPU OOM raises instead). Recreate this device's worker once
                    # and resubmit; the pool's other queued futures fail the same
                    # way and land on the new worker here too.
                    index = used_devices.index(job.device)
                    if owners[future] is executors[index]:
                        executors[index] = ProcessPoolExecutor(max_workers=1)
                    retry = executors[index].submit(_prepare_episode, job)
                    futures[retry] = job
                    owners[retry] = executors[index]
                    pending.add(retry)
                    print(f"[retry] {job.name}: worker died, resubmitted on {job.device}", flush=True)
                    continue
                rows_by_name[job.name] = row
                gib = np.prod(row["shape"], dtype=np.int64) * row["windows"] * 2 / 1024**3
                print(
                    f"[{len(rows_by_name):02d}/{len(jobs):02d}] {job.name}: {row['windows']} windows "
                    f"{tuple(row['shape'])} ({gib:.2f} GiB)  [{time.time() - started:.0f}s]",
                    flush=True,
                )
    finally:
        for executor in executors:
            executor.shutdown(wait=True)

    manifest = {
        "schema_version": 1,
        "fps": args.fps,
        "chunk_length": args.chunk_length,
        "sample_stride": args.sample_stride,
        "resolution": args.resolution,
        "storage": STORAGE,
        "vae": str(args.vae_path),
    }
    manifest_path = output_root / MANIFEST_NAME
    # Merge, never narrow: episodes cached by earlier runs stay in the manifest.
    merged = {}
    if manifest_path.is_file() and not args.overwrite:
        previous = json.loads(manifest_path.read_text())
        if int(previous.get("schema_version", 0)) != 1:
            raise ValueError(f"Unsupported existing window latent schema in {manifest_path}")
        for field in ("fps", "chunk_length", "sample_stride", "resolution", "storage"):
            if str(previous.get(field)) != str(manifest[field]):
                raise ValueError(f"{manifest_path}: {field} changed; rerun with --overwrite")
        merged = {str(row["name"]): row for row in previous["episodes"]}
    merged.update(rows_by_name)
    manifest["episodes"] = [merged[name] for name in sorted(merged)]
    temp_manifest = manifest_path.with_suffix(".json.tmp")
    temp_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    temp_manifest.replace(manifest_path)

    total = sum(np.prod(row["shape"], dtype=np.int64) * row["windows"] for row in manifest["episodes"])
    print(f"\nWrote {manifest_path}  ({len(names)} episodes, {total * 2 / 1024**3:.1f} GiB)")


if __name__ == "__main__":
    main()
