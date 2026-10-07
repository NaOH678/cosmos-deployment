#!/usr/bin/env python3
"""Cache explicit simulation windows using the native Cosmos spatial/VAE path.

Unlike stride-enumerated caches, rows are keyed by start_frames/frame_ids in the
manifest. A consumer must use that mapping, not start_frame as an array offset.
"""

import argparse
import fcntl
import hashlib
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, data):
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(temp, path)


def prepare(raw, resolution):
    import torch

    from cosmos_framework.data.generator.action.transforms import find_closest_target_size, reflection_pad_to_target

    video = torch.from_numpy(raw).permute(1, 0, 2, 3)
    tw, th = find_closest_target_size(*video.shape[-2:], resolution)
    return reflection_pad_to_target({"video": video}, ["video"], True, tw, th)


def source_rgb(root, name, ids):
    """Independent JPEG decode and native three-view composition (no MP4 loss)."""
    import cv2
    import h5py
    import torch

    from tools.prepare_dualhand_joint_video_cache import compose_dualhand_views

    with h5py.File(root / "raw_data/bench2dex_task21" / name / "observations.hdf5") as f:
        views = []
        for cam in ("cam_overhead", "cam_wrist_left", "cam_wrist_right"):
            frames = [cv2.cvtColor(cv2.imdecode(f[f"cameras/{cam}/rgb"][i], 1), cv2.COLOR_BGR2RGB) for i in ids]
            views.append(torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2))
    return compose_dualhand_views(*views).numpy()


def worker(device, names, args, contract):
    import torch

    from cosmos_framework.data.pointflow_window import read_frame
    from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import Wan2pt2VAEInterface

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(device)
    root, out = Path(args.bundle), Path(args.output_root)
    cache = root / "datasets/bench2dex-task21-cosmos-cache"
    windows = json.loads((cache / "window_index.json").read_text())["windows"]
    tokenizer = Wan2pt2VAEInterface(
        vae_path=args.vae_path,
        causal=True,
        encode_chunk_frames={"256": 68, "480": 24, "720": 8, "768": 8},
        encode_exact_durations=[33],
        spatial_compression_factor=16,
        temporal_compression_factor=4,
    )

    def encode(raw):
        prepared = prepare(raw, args.resolution)
        x = prepared["video"].unsqueeze(0).to(device=f"cuda:{device}").float() / 127.5 - 1
        with torch.inference_mode():
            z = tokenizer.encode(x.to(torch.bfloat16)).cpu().contiguous()
        if not torch.isfinite(z).all():
            raise ValueError("Non-finite VAE output")
        return z[0].view(torch.uint16).numpy(), prepared["image_size"].tolist()

    results = []
    for name in names:
        rows = sorted([r for r in windows if r["episode"] == name], key=lambda r: r["start_frame"])
        ids = np.asarray([r["frame_ids"] for r in rows], dtype=np.int64)
        if ids.shape != (len(rows), 33) or not np.all(np.diff(ids, axis=1) == 1):
            raise ValueError(f"{name}: invalid explicit frame indices")
        if len(set(ids[:, 0])) != len(rows):
            raise ValueError(f"{name}: duplicate windows")
        video_path = cache / "video_frames" / f"{name}.npy"
        state_path = cache / "episodes" / f"{name}.npz"
        with np.load(state_path) as state:
            np.testing.assert_array_equal(state["source_frame_ids"][ids], ids)
            assert state["action_valid"][ids[:, :-1]].all()
        record = dict(
            name=name,
            contract=contract,
            start_frames=ids[:, 0].tolist(),
            frame_ids=ids.tolist(),
            split=rows[0]["split"],
            video_sha256=digest(video_path),
            state_sha256=digest(state_path),
        )
        target, meta = out / f"{name}.npy", out / f"{name}.json"
        if meta.exists():
            prior = json.loads(meta.read_text())
            if any(prior.get(k) != v for k, v in record.items()) or digest(target) != prior["sha256"]:
                raise ValueError(f"{name}: stale cache; choose a fresh output directory")
            results.append(prior)
            print(f"GPU {device}: verified existing {name}", flush=True)
            continue
        temp = target.with_suffix(".npy.tmp")
        shape = None
        with temp.open("wb") as stream:
            for offset, indices in enumerate(ids):
                raw = np.stack([read_frame(video_path, int(i)) for i in indices])
                z, image_size = encode(raw)
                if shape is None:
                    shape = (len(rows), *z.shape)
                    np.lib.format.write_array_header_1_0(stream, dict(descr="<u2", fortran_order=False, shape=shape))
                    record.update(shape=list(shape), image_size=image_size, content_hw=list(raw.shape[-2:]))
                assert z.shape == shape[1:]
                stream.write(z.tobytes())
                print(f"GPU {device}: {name} {offset + 1}/{len(rows)}", flush=True)
        # One full independent source window per episode, including JPEG decode,
        # view composition, spatial transform, and fresh batch-one VAE encoding.
        offset = len(rows) // 2
        raw = source_rgb(root, name, ids[offset])
        cached_raw = np.stack([read_frame(video_path, int(i)) for i in ids[offset]])
        np.testing.assert_array_equal(raw, cached_raw)
        fresh, _ = encode(raw)
        actual = read_frame(temp, offset)
        np.testing.assert_array_equal(fresh, actual)
        record["independent_verification"] = dict(
            start_frame=int(ids[offset, 0]),
            source="HDF5 JPEG -> compose_dualhand_views -> native spatial transform -> VAE",
            bit_exact=True,
        )
        os.replace(temp, target)
        record.update(path=target.name, windows=len(rows), file_bytes=target.stat().st_size, sha256=digest(target))
        atomic_json(meta, record)
        results.append(record)
        print(f"GPU {device}: DONE {name}, independent encode bit-exact", flush=True)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True)
    p.add_argument("--vae-path", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    p.add_argument("--resolution", default="480")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    cache = Path(args.bundle) / "datasets/bench2dex-task21-cosmos-cache"
    index = json.loads((cache / "window_index.json").read_text())
    assert index["fps"] == 20 and index["chunk_length"] == 32
    names = sorted({r["episode"] for r in index["windows"]})
    devices = [int(v) for v in args.devices.split(",")]
    assert len(set(devices)) == len(devices)
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".writer.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        contract = dict(
            schema="sim_pointfk_explicit_window_latents_v1",
            fps=20,
            chunk_length=32,
            enumeration="explicit frame_ids; row offset is NOT start_frame",
            resolution=args.resolution,
            storage="bfloat16_bits_in_uint16",
            vae_batch_size=1,
            seed=args.seed,
            window_index_sha256=digest(cache / "window_index.json"),
            state_manifest_sha256=digest(cache / "manifest.json"),
            video_manifest_sha256=digest(cache / "video_manifest.json"),
            vae_sha256=digest(args.vae_path),
        )
        records = []
        with ProcessPoolExecutor(max_workers=len(devices), mp_context=multiprocessing.get_context("spawn")) as pool:
            jobs = [
                pool.submit(worker, device, names[i :: len(devices)], args, contract)
                for i, device in enumerate(devices)
            ]
            for job in jobs:
                records.extend(job.result())
        assert sum(r["windows"] for r in records) == len(index["windows"])
        atomic_json(
            out / "window_manifest.json", dict(contract=contract, episodes=sorted(records, key=lambda r: r["name"]))
        )
        print(f"COMPLETE: {len(records)} episodes, {len(index['windows'])} windows", flush=True)


if __name__ == "__main__":
    main()
