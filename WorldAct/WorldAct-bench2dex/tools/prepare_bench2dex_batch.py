#!/usr/bin/env python3
"""Resume parallel per-episode conversion, optionally waiting for verified downloads."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.prepare_bench2dex import convert


def worker(source, part, task, task_text, width):
    import cv2

    cv2.setNumThreads(1)
    part = Path(part)
    if (part / "manifest.json").exists():
        cached = json.loads((part / "manifest.json").read_text())
        if cached["task"] != task or cached["task_text"] != task_text or cached["view_width"] != width:
            raise ValueError(f"Resume configuration differs from completed cache: {part}")
        return cached
    if part.exists():
        # Keep a failed conversion available for diagnosis, never append to it.
        part.rename(part.with_name(part.name + ".incomplete." + str(time.time_ns())))
    return convert(source, part, task, task_text, width)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--task", default="21_condiment_box_loading")
    p.add_argument("--task-text", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--view-width", type=int, default=640)
    p.add_argument("--expected-files", help="Official file listing JSON; wait until every listed HDF5 appears")
    p.add_argument("--wait-seconds", type=int, default=7200)
    a = p.parse_args()
    src = Path(a.source)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    expected = (
        [
            Path(r["path"]).name
            for r in json.loads(Path(a.expected_files).read_text())
            if r["type"] == "file" and r["path"].endswith(".hdf5")
        ]
        if a.expected_files
        else [x.name for x in sorted(src.glob("*.hdf5"))]
    )
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("Missing or duplicate expected episodes")
    if (out / "manifest.json").exists():
        raise FileExistsError("Completed output already exists")
    pending = set(expected)
    futures = {}
    completed = {}
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        while pending or futures:
            for name in sorted(pending):
                if len(futures) >= a.workers:
                    break
                if (src / name).is_file():
                    future = pool.submit(
                        worker, str(src / name), str(out / "parts" / Path(name).stem), a.task, a.task_text, a.view_width
                    )
                    futures[future] = name
                    pending.remove(name)
            if not futures:
                if time.monotonic() - start > a.wait_seconds:
                    raise TimeoutError(f"Missing {len(pending)} downloads")
                time.sleep(5)
                continue
            done, _ = wait(futures, timeout=5, return_when=FIRST_COMPLETED)
            for f in done:
                name = futures.pop(f)
                completed[name] = f.result()
                print(f"CONVERTED {len(completed)}/{len(expected)} {name}", flush=True)
            progress = {
                "completed": len(completed),
                "expected": len(expected),
                "waiting_for_source": len(pending),
                "running": len(futures),
            }
            (out / "progress.json").write_text(json.dumps(progress, indent=2))
    names = sorted(completed)
    m = completed[names[0]].copy()
    m["episodes"] = []
    videos = []
    (out / "episodes").mkdir(exist_ok=True)
    for name in names:
        e = Path(name).stem
        part = out / "parts" / e
        row = completed[name]
        for key in ["task", "robot_key", "joint_names", "view_width", "action_alignment"]:
            if row[key] != m[key]:
                raise ValueError(f"Inconsistent contract {key}: {name}")
        m["episodes"].extend(row["episodes"])
        shutil.copyfile(part / "episodes" / f"{e}.npz", out / "episodes" / f"{e}.npz")
        vm = json.loads((part / "video_manifest.json").read_text())
        for v in vm["episodes"]:
            v["path"] = str(Path("parts") / e / v["path"])
            videos.append(v)
    m["num_episodes"] = len(m["episodes"])
    m["total_frames"] = sum(e["num_frames"] for e in m["episodes"])
    vm["episodes"] = videos
    (out / "video_manifest.json").write_text(json.dumps(vm, indent=2))
    shutil.copyfile(out / "parts" / Path(names[0]).stem / "runtime_joint_names.json", out / "runtime_joint_names.json")
    (out / "manifest.json").write_text(json.dumps(m, indent=2))
    print("COMPLETE", m["num_episodes"], m["total_frames"], flush=True)


if __name__ == "__main__":
    main()
