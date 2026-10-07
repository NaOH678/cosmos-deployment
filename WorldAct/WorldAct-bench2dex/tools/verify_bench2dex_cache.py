#!/usr/bin/env python3
"""Check every cache header and numeric episode, and write the group split audit."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cosmos_framework.data.generator.action.datasets.bench2dex_dataset import Bench2DexDataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    root = Path(a.cache_root)
    m = json.loads((root / "manifest.json").read_text())
    full = Bench2DexDataset(cache_root=str(root), split="full")
    for i, e in enumerate(full._episodes):
        full._load_arrays(i)
        full._get_video_reader(i)  # checks shape, dtype, physical byte length
    train = Bench2DexDataset(cache_root=str(root), split="train")
    val = Bench2DexDataset(cache_root=str(root), split="val")
    groups = {r["name"]: r["trajectory_group"] for r in m["episodes"]}
    tnames = [e.name for e in train._episodes]
    vnames = [e.name for e in val._episodes]
    tg = {groups[n] for n in tnames}
    vg = {groups[n] for n in vnames}
    assert not tg & vg
    assert len(train) + len(val) == len(full)
    sample = full[0]
    assert sample["action"].shape == (33, 52) and sample["video"].shape[0:2] == (3, 33)
    report = dict(
        cache_root=str(root.resolve()),
        episodes=len(full._episodes),
        trajectory_groups=len(tg | vg),
        frames=sum(e.num_frames for e in full._episodes),
        windows=len(full),
        train_windows=len(train),
        val_windows=len(val),
        train_episodes=tnames,
        val_episodes=vnames,
        train_groups=sorted(tg),
        val_groups=sorted(vg),
        split_seed=42,
        val_ratio=0.1,
        raw_action_dim=52,
        fps=20,
        video_shape=list(sample["video"].shape),
        all_video_headers_checked=True,
    )
    Path(a.output).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, list)}, indent=2))


if __name__ == "__main__":
    main()
