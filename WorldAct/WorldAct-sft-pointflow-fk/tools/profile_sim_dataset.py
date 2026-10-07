"""Measure simulation sample preparation on CPU with the real tokenizer."""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from cosmos_framework.data.generator.action import transforms
from cosmos_framework.data.generator.action.datasets import sim_pointfk_dataset as sim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--cached-video-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    config = OmegaConf.load(args.config)
    dataset_config = config.dataloader_train.dataloader.datasets.sim_pointfk.dataset
    dataset = sim.SimPointFKSFTDataset(
        dataset_config.bundle,
        dataset_config.selection_root,
        "train",
        config.model.config.vlm_config.tokenizer,
        0.0,
        64,
        cached_video_only=args.cached_video_only,
    )
    timings = defaultdict(float)
    counts = defaultdict(int)

    def wrap(name, function):
        def measured(*a, **kw):
            start = time.perf_counter()
            result = function(*a, **kw)
            key = name
            if name == "read_frame":
                key = "read_rgb" if "video_frames" in str(a[0]) else "read_latent"
            timings[key] += time.perf_counter() - start
            counts[key] += 1
            return result

        return measured

    rows = []
    with (
        patch.object(sim, "read_frame", wrap("read_frame", sim.read_frame)),
        patch.object(torch, "load", wrap("load_point_fk", torch.load)),
        patch.object(transforms.transforms_F, "resize", wrap("resize", transforms.transforms_F.resize)),
        patch.object(transforms.transforms_F, "pad", wrap("pad", transforms.transforms_F.pad)),
        patch.object(
            transforms.ActionTransformPipeline,
            "__call__",
            wrap("transform_total", transforms.ActionTransformPipeline.__call__),
        ),
    ):
        # Exclude tokenizer initialization and the first item from the steady summary.
        for i in range(args.samples + 1):
            before = dict(timings)
            start = time.perf_counter()
            sample = dataset[(i * 7) % len(dataset)]
            row = dict(index=(i * 7) % len(dataset), total=time.perf_counter() - start)
            row.update({k: v - before.get(k, 0) for k, v in timings.items()})
            rows.append(row)
        video = sample["video"]
    result = dict(
        samples=rows,
        mean_seconds={k: float(np.mean([r[k] for r in rows[1:]])) for k in rows[-1] if k != "index"},
        video=dict(shape=list(video.shape), dtype=str(video.dtype), bytes=video.numel() * video.element_size()),
        note="Single CPU thread, serial dataset reads; transform_total includes resize/pad. Not an 8-rank throughput benchmark.",
    )
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=2))


if __name__ == "__main__":
    main()
