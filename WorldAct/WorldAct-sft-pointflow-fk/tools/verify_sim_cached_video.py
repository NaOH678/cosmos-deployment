"""Compare full-RGB and cached-video training samples and packing on real windows."""

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from cosmos_framework.callbacks.fk_eval_cases import pack_one, split_dataset
from cosmos_framework.data.generator.action.datasets.sim_pointfk_dataset import SimPointFKSFTDataset


def equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif dataclasses.is_dataclass(a):
        equal(dataclasses.asdict(a), dataclasses.asdict(b))
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b, (a, b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg = OmegaConf.load(args.config)
    source = cfg.dataloader_train.dataloader.datasets.sim_pointfk.dataset
    kw = dict(
        bundle=source.bundle,
        selection_root=source.selection_root,
        split="train",
        tokenizer_config=cfg.model.config.vlm_config.tokenizer,
        cfg_dropout_rate=0.0,
        max_action_dim=64,
    )
    full = SimPointFKSFTDataset(**kw)
    fast = SimPointFKSFTDataset(**kw, cached_video_only=True)
    loader_cfg = OmegaConf.to_container(cfg.dataloader_train, resolve=True)
    checked = []
    for i in (0, 17, len(full) - 1):
        old, new = full[i], fast[i]
        assert new["cached_video_num_frames"] == old["video"].shape[1] == 33
        equal(old["video"][:, :1], new["video"])
        equal(
            {k: v for k, v in old.items() if k != "video"},
            {k: v for k, v in new.items() if k not in ("video", "cached_video_num_frames")},
        )
        # Exercise the actual collator, token budget estimator and packing path.
        old_pack, new_pack = pack_one(loader_cfg, old), pack_one(loader_cfg, new)
        equal(
            {k: v for k, v in old_pack.items() if k != "video"},
            {k: v for k, v in new_pack.items() if k not in ("video", "cached_video_num_frames")},
        )
        checked.append(i)
    eval_frames = {}
    for split in ("train", "val"):
        _, ds = split_dataset(cfg, split)
        eval_frames[split] = ds[0]["video"].shape[1]
        assert eval_frames[split] == 33
    result = dict(
        equal_sample_and_packed_indices=checked,
        eval_rgb_frames=eval_frames,
        note="Exact equality except RGB contains the identical first frame and explicit full-frame-count metadata. GPU model execution not covered.",
    )
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
