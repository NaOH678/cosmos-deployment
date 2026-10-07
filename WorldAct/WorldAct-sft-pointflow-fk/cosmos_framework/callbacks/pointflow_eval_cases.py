"""Deterministic train/validation windows independent of the training iterator."""

import copy
import hashlib
import json
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader, custom_collate_fn
from cosmos_framework.utils.lazy_config import instantiate


@contextmanager
def evaluation_rng(seed):
    """Do not let evaluation alter training's Python/NumPy/Torch RNG streams."""
    py_state, np_state = random.getstate(), np.random.get_state()
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.default_generator.manual_seed(seed)
            if devices:
                torch.cuda.manual_seed(seed)
            yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def stage_windows(dataset, fractions, windows=1, episodes=1):
    """Pick early/middle/late windows from distinct labeled episodes."""
    if windows < 1 or episodes < 1:
        raise ValueError("windows and episodes must be positive")
    result, seen = [], set()
    if len(fractions) != 3 or not 0 < fractions[0] < fractions[1] < fractions[2] < 1:
        raise ValueError("Expected three increasing interior validation fractions")
    for start, length in dataset.get_shuffle_blocks():
        if length < 2:
            continue
        step = 0
        if windows > 1:
            first = dataset[start]["pointflow"]["metadata"]
            second = dataset[start + 1]["pointflow"]["metadata"]
            span = int(np.asarray(first["raw_frame_ids"])[-1]) - int(first["start_frame"])
            stride = int(second["start_frame"]) - int(first["start_frame"])
            if stride <= 0 or span <= 0 or span % stride:
                raise ValueError("Dataset stride cannot represent contiguous prediction windows")
            step = span // stride
        available = length - 1 - (windows - 1) * step
        if available < 4:
            continue
        bases = [start + round(available * fraction) for fraction in fractions]
        indices = [base + window * step for base in bases for window in range(windows)]
        if len(set(bases)) != 3 or bases[0] == start or indices[-1] == start + length - 1:
            continue
        selected = [(idx, dataset[idx]) for idx in indices]
        points = [sample.get("pointflow") for _, sample in selected]
        if all(
            p is not None and len(p["inputs"]["point_ids"]) and np.asarray(p["targets"]["valid"]).any() for p in points
        ):
            if len({str(p["metadata"]["episode"]) for p in points}) != 1:
                raise ValueError("Dataset block crosses episode boundaries")
            for offset in range(0, len(points), windows):
                for left, right in zip(points[offset : offset + windows - 1], points[offset + 1 : offset + windows]):
                    if int(left["metadata"]["raw_frame_ids"][-1]) != int(right["metadata"]["raw_frame_ids"][0]):
                        raise ValueError("Prediction windows are not temporally contiguous")
            episode = str(points[0]["metadata"]["episode"])
            if episode in seen:
                continue
            seen.add(episode)
            result.extend(selected)
            if len(seen) == episodes:
                return result
    raise ValueError(f"Only {len(seen)} labeled episodes support stage windows; requested {episodes}")


def fixed_cases(config, *, count=2, seed=42, val_stage_fractions=None, val_windows=4, val_episodes=1):
    """Save selected map indices and identity; verify them when resuming."""
    root = Path(config.job.path_local) / "pointflow_eval"
    manifest = root / (
        f"fixed_cases_stages_{val_windows}windows.json" if val_stage_fractions is not None else "fixed_cases.json"
    )
    # All ranks select identically. Only the callback's rank-zero writer persists.
    previous = json.loads(manifest.read_text()) if manifest.exists() else None
    rows, batches = [], []
    for split in ("train", "val"):
        # Runtime Cosmos Config is an attrs object, not a mapping. Its loader
        # fields may already be plain dicts after LazyConfig resolves the job.
        loader = getattr(config, "dataloader_" + split)
        loader_cfg = (
            OmegaConf.to_container(loader, resolve=True) if OmegaConf.is_config(loader) else copy.deepcopy(loader)
        )
        dataset_cfgs = loader_cfg["dataloader"]["datasets"]
        if len(dataset_cfgs) != 1:
            raise ValueError("Fixed PointFlow eval currently requires one action dataset")
        dataset_cfg = copy.deepcopy(next(iter(dataset_cfgs.values()))["dataset"])
        dataset_cfg.update(iterable_shuffle=False, cfg_dropout_rate=0.0, use_image_augmentation=False)
        dataset = instantiate(dataset_cfg)
        stage_mode = split == "val" and val_stage_fractions is not None
        expected_count = 3 * val_windows * val_episodes if stage_mode else count
        if len(dataset) < expected_count:
            raise ValueError(f"Not enough {split} windows for fixed PointFlow eval")
        old = [row for row in previous or [] if row["split"] == split]
        if previous is not None and len(old) != expected_count:
            raise ValueError("Existing fixed_cases.json has different case counts")
        if old:
            selected = [(row["index"], dataset[row["index"]]) for row in old]
        elif stage_mode:
            selected = stage_windows(dataset, val_stage_fractions, val_windows, val_episodes)
        else:
            # Bounded GT-only selection: moving window first, then a low-motion
            # window, preferring another episode. Never select using predictions.
            candidates = []
            for idx in np.unique(np.linspace(0, len(dataset) - 1, min(12, len(dataset)), dtype=int)):
                sample = dataset[int(idx)]
                point = sample.get("pointflow")
                if point is None or not len(point["inputs"]["point_ids"]):
                    continue
                valid = np.asarray(point["targets"]["valid"], bool)
                distances = np.linalg.norm(point["targets"]["displacement"], axis=-1)
                score = float(distances[valid].mean()) if valid.any() else -1
                if score >= 0:
                    candidates.append((score, int(idx), sample))
            if len(candidates) < count:
                raise ValueError(f"Not enough labeled {split} windows in deterministic candidate set")
            candidates.sort(key=lambda item: (-item[0], item[1]))
            selected = [(candidates[0][1], candidates[0][2])]
            remaining = candidates[1:]
            remaining.sort(
                key=lambda item: (
                    item[2]["pointflow"]["metadata"]["episode"] == selected[0][1]["pointflow"]["metadata"]["episode"],
                    item[0],
                    item[1],
                )
            )
            selected.extend((item[1], item[2]) for item in remaining[: count - 1])
        for case_index, (idx, sample) in enumerate(selected):
            meta = sample["pointflow"]["metadata"]
            identity = dict(
                split=split,
                index=idx,
                episode=str(meta["episode"]),
                start_frame=int(meta["start_frame"]),
                raw_frame_ids=np.asarray(meta["raw_frame_ids"]).tolist(),
                source_path=str(meta["source_path"]),
                point_ids=np.asarray(sample["pointflow"]["inputs"]["point_ids"]).tolist(),
            )
            case_seed = int.from_bytes(
                hashlib.sha256(f"{seed}:{split}:{identity['episode']}:{identity['start_frame']}".encode()).digest()[:4],
                "little",
            )
            identity.update(case_id=f"{split}_{case_index:02d}", seed=case_seed)
            if stage_mode:
                identity.update(
                    stage=("early", "middle", "late")[(case_index // val_windows) % 3],
                    window_index=case_index % val_windows,
                    window_fraction=float(val_stage_fractions[(case_index // val_windows) % 3]),
                )
            if old and identity != old[case_index]:
                raise ValueError("Fixed eval identity changed; use a new output directory")
            pack_cfg = {k: v for k, v in loader_cfg.items() if k not in ("_target_", "dataloader")}
            pack_cfg.update(max_samples_per_batch=1, max_sequence_length=None)
            inner = torch.utils.data.DataLoader([sample], batch_size=1, num_workers=0, collate_fn=custom_collate_fn)
            batch = next(iter(PackingDataLoader(dataloader=inner, **pack_cfg)))
            rows.append(identity)
            batches.append(batch)
    return rows, batches
