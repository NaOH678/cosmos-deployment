"""Deterministic train/validation windows independent of the training iterator.

The FK counterpart of ``pointflow_eval_cases.py``.  Case selection reads only
*labels* (how far the hand actually moves), never a model prediction, so the
reported numbers cannot be improved by picking easier windows after the fact.
The chosen indices are written to disk and re-checked on resume: a run that
silently evaluated different windows would look like a regression or a jump
depending on which way the selection drifted.
"""

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


def _motion_score(sample):
    """Mean keypoint displacement over the supervised steps, in millimetres.

    Read straight off the labels.  ``-1`` means "no usable labels", which is how
    an episode that slipped out of the annotation set shows up here rather than
    as a silently low-motion case.
    """
    fk = sample.get("fk")
    if fk is None or not len(fk["inputs"]["point_ids"]):
        return -1.0
    valid = np.asarray(fk["targets"]["valid"], bool)
    if not valid.any():
        return -1.0
    return float(np.linalg.norm(np.asarray(fk["targets"]["displacement"], float), axis=-1)[valid].mean() * 1000)


def split_dataset(config, split):
    """The eval-time dataset for one split, built exactly as ``fixed_cases`` builds it.

    Extracted so a rollout indexes the *same* dataset the cases were chosen from.
    A second construction that drifted by one setting would put the rollout's
    windows on different frames than the cases it exists to extend, and nothing
    downstream would notice.
    """
    # Runtime Cosmos Config is an attrs object, not a mapping. Its loader
    # fields may already be plain dicts after LazyConfig resolves the job.
    loader = getattr(config, "dataloader_" + split)
    loader_cfg = OmegaConf.to_container(loader, resolve=True) if OmegaConf.is_config(loader) else copy.deepcopy(loader)
    dataset_cfgs = loader_cfg["dataloader"]["datasets"]
    if len(dataset_cfgs) != 1:
        raise ValueError("Fixed FK eval currently requires one action dataset")
    dataset_cfg = copy.deepcopy(next(iter(dataset_cfgs.values()))["dataset"])
    # cfg_dropout_rate=0: a case whose caption was dropped measures the
    # caption dropout, not the FK branch.
    dataset_cfg.update(iterable_shuffle=False, cfg_dropout_rate=0.0, use_image_augmentation=False)
    return loader_cfg, instantiate(dataset_cfg)


def pack_one(loader_cfg, sample):
    """One dataset sample -> one packed batch, the single-sample path ``fixed_cases`` uses."""
    pack_cfg = {k: v for k, v in loader_cfg.items() if k not in ("_target_", "dataloader")}
    pack_cfg.update(max_samples_per_batch=1, max_sequence_length=None)
    inner = torch.utils.data.DataLoader([sample], batch_size=1, num_workers=0, collate_fn=custom_collate_fn)
    return next(iter(PackingDataLoader(dataloader=inner, **pack_cfg)))


def window_step(dataset):
    """``(span, stride, step)`` in dataset indices.

    ``span`` is a window's raw-frame extent, ``stride`` the raw frames between
    consecutive indices, and ``step == span // stride`` the number of indices
    between two windows that share **exactly one** frame -- the earlier window's
    last predicted frame is the later window's anchor.

    Measured, not hardcoded: it is a product of ``sample_stride`` and
    ``source_stride``, and a hardcoded value would silently tile the wrong frames
    (the failure is a smooth-looking rollout over frames that never existed).

    Tiling by ``step`` is what removes the overlap question. Every predicted frame
    then comes from exactly one window, so there is no "which chunk wins" to
    decide. Anything smaller stacks predicted frames on top of each other.
    """
    first = dataset[0]["fk"]["metadata"]
    span = int(np.asarray(first["raw_frame_ids"])[-1]) - int(first["start_frame"])
    stride = None
    for index in range(1, min(len(dataset), 64)):
        meta = dataset[index]["fk"]["metadata"]
        if meta["episode"] == first["episode"]:
            stride = int(meta["start_frame"]) - int(first["start_frame"])
            break
    if stride is None:
        raise ValueError("no episode with two windows; cannot measure the window spacing")
    if stride <= 0 or span <= 0 or span % stride:
        raise ValueError(f"cannot tile contiguously: span={span} stride={stride}")
    return span, stride, span // stride


def raw_dataset(dataset):
    """The innermost dataset, reached through the index-preserving wrappers.

    ``instantiate`` returns an ``ActionSFTDataset`` -- a transform wrapper -- whose
    indices are 1:1 with the raw dataset's (both ``__len__`` and ``__getitem__``
    delegate) but which does **not** carry the window bookkeeping: ``_cumulative_ends``
    lives on the raw one.  The first version of this assumed the wrapper itself had
    it and died at the first rollout with "dataset does not expose _cumulative_ends".

    Reaching through is only sound while the mapping stays 1:1, so the delegation is
    checked rather than assumed: a wrapper that ever added or dropped samples would
    make every index below address the wrong episode, silently.
    """
    inner, depth = dataset, 0
    while not hasattr(inner, "_cumulative_ends"):
        nxt = getattr(inner, "_dataset", None)
        if nxt is None or depth > 8:
            raise ValueError(
                f"{type(dataset).__name__} has no _cumulative_ends and no _dataset to unwrap; "
                "cannot bound the episode cheaply"
            )
        if len(nxt) != len(inner):
            raise ValueError(
                f"{type(inner).__name__} is not index-preserving ({len(nxt)} inner vs {len(inner)} outer); "
                "indexing the raw dataset would address the wrong windows"
            )
        inner, depth = nxt, depth + 1
    return inner


def episode_index_range(dataset, index):
    """``[start, end)`` dataset indices of the episode holding ``index``.

    Reads the dataset's cumulative window counts rather than probing samples: each
    ``dataset[i]`` decodes latents, so scanning for the episode boundary a sample
    at a time would cost minutes per case.
    """
    import bisect

    ends = raw_dataset(dataset)._cumulative_ends
    if not len(ends):
        raise ValueError("dataset has no windows; cannot bound an episode")
    episode_idx = bisect.bisect_right(ends, index)
    if episode_idx >= len(ends):
        raise ValueError(f"index {index} is past the last window ({int(ends[-1])})")
    start = 0 if episode_idx == 0 else int(ends[episode_idx - 1])
    return start, int(ends[episode_idx])


def rollout_windows(config, *, split="val", base_index, seconds=None, fps=15.0, seed=42):
    """Windows of one episode, one ``step`` apart -- a contiguous rollout.

    Starts at the episode's **first** window, not at ``base_index``: the cases are
    chosen as high-motion windows anywhere in the episode, and a rollout that
    began mid-episode would render a clip the label "complete episode" does not
    describe.

    ``seconds``, when given, keeps only the **middle** that many seconds and drops
    the windows on either side.  An episode runs ~40 s, which makes for a long clip
    to watch and an expensive one to sample; the middle is the fair place to cut
    because it is where the hand is doing the task, not reaching in or settling.
    The cut lands on window boundaries, so the retained span is the closest whole
    number of windows to ``seconds`` -- within one window (32 steps) of the ask.

    Returns ``(identities, batches)``. The windows tile the episode with no gap
    and no duplicated predicted frame, so concatenating their 32 steps in order
    yields the episode's whole trajectory. The one frame they share is the later
    window's anchor, which is ground truth and is therefore not emitted twice.
    """
    loader_cfg, dataset = split_dataset(config, split)
    span, stride, step = window_step(dataset)
    start, end = episode_index_range(dataset, base_index)
    indices = list(range(start, end, step))
    if not indices:
        raise ValueError(f"episode at index {base_index} has no windows at step {step}")
    episode = dataset[start]["fk"]["metadata"]["episode"]
    if seconds is not None:
        # ``step`` is both the index stride and the steps per window, so the step
        # count of the whole tiling is ``len(indices) * step``.
        steps_per_window = int(np.asarray(dataset[start]["fk"]["metadata"]["raw_frame_ids"]).size) - 1
        want = max(1, int(round(float(seconds) * float(fps) / steps_per_window)))
        if want < len(indices):
            drop = (len(indices) - want) // 2
            indices = indices[drop : drop + want]
    identities, batches = [], []
    for index in indices:
        sample = dataset[index]
        meta = sample["fk"]["metadata"]
        if meta["episode"] != episode:
            raise ValueError(f"index {index} left episode {episode}; the range scan is wrong")
        identities.append(
            dict(
                case_id=f"rollout_{len(identities):03d}",
                episode=episode,
                index=index,
                start_frame=int(meta["start_frame"]),
                raw_frame_ids=np.asarray(meta["raw_frame_ids"]).tolist(),
                span=span,
                stride=stride,
                step=step,
            )
        )
        batches.append(pack_one(loader_cfg, sample))
    return identities, batches


def fixed_cases(config, *, count=2, seed=42):
    """Save selected map indices and identity; verify them when resuming."""
    root = Path(config.job.path_local) / "fk_eval"
    manifest = root / "fixed_cases.json"
    # All ranks select identically. Only the callback's rank-zero writer persists.
    previous = json.loads(manifest.read_text()) if manifest.exists() else None
    rows, batches = [], []
    for split in ("train", "val"):
        loader_cfg, dataset = split_dataset(config, split)
        if len(dataset) < count:
            raise ValueError(f"Not enough {split} windows for fixed FK eval")
        old = [row for row in previous or [] if row["split"] == split]
        if previous is not None and len(old) != count:
            raise ValueError("Existing fixed_cases.json has a different case count")
        if old:
            selected = [(row["index"], dataset[row["index"]]) for row in old]
        else:
            # Candidate pool spread over the dataset; the most-moving window
            # first, then distinct episodes ahead of easier same-episode ones.
            candidates = []
            for idx in np.unique(np.linspace(0, len(dataset) - 1, min(12, len(dataset)), dtype=int)):
                sample = dataset[int(idx)]
                score = _motion_score(sample)
                if score >= 0:
                    candidates.append((score, int(idx), sample))
            if len(candidates) < count:
                raise ValueError(f"Not enough labeled {split} windows in the deterministic candidate set")
            candidates.sort(key=lambda item: (-item[0], item[1]))
            selected = [(candidates[0][1], candidates[0][2])]
            remaining = candidates[1:]
            remaining.sort(
                key=lambda item: (
                    item[2]["fk"]["metadata"]["episode"] == selected[0][1]["fk"]["metadata"]["episode"],
                    item[0],
                    item[1],
                )
            )
            selected.extend((item[1], item[2]) for item in remaining[: count - 1])
        for case_index, (idx, sample) in enumerate(selected):
            meta = sample["fk"]["metadata"]
            identity = dict(
                split=split,
                index=idx,
                episode=str(meta["episode"]),
                start_frame=int(meta["start_frame"]),
                raw_frame_ids=np.asarray(meta["raw_frame_ids"]).tolist(),
                point_ids=np.asarray(sample["fk"]["inputs"]["point_ids"]).tolist(),
            )
            case_seed = int.from_bytes(
                hashlib.sha256(f"{seed}:{split}:{identity['episode']}:{identity['start_frame']}".encode()).digest()[:4],
                "little",
            )
            identity.update(case_id=f"{split}_{case_index:02d}", seed=case_seed)
            if old and identity != old[case_index]:
                raise ValueError("Fixed eval identity changed; use a new output directory")
            batch = pack_one(loader_cfg, sample)
            rows.append(identity)
            batches.append(batch)
    return rows, batches
