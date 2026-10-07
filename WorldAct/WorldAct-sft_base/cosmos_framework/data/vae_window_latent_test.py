"""The dataset must hand the model the window's own VAE latent, and refuse a stale cache.

Three things are worth a test rather than a launch:

1. ``sample["vae_latent_cache"]`` is the latent sitting at this window's offset in
   the cache -- not the whole-episode cache, not offset 0.  A wrong offset is a
   plausible-looking video with the wrong content, and nothing downstream catches it.
2. The cache layout is ``bfloat16`` bit patterns in ``uint16``: reading it as
   numbers instead of bits would give finite garbage, so the dtype/header check
   has to be real.
3. A cache whose ``fps``/``chunk_length``/``sample_stride`` disagree with the run
   must raise, because the window enumeration is what the offsets index into.

Run:  PYTHONPATH=. <venv>/bin/python cosmos_framework/data/vae_window_latent_test.py
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.data.generator.action.datasets.singlerighthand_raw_dataset import (
    SingleRightHandRawDataset,
)

RAW_ROOT = "/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/raw_data/singlerighthand_sandwich_100"
CACHE_ROOT = "/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/datasets/singlerighthand-sandwich-100-cosmos-cache"
WINDOW_ROOT = Path(CACHE_ROOT) / "vae_window_latents"
ALLOWLIST = Path(__file__).resolve().parents[2] / "examples/pointflow_sandwich_10_episodes.txt"


def _stub_video(self, episode_idx, observation_indices):
    """Stand-in for ``_load_video``.

    The real one memmaps ``video_frames/<episode>.npy``, and this dev node's
    ``virtiofs`` mount rejects mmap outright (ENODEV) -- the GPU node mounts it
    differently and the production runs do use it.  Nothing under test here reads
    the pixels: the claim is about ``vae_latent_cache``, so a small dummy keeps
    the ``__getitem__`` wiring real while staying runnable off-GPU.
    """
    return torch.zeros(3, len(observation_indices), 8, 8, dtype=torch.uint8)


def build(**overrides):
    kwargs = dict(
        root=RAW_ROOT,
        cache_root=CACHE_ROOT,
        fps=15.0,
        chunk_length=32,
        split="train",
        split_val_ratio=0.2,
        sample_stride=1,
        use_precomputed_video=True,
        video_decoder="opencv",
        episode_allowlist=str(ALLOWLIST),
        vae_window_latent_root=str(WINDOW_ROOT),
    )
    kwargs.update(overrides)
    return SingleRightHandRawDataset(**kwargs)


def load_array(name: str) -> np.ndarray:
    """The whole ``uint16`` payload for one episode, from the manifest's own path."""
    manifest = json.loads((WINDOW_ROOT / "window_manifest.json").read_text())
    row = next(r for r in manifest["episodes"] if r["name"] == name)
    array = np.load(WINDOW_ROOT / row["path"])
    assert array.dtype == np.uint16, array.dtype
    return array


def read_reference(name: str, offset: int) -> torch.Tensor:
    """Independent read of the same window: whole array, then index.

    Deliberately not the seek-and-read the dataset does -- loading the array and
    indexing it is a different route to the same value, which is what makes this
    a check of the offset arithmetic rather than a restatement of it.
    ``mmap_mode`` is not an option: this node's virtiofs mount rejects mmap.
    """
    return torch.from_numpy(np.array(load_array(name)[offset])).view(torch.bfloat16).unsqueeze(0)


def main():
    # Patch for the whole run: the stub has to still be in place when __getitem__
    # is called, not just during construction.
    SingleRightHandRawDataset._load_video = _stub_video

    dataset = build()
    sample = dataset[0]
    latent = sample["vae_latent_cache"]
    # Read the identity from the dataset's own bookkeeping rather than from the
    # sample: an FK-free dataset has no "fk" key, and the episode/window pair is
    # exactly what the latent lookup is supposed to agree with.
    episode_idx, offset = dataset._resolve_index(0)
    name = dataset._episodes[episode_idx].name

    row = dataset._vae_window_rows[name]
    expected_shape = (1, *row["shape"])
    assert tuple(latent.shape) == expected_shape, f"{tuple(latent.shape)} != {expected_shape}"
    assert latent.dtype == torch.bfloat16, latent.dtype
    print(f"✅ 形状/类型: {tuple(latent.shape)} {latent.dtype}  (episode {name}, window {offset})")

    reference = read_reference(name, offset)
    assert torch.equal(latent, reference), "缓存读到的不等于该窗口位置的真实内容"
    print(f"✅ 内容等于该 window_offset 的独立读取  (max|diff| = {(latent.float() - reference.float()).abs().max():.6f})")

    # A different offset must give different content -- otherwise "read the right
    # window" is not actually being tested, since offset 0 would pass too.
    other = dataset._read_window_latent(name, offset + 1)
    assert not torch.equal(latent, other), "相邻窗口内容相同，说明偏移没生效"
    print("✅ 相邻窗口内容不同（偏移确实生效，不是恒读 offset 0）")

    # Teeth: reading the uint16 payload as numbers must NOT equal the bf16 view.
    raw = load_array(name)[offset]
    as_numbers = torch.from_numpy(np.array(raw).astype(np.int32))
    assert not torch.equal(as_numbers.view(-1)[:8], latent.view(-1)[:8]), "位模式与数值恰好相同，测试无意义"
    print("✅ 位模式 ≠ 数值读取（错读会得到有限但错误的数，不会被 NaN 挡住）")

    # The collate seam.  This is the shape contract the model's read branch is
    # written against, and it is not the same as the dataset's: the collator emits
    # a LIST of 6-dim tensors, which the model squeezes back to the 5-dim
    # [1,C,T,H,W] that the online `_encode_vision_item` returns.  A mismatch here
    # would be silently broadcast rather than raised.
    from cosmos_framework.data.generator.dataflow.collators import VFMListCollator

    batch = VFMListCollator().collate([dataset[i] for i in (0, 1)])
    collated = batch["vae_latent_cache"]
    assert isinstance(collated, (list, tuple)), type(collated)
    assert len(collated) == 2, len(collated)
    for item in collated:
        assert isinstance(item, torch.Tensor) and item.ndim == 6, tuple(item.shape)
        assert item.shape == (1, 1, *row["shape"]), tuple(item.shape)
    squeezed = collated[0].squeeze(0)
    assert squeezed.shape == (1, *row["shape"]), tuple(squeezed.shape)
    print(f"✅ collate 后为 list[{len(collated)}]，每项 {tuple(collated[0].shape)} -> squeeze 回 {tuple(squeezed.shape)}")
    print("   ↑ 与在线 _encode_vision_item 的返回形状一致（模型侧读的就是这个）")

    # Stale cache: a run whose chunk_length disagrees must refuse to start.
    for field, bad in (("fps", 30.0), ("chunk_length", 16), ("sample_stride", 2)):
        try:
            build(**{field: bad})
        except ValueError as error:
            assert field in str(error) and "regenerate" in str(error), error
            print(f"✅ 过期缓存被拒: 数据集设为 {field}={bad} -> {str(error).split(':', 1)[1].strip()[:78]}")
        else:
            raise AssertionError(f"{field}={bad} 应该被拒绝")

    # No cache configured -> no key, and the model falls back to the VAE.
    plain = build(vae_window_latent_root=None)
    assert "vae_latent_cache" not in plain[0]
    print("✅ 不配缓存时不产生该键（回落在线的 VAE 路径）")

    # A manifest missing its per-episode storage field must not be read as numbers.
    with tempfile.TemporaryDirectory() as tmp:
        tampered = json.loads((WINDOW_ROOT / "window_manifest.json").read_text())
        # Strip it from EVERY episode: the dataset picks its own episode order from
        # the split seed, so tampering with one row can miss the selected set and
        # make the check pass for the wrong reason.
        for row in tampered["episodes"]:
            row.pop("storage")
        (Path(tmp) / "window_manifest.json").write_text(json.dumps(tampered))
        for row in tampered["episodes"]:
            (Path(tmp) / row["path"]).symlink_to(WINDOW_ROOT / row["path"])
        # The per-episode storage check is lazy -- it runs when a window is read,
        # not at construction (unlike the fps/chunk_length/sample_stride check,
        # which is about the whole cache and can be settled up front).
        try:
            build(vae_window_latent_root=tmp)[0]
        except ValueError as error:
            assert "storage" in str(error), error
            print(f"✅ 缺 storage 字段的 manifest 被拒（读取时）: {str(error)[:60]}")
        else:
            raise AssertionError("缺 storage 字段应该被拒绝")

    print("\nPASS")


if __name__ == "__main__":
    main()
