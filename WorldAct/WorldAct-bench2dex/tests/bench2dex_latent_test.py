"""Cache corruption and collation regressions without loading the VAE."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from cosmos_framework.data.generator.action.datasets.bench2dex_latent_cache import (
    Bench2DexLatentCache,
    sha256,
)
from cosmos_framework.utils.cached_vision_latents import unpack_cached_vision_latents


@pytest.fixture
def cache(tmp_path):
    (tmp_path / "episodes").mkdir()
    for file in ["manifest.json", "video_manifest.json", "vae.pt"]:
        (tmp_path / file).write_text("{}")
    np.savez(tmp_path / "episodes/ep.npz", state=np.zeros((36, 52)))
    ds = SimpleNamespace(
        _chunk_length=32,
        _sample_stride=1,
        _cache_root=tmp_path,
        _episodes=[SimpleNamespace(name="ep")],
        _starts=[np.array([1, 3])],
        _video_cache_rows={"ep": {"shape": [36, 3, 720, 640]}},
    )
    contract = dict(
        schema="bench2dex_valid_window_latents_v1",
        fps=20,
        chunk_length=32,
        sample_stride=1,
        resolution="480",
        storage="bfloat16_bits_in_uint16",
        vae_batch_size=1,
        state_manifest_sha256=sha256(tmp_path / "manifest.json"),
        video_manifest_sha256=sha256(tmp_path / "video_manifest.json"),
        vae_sha256=sha256(tmp_path / "vae.pt"),
    )
    values = torch.zeros((2, 48, 9, 40, 40), dtype=torch.bfloat16)
    values[1] = 2
    np.save(tmp_path / "ep.npy", values.view(torch.uint16).numpy())
    row = dict(
        contract=contract,
        name="ep",
        start_frames=[1, 3],
        state_sha256=sha256(tmp_path / "episodes/ep.npz"),
        path="ep.npy",
        shape=list(values.shape),
        file_bytes=(tmp_path / "ep.npy").stat().st_size,
    )
    m = dict(contract=contract, episodes=[row])
    (tmp_path / "window_manifest.json").write_text(json.dumps(m))
    return ds, tmp_path, m


def test_cached_window_index_and_dtype(cache):
    ds, p, _ = cache
    r = Bench2DexLatentCache(ds, p, p / "vae.pt")
    assert r.read("ep", 1).dtype == torch.bfloat16
    assert (r.read("ep", 1) == 2).all()
    with pytest.raises(IndexError):
        r.read("ep", 2)


@pytest.mark.parametrize("fault", ["start", "vae", "truncated", "state"])
def test_reject_mismatched_cache(cache, fault):
    ds, p, m = cache
    if fault == "start":
        m["episodes"][0]["start_frames"] = [0, 2]
        (p / "window_manifest.json").write_text(json.dumps(m))
    elif fault == "vae":
        (p / "vae.pt").write_text("new weights")
    elif fault == "state":
        (p / "episodes/ep.npz").write_text("changed state")
    else:
        with (p / "ep.npy").open("r+b") as f:
            f.truncate(200)
    with pytest.raises(ValueError):
        Bench2DexLatentCache(ds, p, p / "vae.pt")


def test_collated_shapes_and_count_guard():
    v = torch.zeros((1, 3, 33, 640, 640), device="meta")
    z = torch.zeros((1, 48, 9, 40, 40), dtype=torch.bfloat16, device="meta")
    for batch in [z, z.unsqueeze(0), [z], [z.unsqueeze(0)]]:
        assert unpack_cached_vision_latents(batch, [v])[0].shape == z.shape
    with pytest.raises(ValueError):
        unpack_cached_vision_latents([z, z], [v])
    with pytest.raises(ValueError):
        unpack_cached_vision_latents([z[:, :, :8]], [v])
