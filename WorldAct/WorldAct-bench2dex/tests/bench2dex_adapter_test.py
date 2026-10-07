"""CPU contract tests: real converter, windows, and RPC joint mapping."""

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from cosmos_framework.data.generator.action.datasets.bench2dex_dataset import (
    Bench2DexDataset,
)
from cosmos_framework.inference.robot_policy.bench2dex import Bench2DexPolicy
from cosmos_framework.utils.bench2dex_contract import (
    CAMERAS,
    JOINT_NAMES,
    ROBOT,
    compose_rgb,
    permutation,
)
from tools.prepare_bench2dex import convert


def make_source(path, name="episode_000000"):
    p = path / f"{name}.hdf5"
    names = list(reversed(JOINT_NAMES))
    anames = list(JOINT_NAMES[7:] + JOINT_NAMES[:7])
    n = 80
    state = np.arange(n, dtype=np.float32)[:, None] + np.arange(52, dtype=np.float32)[None, :] / 100
    action = state + 100
    with h5py.File(p, "w") as f:
        for k, v in {
            "robot_key": ROBOT,
            "scene_name": "task",
            "fps": 20,
            "step_stride": 3,
            "homing_start_sim_step": 210,
        }.items():
            f[f"meta/{k}"] = v
        f["robot/joint_names"] = np.asarray(names, dtype="S")
        f["action/action_names"] = np.asarray(anames, dtype="S")
        f["robot/qpos"] = state[:, permutation(JOINT_NAMES, names)]
        f["action/commanded"] = action[:, permutation(JOINT_NAMES, anames)]
        valid = np.ones(n, bool)
        valid[0] = False
        valid[40] = False
        f["action/action_valid"] = valid
        f["action/control_mode"] = "joint_position"
        f["action/action_type"] = "absolute"
        f["time/sim_step"] = np.arange(n) * 3
        for j, c in enumerate(CAMERAS):
            image = np.zeros((n, 8, 12, 3), np.uint8)
            image[..., j] = 200
            f[f"cameras/{c}/rgb"] = image
    return p, state, action


def test_conversion_reorder_homing_and_valid_windows(tmp_path):
    p, state, action = make_source(tmp_path)
    out = tmp_path / "cache"
    convert(p, out, "task", "load objects", 32)
    data = Bench2DexDataset(cache_root=str(out), split="full")
    with np.load(out / "episodes/episode_000000.npz") as x:
        np.testing.assert_array_equal(x["state"], state[:70])
        np.testing.assert_array_equal(x["action"][1:40], action[1:40])
    assert len(data) == 8  # starts 1..8; start9's horizon includes invalid action40
    a = data[0]
    assert a["action"].shape == (33, 52) and a["video"].shape == (3, 33, 36, 32)
    np.testing.assert_array_equal(a["action"][0], state[1])
    np.testing.assert_array_equal(a["action"][1], action[1])
    assert a["conditioning_fps"].item() == 20 and a["domain_id"].item() == 28
    assert data.get_shuffle_blocks() == [(0, 8)]


def test_strict_input_contract(tmp_path):
    with pytest.raises(ValueError):
        permutation(["wrong"] * 52)
    p, _, _ = make_source(tmp_path)
    with pytest.raises(ValueError, match="wrong robot or task"):
        convert(p, tmp_path / "bad", "different", "text", 32)
    with h5py.File(p, "r+") as f:
        f["meta/homing_start_sim_step"][...] = -1
    with pytest.raises(ValueError, match="homing"):
        convert(p, tmp_path / "missing", "task", "text", 32)


def test_shared_rgb_layout():
    images = {c: np.full((6, 8, 3), v, np.uint8) for c, v in zip(CAMERAS, [10, 20, 30])}
    x = compose_rgb(images, 32)
    assert x.shape == (3, 36, 32)
    assert np.all(x[:, 0:24] == 10)
    assert np.all(x[:, 24:, :16] == 20) and np.all(x[:, 24:, 16:] == 30)


def test_inference_joint_roundtrip_and_execution_horizon():
    # No GPU or checkpoint needed: exercise the actual public RPC adapter.
    p = Bench2DexPolicy.__new__(Bench2DexPolicy)
    p.runtime_names = list(reversed(JOINT_NAMES))
    p.expected_runtime_names = list(p.runtime_names)
    p.to_model = permutation(p.runtime_names)
    p.to_runtime = permutation(JOINT_NAMES, p.runtime_names)
    p.execution_horizon = 4
    runtime = np.arange(52, dtype=np.float32)

    def infer(images, state):
        np.testing.assert_array_equal(state, runtime[p.to_model])
        return np.tile(state + 0.1, (32, 1)), 0.0

    p._infer_native = infer
    obs = {
        "joint_names": list(p.runtime_names),
        "joint_action": {"qpos": runtime},
        "observation": {c: {"rgb": np.zeros((8, 8, 3), np.uint8)} for c in CAMERAS},
    }
    np.testing.assert_allclose(p.get_action(obs), np.tile(runtime + 0.1, (4, 1)))
    obs["joint_names"] = list(JOINT_NAMES)
    with pytest.raises(ValueError, match="order differs"):
        p.get_action(obs)


def test_episode_split_disjoint(tmp_path):
    source = tmp_path / "sources"
    source.mkdir()
    for i in range(10):
        make_source(source, f"episode_{i:06d}")
    out = tmp_path / "cache"
    convert(source, out, "task", "text", 32)
    train = Bench2DexDataset(cache_root=str(out), split="train")
    val = Bench2DexDataset(cache_root=str(out), split="val")
    assert len(train._episodes) == 9 and len(val._episodes) == 1
    assert not {e.name for e in train._episodes} & {e.name for e in val._episodes}


def test_train_inference_batch_parity(tmp_path):
    import torch

    from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline

    p, _, _ = make_source(tmp_path)
    out = tmp_path / "cache"
    convert(p, out, "task", "load objects", 32)
    dataset = Bench2DexDataset(cache_root=str(out), split="full")
    raw = dataset[0]
    state = raw["action"][0].numpy().copy()
    transform = ActionTransformPipeline(max_action_dim=64, format_prompt_as_json=True)
    training = transform(raw, "480")
    policy = Bench2DexPolicy.__new__(Bench2DexPolicy)
    policy.device = "cpu"
    policy.view_width = 32
    policy.input_video_key = "video"
    policy.config = SimpleNamespace(
        model=SimpleNamespace(
            native_chunk_size=32,
            native_action_dim=52,
            max_action_dim=64,
            task="load objects",
            resolution="480",
            domain_name="bench2dex_wuji",
        ),
        deployment=SimpleNamespace(action_rate_hz=20.0),
    )
    with h5py.File(p) as f:
        images = {c: f[f"cameras/{c}/rgb"][1] for c in CAMERAS}
    inference = policy._build_batch(images, state)
    torch.testing.assert_close(training["video"][:, 0], inference["video"][0][0][:, 0])
    torch.testing.assert_close(training["action"][0], inference["action"][0][0][0])
    assert training["action"].shape == (33, 64)
    assert inference["action"][0][0][1:].count_nonzero().item() == 0
    assert training["ai_caption"] == json.loads(inference["ai_caption"][0])
    assert training["sequence_plan"] == inference["sequence_plan"][0]


def test_augmentation_variants_stay_in_same_split(tmp_path):
    source = tmp_path / "sources"
    source.mkdir()
    for i in range(10):
        make_source(source, f"episode_{i:06d}")
        make_source(source, f"episode_{i:06d}_1")
    out = tmp_path / "cache"
    convert(source, out, "task", "text", 32)
    train = Bench2DexDataset(cache_root=str(out), split="train")
    val = Bench2DexDataset(cache_root=str(out), split="val")
    assert len(train._episodes) == 18 and len(val._episodes) == 2
    assert not {e.name[:14] for e in train._episodes} & {e.name[:14] for e in val._episodes}


def test_runtime_names_are_mandatory():
    p = Bench2DexPolicy.__new__(Bench2DexPolicy)
    with pytest.raises(ValueError, match="Missing runtime"):
        p.get_action({})


def test_normalized_training_inference_parity_and_inverse(tmp_path):
    import torch

    from cosmos_framework.data.generator.action.action_processing import ActionProcessor
    from cosmos_framework.data.generator.action.datasets.action_sft_dataset import get_action_bench2dex_sft_dataset
    from tools.compute_bench2dex_action_stats import compute_stats

    src = tmp_path / "sources"
    src.mkdir()
    for i in range(10):
        make_source(src, f"episode_{i:06d}")
    root = tmp_path / "cache"
    convert(src, root, "task", "load objects", 32)
    stats = compute_stats(root)
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps(stats))
    dataset = get_action_bench2dex_sft_dataset(cache_root=str(root), action_stats_path=str(stats_path))
    raw = dataset._dataset[0]
    training = dataset[0]
    normalizer = dataset._dataset.action_normalizer
    assert normalizer is not None
    torch.testing.assert_close(training["action_raw"], raw["action"])
    torch.testing.assert_close(training["action"][:, :52], normalizer.normalize_action(raw["action"]))
    assert training["action"][:, 52:].count_nonzero() == 0
    torch.testing.assert_close(
        ActionProcessor.postprocess_action(training["action"], training["action_processing_record"]),
        raw["action"],
        atol=2e-5,
        rtol=1e-5,
    )
    policy = Bench2DexPolicy.__new__(Bench2DexPolicy)
    policy.action_normalizer = normalizer
    policy.device = "cpu"
    policy.view_width = 32
    policy.input_video_key = "video"
    policy.config = SimpleNamespace(
        model=SimpleNamespace(
            native_chunk_size=32,
            native_action_dim=52,
            max_action_dim=64,
            task="load objects",
            resolution="480",
            domain_name="bench2dex_wuji",
        ),
        deployment=SimpleNamespace(action_rate_hz=20.0),
    )
    images = {c: np.zeros((8, 12, 3), np.uint8) for c in CAMERAS}
    inference = policy._build_batch(images, raw["action"][0].numpy())
    torch.testing.assert_close(inference["action"][0][0][0], training["action"][0])
    assert inference["action"][0][0][1:].count_nonzero() == 0
    restored = ActionProcessor.postprocess_action(training["action"], inference["action_processing_record"][0])
    torch.testing.assert_close(restored, raw["action"], atol=2e-5, rtol=1e-5)


def test_stats_exclude_validation_invalid_but_include_all_valid_frames(tmp_path):
    from tools.compute_bench2dex_action_stats import compute_stats

    src = tmp_path / "sources"
    src.mkdir()
    for i in range(10):
        make_source(src, f"episode_{i:06d}")
    root = tmp_path / "cache"
    convert(src, root, "task", "text", 32)
    before = compute_stats(root)
    val = Bench2DexDataset(cache_root=str(root), split="val")
    for e in val._episodes:
        path = root / "episodes" / f"{e.name}.npz"
        with np.load(path) as z:
            data = dict(z)
        data["state"][:] = 1e6
        data["action"][:] = 1e6
        np.savez(path, **data)
    # Only starts 1..8 form windows. Invalid frames 0/40 must not enter statistics.
    for name in before["train_episodes"]:
        path = root / "episodes" / f"{name}.npz"
        with np.load(path) as z:
            data = dict(z)
        data["action"][[0, 40]] = 1e6
        data["state"][[0, 40]] = 1e6
        np.savez(path, **data)
    after = compute_stats(root)
    for key in ["action", "state", "offset", "scale"]:
        assert before[key] == after[key]
    assert before["action"]["count"] == 9 * 68
    assert before["state"]["count"] == 9 * 68
    # Valid frame 65 is outside any window but must be counted, matching Bench2Dex finalize.
    path = root / "episodes" / f"{before['train_episodes'][0]}.npz"
    with np.load(path) as z:
        data = dict(z)
    data["state"][65] = 1e6
    np.savez(path, **data)
    state_changed = compute_stats(root)
    assert state_changed["state"] != after["state"]
    assert state_changed["offset"] == after["offset"]
    assert state_changed["scale"] == after["scale"]
    data["action"][65] = 1e6
    np.savez(path, **data)
    assert compute_stats(root)["action"] != after["action"]


def test_stats_contract_and_checkpoint_guard(tmp_path):
    import hashlib

    from omegaconf import OmegaConf

    from cosmos_framework.inference.robot_policy.bench2dex import load_policy_action_normalizer
    from cosmos_framework.utils.bench2dex_normalization import load_bench2dex_normalizer
    from tools.compute_bench2dex_action_stats import compute_stats
    from tools.prepare_bench2dex_normalized_run import prepare

    src, _, _ = make_source(tmp_path)
    root = tmp_path / "cache"
    convert(src, root, "task", "text", 32)
    stats = compute_stats(root, split_val_ratio=0)
    path = tmp_path / "stats.json"
    path.write_text(json.dumps(stats))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    load_bench2dex_normalizer(path, digest)
    with pytest.raises(ValueError, match="SHA256"):
        load_bench2dex_normalizer(path, "0" * 64)
    with pytest.raises(ValueError, match="contract mismatch"):
        Bench2DexDataset(
            cache_root=str(root), split="full", split_val_ratio=0, action_stats_path=str(path), chunk_length=16
        )
    stats["joint_names"].reverse()
    path.write_text(json.dumps(stats))
    with pytest.raises(ValueError, match="contract"):
        load_bench2dex_normalizer(path)
    stats["joint_names"].reverse()
    path.write_text(json.dumps(stats))
    run = tmp_path / "run"
    snapshot, digest = prepare(path, run)
    config = run / "cosmos3_action/action_sft/action_policy_bench2dex_edge/config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("{}\n")
    with pytest.raises(ValueError, match="new OUTPUT_ROOT"):
        prepare(path, run)
    with pytest.raises(ValueError, match="unnormalized"):
        load_policy_action_normalizer({"config_file": str(config), "action_stats_path": str(path)})
    cfg = {
        "dataloader_train": {
            "dataloader": {
                "datasets": {
                    "bench2dex": {
                        "dataset": {
                            "action_stats_path": str(snapshot),
                            "action_stats_sha256": digest,
                        }
                    }
                }
            }
        }
    }
    OmegaConf.save(cfg, config)
    assert load_policy_action_normalizer({"config_file": str(config), "action_stats_path": str(path)}) is not None
    assert prepare(path, run)[1] == digest
    with np.load(root / "episodes/episode_000000.npz") as z:
        data = dict(z)
    data["action"][1, 0] += 0.1
    np.savez(root / "episodes/episode_000000.npz", **data)
    with pytest.raises(ValueError, match="source data changed"):
        Bench2DexDataset(cache_root=str(root), split="full", split_val_ratio=0, action_stats_path=str(path))


def test_normalizer_keeps_tails_and_rejects_bad_scales(tmp_path):
    import torch

    from cosmos_framework.utils.bench2dex_normalization import Bench2DexJointNormalizer, load_bench2dex_normalizer
    from tools.compute_bench2dex_action_stats import compute_stats

    normalizer = Bench2DexJointNormalizer((0.2,) * 52, (0.05,) * 52)
    action = torch.full((3, 52), 100.0)
    assert normalizer.normalize_action(action).max() > 1
    torch.testing.assert_close(normalizer.denormalize_action(normalizer.normalize_action(action)), action)
    src, _, _ = make_source(tmp_path)
    root = tmp_path / "cache"
    convert(src, root, "task", "text", 32)
    stats = compute_stats(root, split_val_ratio=0)
    assert min(stats["scale"]) >= 0.05
    # State is far from commanded action in this fixture: no state-union or tail expansion.
    assert stats["normalized_abs_max"]["state"] > 1
    stats["scale"][0] = 0
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(stats))
    with pytest.raises(ValueError, match="positive scales"):
        load_bench2dex_normalizer(path)


def test_action_only_reference_and_legacy_compatibility(tmp_path):
    import torch

    from cosmos_framework.utils.bench2dex_normalization import (
        LEGACY_CONTRACT,
        action_quantile_parameters,
        load_bench2dex_normalizer,
    )
    from tools.compute_bench2dex_action_stats import compute_stats
    from tools.prepare_bench2dex_normalized_run import prepare

    src, _, _ = make_source(tmp_path)
    root = tmp_path / "cache"
    convert(src, root, "task", "text", 32)
    stats = compute_stats(root, split_val_ratio=0)
    low, high = [np.asarray(stats["action"][k], np.float32) for k in ("q01", "q99")]
    np.testing.assert_array_equal(stats["offset"], (low + high) / 2)
    np.testing.assert_array_equal(stats["scale"], np.maximum((high - low) / 2, 0.05))
    # Constant channels must stay finite, invertible, and are not masked out.
    off, scale = action_quantile_parameters({"q01": [0.3] * 52, "q99": [0.3] * 52})
    np.testing.assert_array_equal(scale, np.full(52, 0.05, dtype=np.float32))
    path = tmp_path / "stats.json"
    stats["offset"][0] += 0.01
    path.write_text(json.dumps(stats))
    with pytest.raises(ValueError, match="differs from action"):
        load_bench2dex_normalizer(path)
    # Legacy checkpoints use stored union/tail parameters, not newly resolved q01/q99.
    stats["schema"], stats["method"] = LEGACY_CONTRACT
    stats["offset"] = [10.0] * 52
    stats["scale"] = [2.0] * 52
    path.write_text(json.dumps(stats))
    norm, _, _ = load_bench2dex_normalizer(path)
    torch.testing.assert_close(norm.normalize_action(torch.full((2, 52), 12.0)), torch.ones(2, 52))
    with pytest.raises(ValueError, match="New runs require action-only"):
        prepare(path, tmp_path / "new_run")
