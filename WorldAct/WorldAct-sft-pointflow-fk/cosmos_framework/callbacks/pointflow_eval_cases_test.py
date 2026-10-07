import json
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from cosmos_framework.callbacks import pointflow_eval_cases as cases


def test_stage_windows_stay_in_one_episode():
    class Dataset:
        def get_shuffle_blocks(self):
            return [(0, 2), (2, 101)]

        def __getitem__(self, idx):
            return {
                "pointflow": {
                    "inputs": {"point_ids": np.arange(3)},
                    "targets": {"valid": np.ones((2, 3), bool)},
                    "metadata": {"episode": "episode2"},
                }
            }

    selected = cases.stage_windows(Dataset(), (0.2, 0.5, 0.8))
    assert [idx for idx, _ in selected] == [22, 52, 82]
    with pytest.raises(ValueError):
        cases.stage_windows(Dataset(), (0, 0.5, 1))


def test_rng_restored_on_exception():
    import random

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    expected = (random.random(), np.random.rand(), torch.rand(1))
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    with pytest.raises(RuntimeError):
        with cases.evaluation_rng(19):
            random.random()
            np.random.rand()
            torch.rand(1)
            raise RuntimeError("test")
    actual = (random.random(), np.random.rand(), torch.rand(1))
    assert expected[:2] == actual[:2]
    torch.testing.assert_close(expected[2], actual[2])


@pytest.mark.parametrize("config_kind", ["omegaconf", "cosmos_dict", "cosmos_omegaconf"])
def test_fixed_case_resume_identity(tmp_path, monkeypatch, config_kind):
    class Dataset:
        def __init__(self, split):
            self.split = split

        def __len__(self):
            return 8

        def __getitem__(self, i):
            return {
                "pointflow": {
                    "metadata": {
                        "episode": f"{self.split}_{i // 4}",
                        "start_frame": i,
                        "raw_frame_ids": np.arange(i, i + 3),
                        "source_path": "/source",
                    },
                    "inputs": {"point_ids": np.arange(3)},
                    "targets": {"valid": np.ones((2, 3), bool), "displacement": np.full((2, 3, 3), i * 0.01)},
                }
            }

    def instantiate(cfg):
        assert cfg["cfg_dropout_rate"] == 0
        assert cfg["iterable_shuffle"] is False
        return Dataset(cfg["split"])

    monkeypatch.setattr(cases, "instantiate", instantiate)
    monkeypatch.setattr(cases, "PackingDataLoader", lambda dataloader, **kw: [dataloader.dataset[0]])

    def loader(split):
        return {"dataloader": {"datasets": {"single": {"dataset": {"split": split}}}}}

    cfg = OmegaConf.create(
        {"job": {"path_local": str(tmp_path)}, "dataloader_train": loader("train"), "dataloader_val": loader("val")}
    )
    if config_kind != "omegaconf":
        from cosmos_framework.utils.config import Config, JobConfig

        monkeypatch.setenv("IMAGINAIRE_OUTPUT_ROOT", str(tmp_path))
        train, val = loader("train"), loader("val")
        if config_kind == "cosmos_omegaconf":
            train, val = OmegaConf.create(train), OmegaConf.create(val)
        cfg = Config(
            model={},
            optimizer={},
            scheduler={},
            dataloader_train=train,
            dataloader_val=val,
            job=JobConfig(project="test", group="pointflow", name="eval"),
        )
    rows, _ = cases.fixed_cases(cfg)
    assert [r["case_id"] for r in rows] == ["train_00", "train_01", "val_00", "val_01"]
    assert rows[0]["index"] == 7
    root = Path(cfg.job.path_local) / "pointflow_eval"
    root.mkdir(parents=True)
    (root / "fixed_cases.json").write_text(json.dumps(rows))
    resumed, _ = cases.fixed_cases(cfg)
    assert resumed == rows
    rows[0]["point_ids"][0] = 999
    (root / "fixed_cases.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="identity changed"):
        cases.fixed_cases(cfg)


def test_four_contiguous_windows_per_stage():
    class Dataset:
        def get_shuffle_blocks(self):
            return [(0, 500)]

        def __getitem__(self, idx):
            return {
                "pointflow": {
                    "metadata": {
                        "episode": "one",
                        "start_frame": idx * 2,
                        "raw_frame_ids": np.arange(idx * 2, idx * 2 + 65, 2),
                    },
                    "inputs": {"point_ids": np.arange(3)},
                    "targets": {"valid": np.ones((32, 3), bool)},
                }
            }

    selected = cases.stage_windows(Dataset(), (0.2, 0.5, 0.8), windows=4)
    assert len(selected) == 12
    for offset in (0, 4, 8):
        indices = [idx for idx, _ in selected[offset : offset + 4]]
        assert np.diff(indices).tolist() == [32, 32, 32]
        assert indices[0] > 0 and indices[-1] < 499


def test_stage_windows_cover_distinct_episodes():
    class Dataset:
        def get_shuffle_blocks(self):
            return [(0, 500), (500, 500), (1000, 500)]

        def __getitem__(self, idx):
            episode, frame = divmod(idx, 500)
            return {
                "pointflow": {
                    "metadata": {
                        "episode": str(episode),
                        "start_frame": frame * 2,
                        "raw_frame_ids": np.arange(frame * 2, frame * 2 + 65, 2),
                    },
                    "inputs": {"point_ids": np.arange(3)},
                    "targets": {"valid": np.ones((32, 3), bool)},
                }
            }

    selected = cases.stage_windows(Dataset(), (0.2, 0.5, 0.8), windows=1, episodes=3)
    assert [idx for idx, _ in selected] == [100, 250, 399, 600, 750, 899, 1100, 1250, 1399]
    with pytest.raises(ValueError, match="requested 4"):
        cases.stage_windows(Dataset(), (0.2, 0.5, 0.8), episodes=4)
