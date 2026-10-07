"""CPU regression tests for paired PointFlow media."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

spec = importlib.util.spec_from_file_location("pointflow_visualize", Path(__file__).with_name("pointflow_visualize.py"))
visualize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(visualize)


def test_perfect_prediction_media(tmp_path):
    t, n = 7, 4
    xyz = np.tile([0.0, 0.0, 1.0], (n, 1))
    gt = np.zeros((t, n, 3), dtype=np.float32)
    gt[:, 0, 0] = np.linspace(0, 0.06, t)
    uv = np.full((t, n, 2), 30.0)
    uv[:, 0, 0] += np.arange(t)
    valid = np.ones((t, n), bool)
    record = dict(
        xyz0=xyz,
        flow=gt,
        flow_gt=gt,
        valid=valid,
        moving=np.array([True, False, False, False]),
        frames_bgr=[np.zeros((64, 96, 3), np.uint8) for _ in range(t)],
        gt_uv=uv,
        pred_uv=uv,
        fps=15.0,
    )
    metrics = visualize.trajectory_metrics(gt, gt, valid, record["moving"])
    assert metrics["all_ade_mm"] == 0
    assert metrics["moving_count"] == t - 1
    paths, diagnostics = visualize.render_case(record, None, tmp_path, max_points=4)
    assert diagnostics["displayed_points"] == 4
    assert Image.open(paths["comparison"]).size == (288, 256)
    assert Path(paths["video"]).suffix == ".mp4"
    video, fps = visualize.read_clip(paths["video"])
    assert len(video) == t and fps == pytest.approx(15.0, abs=0.5)
    assert Path(paths["error_map"]).is_file()


def test_invalid_labels_excluded():
    gt = np.zeros((3, 2, 3))
    pred = gt.copy()
    pred[:, 0, 0] = 0.01
    pred[:, 1] = 999
    valid = np.array([[True, False]] * 3)
    metrics = visualize.trajectory_metrics(pred, gt, valid, np.array([True, False]))
    assert np.isclose(metrics["all_ade_mm"], 10)
    assert metrics["static_count"] == 0


def test_cosmos_raw_frame_alignment(tmp_path):
    import json
    from types import SimpleNamespace

    import cv2

    from cosmos_framework.callbacks.pointflow_eval import make_preview

    video = tmp_path / "head.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (96, 64))
    assert writer.isOpened()
    for i in range(5):
        writer.write(np.full((64, 96, 3), i * 40, np.uint8))
    writer.release()
    (tmp_path / "COMPLETE.json").write_text(
        json.dumps(
            dict(
                video=str(video),
                inference_width=96,
                inference_height=64,
                coordinate="camera_depthanythingv3",
                metric_scale=True,
            )
        )
    )
    np.save(tmp_path / "frame_indices.npy", np.array([0, 2, 4]))
    np.save(tmp_path / "uv_px.npy", np.full((3, 2, 2), [48.0, 32.0]))
    k = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
    np.save(tmp_path / "intrinsics.npy", np.tile(k, (3, 1, 1)))
    sample = dict(
        metadata=dict(source_path=str(tmp_path), raw_frame_ids=np.array([0, 2, 4]), timing=SimpleNamespace(fps=15)),
        inputs=dict(
            point_ids=np.array([0, 1]),
            anchor_xyz=np.array([[0.0, 0.0, 1.0]] * 2),
            anchor_uv=np.array([[48.0, 32.0]] * 2),
        ),
        targets=dict(displacement=np.zeros((2, 2, 3)), valid=np.ones((2, 2), bool)),
    )
    record = make_preview(sample, np.zeros((2, 2, 3)), 0.5, width=96)
    assert record["flow"].shape == (3, 2, 3)
    assert np.allclose(record["pred_uv"], record["gt_uv"])
    assert np.allclose([frame.mean() for frame in record["frames_bgr"]], [0, 80, 160], atol=3)


def test_error_curve_and_zero_baseline(tmp_path):
    import json

    from cosmos_framework.callbacks.pointflow_eval import save_error_curve

    gt = np.zeros((4, 2, 3))
    gt[:, 0, 0] = np.arange(4) * 0.01
    valid = np.ones((4, 2), bool)
    record = dict(flow=gt, flow_gt=gt, valid=valid, moving=np.array([True, False]), fps=15)
    path = save_error_curve(record, tmp_path)
    assert Path(path).is_file()
    curves = json.loads((tmp_path / "error_curve.json").read_text())
    assert curves["pred_moving_mm"] == [0, 0, 0]
    np.testing.assert_allclose(curves["Zero_moving_mm"], [10, 20, 30])


def test_three_stage_viewer(tmp_path):
    frames = [np.full((32, 32, 3), value, np.uint8) for value in (10, 20, 30)]
    video = visualize.write_h264(frames, tmp_path / "clip.mp4", 15)
    cases = [dict(stage=stage, video=video, label=f"{stage} <episode>") for stage in ("early", "middle", "late")]
    path = visualize.stage_viewer(cases, tmp_path / "stages.html")
    page = Path(path).read_text()
    assert page.count("data:video/mp4;base64,") == 3
    assert "autoplay loop muted playsinline" in page
    assert 'type="range" min="0" max="2" step="1"' in page
    assert 'id="stage-0" hidden' not in page
    assert 'id="stage-2" hidden' in page
    assert "&lt;episode&gt;" in page


def test_grouped_stage_viewer_grid(tmp_path):
    import json

    frames = [np.full((32, 32, 3), value, np.uint8) for value in (10, 20, 30)]
    video = visualize.write_h264(frames, tmp_path / "clip.mp4", 15)
    cases = [
        dict(stage=stage, kind=kind, video=video, label=f"{kind} {stage}")
        for stage in ("early", "middle", "late")
        for kind in ("conditional", "joint", "joint_dream")
    ]
    path = visualize.stage_viewer(cases, tmp_path / "stages.html")
    page = Path(path).read_text()
    assert page.count("data:video/mp4;base64,") == 9
    assert 'type="range"' not in page
    assert page.count("<tr>") == 4  # header + one row per kind
    saved = json.loads((tmp_path / "stages.json").read_text())
    assert [entry["kind"] for entry in saved][:3] == ["conditional", "joint", "joint_dream"]


def test_stitch_keeps_each_boundary_once(tmp_path):
    clips = []
    for window in range(4):
        frames = [np.full((64, 480, 3), window * 40 + step * 5, np.uint8) for step in range(3)]
        clips.append(visualize.write_h264(frames, tmp_path / f"window{window}.mp4", 15))
    path = visualize.stitch_windows(clips, tmp_path / "joined.mp4")
    frames, fps = visualize.read_clip(path)
    assert len(frames) == 9
    assert fps == pytest.approx(15.0, abs=0.5)
    # Windows 2..4 drop their duplicated first frame; window 1 keeps all three.
    # H.264 is lossy, so the flat colours only need to survive to within a code step.
    assert [int(frame[0, 0, 0]) for frame in frames] == pytest.approx([0, 5, 10, 45, 50, 85, 90, 125, 130], abs=4)


def test_video_ablation_transforms(tmp_path):
    import torch

    from cosmos_framework.callbacks.pointflow_eval import (
        PointFlowEvalCallback,
        freeze_video_first_frame_,
        zero_video_,
    )

    video = torch.arange(2 * 3 * 4 * 2 * 2, dtype=torch.float32).reshape(2, 3, 4, 2, 2)  # [B,C,T,H,W]
    batch = {"video": [video.clone()]}
    zero_video_(batch)
    assert torch.all(batch["video"][0] == 0)

    batch = {"video": video.clone()}
    freeze_video_first_frame_(batch)
    assert torch.equal(batch["video"], video[:, :, :1].expand_as(video))
    assert batch["video"].shape == video.shape

    # The precomputed latent cache must be dropped, else the model reads the
    # un-ablated latents and the pixel-space ablation is a no-op.
    batch = {"video": video.clone(), "vae_latent_cache": torch.zeros(1)}
    zero_video_(batch)
    assert "vae_latent_cache" not in batch
    batch = {"video": video.clone(), "vae_latent_cache": torch.zeros(1)}
    freeze_video_first_frame_(batch)
    assert "vae_latent_cache" not in batch

    # [C,T,H,W] unbatched form works too
    single = video[0].clone()
    batch = {"video": single.clone()}
    freeze_video_first_frame_(batch)
    assert torch.equal(batch["video"], single[:, :1].expand_as(single))

    cb = PointFlowEvalCallback(ablate_modes="action,video,first_frame")
    assert cb.ablate_modes == ["action", "video", "first_frame"]
    cb = PointFlowEvalCallback(ablate_action="true")
    assert cb.ablate_modes == ["action"]
    try:
        PointFlowEvalCallback(ablate_modes="bogus")
        raise AssertionError("unknown mode must raise")
    except ValueError:
        pass
