"""CPU oracle for pair-specific THW/THWZ RoPE; not a training attention backend.

Run with the shared venv and LD_LIBRARY_PATH=''. No model weights or run files
are modified. --scan samples anchor geometry only, for exploratory calibration.
"""

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch

DEPTH_PAIRS = (49, 50, 52, 53, 55, 56, 58, 59)


def phases(positions, inverse_depth, *, depth_enabled=True):
    """positions [N,3], inverse_depth [N], split-half rotary convention.

    z = (1/Z - 1 m^-1) / (1 m^-1). Frequencies are a provisional sweep,
    not a measured optimum. The common offset cancels in geometric pairs.
    """
    channels = torch.arange(64, device=positions.device)
    axes = torch.zeros(64, dtype=torch.long, device=positions.device)
    axes[1:60:3] = 1
    axes[2:60:3] = 2
    frequencies = 1e8 ** (-channels.to(positions.dtype) / 64)
    phase3 = positions[:, axes] * frequencies
    phase4 = phase3.clone()
    if depth_enabled:
        depth_frequencies = torch.logspace(
            math.log10(0.125),
            math.log10(4.0),
            8,
            dtype=positions.dtype,
            device=positions.device,
        )
        phase4[:, DEPTH_PAIRS] = (inverse_depth[:, None] - 1.0) * depth_frequencies
    return phase3, phase4, axes


def rotate(value, phase):
    """value [heads,N,128]; pair c joins dimensions c and c+64."""
    first, second = value.chunk(2, dim=-1)
    cos, sin = phase.cos()[None], phase.sin()[None]
    return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)


def scores(q, k, phase3, phase4):
    # Explicit repeat is for the CPU oracle, not the intended GQA GPU backend.
    k = k.repeat_interleave(q.shape[0] // k.shape[0], dim=0)
    return (
        rotate(q, phase3) @ rotate(k, phase3).transpose(-1, -2) / math.sqrt(128),
        rotate(q, phase4) @ rotate(k, phase4).transpose(-1, -2) / math.sqrt(128),
    )


def dense_reference(q, k, v, phase3, phase4, geometry, allowed):
    s3, s4 = scores(q, k, phase3, phase4)
    pair = geometry[:, None] & geometry[None, :]
    score = torch.where(pair[None], s4, s3).masked_fill(~allowed[None], -torch.inf)
    values = v.repeat_interleave(q.shape[0] // v.shape[0], dim=0)
    return score.softmax(-1) @ values, score


def partitioned_reference(q, k, v, phase3, phase4, geometry, allowed):
    """Exact two-part softmax merger, handling empty groups explicitly.

    This oracle still materializes scores. A production implementation must
    compute the groups directly with fused attention and validate its backward.
    """
    s3, s4 = scores(q, k, phase3, phase4)
    values = v.repeat_interleave(q.shape[0] // v.shape[0], dim=0)
    rows = []
    for i in range(q.shape[1]):
        groups = [allowed[i] & ~geometry, allowed[i] & geometry] if geometry[i] else [allowed[i]]
        outputs, lse = [], []
        for group in groups:
            if not group.any():
                continue
            local4 = bool(geometry[i] and geometry[group].all())
            score = (s4 if local4 else s3)[:, i, group]
            outputs.append((score.softmax(-1)[..., None] * values[:, group]).sum(1))
            lse.append(score.logsumexp(-1))
        weight = torch.stack(lse, dim=-1).softmax(-1)
        rows.append((torch.stack(outputs, dim=-2) * weight[..., None]).sum(-2))
    return torch.stack(rows, dim=1)


def check():
    torch.manual_seed(42)
    n = 19
    q = torch.randn(4, n, 128, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, n, 128, dtype=torch.float64, requires_grad=True)
    v = torch.randn(2, n, 128, dtype=torch.float64, requires_grad=True)
    positions = torch.rand(n, 3, dtype=torch.float64) * 10
    rho = torch.linspace(0.7, 2.0, n, dtype=torch.float64)
    phase3, phase4, axes = phases(positions, rho)
    axes4 = axes.clone()
    axes4[list(DEPTH_PAIRS)] = 3
    assert axes4.bincount().tolist() == [24, 16, 16, 8]
    assert torch.equal(phase3[:, axes == 0], phase4[:, axes == 0])
    # Two ragged samples, each with causal text then a full generation segment.
    allowed = torch.zeros(n, n, dtype=torch.bool)
    for start, end in [(0, 10), (10, 19)]:
        allowed[start:end, start:end] = True
        allowed[start : start + 2, start:end] = False
        allowed[start : start + 2, start : start + 2] = torch.ones(2, 2, dtype=torch.bool).tril()
    # Both point and FK use True; neither text, video nor action does.
    geometry = torch.tensor([False] * n)
    geometry[[4, 6, 9, 13, 14, 18]] = True
    dense, score = dense_reference(q, k, v, phase3, phase4, geometry, allowed)
    grouped = partitioned_reference(q, k, v, phase3, phase4, geometry, allowed)
    torch.testing.assert_close(dense, grouped, atol=1e-12, rtol=1e-12)
    probe = torch.randn_like(dense)
    grad1 = torch.autograd.grad((dense * probe).sum(), (q, k, v), retain_graph=True)
    grad2 = torch.autograd.grad((grouped * probe).sum(), (q, k, v), retain_graph=True)
    for a, b in zip(grad1, grad2, strict=True):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
    changed = rho.clone()
    changed[4] += 0.3
    _, phase_changed, _ = phases(positions, changed)
    _, score_changed = dense_reference(q, k, v, phase3, phase_changed, geometry, allowed)
    outside = ~(geometry[:, None] & geometry[None, :]) & allowed
    assert torch.equal(score[:, outside], score_changed[:, outside])
    assert (score[:, 4, 6] - score_changed[:, 4, 6]).abs().max() > 1e-5
    # A common inverse-depth shift cancels in GxG, and never enters other scores.
    _, shifted, _ = phases(positions, rho + 3.0)
    shifted_out, _ = dense_reference(q, k, v, phase3, shifted, geometry, allowed)
    torch.testing.assert_close(dense, shifted_out, atol=1e-12, rtol=1e-12)
    # Disabling local rotation restores baseline exactly; no geometry also does.
    baseline = scores(q, k, phase3, phase3)[0].masked_fill(~allowed[None], -torch.inf)
    baseline = baseline.softmax(-1) @ v.repeat_interleave(2, dim=0)
    for mask, p4 in [(geometry, phase3), (torch.zeros_like(geometry), phase4)]:
        output = partitioned_reference(q, k, v, phase3, p4, mask, allowed)
        torch.testing.assert_close(output, baseline, atol=1e-12, rtol=1e-12)
    # No non-geometry keys: the merger must avoid softmax(-inf,-inf).
    all_geo = torch.ones(n, dtype=torch.bool)
    full = torch.ones(n, n, dtype=torch.bool)
    a, _ = dense_reference(q, k, v, phase3, phase4, all_geo, full)
    b = partitioned_reference(q, k, v, phase3, phase4, all_geo, full)
    torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
    # An unrelated sample must not affect the first sample's output.
    altered_v = v.detach().clone()
    altered_v[:, 10:] += 100
    separated, _ = dense_reference(q, k, altered_v, phase3, phase4, geometry, allowed)
    torch.testing.assert_close(dense[:, :10], separated[:, :10], atol=1e-12, rtol=1e-12)
    return {
        "axis_counts": axes4.bincount().tolist(),
        "depth_pair_indices": DEPTH_PAIRS,
        "output_max_error": (dense - grouped).abs().max().item(),
        "gradient_max_error": max((a - b).abs().max().item() for a, b in zip(grad1, grad2, strict=True)),
        "checks": [
            "GQA",
            "ragged_sample_isolation",
            "causal_text_mask",
            "empty_groups",
            "baseline_recovery",
            "time_channels_unchanged",
            "local_depth_sensitivity",
            "non_geometry_logits_unchanged",
            "common_depth_shift_invariance",
            "backward_equivalence",
        ],
        "gpu_benchmark": "not run; prototype is a CPU oracle, not a fused implementation",
    }


def scan(cache_root, fk_root, fk_transform):
    """24 deterministic windows from 8 episodes; descriptive, NOT train-only fitting."""
    spec = importlib.util.spec_from_file_location("fk_transform", fk_transform)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    point, fk, entries = [], [], []
    for path in sorted(cache_root.glob("episode_*.npz"))[:8]:
        annotation = fk_root / path.stem / "annotations/wuji_fk21.npz"
        with np.load(annotation, allow_pickle=False) as data:
            assert str(data["coordinate_frame"]) == "Link_Base"
            assert str(data["units"]) == "metre"
            assert bool(data["side_is_observed"][1])
            assert "qpos_was_clipped" not in data or not bool(data["qpos_was_clipped"])
            positions = data["positions"][:, 1]
        with np.load(path, allow_pickle=False) as data:
            windows = sorted(int(k.split("/")[0]) for k in data.files if k.endswith("/anchor_xyz"))
            for index in [0, len(windows) // 2, len(windows) - 1]:
                window = str(windows[index])
                z_point = data[f"{window}/anchor_xyz"][:, 2]
                frame = int(data[f"{window}/raw_frame_ids"][0])
                z_fk = module.base_to_camera(positions[frame])[:, 2]
                if not all(np.isfinite(z).all() and (z > 0).all() for z in [z_point, z_fk]):
                    raise ValueError(f"Invalid anchor depth in {path.stem}/{window}")
                point.extend(z_point.tolist())
                fk.extend(z_fk.tolist())
                entries.append({"episode": path.stem, "window": int(window), "source_frame": frame})
    quantiles = [0, 0.01, 0.5, 0.99, 1]
    return {
        "scope": "24 anchor windows from first 8 sorted episodes, not split-filtered; no fitted training parameters",
        "quantiles": quantiles,
        "point_depth_m": np.quantile(point, quantiles).tolist(),
        "fk_depth_m": np.quantile(fk, quantiles).tolist(),
        "point_inverse_depth_per_m": np.quantile(1 / np.array(point), quantiles).tolist(),
        "fk_inverse_depth_per_m": np.quantile(1 / np.array(fk), quantiles).tolist(),
        "windows": entries,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scan", action="store_true")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("/data/shichaojian/datasets/singlerighthand-sandwich-100-cosmos-cache/pointflow_windows"),
    )
    parser.add_argument("--fk-root", type=Path, default=Path("/data/shichaojian/raw_data/sandwich_fk21"))
    parser.add_argument(
        "--fk-transform",
        type=Path,
        default=Path("/mnt/afs/WorldAct-cosmos3-edge-droid-sft_base/cosmos_framework/data/fk_camera_extrinsic.py"),
    )
    args = parser.parse_args()
    result = {"correctness": check()}
    if args.scan:
        result["anchor_depth_scan"] = scan(args.cache_root, args.fk_root, args.fk_transform)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
