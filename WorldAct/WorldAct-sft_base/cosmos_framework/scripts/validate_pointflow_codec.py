"""Task 3 CUDA check: real geometry -> point tokens -> per-point velocity."""

import argparse
import json
from pathlib import Path

import torch

from cosmos_framework.data.pointflow_dataset import PointFlowWindowDataset, collate_pointflow_windows
from cosmos_framework.data.pointflow_window import PointFlowTiming
from cosmos_framework.model.generator.pointflow_codec import PointFlowCodec
from cosmos_framework.model.generator.pointflow_geometry import SonataGeometryEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This check requires a GPU; use pointflow_codec_test for CPU verification")
    torch.manual_seed(args.seed)
    timing = PointFlowTiming()
    samples = []
    for episode, start, budget in [
        ("episode_0013_20260731_133649", 596, 512),
        ("episode_0015_20260730_170501", 1120, 256),
        ("episode_0014_20260731_133743", 596, 256),
    ]:
        samples.append(
            PointFlowWindowDataset(args.data_root, [episode], timing=timing, max_points=budget, seed=args.seed)[start]
        )
    inputs = {k: v.cuda() for k, v in collate_pointflow_windows(samples)["inputs"].items()}
    encoder = SonataGeometryEncoder(args.checkpoint, stage=args.stage).cuda().eval()
    codec = PointFlowCodec(timing=timing, geometry_dim=encoder.output_dim).cuda()
    geometry = encoder(inputs)
    noise = torch.randn(timing.steps, len(inputs["point_ids"]), 3, device="cuda", requires_grad=True)
    sigma = torch.tensor([0.2, 0.5, 0.8], device="cuda")
    tokens = codec(geometry, noise, sigma)
    # Identity stand-in for the not-yet-integrated Cosmos point hidden states.
    velocity = codec.decode(geometry, inputs, tokens["noisy_tokens"], noise, sigma)
    assert velocity.shape == noise.shape and torch.isfinite(velocity).all()
    loss = velocity.square().mean() + tokens["anchor_tokens"].square().mean()
    loss.backward()
    for grad in (
        noise.grad,
        encoder.backbone.embedding.stem.linear.weight.grad,
        codec.motion_encoder[0].weight.grad,
        codec.decoder[-1].weight.grad,
    ):
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in encoder.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in codec.parameters())
    report = {
        "task": 3,
        "status": "PASS",
        "gpu": torch.cuda.get_device_name(),
        "point_offsets": inputs["point_offsets"].tolist(),
        "cluster_offsets": geometry["cluster_offsets"].tolist(),
        "anchor_shape": list(tokens["anchor_tokens"].shape),
        "noisy_token_shape": list(tokens["noisy_tokens"].shape),
        "velocity_shape": list(velocity.shape),
        "cosmos_attention": "NOT_IMPLEMENTED",
        "loss": "synthetic gradient probe, not training loss",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
