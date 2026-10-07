"""Task 2 check: Dataset -> packed inputs -> SonataGeometryEncoder."""

import argparse
import json
from pathlib import Path

import torch

from cosmos_framework.data.pointflow_dataset import PointFlowWindowDataset, collate_pointflow_windows
from cosmos_framework.data.pointflow_window import PointFlowTiming
from cosmos_framework.model.generator.pointflow_geometry import SonataGeometryEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", type=int, default=3)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    model = SonataGeometryEncoder(args.checkpoint, stage=args.stage).to(args.device).eval()
    timing = PointFlowTiming()

    def sample(episode, start, budget):
        return PointFlowWindowDataset(args.data_root, [episode], timing=timing, max_points=budget, seed=args.seed)[
            start
        ]

    empty = sample("episode_0015_20260730_170501", 1120, 256)
    if not empty["metadata"]["empty_anchor"]:
        raise ValueError("Expected known empty-anchor test window")
    inputs = collate_pointflow_windows([empty, empty])["inputs"]
    output = model({k: v.to(args.device) for k, v in inputs.items()})
    assert output["cluster_features"].shape == (0, model.output_dim)
    assert output["cluster_offsets"].tolist() == [0, 0]
    output["cluster_features"].sum().backward()
    assert all(p.grad is not None and torch.all(p.grad == 0) for p in model.parameters() if p.requires_grad)
    model.zero_grad(set_to_none=True)
    report = {"stage": args.stage, "strict_load": "PASS", "empty_batch_backward": "PASS", "gpu_mixed_batch": "NOT_RUN"}
    if args.device == "cuda":
        a = sample("episode_0013_20260731_133649", 596, 512)
        b = sample("episode_0014_20260731_133743", 596, 256)
        inputs = collate_pointflow_windows([a, empty, b])["inputs"]
        inputs = {k: v.cuda() for k, v in inputs.items()}
        output = model(inputs)
        assert torch.equal(output["cluster_batch"][output["original_to_cluster"]], inputs["point_batch"])
        assert output["cluster_offsets"][0] == output["cluster_offsets"][1]
        assert torch.isfinite(output["cluster_features"]).all()
        loss = (output["cluster_features"] * torch.randn_like(output["cluster_features"])).mean()
        loss.backward()
        grad = model.backbone.embedding.stem.linear.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        report.update(
            gpu_mixed_batch="PASS",
            cluster_offsets=output["cluster_offsets"].tolist(),
            feature_shape=list(output["cluster_features"].shape),
        )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
