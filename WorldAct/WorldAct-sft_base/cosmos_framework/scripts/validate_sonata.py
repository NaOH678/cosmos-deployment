"""Validate a local Sonata checkpoint without launching Cosmos training."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.backward and args.device != "cuda":
        parser.error("--backward requires --device cuda; CPU mode checks loading and transforms only")

    import numpy as np
    import torch

    from cosmos_framework.auxiliary import sonata

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is visible; GPU validation was not performed")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = sonata.model.PointTransformerV3(**checkpoint["config"])
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    # Synthetic plane: meaningful unit normals and RGB, not a production dataset adapter.
    side = 32
    x, y = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    coord = np.stack((x.ravel() * 0.03, y.ravel() * 0.03, np.ones(side * side)), axis=1).astype(np.float32)
    color = np.random.randint(0, 256, (len(coord), 3)).astype(np.float32)
    normal = np.tile(np.array([[0, 0, -1]], dtype=np.float32), (len(coord), 1))
    point = sonata.transform.default()({"coord": coord.copy(), "color": color, "normal": normal})
    assert point["feat"].shape[1] == checkpoint["config"]["in_channels"] == 9
    torch.testing.assert_close(point["feat"][:, :3], point["coord"])
    assert torch.isfinite(point["feat"]).all()
    assert ((point["feat"][:, 3:6] >= 0) & (point["feat"][:, 3:6] <= 1)).all()
    assert point["inverse"].shape == (len(coord),)

    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "torch": torch.__version__,
        "seed": args.seed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint_tensors": len(checkpoint["state_dict"]),
        "missing_keys": incompatible.missing_keys,
        "unexpected_keys": incompatible.unexpected_keys,
        "strict_load": "PASS",
        "preprocessing": "PASS",
        "input_feature_shape": list(point["feat"].shape),
        "gpu_forward": "NOT_RUN",
        "gpu_backward": "NOT_RUN",
    }
    if args.device == "cuda":
        model = model.cuda()
        point = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in point.items()}
        with torch.set_grad_enabled(args.backward):
            output = model(point)
            assert torch.isfinite(output.feat).all()
            # Compose all pooling inverses without destroying the upstream hierarchy.
            mapping = torch.arange(len(output.feat), device="cuda")
            level = output
            while "pooling_parent" in level:
                mapping = mapping[level.pooling_inverse]
                level = level.pooling_parent
            raw_mapping = mapping[point["inverse"]]
            assert raw_mapping.shape == (len(coord),)
            result["output_feature_shape"] = list(output.feat.shape)
            result["original_to_cluster_shape"] = list(raw_mapping.shape)
            result["gpu_forward"] = "PASS"
            if args.backward:
                output.feat.float().square().mean().backward()
                grad = model.embedding.stem.linear.weight.grad
                assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                result["gpu_backward"] = "PASS"
        torch.cuda.synchronize()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
