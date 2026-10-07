"""Task 6 real dense -> mixed collator -> clean/noised containers -> Cosmos packer."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.data.generator.action.pointflow_source import PointFlowSource
from cosmos_framework.data.generator.dataflow.collators import VFMListCollator
from cosmos_framework.data.generator.sequence_packing import SequencePlan, pack_input_sequence
from cosmos_framework.data.pointflow_batch import PointFlowNoised, build_pointflow_batch, pointflow_token_upper_bound
from cosmos_framework.data.pointflow_window import PointFlowTiming
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean, GenerationDataNoised


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    source = PointFlowSource(args.manifest, timing=PointFlowTiming(), max_points=256, seed=args.seed)
    name, row = next((name, row) for name, row in source.entries.items() if row is not None)
    first = source.load(name, np.arange(0, 65, 2), 30, row["video_size_wh"])
    second = source.load(name, np.arange(2, 67, 2), 30, row["video_size_wh"])
    # Explicit empty-anchor sample, while preserving the real timeline/metadata.
    from copy import deepcopy

    empty = deepcopy(first)
    for key in (
        "point_ids",
        "anchor_xyz",
        "anchor_uv",
        "normal",
        "color",
        "coord",
        "feat",
        "grid_coord",
        "original_to_voxel",
        "voxel_representatives",
    ):
        empty["inputs"][key] = empty["inputs"][key][:0]
    empty["targets"] = {key: value[:, :0] for key, value in empty["targets"].items()}
    samples = []
    for point in (first, None, empty, second):
        samples.append(
            {
                "video": torch.zeros(3, 33, 32, 32),
                "pointflow": point,
                "sequence_plan": SequencePlan(
                    has_text=True,
                    has_vision=True,
                    has_action=True,
                    has_point=point is not None and len(point["inputs"]["point_ids"]) > 0,
                ),
            }
        )
    batch = VFMListCollator().collate(samples)
    point = build_pointflow_batch(batch["pointflow"], batch_size=4)
    assert point.labeled.tolist() == [True, False, True, True]
    assert point.has_point.tolist() == [True, False, False, True]
    clean = GenerationDataClean(
        batch_size=4,
        is_image_batch=False,
        pointflow=point,
        x0_tokens_vision=[torch.zeros(1, 4, 9, 2, 2) for _ in samples],
        x0_tokens_action=[torch.zeros(32, 32) for _ in samples],
    )
    state = PointFlowNoised(
        torch.randn_like(point.displacement),
        torch.randn_like(point.displacement),
        torch.zeros_like(point.displacement),
        torch.ones(4) * 0.5,
    )
    state.validate(point)
    noised = GenerationDataNoised(4, torch.empty(0), torch.empty(0), torch.empty(0), pointflow=state)
    sequence = pack_input_sequence(
        batch["sequence_plan"],
        [[1, 2]] * 4,
        clean,
        torch.ones(4) * 0.5,
        {"eos_token_id": 3, "start_of_generation": 4, "end_of_generation": 5},
    )
    assert sequence.pointflow_data is point and sequence.point is None
    gpu = "not available; CPU contract passed"
    if torch.cuda.is_available():
        sequence.to_cuda()
        noised.pointflow.to_cuda()
        noised.pointflow.validate(sequence.pointflow_data)
        assert sequence.pointflow_data.valid.is_cuda
        assert sequence.pointflow_data.inputs["point_ids"].dtype == torch.int64
        assert sequence.pointflow_data.valid.dtype == torch.bool
        assert sequence.pointflow_data.displacement.dtype == torch.float32
        gpu = torch.cuda.get_device_name()
    report = {
        "task": 6,
        "status": "PASS",
        "gpu": gpu,
        "point_offsets": point.inputs["point_offsets"].tolist(),
        "voxel_offsets": point.inputs["voxel_offsets"].tolist(),
        "has_point": point.has_point.tolist(),
        "labeled": point.labeled.tolist(),
        "displacement_shape": list(point.displacement.shape),
        "reserved_point_tokens": [pointflow_token_upper_bound(s["pointflow"]) for s in samples],
        "base_sample_lens": sequence.sample_lens,
        "validation": "real dense windows, synthetic video/action latents and noised state; no learned point insertion or joint loss",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
