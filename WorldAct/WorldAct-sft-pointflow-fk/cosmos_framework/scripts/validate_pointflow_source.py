"""Task 5: real video/action/dense sample and resize validation, without model loading."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from cosmos_framework.data.generator.action.datasets.singlerighthand_raw_dataset import SingleRightHandRawDataset
from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in json.loads((args.cache_root / "manifest.json").read_text())["episodes"]:
        name = row["name"]
        dense = args.data_root / name
        source = None
        if (dense / "COMPLETE.json").exists():
            meta = json.loads((dense / "COMPLETE.json").read_text())
            dimensions = []
            for camera in ("head", "right_wrist"):
                cap = cv2.VideoCapture(str(args.raw_root / name / "videos" / f"{camera}.mp4"))
                try:
                    if not cap.isOpened():
                        raise ValueError(f"Cannot open {name}/{camera}")
                    dimensions.append((int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
                finally:
                    cap.release()
            (hw, hh), (ww, wh) = dimensions
            width = min(hw, ww)
            head_h, wrist_h = max(1, round(hh * width / hw)), max(1, round(wh * width / ww))
            sx, sy = width / meta["inference_width"], head_h / meta["inference_height"]
            source = {
                "path": str(dense.resolve()),
                "video_size_wh": [width, head_h + wrist_h],
                "uv_to_video": [[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2 + wrist_h]],
            }
        rows.append({"name": name, "pointflow_source": source})
    manifest = args.output / "mixed_manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "episodes": rows}, indent=2) + "\n")
    dataset = SingleRightHandRawDataset(
        root=str(args.raw_root),
        cache_root=str(args.cache_root),
        split="full",
        video_decoder="opencv",
        pointflow_manifest=str(manifest),
        pointflow_max_points=512,
        pointflow_seed=args.seed,
    )
    transform = ActionTransformPipeline(
        tokenizer_config=None,
        append_viewpoint_info=False,
        append_duration_fps_timestamps=False,
        append_resolution_info=False,
    )
    results = []
    start = 0
    seen = set()
    for episode, length in zip(dataset._episodes, dataset._valid_windows, strict=True):
        labeled = dataset._pointflow_source.entries[episode.name] is not None
        if length and labeled not in seen:
            sample = transform(dataset[start], "256")
            point = sample["pointflow"]
            record = {
                "episode": episode.name,
                "labeled": labeled,
                "video_shape": list(sample["video"].shape),
                "action_shape": list(sample["action"].shape),
                "raw_frame_ids": sample["raw_frame_ids"].tolist(),
            }
            if point is not None:
                np.testing.assert_array_equal(point["metadata"]["raw_frame_ids"], sample["raw_frame_ids"])
                assert point["targets"]["displacement"].shape == (32, len(point["inputs"]["point_ids"]), 3)
                uv = point["inputs"]["anchor_uv"]
                projected = np.c_[uv, np.ones(len(uv))] @ point["metadata"]["uv_to_video"].T
                w, h = point["metadata"]["video_size_wh"]
                assert np.all((projected >= -0.5) & (projected < np.array([w, h])))
                record.update(points=len(uv), uv_to_video=point["metadata"]["uv_to_video"].tolist())
            results.append(record)
            seen.add(labeled)
        start += length
        if len(seen) == 2:
            break
    assert True in seen, "No labeled sample validated"
    report = {
        "task": 5,
        "status": "PASS",
        "samples": results,
        "training_batch_and_model": "task 6 onward",
        "video_decoder": "opencv",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
