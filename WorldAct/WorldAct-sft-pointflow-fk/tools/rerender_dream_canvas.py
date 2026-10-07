"""Offline re-render of dream-canvas overlays with the corrected projection.

The dream render in pointflow_eval.make_preview used to scale tracker->canvas
coordinates by the PADDED canvas (metadata["video_size_wh"], 736 rows) while the
model's true canvas is the decoded dream frame itself (704 rows after
_remove_padding_from_latent floors 716 content rows to 44 latent rows). This
script replays the render on already-saved artifacts (dream.mp4 + the joint
prediction npz + the dataset sample) so historical eval steps can be fixed
without re-running the model.

Outputs are written as comparison_fixed.mp4 / comparison_fixed.png /
error_map_fixed.png next to the originals, which are left untouched. Cases that
already have comparison_fixed.mp4 are skipped, so reruns are cheap.

Example:
  PYTHONPATH=. python tools/rerender_dream_canvas.py \
    --run /data/shichaojian/runs/perpoint_strat500_framescale_20260930
"""

import argparse
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

RUN_CONFIG = "cosmos3_action/action_sft/action_policy_singlerighthand_edge"


def load_dataset(config_yaml: Path, split: str):
    from omegaconf import OmegaConf

    from cosmos_framework.utils.lazy_config import instantiate

    cfg = OmegaConf.load(config_yaml)
    loader = OmegaConf.to_container(getattr(cfg, f"dataloader_{split}"), resolve=True)
    ds_cfg = dict(next(iter(loader["dataloader"]["datasets"].values()))["dataset"])
    ds_cfg.update(iterable_shuffle=False, cfg_dropout_rate=0.0, use_image_augmentation=False)
    return instantiate(ds_cfg)


def read_video(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--max-points", type=int, default=500)
    parser.add_argument("--cases", nargs="*", default=None, help="case ids like val_00; default all")
    args = parser.parse_args()

    from cosmos_framework.callbacks.pointflow_eval import make_preview
    from cosmos_framework.callbacks.pointflow_visualize import render_case

    run = args.run
    eval_root = run / RUN_CONFIG / "pointflow_eval"
    fixed = json.loads((eval_root / "fixed_cases_stages_4windows.json").read_text())
    case_rows = {r["case_id"]: r for r in fixed}
    datasets: dict[str, object] = {}

    step_dirs = sorted(eval_root.glob("step_*/"))
    done, skipped, failed = 0, 0, 0
    for step_dir in step_dirs:
        for joint_dir in sorted(step_dir.glob("*_joint")):
            dream_dir = joint_dir / "dream_canvas"
            dream_mp4 = dream_dir / "dream.mp4"
            target = dream_dir / "comparison_fixed.mp4"
            if not dream_mp4.is_file():
                continue
            case_id = joint_dir.name[: -len("_joint")]
            if args.cases and case_id not in args.cases:
                continue
            if target.exists():
                skipped += 1
                continue
            tag = f"{step_dir.name}/{joint_dir.name}"
            try:
                row = case_rows[case_id]
                split = row["split"]
                if split not in datasets:
                    datasets[split] = load_dataset(run / RUN_CONFIG / "config.yaml", split)
                sample = datasets[split][int(row["index"])]
                pointflow = sample["pointflow"]
                if isinstance(pointflow, list):
                    pointflow = pointflow[0]
                prediction = np.load(joint_dir / "prediction.npz")
                frames = read_video(dream_mp4)
                record = make_preview(
                    pointflow,
                    prediction["flow"][1:],  # saved flow prepends the zero anchor row; make_preview re-adds it
                    1.0,
                    prediction_kind=f"joint_dream_canvas_rerender_{step_dir.name}",
                    frames_bgr_override=frames,
                )
                with tempfile.TemporaryDirectory(dir=dream_dir) as tmp:
                    paths, _ = render_case(record, None, tmp, max_points=args.max_points, make_video=True)
                    video = Path(paths["video"])
                    video.replace(target)
                    for name in ("comparison.png", "error_map.png"):
                        extra = video.parent / name
                        if extra.is_file():
                            extra.replace(dream_dir / f"{extra.stem}_fixed.png")
                done += 1
                print(f"[done] {tag}", flush=True)
            except Exception as exc:  # one bad case must not stop the batch
                failed += 1
                print(f"[FAIL] {tag}: {exc}", flush=True)
    print(f"finished: {done} rendered, {skipped} skipped (already fixed), {failed} failed")


if __name__ == "__main__":
    main()
