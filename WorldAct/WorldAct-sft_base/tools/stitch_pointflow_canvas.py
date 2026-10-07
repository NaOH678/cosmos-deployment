"""Backfill stitched stage videos for joint / dream-canvas PointFlow eval cases.

The online eval callback only started stitching the joint-rollout and
dream-canvas comparisons after step_0003000 of the labeled-29 run, so earlier
(and concurrently produced) step directories hold the per-window clips but no
``{stage}_stitched_joint.mp4`` / ``{stage}_stitched_joint_dream.mp4``.  This
script redoes the stitching offline (CPU only) and regenerates
``validation_stages.html`` with one row per canvas.

Usage:
    python tools/stitch_pointflow_canvas.py --eval-dir <run>/.../pointflow_eval [--overwrite]
"""

import argparse
import importlib.util
import json
from pathlib import Path

# Load the module directly: importing the cosmos_framework.callbacks package
# pulls the whole training stack, which an offline CPU-only tool does not need.
_spec = importlib.util.spec_from_file_location(
    "pointflow_visualize", Path(__file__).resolve().parent.parent / "cosmos_framework/callbacks/pointflow_visualize.py"
)
_visualize = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_visualize)
stage_viewer = _visualize.stage_viewer
stitch_windows = _visualize.stitch_windows

STAGES = ("early", "middle", "late")
# (kind, per-case comparison clip relative to the step directory)
KIND_CLIPS = {
    "joint": "{case}_joint/comparison.mp4",
    "joint_dream": "{case}_joint/dream_canvas/comparison.mp4",
}
KIND_LABELS = {
    "conditional": "conditional | clean GT video/action",
    "joint": "joint rollout | first frame + state action clean",
    "joint_dream": "joint rollout on the DREAMED video",
}


def _val_cases_by_stage(eval_dir: Path):
    cases_path = eval_dir / "fixed_cases_stages_4windows.json"
    cases = json.loads(cases_path.read_text())
    by_stage = {stage: [] for stage in STAGES}
    for case in cases:
        if case.get("split") == "val" and case.get("stage") in by_stage:
            by_stage[case["stage"]].append(case)
    return by_stage


def _stitch_step(step_dir: Path, by_stage, overwrite: bool):
    entries = []
    stitched_any = False
    missing = []
    for stage in STAGES:
        cases = by_stage[stage]
        conditional = step_dir / f"{stage}_stitched.mp4"
        if conditional.is_file():
            entries.append(
                dict(stage=stage, kind="conditional", video=str(conditional), label=f"{stage} | {KIND_LABELS['conditional']}")
            )
        for kind, pattern in KIND_CLIPS.items():
            output = step_dir / f"{stage}_stitched_{kind}.mp4"
            clips = [step_dir / pattern.format(case=case["case_id"]) for case in cases]
            absent = [str(clip) for clip in clips if not clip.is_file()]
            if absent:
                missing.extend(absent)
                continue
            if overwrite or not output.is_file():
                stitch_windows([str(clip) for clip in clips], output)
                stitched_any = True
            entries.append(
                dict(
                    stage=stage,
                    kind=kind,
                    video=str(output),
                    label=f"{stage} | {KIND_LABELS[kind]} | "
                    f"frame {cases[0]['start_frame']}-{cases[-1]['start_frame']} | 4 windows",
                )
            )
    viewer = None
    kinds_done = {entry["kind"] for entry in entries}
    if entries and not missing and (stitched_any or overwrite or kinds_done != {"conditional"}):
        html_path = step_dir / "validation_stages.html"
        existing_kinds = set()
        if html_path.is_file() and not stitched_any and not overwrite:
            try:
                existing_kinds = {
                    entry.get("kind", "conditional")
                    for entry in json.loads(html_path.with_suffix(".json").read_text())
                }
            except (OSError, json.JSONDecodeError):
                existing_kinds = set()
        if stitched_any or overwrite or existing_kinds != kinds_done:
            viewer = stage_viewer(entries, html_path)
    return stitched_any, viewer, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval-dir", required=True, type=Path, help="pointflow_eval directory of a run")
    parser.add_argument("--overwrite", action="store_true", help="re-stitch even when outputs exist")
    args = parser.parse_args()

    eval_dir = args.eval_dir
    by_stage = _val_cases_by_stage(eval_dir)
    steps = sorted(eval_dir.glob("step_[0-9]*"))
    print(f"{len(steps)} step directories under {eval_dir}")
    done = skipped = 0
    for step_dir in steps:
        if not step_dir.is_dir():
            continue
        stitched, viewer, missing = _stitch_step(step_dir, by_stage, args.overwrite)
        if missing:
            print(f"[skip] {step_dir.name}: missing {len(missing)} clip(s), e.g. {missing[0]}")
            skipped += 1
            continue
        mark = "stitched" if stitched else "ok"
        print(f"[{mark}] {step_dir.name}" + (f" viewer={viewer}" if viewer else ""))
        done += 1
    print(f"done: {done} steps complete, {skipped} skipped (incomplete cases)")


if __name__ == "__main__":
    main()
