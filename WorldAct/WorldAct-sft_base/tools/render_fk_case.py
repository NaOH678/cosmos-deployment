#!/usr/bin/env python3
"""Draw a fixed-case FK eval's figures offline, from the ``prediction.npz`` it left behind.

The eval writes ``prediction.npz`` + ``metrics.json`` for every case on every
validation pass; the figures and the mp4 are derived from exactly those arrays.
Rendering them inside the trainer costs ~128 matplotlib frames per pass -- real
time against a single-GPU step, spent on steps nobody will open.  With
``FK_EVAL_FIGURES=false`` (the default) the trainer writes only the data, and this
draws the same figures later, for any step, without touching the GPU.

Output is byte-for-byte the layout the trainer produced: ``comparison.png``,
``error_curve.png`` and ``comparison.mp4``, via the same
``fk_visualize.render_case`` / ``write_video`` the callback calls.

    PYTHONPATH=. python tools/render_fk_case.py --case-dir <...>/fk_eval/step_0001000/val_00
    PYTHONPATH=. python tools/render_fk_case.py --eval-dir <...>/fk_eval --steps 1000,2000
    PYTHONPATH=. python tools/render_fk_case.py --eval-dir <...>/fk_eval --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

STAGES = ("comparison.png", "error_curve.png", "comparison.mp4")


def load_case(case_dir: Path) -> dict:
    """``prediction.npz`` + ``metrics.json`` -> the dict ``fk_visualize`` expects.

    ``anchor``/``target``/``prediction`` are the arrays the figures are drawn from;
    the rest is provenance the figures label themselves with.
    """
    data = np.load(case_dir / "prediction.npz", allow_pickle=True)

    def text(key, default=""):
        return str(data[key]) if key in data.files else default

    case = dict(
        prediction=np.asarray(data["prediction"], np.float64),
        target=np.asarray(data["target"], np.float64),
        anchor=np.asarray(data["anchor"], np.float64),
        valid=np.asarray(data["valid"], bool) if data["valid"] is not None else None,
        case_id=case_dir.name,
        episode=text("episode"),
        frame_ids=np.asarray(data["frame_ids"], np.int64) if "frame_ids" in data.files else [],
    )
    metrics_path = case_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"{case_dir} has no metrics.json -- the figures title themselves with its ADE, "
            "and recomputing it here would let the picture and the number drift apart"
        )
    case["metrics"] = json.loads(metrics_path.read_text())
    return case


def render(case_dir: Path, *, video: bool, force: bool):
    """Write the figures for one case; skips work that is already on disk."""
    from cosmos_framework.callbacks import fk_visualize

    wanted = STAGES if video else STAGES[:2]
    if not force and all((case_dir / name).is_file() for name in wanted):
        return "skipped (already drawn)"
    case = load_case(case_dir)
    title = f"{case['case_id']} ({case['episode']})" if case["episode"] else case["case_id"]
    fk_visualize.render_case(case, case_dir, title=title)
    if video:
        # Only when asked: this is the ~32-frame matplotlib loop, an order of
        # magnitude more work than the two stills.
        fk_visualize.write_video(case, case_dir)
    return "drew " + ", ".join(wanted)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case-dir", type=Path, help="one case directory holding prediction.npz")
    ap.add_argument("--eval-dir", type=Path, help="a fk_eval directory; renders its cases")
    ap.add_argument("--steps", help="comma-separated steps, e.g. 1000,9000 (default: all)")
    ap.add_argument("--cases", default="", help="comma-separated case names, e.g. val_00,val_01")
    ap.add_argument("--all", action="store_true", help="every step found")
    ap.add_argument("--video", action="store_true", help="also render comparison.mp4 (slow)")
    ap.add_argument("--force", action="store_true", help="redraw even if the files exist")
    args = ap.parse_args()

    if bool(args.case_dir) == bool(args.eval_dir):
        print("ERROR: pass exactly one of --case-dir or --eval-dir", file=sys.stderr)
        return 2

    if args.case_dir:
        print(f"{args.case_dir}: {render(args.case_dir, video=args.video, force=args.force)}")
        return 0

    steps = sorted(int(p.name.split("_")[1]) for p in args.eval_dir.glob("step_*") if p.is_dir())
    if args.steps:
        wanted = {int(x) for x in args.steps.split(",") if x.strip()}
        missing = wanted - set(steps)
        if missing:
            print(f"ERROR: no such step(s): {sorted(missing)}; have {steps}", file=sys.stderr)
            return 2
        steps = [s for s in steps if s in wanted]
    elif not args.all:
        # Default to the newest, which is what a "how is it doing" look wants.
        steps = steps[-1:]
        print(f"(rendering step {steps[0]} only; --all for every step, --steps for a list)")

    names = [x for x in args.cases.split(",") if x.strip()]
    drawn = 0
    for step in steps:
        for case_dir in sorted((args.eval_dir / f"step_{step:07d}").iterdir()):
            if not case_dir.is_dir() or (names and case_dir.name not in names):
                continue
            if not (case_dir / "prediction.npz").is_file():
                continue
            print(f"step {step} {case_dir.name}: {render(case_dir, video=args.video, force=args.force)}")
            drawn += 1
    print(f"\n{drawn} case(s) under {args.eval_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
