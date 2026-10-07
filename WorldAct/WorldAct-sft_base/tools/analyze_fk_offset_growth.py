#!/usr/bin/env python3
"""Does the FK error accumulate as a per-step bias, or as a random walk?

``fk_sampler_probe.py`` established that the sampled trajectory's error is
94-98% whole-hand translation (``off%``), that the hand *shape* is intact, and
that the one-step fit is nearly exact (cosine +0.997, magnitude ratio 0.99).
It also showed the compression is flat in sigma and that swapping sampler, step
count and shift moves nothing.  So the fit is not the problem and the sampler is
not the problem -- what is left is *accumulation*: a small per-step error that
the 32-step rollout integrates into a whole-hand drift.

That leaves one question this tool answers, and it is a question about shape,
not size: does the cumulative offset grow **linearly** in t (a systematic bias,
every step adds the same wrong vector -- fixable by removing the bias) or like
**sqrt(t)** (independent per-step noise, the errors cancel in expectation and
only their variance accumulates -- not fixable by a per-step correction)?

The discriminator is the per-step increment.  Write the displacement as
``d[t] = x[t] - x[0]``, so the per-step motion is ``v[t] = d[t] - d[t-1]``
(with ``d[-1] = 0``).  Then

    offset[31]  ~=  sum_t v_bias[t]        (linear   => v_bias roughly constant)
    offset[31]  ~=  sqrt(sum_t v_var[t])   (random   => v_bias ~ 0, v_var drives it)

so it is enough to print ``v_bias[t]`` next to the cumulative curve: a flat,
non-zero ``v_bias`` is the linear case, a near-zero and sign-fluctuating one is
the random-walk case.  The fit R^2 of ``offset`` against ``t`` and against
``sqrt(t)`` is printed as a second, cruder read on the same thing.

Reads only ``prediction.npz`` -- no GPU, no checkpoint, no model.

    PYTHONPATH=. python tools/analyze_fk_offset_growth.py
    PYTHONPATH=. python tools/analyze_fk_offset_growth.py --run fk-singlerighthand-edge-7 --step 4000
    PYTHONPATH=. python tools/analyze_fk_offset_growth.py --cases val_00 val_01
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

REL = "cosmos3_action/action_sft/action_policy_fk_singlerighthand_edge"
RUNS_ROOT = "/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/runs/cosmos"
DEFAULT_CASES = ("train_00", "train_01", "val_00", "val_01")
LAST_RUN_POINTER = Path("/tmp/fk_last_run")


def resolve_run(value: str | None) -> Path:
    """A run name, a full path, or nothing -- in which case the last started run."""
    if not value:
        if LAST_RUN_POINTER.is_file():
            value = LAST_RUN_POINTER.read_text().strip()
        else:
            value = str(Path(RUNS_ROOT) / "fk-singlerighthand-edge-8")
    path = Path(value)
    return path if path.is_absolute() else Path(RUNS_ROOT) / path


def resolve_step(eval_dir: Path, wanted: int | None) -> str:
    """The ``step_%07d`` directory to read, newest by default.

    Newest by *name* rather than by mtime: the eval writes its cases in parallel,
    so the most recently written ``metrics.json`` is not the highest step.
    """
    steps = sorted(int(p.name[len("step_"):]) for p in eval_dir.glob("step_*") if p.is_dir())
    if not steps:
        raise SystemExit(f"no step_* directories under {eval_dir}")
    if wanted is None:
        return f"step_{steps[-1]:07d}"
    match = [s for s in steps if s == wanted]
    if not match:
        raise SystemExit(f"step {wanted} not evaluated; have {steps}")
    return f"step_{wanted:07d}"


def load_case(eval_dir: Path, step: str, case: str) -> dict | None:
    path = eval_dir / step / case / "prediction.npz"
    if not path.is_file():
        return None
    data = np.load(path, allow_pickle=True)
    return {
        "pred": data["prediction"].astype(np.float64),
        "gt": data["target"].astype(np.float64),
        "valid": data["valid"],
        "names": [str(n) for n in data["keypoint_names"]],
    }


def per_step(case: dict) -> dict:
    """Per-timestep error decomposed into common-mode offset and scatter, in mm.

    ``offset`` is the length of the *mean* keypoint error (how far the hand as a
    whole is displaced); ``scatter`` is the mean residual after removing it (how
    wrong the shape is).  Their sum bounds ``ade`` from above; the probe's
    ``off%`` is ``offset / ade``.
    """
    pred, gt, valid = case["pred"] * 1000.0, case["gt"] * 1000.0, case["valid"]
    err = pred - gt                                             # [T,K,3] mm
    n = valid.sum(-1, keepdims=True).clip(min=1).astype(np.float64)
    mean_err = np.where(valid[..., None], err, 0.0).sum(1, keepdims=True) / n[..., None]
    resid = np.where(valid[..., None], err - mean_err, 0.0)

    ade = (np.linalg.norm(err, axis=-1) * valid).sum(-1) / n[:, 0]
    offset = np.linalg.norm(mean_err, axis=-1)[:, 0]
    scatter = (np.linalg.norm(resid, axis=-1) * valid).sum(-1) / n[:, 0]

    # Per-step motion: d[t] - d[t-1], with d[-1] = 0 because d[0] is already
    # measured from the anchor.
    v_pred = np.diff(pred, axis=0, prepend=np.zeros_like(pred[:1]))
    v_gt = np.diff(gt, axis=0, prepend=np.zeros_like(gt[:1]))
    v_err = v_pred - v_gt
    v_mean = np.where(valid[..., None], v_err, 0.0).sum(1, keepdims=True) / n[..., None]
    v_resid = np.where(valid[..., None], v_err - v_mean, 0.0)
    return {
        "ade": ade,
        "offset": offset,
        "scatter": scatter,
        "v_bias": np.linalg.norm(v_mean, axis=-1)[:, 0],
        "v_scatter": (np.linalg.norm(v_resid, axis=-1) * valid).sum(-1) / n[:, 0],
        "v_mean": v_mean[:, 0],          # [T,3] the per-step common-mode error *vector*
        "mean_err": mean_err[:, 0],      # [T,3] hand-mean error (common mode)
        "err_final": err[-1],            # [K,3] per-keypoint error at the last step
        "valid": valid,
        "names": case["names"],
    }


def fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares slope/intercept, plus R^2, for y against x."""
    if x.size < 2 or np.ptp(x) == 0:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(((y - predicted) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return float(slope), (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def summarise(curve: dict) -> dict:
    """The numbers that answer "which shape is this?", for one curve or a pool."""
    off = curve["offset"]
    t = np.arange(off.size, dtype=float)
    final = float(off[-1])
    slope, r2_lin = fit(t, off)
    _, r2_sqrt = fit(np.sqrt(t + 1.0), off)
    v_mean = curve["v_mean"]
    walked = float(np.linalg.norm(v_mean.sum(0)))
    travelled = float(np.linalg.norm(v_mean, axis=-1).sum())
    steps = np.diff(off)
    return {
        "final": final,
        "slope": slope,
        "r2_lin": r2_lin,
        "r2_sqrt": r2_sqrt,
        "walked": walked,
        "travelled": travelled,
        "coherence": walked / travelled if travelled > 0 else float("nan"),
        "reach90": int(np.argmax(off >= 0.9 * final)) + 1,
        "maxjump": float(np.abs(steps).max()) if steps.size else float("nan"),
        "jump_at": int(np.abs(steps).argmax()) + 2 if steps.size else 0,
        "off_pct": 100.0 * final / float(curve["ade"][-1]) if curve["ade"][-1] > 0 else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", help="run name or path (default: the last started run)")
    parser.add_argument("--step", type=int, help="eval step (default: newest)")
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--pool", action="store_true",
                        help="also print the average over --cases.  The cases do not share a "
                             "shape, so the average is a curve no case has; it is a summary, "
                             "never the headline.")
    parser.add_argument("--json", help="also write the pooled curve here as JSON")
    args = parser.parse_args()

    run = resolve_run(args.run)
    eval_dir = run / REL / "fk_eval"
    if not eval_dir.is_dir():
        raise SystemExit(f"no fk_eval under {run}")
    step = resolve_step(eval_dir, args.step)

    loaded = [(c, load_case(eval_dir, step, c)) for c in args.cases]
    loaded = [(c, d) for c, d in loaded if d is not None]
    if not loaded:
        raise SystemExit(f"no prediction.npz for {args.cases} under {eval_dir / step}")

    print(f"run  : {run}")
    print(f"step : {step}")
    print(f"cases: {', '.join(c for c, _ in loaded)}   (units mm)")
    print()

    curves = {c: per_step(d) for c, d in loaded}
    names = curves[loaded[0][0]]["names"]

    # Per-case first, always -- and the summary comes before any table.
    #
    # The cases do not share a shape.  On one checkpoint val_00's offset grows
    # across all 32 steps while val_01's is flat past t~12, so an average over
    # them is a curve no case has: it showed a saturation at t=12 that val_00
    # never had, and reading it as "the damage is done in the first 12 steps"
    # was exactly that artefact.  Pooling is still available (--pool) but it is
    # labelled and never the first thing printed.
    print("PER-CASE SUMMARY")
    print(f"   {'case':>10s}  {'offset[32]':>10s}  {'90% at':>6s}  {'R2(t)':>6s}  {'R2(sqrt)':>9s}"
          f"  {'coherence':>9s}  {'off%':>5s}")
    for c, _ in loaded:
        s = summarise(curves[c])
        print(f"   {c:>10s}  {s['final']:10.2f}  {s['reach90']:6d}  {s['r2_lin']:6.3f}  {s['r2_sqrt']:9.3f}"
              f"  {s['coherence']:9.3f}  {s['off_pct']:4.0f}%")
    print()

    groups = [(c, curves[c]) for c, _ in loaded]
    if args.pool:
        merged = {k: np.mean([curves[c][k] for c, _ in loaded], axis=0)
                  for k in ("ade", "offset", "scatter", "v_bias", "v_scatter", "v_mean", "err_final")}
        groups.insert(0, (f"POOLED({len(loaded)})", merged))

    for label, curve in groups:
        print(f"===== {label} =====")
        print("CUMULATIVE ERROR vs TIME STEP")
        print("   t     ade   offset  scatter   off%    v_bias  v_scatter")
        for i in range(curve["offset"].size):
            pct = 100.0 * curve["offset"][i] / curve["ade"][i] if curve["ade"][i] > 0 else float("nan")
            print(f"  {i + 1:2d}  {curve['ade'][i]:6.1f}  {curve['offset'][i]:7.1f}  {curve['scatter'][i]:7.1f}"
                  f"  {pct:5.0f}%  {curve['v_bias'][i]:8.2f}  {curve['v_scatter'][i]:9.2f}")
        print()

        # The shape question.  R^2 close to 1 against t means a constant per-step
        # bias; against sqrt(t) means independent per-step noise.  Then the same
        # question read directly off the increments: the per-step common-mode
        # error has some length every step, and what decides whether it
        # accumulates is whether those vectors point the *same way*.
        # Coherence = |sum of the vectors| / sum of their lengths.
        s = summarise(curve)
        print("SHAPE OF THE GROWTH")
        print(f"   offset ~ t        R^2 {s['r2_lin']:5.3f}   (slope {s['slope']:+.2f} mm/step)")
        print(f"   offset ~ sqrt(t)  R^2 {s['r2_sqrt']:5.3f}")
        print()
        print("PER-STEP BIAS  (the direct read)")
        print(f"   sum_t |v_bias|         {s['travelled']:7.2f} mm   (distance walked)")
        print(f"   |sum_t v_bias|         {s['walked']:7.2f} mm   (net displacement)")
        print(f"   coherence              {s['coherence']:7.3f}      <- 1 = fixed bias (integrates), 0 = cancelling noise")
        print(f"   cumulative offset[32]  {s['final']:7.2f} mm")
        print(f"   v_bias mean / std      {curve['v_bias'].mean():7.2f} / {curve['v_bias'].std():.2f} mm")
        print(f"   largest single jump    {s['maxjump']:7.2f} mm at t={s['jump_at']}")
        print(f"   90% of final reached   at t={s['reach90']} of 32")
        print()

        # Which keypoint carries the final offset -- one finger carrying it all
        # would point at a tokenisation or indexing problem rather than a
        # distributional one.  Projected onto the common-mode direction, so the
        # column reads as "how much of this keypoint's error is the hand being
        # in the wrong place", leaving the rest as shape error.
        print("WHERE THE FINAL OFFSET SITS  (per-keypoint error at t=32, mm)")
        err_k = curve["err_final"]
        common = err_k.mean(0)
        unit = common / max(np.linalg.norm(common), 1e-9)
        order = np.argsort(-np.linalg.norm(err_k, axis=-1))
        print(f"   {'keypoint':>14s}  {'|err|':>6s}  {'along common':>12s}  {'shape-only':>10s}")
        for i in order[:5]:
            along = float(np.dot(err_k[i], unit))
            print(f"   {names[i]:>14s}  {np.linalg.norm(err_k[i]):6.2f}  {along:12.2f}"
                  f"  {float(np.linalg.norm(err_k[i] - along * unit)):10.2f}")
        spread = np.linalg.norm(err_k - common, axis=-1)
        print(f"   {'(hand mean)':>14s}  {np.linalg.norm(common):6.2f}  {np.linalg.norm(common):12.2f}  "
              f"{spread.mean():10.2f}  = mean deviation from it")
        print()

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"run": str(run), "step": step,
             "per_case": {c: {"t": list(range(1, curves[c]["offset"].size + 1)),
                              **{k: v.tolist() for k, v in curves[c].items()
                                 if k not in ("mean_err", "valid", "names", "err_final")},
                              "summary": summarise(curves[c])}
                          for c, _ in loaded}},
            indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
