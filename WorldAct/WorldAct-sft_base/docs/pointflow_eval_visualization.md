# Fixed PointFlow fitting evaluation

> **Motion selection (2026-09-14).** `POINTFLOW_SELECT_MOTION_FRACTION` restricts the anchor
> points to the fastest-moving fraction *before* PTv3, and this callback reads its cases
> through the dataset, so it is affected too: only the selected points are evaluated.
> `trajectory_metrics` splits `all` / `moving` / `static` at a 1 cm threshold, and every
> selected point moves well past it, so `moving == all`, `static_count` collapses to zero
> and `static_drift_mm` stops being reported. The rendered panels lose their static
> context points for the same reason. **Metrics are therefore not comparable across runs
> that differ in this setting** — `all_ade_mm` means "all 8192 points" in one and "the 410
> moving points" in the other. Details: `pointflow_motion_selection_20260914.md`.

The single-right-hand recipe runs validation every 100 optimizer iterations
without startup validation. Its PointFlow callback now evaluates **two fixed
training windows and three fixed validation stages (four windows each)**, independent of the
training loader's position. It caches CPU batches, writes indices, episode,
raw frame IDs, point IDs and seeds to `pointflow_eval/fixed_cases_stages_4windows.json`, and
verifies their identity on restart. Selection scans at most 12 evenly spaced
windows per split, prefers a moving window and a lower-motion window from a
different episode when available for training cases. Validation instead uses
20%, 50%, and 80% of the legal window-start range in one labeled episode,
skipping episodes too short for three distinct interior starts. This avoids
episode boundaries. It does not guarantee that every selected window has
large motion. The old fixed_cases.json is preserved. This is a diagnostic subset, not full-set
evaluation. Caption dropout and image augmentation are disabled for cases.

## Prediction and conditions

Each case starts from standard Gaussian PointFlow noise with a fixed
case-specific seed. Euler integration runs from sigma=1 to sigma=0 with
16 steps using Cosmos's v = epsilon - clean convention:

    x_next = x - v_pred / steps
    displacement_meters = x_final * pointflow_displacement_scale

Future PointFlow labels and validity masks are zeroed in the packed input
before the network forward. They are used only for metrics, selecting
displayed points, and GT rendering. The normal Cosmos packing, geometry,
attention and decoder are reused. No alternate attention implementation is
introduced.

**Conditions are clean GT video/action tokens, text, and anchor geometry.**
The modality packing/attention connectivity remains unchanged. This evaluates
conditional fitting/generalization, not history-only deployment or joint
video/action generation. The caption and artifact metadata say so explicitly.

The first evaluation after each process start also samples 32 steps with
the exact same initial noise. It saves `flow_comparison` and
`comparison_steps` in the NPZ, per-case 32-step metrics, the valid-point
16/32 prediction difference in mm, and both curves. This diagnoses sampling
sensitivity rather than automatically choosing a step count. Subsequent
evaluations use 16 steps only. All ranks participate in model forwards
inside the trainer's eval/EMA context; only rank zero renders and logs.
Python, NumPy, CPU Torch and current-device CUDA RNG states are restored.

## Media and metrics

LingBot-style GT / prediction / overlay RGB panels, ADE error maps and GIFs
are retained. GIFs now show every state at 1x playback speed. Small 1-pixel dots and short trails on at most 16 moving points
avoid obscuring the RGB. Selection and colors are paired across GT/prediction.

The adapter reads exact `raw_frame_ids`, including non-unit strides.
GT overlays use saved Track4World UV; predictions use saved per-frame camera
intrinsics. Rendering uses source head RGB, not the composed Cosmos canvas.
Future camera parameters are rendering metadata, not model inputs.

ADE/FDE are reported in millimeters for all/moving/static valid points,
excluding the prepended zero-displacement anchor. Moving means at least
10 mm displacement in valid GT states. Metrics include a zero-displacement
baseline and static-point predicted drift. Error curves compare prediction,
zero baseline and (on the first eval) 32-step results at each physical time.
Missing-valid-point curve entries are null, not misleading zeros.

Artifacts:

    <job.path_local>/pointflow_eval/fixed_cases_stages_4windows.json
    <job.path_local>/pointflow_eval/step_0000100/validation_stages.html
    <job.path_local>/pointflow_eval/step_0000100/train_00/
        prediction.npz
        metrics.json
        comparison.png
        error_map.png
        comparison.mp4
        error_curve.png
        error_curve.json

Other case names: train_01, val_00–val_03 (early), val_04–val_07 (middle), val_08–val_11 (late).
The single W&B HTML panel `pointflow/val_stages` contains a three-position
slider below the media to switch between these independent validation clips.
It displays episode, raw start frame and source time range. The HTML is
self-contained and can also be opened locally; windows are not stitched into
a supposedly continuous prediction. W&B panel interaction still needs a
check in the actual W&B frontend.
W&B keys: `pointflow/<case_id>/{comparison,error_map,video,error_curve,table}`
and per-case scalar metrics. Local files are written even if W&B is disabled.
The callback reuses the main W&B lifecycle.

CPU tests cover integration sign/scale, fixed-noise parity, shape validation,
future-label isolation at the model sampler boundary, fixed-case restart
identity, RNG restoration, frame alignment, rendering and baseline curves.
Full GPU model sampling must still be validated in the training environment.

Validation stages now concatenate four consecutive 32-frame predictions, removing repeated boundary frames. Each window uses fresh ground-truth conditions; this is not autoregressive rollout. The stage GIF labels the current window and plays at normal speed. Model and training horizons remain unchanged. Validation therefore evaluates 12 windows (plus training cases).
