# FK modality — single right hand, Edge-DROID

The robot hand's **21 keypoints as a third modality** alongside video and action:
a learned branch that predicts where the hand goes, denoised inside the same pass
as the video.

Written from the `WorldAct-cosmos3-edge-droid-sft_mano` worktree. The design of
record is that worktree's `docs/fk_modality_design.md`; this is the implementation
view — what was added, how it fits together, what was measured, and what will bite
you.

**Everything here is either read off the code or off a measurement.** Where a claim
is an inference rather than a measurement it says so.

---

## Contents

1. [What the modality is](#1-what-the-modality-is)
2. [Architecture](#2-architecture)
3. [The data pipeline, end to end](#3-the-data-pipeline-end-to-end)
4. [Inference: four arms](#4-inference-four-arms)
5. [Evaluation](#5-evaluation)
6. [Rendering](#6-rendering)
7. [Tests and what each one guards](#7-tests-and-what-each-one-guards)
8. [Gotchas](#8-gotchas)
9. [Runs and results](#9-runs-and-results)
10. [Walkthrough: from a checkpoint to a video](#10-walkthrough-from-a-checkpoint-to-a-video)
11. [Unverified, open, and next](#11-unverified-open-and-next)
12. [Glossary](#12-glossary)

---

## 1. What the modality is

| | |
|---|---|
| Signal | 21 keypoints — wrist plus four per finger |
| Units | **camera-frame metres**, as a displacement **relative to the window's anchor frame** |
| Shape | `[32, 21, 3]` — 32 future steps |
| Coordinate | `camera_d435_real` |
| Anchor | the window's frame 0, ground truth |

### 1.1 Two decisions carry most of the weight

**Camera frame, not base frame.** The labels are produced by transforming FK
output through `base_to_camera`, which is numerically identical to
`roll_about_z(180) @ base_to_head_camera(URDF)` — verified to 4e-13. So training's
frame, the frame inside `prediction.npz`, and the frame the projection tools work
in are one frame. Nothing has to be fitted to compare them, and
`anchor + prediction[t]` is already the camera-frame position.

**No uv.** The image-plane position is derivable from the 3D position, so feeding
it as a token feature or a position encoding would let the model shortcut learning
3D. The 3D→camera transform itself is *not* skipped — it is how the labels are
made — so rendering the skeleton into the image plane is legitimate. It is a
diagnostic about the output, not an input the model was given; that distinction is
why `render_fk_projection.py` lives in `tools/` rather than inside the eval.

### 1.2 The scale is a property of the episode selection

`fk_displacement_scale` is metres per model unit: the pooled per-element std of the
window displacement over the **selected episodes**. Training divides by it, so the
branch regresses in unit-scale space like every other modality here (the video
latent measures std 0.872; the action is normalised by `quantile_rot`).

| episode set | `fk_displacement_scale` | n |
|---|---|---|
| `examples/pointflow_sandwich_10_episodes.txt` (10) | **0.074450** | 26,522,496 |
| `examples/singlerighthand_101_episodes.txt` (101) | **0.083745** | 202,138,272 |

The two are 12% apart because the wider set carries larger hand motion. Change the
allowlist and it must be re-measured
(`tools/scan_fk_displacement_scale.py --episodes <allowlist>`); **a checkpoint
trained under one value cannot be resumed onto the other.**

Why it matters rather than being bookkeeping: the RF target is `v = ε − x₁` with
`x₁ = displacement / scale`. At unit scale the two terms are comparable; at a
scale 12% too large the signal is 11% smaller against a unit-variance noise term,
so the model spends proportionally more of its capacity on the noise. In the limit
`scale → ∞` the target carries no signal and the model learns to predict zero —
i.e. the stationary baseline.

> **Measured exception.** Run `-10` trained under the wrong value (0.083745 while
> its ten-episode data measured 0.074450) and it made **no measurable difference**:
> `-10` and `-10-2` land within 0.1–0.2 mm of each other at matched steps on the
> comparable windows (§9). The mismatch is real as a configuration error and
> negligible as a 12% perturbation. Do not repeat this as evidence that the scale
> does not matter — it is evidence that 12% is small.

---

## 2. Architecture

### 2.1 FK rides the existing packed sequence

FK is **not** a second model. It is a sibling field on `PackedSequence`, consumed
by the same `denoise()` call as vision and action:

```
PackedSequence
├── vision      [B, C, T, H, W]                 ─┐
├── action      [B, ...]                         │
├── sound       ...                              ├──> one denoise() pass
├── fk_data     FKBatch      labels + anchors    │    one attention
├── fk          FKTokenPayload  learned tokens   │
└── fk_noised   FKNoised     noisy state + sigma ─┘
```

`denoise()` reads `data_batch_packed.fk_noised` and returns `out_net["preds_fk"]`
alongside `out_net["preds_vision"]`. The packer sets `packed.fk_data` from the
batch, so FK is present from the packing stage onward — the same place vision and
action enter.

The consequence worth internalising: **the FK tokens attend to the video tokens**,
which is what makes the joint arm possible at all, and also what makes the 25× LR
multiplier (§2.3) a shared-weight concern rather than a private one.

### 2.2 Modules

**Data**

| File | What it does |
|---|---|
| `data/generator/action/fk_source.py` | Reads the FK annotation tree and produces camera-frame metres via `base_to_camera`. Owns `coordinate: camera_d435_real`. |
| `data/fk_camera_extrinsic.py` | The base→camera transform as a constant. Numerically equal to `roll_about_z(180) @ base_to_head_camera(URDF)`. |
| `data/fk_window.py` | Window slicing over an episode's annotations. |
| `data/fk_batch.py` | `FKBatch` / `FKNoised`. Per-sample ragged keypoint sets, **padded rather than dropped** — a sample with fewer points keeps its slot — with the token upper bound the branch is sized against. |

**Model**

| File | What it does |
|---|---|
| `model/generator/fk_sequence.py` | The token layout. Groups steps into tokens (`fk_steps_per_token = 4`, so 32 steps → **8 FK tokens**). Exists so the branch's shape and the packer's cannot drift apart. |
| `model/generator/fk_branch.py` | The learned branch: an embedding over the 21 keypoints and a sigma-conditioned MLP. |
| `model/generator/fk_training.py` | `fk_add_noise` (divides the target by the scale), `fk_loss` (mse or charbonnier), `fk_ade`. |
| `model/generator/fk_sampling.py` | The sigma grid including the **shift** the branch is trained under, `sample_displacement` (Euler, FK alone), `joint_layout` / `reassemble_fk`. |
| `model/generator/mot/cosmos3_vfm_network.py` | `install_fk` — attaches the branch to the real network. |
| `model/generator/omni_mot_model.py` | FK noising and loss in the training path; `sample_fk`, `_fk_fitting_context`, `_sample_joint`, `_make_joint_velocity`, `_seed_vision_from_noise` for inference. |

**Eval and rendering** — §5 and §6.

### 2.3 Training

```python
# fk_training.fk_add_noise
target = clean.displacement.float() / scale     # metres -> model units
x_sigma = sigma * epsilon + (1 - sigma) * target
velocity_target = epsilon - target

# fk_training.fk_loss
loss = charbonnier(pred, velocity_target, eps)   # or squared error
```

Three settings deserve their reasons written down:

**`independent_fk_schedule=True`.** FK's sigma is drawn **independently of the
video's**. Sharing one sigma per sample lets the model learn "the video is clean ⇒
σ≈0 ⇒ return the input unchanged", and the eval's condition — clean video, FK from
pure noise — is exactly what fires that shortcut: the sampled trajectory collapses
onto its own initial noise, which is the "scattered, no hand shape" failure. Drawing
FK's sigma independently keeps the marginal over sigma identical and puts that
condition back inside the training distribution. The PointFlow branch hit the same
bug and fixed it the same way.

**`optimizer.lr_multipliers.fk_branch = 25.0`.** The branch trains at 25× the base
LR. FK tokens pass through the same attention as vision, so the shared weights
receive FK gradients at 25× the step size. This is the leading suspect for the
video-loss regression measured in §5.5 — untested, and cheap to test.

**Charbonnier vs MSE.** `sqrt(x² + ε²) − ε`: L2 inside `ε`, L1 outside,
differentiable at zero. It is a *switch to compare*, not a correction known to help
— L1 objectives are higher-variance. Across every mse-vs-charbonnier pair here it
produces the same signature: better on the training windows, worse on `val_01`.

The masked MSE is not the number to read. Its scale is dominated by the
unit-variance noise term, so a prediction of zero already scores ~1.0; and a
charbonnier run's `fk_loss` is on a different scale from an mse run's. `metrics`
carries `fk_mse_loss` alongside for exactly that reason.

### 2.4 Sampling

`fk_sampling.shifted_sigmas` builds the sigma grid with a **shift** the branch is
trained under. Sampling unshifted walks out of that distribution, so `sample_fk`
defaults `shift=None` → resolves the trained value; the `shift=` argument is a knob
to sweep, not one to set by reasoning.

The FK-only path uses `sample_displacement` (Euler) or UniPC. The joint path uses
`self.sampler` — the deployed video sampler — because the FK tokens are meant to be
denoised *inside* that loop, and evaluating them with a different integrator
measures a trajectory the deployed system never runs.

> **UniPC's container contract.** `UniPCSampler.forward` takes `noise` and `seed`
> and requires **both to be lists or both to be scalars**, of the same length. It
> forwards both into `run_multiseed`, which asserts its kwargs are all-list or
> all-non-list. A list of per-sample latents with a scalar seed makes it build a
> single scheduler while the latents arrive as a list, and the failure is
> `AssertionError: All kwargs must be lists or all must be non-lists, cannot mix`.
> The joint arm passes one seed per sample.

---

## 3. The data pipeline, end to end

### 3.1 One window

`sample_stride=1`, `chunk_length=32`, `source_stride = 30/15 = 2`.

```
raw frame index (30 Hz)     w   w+2  w+4 ...................... w+64
                             │    │    │                            │
observations (33 frames)     ●────●────●──── ... ────────────────●
                             │    │    │                            │
predicted steps (32)         │    ●────●──── ... ────────────────●
                             │
anchor (ground truth)        ●

frame_ids = [w, w+2, …, w+64]        len 33
prediction / target:  [32, 21, 3]
step t belongs on frame_ids[t+1]
```

The anchor is **given**; the 32 steps are what the branch produces. `frame_ids`
has `steps + 1` entries and the renderers depend on that: step `t` → `frame_ids[t+1]`.

### 3.2 Tiling: `step = span // stride`

`fk_eval_cases.window_step(dataset)` returns `(span, stride, step)` measured off
the dataset — not hardcoded, because it is a product of `sample_stride` and
`source_stride` and a wrong value tiles the wrong frames silently.

```
span   = raw_frame_ids[-1] - start_frame     # 64 raw frames
stride = raw frames between consecutive indices   # 2
step   = span // stride                      # 32 dataset indices = ONE window

window k        covers raw w+2 … w+64      (32 predicted frames)
window k+step   covers raw w+66 … w+128
                        ↑
              exactly one model step apart: contiguous, no gap, no overlap
```

The single frame two windows share is the **later window's anchor**, which is
ground truth — so it is not emitted twice, and no predicted frame comes from two
windows. **There is no "which chunk wins" question to resolve.**

This is the same rule the PointFlow line uses (`pointflow_eval_cases.py:51`), which
is where its `129 frames = 4 windows × 33 − 3 shared boundaries` comes from. Dense
windows (`sample_stride=1`) overlap 32-fold and *create* the ambiguity; the dense
enumeration is for training (where overlapping samples are data augmentation and
nothing has to be stitched), not for rendering.

### 3.3 The composed canvas, and the two rows nobody mentions

The dataset stacks views before the model sees them:

```
_compose_views(head, wrist):
    target_width = min(head_w, wrist_w) = 640
    wrist 848×480 -> width 640 -> 362 rows
    head  640×480              -> 480 rows
    cat(wrist, head)           -> 842 × 640        <- the composite
```

then resized aspect-preserving to the target and reflection-padded
(`find_closest_target_size` + `reflection_pad_to_target`, `resolution="480"`).

> **⚠️ The model is fed only the top 44 of the canvas's 46 latent rows**
> (704 of 736 px). The cache stores 46 rows; the packer hands the model 44.
>
> **Measured, not inferred**: the rollout's conditioning frame equals the cache
> window cut to its top 44 rows with a mean absolute difference of exactly
> **0.000000**.

This is the single most expensive thing in the pipeline to re-derive, and it
poisons any renderer that assumes the cache geometry. Concretely: derive the canvas
crop from the **width**, which is never cropped. Using `min(h/842, w/640)` reads the
scale as 0.836 instead of the true 0.85, which puts `head_top` at 303 instead of 308
and mixes five wrist rows into the head view — a plausible-looking picture that is
subtly wrong everywhere. `render_fk_projection.canvas_crop` uses the width.

Second consequence: the generated panel is not the same geometry as the real one.
`render_fk_rollout.generated_v_scale` corrects for it — the head block that survives
the crop (308..704 = 396 rows) is resized to 480 where the true mapping is
408 → 480, so the generated panel is stretched by 408/396 = 1.0303 and a skeleton
projected with the real intrinsics lands progressively high, ~14 px at the frame
bottom. The real-video panel is untouched; it never went through the model.

### 3.4 The dataset wrapper chain

```
instantiate(dataset_cfg)
  -> ActionSFTDataset            transform wrapper; 1:1 delegation, NO _cumulative_ends
     -> SingleRightHandRawDataset   window bookkeeping lives here
```

`ActionSFTDataset.__len__` and `__getitem__` both delegate, so the index spaces
match — but the attribute needed to map an index to an episode
(`_cumulative_ends`) is one level down. Anything that needs it must unwrap through
`_dataset`, asserting each layer is index-preserving; `fk_eval_cases.raw_dataset()`
does exactly that. A wrapper that ever added or dropped samples would make every
index below address the wrong episode, silently.

`dataset[i]` is **expensive** — it decodes latents — so never scan for an episode
boundary a sample at a time. `episode_index_range` reads the cumulative counts.

### 3.5 Windows, val, and eval cases

**Enumerating windows.** `sample_stride=1` means almost every raw frame starts a
window. For an episode of `F` raw frames: `valid_windows = F - 64`, each advancing
one raw frame.

**The val split** is carved from the **same allowlist**:

```python
num_val  = round(len(all_episodes) * split_val_ratio)     # round(10 * 0.2) = 2
order    = torch.randperm(n, generator=manual_seed(split_seed))
selected = order[:num_val] if split == "val" else order[num_val:]
```

`split_val_ratio` defaults to 0.03, which **rounds to zero val episodes on a
ten-episode set** — leaving the val loader empty and the periodic eval with nothing
to run. The recipe sets 0.2. The ten-episode val episodes are
`episode_0015_20260730_170501` and `episode_0018_20260730_170833`.

**Eval cases** (`fk_eval_cases.fixed_cases`): per split, take up to 12 candidate
indices spread by `np.linspace`, score each by `_motion_score` — **mean keypoint
displacement in mm, read straight off the labels** — pick the highest-motion one
first, then prefer *distinct episodes* over same-episode candidates. Labels only,
never a model prediction, so the reported numbers cannot be improved by picking
easier windows after the fact. The chosen indices are written to
`fk_eval/fixed_cases.json` and re-checked on resume; a run that silently evaluated
different windows would look like a regression or a jump depending on which way the
selection drifted.

---

## 4. Inference: four arms

Read this section before quoting any number.

| arm | where | video | action | FK | length |
|---|---|---|---|---|---|
| **Conditional** (default) | `fk_eval/<case>/` | **clean** | **clean** | from noise | 2.1 s |
| **Joint** (`FK_EVAL_JOINT=1`) | `fk_eval/<case>/joint/` | **from noise** | clean | from noise | 2.1 s |
| **Joint + action** (`JOINT_ACTION=1`) | `fk_eval/<case>/joint_action/` | from noise | **from noise** | from noise | 2.1 s |
| **Rollout** (`FK_ROLLOUT=1`) | `fk_rollout*/step_X/<episode>/` | per `FK_ROLLOUT_GENERATED` | clean | from noise | 17 s |

### 4.1 Why the conditional arm is a ceiling

`_fk_fitting_context` packs the conditioning at sigma=0:

```python
input_timesteps = torch.zeros((clean.batch_size, 1))       # every conditioning token
packed = self._pack_input_sequence(plans, text, clean, input_timesteps)
packed.fk_data = replace(data, displacement=zeros, valid=zeros, labeled=zeros)
# vision_sigma is None  ->  the noising branch is skipped entirely
```

and `sample_fk` then draws `state = randn([32, 21, 3])` and integrates.

So in the conditional arm:

- the video is handed over **clean for the whole 33-frame window, including the 32
  future frames** — the model can simply read where the hand goes;
- the action is clean;
- FK's own labels are zeroed and only its anchor is real.

It is a legitimate diagnostic of "how well does the branch fit when the
conditioning is perfect", and it is **not the deployed configuration**. On a robot
the video and the action are generated too.

**There is no history before the window.** Each window starts at its own anchor and
sees nothing earlier, which is also why the skeleton snaps back to the recorded
pose every 32 steps.

### 4.2 The joint arm is a sampling switch

Training already puts both modalities into one packed sequence and noises them
together, so the model has always been able to denoise them jointly. `generate_video=True`
routes `sample_fk` to `_sample_joint` / `_make_joint_velocity`, which build a flat
`[vision | FK]` state per sample, seed the vision from noise at the condition frames
(`cond_mask * x0 + (1 − cond_mask) * pure`), and step both on one sigma.

**No retraining is needed to ask the question.** If the answer is bad, *then* the
training side might be worth changing — but the question is free.

One honest caveat: with `independent_fk_schedule=True` the training marginal has FK's
sigma independent of the video's, whereas the joint arm shares the loop sigma. That
is a sampling-distribution mismatch, and it is precisely what the experiment
measures.

`joint_action` adds the future action to the same joint state — the deployed
configuration, where only the anchor frame is still handed over.

### 4.3 The window's observed frames

A window observes `steps + 1 = 33` frames and the causal VAE decodes one pixel frame
per observed frame (`1 + (T_lat − 1) × 4 = 33` for `T_lat = 9`). So the generated
frame for step `t` is index **`t + 1`**, exactly the offset the real video already
uses via `frame_ids[t+1]`. Indexing `t` puts the anchor frame on step 1 and shifts
the whole panel one frame early — which looks like a bad model, not an off-by-one.

---

## 5. Evaluation

### 5.1 Metrics

The reported number is `all_ade_mm` against **`zero_all_ade_mm`**, the
stationary-trajectory baseline (`trajectory_metrics(zeros, target, valid)`).

`ratio_to_zero < 1` means the model beats "predict that the hand does not move".
That is the only threshold in this line that means anything on its own — the loss
cannot supply it (§2.3).

`metrics.json` also carries `final_ade_mm`, `per_step_ade_mm`, the 21 per-keypoint
ADEs, and — on the first eval of a run — an euler-16 comparison with
`sampling_difference_mm`, so one eval answers "was the solver the binding
constraint" without a second run.

### 5.2 Comparability rules

These have all cost real time to learn:

1. **`zero_all_ade_mm` is a pure data property.** It is computed from the targets
   and knows nothing about any model. Two runs whose `zero` differs are **not**
   evaluating the same window, whatever the case names say. Always check it first.
2. **`train_00` / `train_01` are in the training set.** They measure memorisation.
   Only `val_00` / `val_01` speak to generalisation, and with a ten-episode set
   there are only two of them.
3. **The rollout's `zero` is ~820 mm** against the fixed windows' 20–240 mm: it
   covers a stretch where the hand travels far and every window resets to its own
   anchor. **Ratios do not compare across the two**, and the rollout's absolute ADE
   is *worse* (54–82 mm vs 23 mm) for that reason.
4. **FK loss values do not compare across runs** — scale and objective have both
   changed. **ADE and `ratio_to_zero` do.**
5. **`val_00` is not stable across runs.** Between `-9`/`-10-2` and `-10` it is a
   different window (zero 224.62 vs 207.68) and a much easier one (ratio ~0.05 vs
   ~0.20). Comparing `-10`'s `val_00` to anything else compares windows, not models.

### 5.3 Measured — `-101` @ step 10000, same windows

| case | conditional | joint |
|---|---|---|
| val_00 | 11.3 mm, ratio 0.048 | 53.9 mm, ratio 0.228 |
| val_01 | 3.3 mm, ratio 0.168 | 50.5 mm, **ratio 2.613** |

`val_01`'s ratio above 1 means the joint arm is worse than predicting that the hand
does not move. The conditional arm's good numbers were almost entirely propped up
by being handed the answer video.

`-101`'s eight-window rollout: ADE 33.99 / 26.55 mm, ratio 0.04 / 0.03 against a
~820 mm baseline.

### 5.4 What the branch costs the other modalities

Baseline `singlerighthand-dropper-edge-droid-50k-aot` — the same recipe with **no
`fk_branch`** and `fk_root=None`. Verified comparable: the same 101 episodes, the
same `max_samples_per_batch=32`, the same base checkpoint, the same base LR. Only
`cycle_lengths` differs (50000 vs 10000), which is why the early window is the one
that matters.

| step window | video loss, no FK | video loss, with FK | change | LR ratio |
|---|---|---|---|---|
| 100-500 | 0.1490 | 0.1684 | **+13.0%** | **1.00×** |
| 1000-2000 | 0.1315 | 0.1493 | +13.6% | 0.95× |
| 3000-4000 | 0.1228 | 0.1377 | +12.1% | 0.74× |
| 5000-6000 | 0.1177 | 0.1335 | +13.4% | 0.43× |
| 9000-10000 | 0.1099 | 0.1261 | +14.8% | 0.01× |

**+13% in the first window, where the learning rates match exactly**, and flat
across the whole run — so it is the cost of adding the branch, not a schedule
artefact. At the last window the FK run's LR is 1% of the baseline's, i.e. it has
annealed further and should be *better*, and it is still +14.8%; the confound runs
against the effect, so the true cost is at least what is measured.

Action loss sits on the noise floor in both (0.0012 vs 0.0016, 100× smaller than
video) and is not worth reading.

Leading suspect: `lr_multipliers.fk_branch = 25.0` (§2.3). Untested.

---

## 6. Rendering

| tool | output | notes |
|---|---|---|
| `tools/render_fk_case.py` | `comparison.png`, `error_curve.png`, `comparison.mp4` for a fixed case | **Offline**, from `prediction.npz`. `--eval-dir <...>/fk_eval --all --video`, or `--case-dir` for one |
| `tools/render_fk_projection.py` | `fk_projection.mp4` + a still sheet, two panels | Needs a GPU (VAE decode). `--case-dir <case>`, `--only animation\|projection`, `--stride N`, `--no-generated` |
| `tools/render_fk_rollout.py` | `rollout.mp4` (the animation) and `projection.mp4` (two panels) | Needs a GPU when the rollout generated video. `--rollout <rollout.npz>`, `--vae`, `--stride` |
| `tools/render_fk_joint.sh` | wrapper | Takes **no path argument** — it finds the eval itself. `ARM=joint\|joint_action`, `ROOT=<eval root>`, `OUT_ROOT=` |

**Why the figures are offline.** `FK_EVAL_FIGURES` defaults to false. The figures
are matplotlib and the eval renders ~128 of them per validation pass (4 cases × 32
frames for the mp4, plus the stills) — measured at **~0.35 s/step** at
`validation_iter=100`, about 9% of a single-GPU step, spent on steps nobody opens.
The trainer writes `prediction.npz` + `metrics.json` (kilobytes, milliseconds) and
the figures are drawn later, for any step.

### 6.1 How to read the two panels

```
┌──────────────────────────┬──────────────────────────┐
│  REAL                    │  GENERATED (i2v)         │
│  the recorded camera     │  the video the model made│
│                          │                          │
│   ── green : GT skeleton │   ── green : GT skeleton │
│   ── red   : prediction  │   ── red   : prediction  │
└──────────────────────────┴──────────────────────────┘
        the SAME two skeletons, the SAME frame
```

The panels differ **only in the background**. That is the whole design: it
separates the two ways a joint arm can fail.

| left (real) | right (generated) | reading |
|---|---|---|
| red tracks green | red tracks green | both fine |
| red does not track | red tracks | **video generation is the bottleneck** — FK is faithfully following the video it was given |
| red does not track | red does not track | **FK's own conditioning is the problem** |

**Check the green on the LEFT panel first.** If GT is not on the hand, the
projection is wrong and nothing else in the picture means anything. Both tools
print this, and `render_fk_projection_test.py` checks it numerically (672/672 GT
keypoints inside 640×480 on a real case).

### 6.2 The seam is real and is not smoothed

Each window generates independently, from its own noise, conditioned on its own
anchor. So:

- the skeleton **snaps back to the recorded pose** every 32 steps,
- the generated panel **jumps** at each seam.

This is a property of windowed generation, not a rendering fault. The PointFlow
stage clips have exactly the same property and label each window
`independent prediction` rather than hiding it; `render_fk_rollout.py` does the
same. Removing the seam would mean conditioning each window on the previous
window's last generated frame, which the model was never trained to do (training
always conditions on frame 0) — a training-side change, not a rendering one.

Renders go under `shichaojian/renders/` — not `/tmp`, and not inside the run
directory, which gets pruned.

---

## 7. Tests and what each one guards

They are **standalone scripts, not pytest**. This venv has no `xdist` and the root
`conftest.py` fails to load, so `pytest` reports "no tests ran"; run them directly:

```bash
PYTHONPATH=. <sft venv>/bin/python <file>
```

| File | Guards against |
|---|---|
| `fk_batch_test.py` | ragged per-sample offsets; a recycled buffer being shared between two batches; mixed timing accepted silently |
| `fk_source_test.py` | the annotation → camera-frame metres path |
| `fk_sequence_test.py` | token layout vs the packer's |
| `fk_branch_test.py` | the branch's dtype/device behaviour |
| `fk_independent_schedule_test.py` | FK's sigma actually being drawn independently |
| `fk_joint_sampling_test.py` | `joint_layout` / `reassemble_fk` round-trips. **Teeth**: a wrong flatten-and-reshape reassembly produces the *correct shape* `(4, 47, 3)` with **162 of 188 values wrong** — the failure is invisible to a shape check |
| `fk_eval_test.py` | the eval callback's recording |
| `render_fk_projection_test.py` | the projection's arithmetic. **Teeth**: `project()` takes the **camera** frame; the neighbouring `render_fk_overlay.project_episode` takes the **base** frame and transforms internally. Reaching for the wrong one applies the transform twice and lands the skeleton 343 px off — *near* the hand, which reads as a bad model rather than a bad call |
| `vae_window_latent_test.py` | the window latent cache's packing |
| `dataset_config_wiring_test.py` | the recipe's dataset config reaching the dataset |
| `fk_sparse_key_test.py` | sparse-key handling in the dataflow |
| `fk_displacement_scale_test.py` | the scale measurement |

---

## 8. Gotchas

Each of these cost time. The symptom is listed because that is what you will see.

| Symptom | Cause |
|---|---|
| `ListConfig indices must be integers or slices, not NoneType` at some step | `scheduler.cycle_lengths` shorter than `max_iter`. `find_in_interval` returns `None` past the last cycle and `schedule` indexes `lr_warm_up_steps[None]`. **The LR schedule is a list of cycles; raising `max_iter` alone trains to the old bound and then dies.** A run of this recipe was lost at step 10001 to exactly that. |
| Every predicted displacement is systematically shrunk or stretched, no error | `fk_displacement_scale` at eval time differs from what the checkpoint trained under. The sampler multiplies by whatever the *config* says. `run_fk_joint_eval.sh` refuses to start on a mismatch; `SCALE=` overrides. |
| `dataset does not expose _cumulative_ends` | The dataset is an `ActionSFTDataset` wrapper; the bookkeeping is on the inner `SingleRightHandRawDataset`. Unwrap through `_dataset` (§3.4). |
| `decoded (704, 544) is smaller than the content 716x544` | Canvas constants hardcoded from the 46-row cache instead of derived. The model sees 44 rows (§3.3). |
| `decoded 33 frames for 32 steps` | Off-by-one indexing of the generated panel: a window observes `steps + 1` frames (§4.3). |
| `8 vision windows x 64 steps != 256 predicted frames` | `window_frames` conflated with the index stride. The steps per window are `len(raw_frame_ids) − 1`; derive from the shapes. |
| `All kwargs must be lists or all must be non-lists, cannot mix` | UniPC's container contract (§2.4). |
| The skeleton is drawn progressively too high on the generated panel only | The model's cropped canvas (§3.3) — `generated_v_scale`. |
| ffmpeg writes a 0-byte file and reports no error | `macro_block_size=1` lets the canvas land on an odd height; libx264 with yuv420p requires even dimensions. |
| Edits appear to have no effect after a launch | The shared venv's editable `cosmos_framework` points at the sibling worktree; `PYTHONPATH` is what makes this worktree win. Run the launcher from its own worktree root — it now checks. |
| A pasted command fails with `-bash: eval/step_...: No such file or directory` | The terminal wrapped a ~150-character path and bash read the tail as a command. This is why the launchers take no path arguments. |
| `OSError: [Errno 19] No such device` on `np.load(..., mmap_mode="r")` | GPFS/virtiofs on this mount. Read the `.npy` header when only the shape is needed. |
| `import torch` takes minutes | Also virtiofs. Slow, not deadlocked — but do not `grep -rn` the whole repo at the same time. |

---

## 9. Runs and results

| run | episodes | what changed |
|---|---|---|
| `-9` | 10 | scale 0.074450 (**correct for this set**), mse, save_iter 1000 / validation_iter 500 |
| `-10` | 10 | **charbonnier**; scale read the 101-episode value (0.083745) because it was not pinned on the command line — `-9` had pinned its own explicitly |
| `-101` | 101 | the full set; **crashed at step 10001** on the `cycle_lengths` bug |
| `-101-2` | 101 | only `cycle_lengths=[20000]` differs. Restarted from base rather than resumed — resuming would re-derive the LR for the first 10000 steps under a different cycle and jump the schedule |
| `-10-2` | 10 | scale corrected to 0.074450, 1 GPU (global batch 32 — **not** `-10`'s 256) |

### 9.1 The scale error had no measurable effect

Comparable windows (`zero` identical across the runs): `train_00` 198.80,
`train_01` 33.96, `val_01` 24.72. At step 9000:

| case | `-9` mse/8×256 | `-10` charb/8×256 | `-10-2` charb/1×32 |
|---|---|---|---|
| train_00 | 13.03 (.066) | 12.12 (.061) | **12.22 (.061)** |
| train_01 | 6.76 (.199) | 5.24 (.154) | **5.25 (.154)** |
| val_01 | 8.99 (.364) | 10.00 (.405) | **9.86 (.399)** |

`-10` and `-10-2` agree to 0.1–0.2 mm. **The 12% scale mismatch is a real
configuration error with no measurable consequence** — an important calibration of
how much a 12% scale perturbation actually costs.

What does separate them from `-9` is the objective: charbonnier is better on the
training windows (`train_01` 6.48 → 4.93 at step 10000, −24%) and worse on `val_01`
(8.73 → 9.66, +11%). The same signature appears in every mse-vs-charbonnier pair
here.

### 9.2 `-10`'s `val_00` is a different window

```
               -9 / -10-2  (zero 224.62)      -10  (zero 207.68)
@9000           44.96 / 43.84  (ratio ~0.20)    8.85  (ratio 0.043)
```

`-10`'s `val_00` ratio stays at 0.04–0.10 where the others' is 0.19–0.26. Its
headline `val_00` number was flattered by window selection, not earned by the
model. This is exactly the failure mode rule 1 in §5.2 exists to catch.

---

## 10. Walkthrough: from a checkpoint to a video

```bash
cd /mnt/shared-storage-gpfs2/.../WorldAct-cosmos3-edge-droid-sft_mano

# 1. Sample an existing checkpoint. No training: the checkpoint is symlinked into a
#    fresh root and max_iter pinned to its own iteration, so the start validation
#    runs and nothing else. JOINT_ACTION=1 denoises the future action too, which is
#    the deployed configuration.
JOINT_ACTION=1 \
SOURCE_RUN=fk-singlerighthand-edge-10 OUT=fk-joint-eval-10 ITER=9500 \
ALLOWLIST=$PWD/examples/pointflow_sandwich_10_episodes.txt \
bash tools/run_fk_joint_eval.sh

#    -> <OUT>/.../fk_eval/step_X/<case>/joint_action/prediction.npz   (with the vision latent)
#    -> <OUT>/.../fk_rollout_joint_action/step_X/<episode>/rollout.npz

# 2. Render. Needs a GPU: the generated panel is a VAE decode. Takes no path.
ARM=joint_action ROOT=<the OUT root> bash tools/render_fk_joint.sh

#    -> renders/fk_joint/joint_action/<case>/fk_projection.mp4        (2.1 s, two panels)
#    -> renders/fk_rollout/joint_action/<episode>/projection.mp4      (17 s, two panels)

# 3. Read it. Green on the LEFT first, then the right panel (§6.1).
```

To train instead:

```bash
bash tools/run_fk_10_2.sh        # 10 episodes, corrected scale
bash tools/run_fk_101.sh         # the 101-episode set
```

To re-measure the scale after changing the episode set:

```bash
PYTHONPATH=. <sft venv>/bin/python tools/scan_fk_displacement_scale.py \
    --episodes examples/<new allowlist>.txt
```

---

## 11. Unverified, open, and next

**Not verified.** Say so rather than assuming:

- **The joint arm has never been validated against a reference.** It runs, it
  produces plausible generated video, and its numbers are worse than the
  conditional arm's. Whether that gap is *entirely* the loss of future-video
  leakage, or partly a defect in the joint sampling, is not established.
- **`lr_multipliers.fk_branch = 25.0` has not been ablated.** It is a hypothesis
  for the +13% video-loss cost (§5.4), not a finding.
- **The generated video was never checked for spatial faithfulness of the hand.**
  The background is recognisably the right scene; whether the *hand* inside it is
  in the right place is what decides whether FK's joint-arm failure is a video
  problem or a branch problem, and it needs the two-panel video watched frame by
  frame rather than sampled.

**Open.**

- The two-segment arm's numbers (`joint/`) do not exist for the six-checkpoint
  sweep — `joint_action=True` skips that arm — so "what was the clean action
  worth" is unmeasured. It needs the same sweep re-run with `JOINT_ACTION=0`, and
  the sweep script's `OUT` does not currently name the arm, so two runs would
  collide by suffix only.
- `val_01` is non-monotonic across checkpoints (e.g. `-10`: 1.08 → **5.57** → 0.94)
  and mostly loses to the stationary baseline. Ten episodes is a very small
  training set and the branch appears to overfit it. This is the most useful
  unexplained number in the line.

---

## 12. Glossary

| Term | Meaning here |
|---|---|
| **anchor** | A window's frame 0 — ground truth, given to the model, and what every displacement is relative to |
| **window** | One training/eval sample: 33 observed frames (at 30 Hz indices) → 32 predicted steps |
| **span** | A window's raw-frame extent, 64 |
| **stride** | Raw frames between consecutive dataset indices, 2 (`source_stride`) |
| **step** | `span // stride` = 32 dataset indices — one full window. The tiling interval |
| **arm** | An inference configuration (conditional / joint / joint_action / rollout) |
| **zero baseline** | `zero_all_ade_mm` — the ADE of predicting no motion. A pure data property |
| **ratio** | `all_ade_mm / zero_all_ade_mm`. Below 1 means beating "the hand does not move" |
| **DCP** | The distributed checkpoint format these runs write |
| **composite / canvas** | wrist stacked over head, 842×640 before the model's resize |
