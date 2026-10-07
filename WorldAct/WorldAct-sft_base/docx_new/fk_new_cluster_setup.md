# FK modality — new-cluster setup

Everything needed to run the FK line on a fresh cluster: what to copy, what to
regenerate, what to remap, and how to check it worked.

**Paths here are relative to the repository root.** The existing cluster has a
long absolute prefix baked into ~40 files; §2 is about finding and replacing it.
Read this alongside [`action_policy_fk_singlerighthand_edge_posttrain.md`](../docs/action_policy_fk_singlerighthand_edge_posttrain.md),
which explains the modality itself.

---

## Contents

1. [What this line is](#1-what-this-line-is)
2. [⚠️ The absolute-path problem](#2-️-the-absolute-path-problem)
3. [Environment](#3-environment)
4. [Repository and worktree layout](#4-repository-and-worktree-layout)
5. [Data: what to copy, what to rebuild](#5-data-what-to-copy-what-to-rebuild)
6. [Configuration reference](#6-configuration-reference)
7. [Architecture in one page](#7-architecture-in-one-page)
8. [Commands](#8-commands)
9. [Verification checklist](#9-verification-checklist)
10. [Cluster-specific gotchas](#10-cluster-specific-gotchas)

---

## 1. What this line is

`cosmos-framework` fine-tuned into a robot policy for a single right hand, with a
**third modality added**: the hand's 21 keypoints as camera-frame displacement,
predicted by a learned branch inside the same denoise pass as the video.

| | |
|---|---|
| Recipe | `action_policy_singlerighthand_edge` (and its FK variant) |
| Base | Cosmos3 Edge-DROID |
| Data | sandwiches, single right hand, ~101 episodes |
| Modality | `[32, 21, 3]` camera-frame metres, relative to the window's anchor |
| Branch | `cosmos_framework/model/generator/fk_branch.py` |

The design of record is `docs/fk_modality_design.md` **in the `_mano` worktree** —
see §4 for why that matters.

---

## 2. ⚠️ The absolute-path problem

**This is the thing that will cost the most time.** ~80 absolute paths referencing
the old cluster are hardcoded across **40 files**. Nothing will run until they are
remapped.

### 2.1 Find them

```bash
grep -rn "/mnt/shared-storage-gpfs2" \
    --include="*.py" --include="*.sh" --include="*.toml" \
    cosmos_framework/ tools/ examples/ docs/
```

### 2.2 The roots they reference

| Old root | What it holds | Count |
|---|---|---|
| `<OLD>/models/` | base DCP checkpoint, Wan2.2 VAE, Edge-DROID processor bundle | 16 |
| `<OLD>/raw_data/` | the raw episodes, and the FK annotation tree | 15 |
| `<OLD>/runs/` | training outputs (checkpoints, evals, renders) | 11 |
| `<OLD>/datasets/` | the cosmos cache (state/action npz, video frames, VAE window latents) | 10 |
| `<OLD>/` (bare) | loose files written to the storage root | 10 |
| `<OLD>/WorldAct-cosmos3-edge-droid-sft/` | **the sibling worktree — its `.venv` is the interpreter** | 7 |
| `<OLD>/renders/` | rendered videos | 3 |
| `<OLD>/wuji-mjlab/`, `<OLD>/mjlib/`, `<OLD>/fk_calib/` | recalibration-only assets — see §2.2d | 5 |

where `<OLD>` = `/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian`.

### 2.2b Where each root goes on the new cluster

| `<OLD>/…` | New | Notes |
|---|---|---|
| `<OLD>/WorldAct-*` | **`/mnt/afs/WorldAct-*`** | the worktrees. `HOME` on the new cluster is `/mnt/afs`, and the worktrees live directly under it. **The venv is one of these** — and `sed` alone does not fix it, see §3.1 |
| `<OLD>/models/` | `/data/shichaojian/models/` | |
| `<OLD>/raw_data/` | `/data/shichaojian/raw_data/` | |
| `<OLD>/datasets/` | `/data/shichaojian/datasets/` | the cosmos cache |
| `<OLD>/runs/` | `/data/shichaojian/runs/` | training outputs |
| `<OLD>/renders/` | `/data/shichaojian/renders/` | |
| `<OLD>/` bare (loose files) | `/data/shichaojian/` | |
| `<OLD>/wuji-mjlab/` | `/data/shichaojian/wuji-mjlab/` | upstream robot repo (URDF + mesh). **Only needed to recalibrate the extrinsic or re-verify the FK export** — training and eval never read it. See §2.2d |
| `<OLD>/mjlib/` | *don't migrate* | a `pip --target` site-packages dir (mujoco 3.12.0 + lmdb). Rebuild instead: §2.2d |
| `<OLD>/fk_calib/` | *don't migrate* | an **output** dir (`run_calib.sh:11`), and it does not exist right now — `mkdir -p` recreates it |

**The two rules differ, and the order matters:** worktree paths go to `$HOME`
(`/mnt/afs`), everything else goes to `/data/shichaojian`. A blanket replace of
`<OLD>` → `/data/shichaojian` would put the venv and the worktrees in the wrong
place.

### 2.2c The replace, in the order that works

```bash
cd <the worktree you are migrating>
OLD=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian

# 1. worktrees FIRST -- including the venv path, which is one of them.
grep -rl "$OLD/WorldAct-" --include="*.py" --include="*.sh" --include="*.toml" . \
  | xargs -r sed -i "s|$OLD/WorldAct-|/mnt/afs/WorldAct-|g"

# 2. everything else.
grep -rl "$OLD" --include="*.py" --include="*.sh" --include="*.toml" . \
  | xargs -r sed -i "s|$OLD|/data/shichaojian|g"

# 3. confirm nothing is left.
grep -rn "$OLD" --include="*.py" --include="*.sh" --include="*.toml" . ; echo "expect: no output"
```

Reversing 1 and 2 rewrites the worktree paths to `/data/shichaojian/WorldAct-*`,
which is wrong and silent. Run them in the order above.

### 2.2d `wuji-mjlab` / `mjlib` / `fk_calib` — the three recalibration-only roots

None of these three is needed to **train or evaluate FK**. The 21 keypoints are
precomputed offline into `raw_data/sandwich_fk21/episode_*/` (one directory per episode,
see §5.1); the training and eval code reads only those. Grepping `cosmos_framework/` for these names returns exactly
one hit — a docstring in `cosmos_framework/data/generator/action/fk_source.py:15` —
and no import. They matter only if you want to **redo the camera-extrinsic calibration**
or **re-verify that the stored annotations really are what the upstream FK gives**.

| Root | What it actually is | Who reads it |
|---|---|---|
| `wuji-mjlab/` | The **upstream robot repository** — URDF (`marvin_wuji_d435_description/urdf/marvin_wuji_d435_complete.urdf`), meshes, and the qpos→keypoint replay reader. The FK annotations *are* this pipeline's output, reproduced bit-exactly. | `tools/export_fk_camera_extrinsic.py:43` (derives the extrinsic from the URDF, and **md5-checks it against our committed copy** in `cosmos_framework/data/fk_camera_extrinsic.py`); `tools/replay_verify_fk.py:27`; `tools/fk_hand_mesh.py:29`; the `render_*` mesh tools |
| `mjlib/` | Not our code and not source — a `pip install --target` **site-packages directory** holding `mujoco 3.12.0` + `lmdb` (hence `mujoco.libs/`, `mujoco-3.12.0.dist-info`). Injected via `PYTHONPATH=<ROOT>/mjlib` because the training venv does not carry mujoco. | every tool that runs MuJoCo forward kinematics or rendering |
| `fk_calib/` | An **output** directory, not an input — `tools/run_calib.sh:11` sets `OUT=<OLD>/fk_calib` and drops the calibration log there. | written only by `run_calib.sh` |

**The URDF exists in two places, and neither is in git.** Check this before deciding
what to copy:

| Copy | Path | Role |
|---|---|---|
| primary | `WorldAct-lingbot-va-mano/lingbot-va/wan_va/fk/assets/marvin_wuji_d435_description/` | what `verify_fk_camera_projection.py:26` (`URDF`) reads |
| mirror | `wuji-mjlab/marvin_wuji_d435_description/urdf/` | read only for the md5 cross-check |

Both hash to `d687e8f52df18839f4f99efe8c12a909`, matching the committed `URDF_MD5`
(`cosmos_framework/data/fk_camera_extrinsic.py:32`), and both carry the full
`meshes/` tree. **`git ls-files` on either returns 0 entries** — they are untracked
data, so they do not travel with the repository and must be copied explicitly.

Consequences for the migration:

* **`wuji-mjlab/` is self-contained and is the one to copy.** It holds the mirror URDF,
  the `meshes/` tree, the replay reader package (`src/`, `requirements-replay.txt`) and
  `tianji_wuji_data/`. Copying it covers every recalibration and rendering tool
  *without* also migrating the `WorldAct-lingbot-va-mano` worktree. The only thing lost
  is the primary/mirror skew check, which the committed `URDF_MD5` already pins.
* The mirror check is guarded (`if MIRROR_URDF.is_file():`,
  `tools/export_fk_camera_extrinsic.py:59`) — a missing mirror is skipped silently, so
  its absence will not announce itself.
* If neither copy is migrated, `verify_fk_camera_projection.py` itself stops working,
  and with it every `render_*` tool. Training is unaffected either way.
* **`mjlib/` should not be copied** — it is a build artefact of a specific Python 3.13
  ABI (note the `cpython-313` `.so` files) and may not even be importable under the new
  cluster's interpreter. Rebuild it instead:
  ```bash
  pip install --target=/data/shichaojian/mjlib mujoco==3.12.0 lmdb
  ```
  The tools already search `ROOT / "mjlib"` then `/tmp/mjlib` (e.g.
  `tools/render_hand_surface.py:39`), so putting it back at
  `/data/shichaojian/mjlib` needs no code change.
* **`fk_calib/` need not exist.** It does not exist on the old cluster right now either;
  `run_calib.sh` does `mkdir -p "$OUT"` itself. Create it empty at
  `/data/shichaojian/fk_calib` if you run that script, or expect a one-time `mkdir`.
  Note `tools/calibrate_extrinsic_v2.py:347` uses a *different* scratch dir
  (`/tmp/fk_calib`) — that one is deliberate local scratch and can stay in `/tmp`.

**No code change is needed beyond the `sed`.** Every render/calibration tool resolves
these through a hardcoded storage root, not a repo-relative one:

```python
ROOT = Path("/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian")
MJLAB = ROOT / "wuji-mjlab"          # tools/render_hand_surface.py:34-35, and 5 siblings
```

Rule 2 of §2.2c rewrites that literal to `/data/shichaojian`, so `ROOT / "wuji-mjlab"`
and `ROOT / "mjlib"` both land on the intended destinations automatically. Verify with:

```bash
grep -rn 'ROOT = Path(' tools/ | grep -v '/data/shichaojian' ; echo "expect: no output"
```

### 2.3 Where they live, by file — worst first

| File | Count | Why it is hardcoded |
|---|---|---|
| `examples/launch_sft_action_policy_fk_singlerighthand_edge.sh` | 7 | the sibling venv path, dataset roots |
| `examples/launch_sft_action_policy_singlerighthand_nano.sh` | 6 | same |
| `examples/launch_sft_action_policy_singlerighthand_edge.sh` | 6 | same |
| `cosmos_framework/model/generator/fk_independent_schedule_test.py` | 6 | test fixtures point at the real cache |
| `cosmos_framework/data/generator/action/datasets/dataset_config_wiring_test.py` | 6 | same |
| `cosmos_framework/configs/base/experiment/action/posttrain_config/fk_displacement_scale_test.py` | 6 | same |
| `tools/run_calib.sh` | 4 | calibration assets |
| `tools/verify_fk_vs_pointcloud.py`, `tools/verify_fk_camera_projection.py`, `tools/render_fk_projection.py`, `tools/render_fk_joint.sh`, `tools/fk_gpu_smoke.py`, `tools/bench_fk_dataloader.py` | 3 each | `RAW_ROOT` / VAE constants at module top |
| ~28 more under `tools/` | 1–2 each | one `RAW_ROOT`-style constant each |
| `cosmos_framework/data/fk_camera_extrinsic.py` | 1 | the URDF path |

**Most of the `tools/` ones are a single module-level constant** — `RAW_ROOT`,
`VAE_DEFAULT`, `DEFAULT_ROOT` — near the top of the file. That is what to look for.

### 2.4 What is *already* configurable

The **training config** does not hardcode data paths; it reads them from the
environment via `oc.env`. Those you set in the launcher or the shell — see §6. The
hardcoded ones are concentrated in `tools/` and in **tests**.

> **A test that silently skips is worse than one that fails.** Several tests point
> at the real cache and will report success while doing nothing if the path is
> gone. After remapping, confirm each one actually ran (they print counts) rather
> than only checking the exit code.

---

## 3. Environment

### 3.1 Interpreter

The launchers pin an interpreter rather than inheriting `python` from `PATH`:

```bash
# examples/launch_sft_action_policy_fk_singlerighthand_edge.sh
SIBLING_VENV="/mnt/afs/WorldAct-cosmos3-edge-droid-sft/.venv"
PYTHON_BIN="${PYTHON_BIN:-$SIBLING_VENV/bin/python}"
```

**Update that path** (it is a worktree path, so it goes to `$HOME`, not
`/data/shichaojian` — see §2.2b). The venv was built for the sibling worktree; it carries
`cosmos_framework` as an **editable install whose `.pth` hard-codes that worktree's
path**. `PYTHONPATH=.` is what makes the current worktree's sources win — an
implicit precedence, which is why the launcher also imports `cosmos_framework` once
and errors out if it resolved outside the working directory. Do not remove that
check; without it, a launch from the wrong directory trains this worktree's config
against the sibling's code and the symptom is "my edits have no effect".

> **⚠️ `sed` does not fix the venv.** The editable install lives in the venv as a
> `.pth` / `__editable__*` file under
> `<venv>/lib/python3.*/site-packages/`, and it hardcodes the **old** sibling
> worktree path as an absolute string. Rewriting source files leaves it pointing at
> a directory that does not exist on the new cluster. Either rebuild the venv
> (`just install` at the new sibling worktree) or edit that file — and then re-run
> the check in §9 step 1, which is what catches it.

```bash
# find it
grep -rl "shared-storage-gpfs2" /mnt/afs/WorldAct-cosmos3-edge-droid-sft/.venv/lib/*/site-packages/ 2>/dev/null
```

### 3.2 Always `PYTHONPATH=.`

Running anything directly:

```bash
PYTHONPATH=. <venv>/bin/python tools/<script>.py
```

Without it, `python tools/foo.py` puts `tools/` on `sys.path[0]` and the editable
install wins for `cosmos_framework` — you silently execute the sibling worktree's
code.

### 3.3 Container / CUDA

In an NGC or PyTorch container, `LD_LIBRARY_PATH` must be cleared before any
`python` call or `torch._C` fails to import:

```bash
export LD_LIBRARY_PATH=''
```

The launcher sets thread limits (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`) and `TORCHINDUCTOR_*` — keep them.

### 3.4 Tests are scripts, not pytest

The venv has no `pytest-xdist`, and the repository's root `conftest.py` fails to
load without it, so `pytest` reports **"no tests ran"** — which looks like success.
Run the `*_test.py` files directly:

```bash
PYTHONPATH=. <venv>/bin/python cosmos_framework/model/generator/fk_joint_sampling_test.py
```

Each prints `PASS` and its own checks. See §9.

---

## 4. Repository and worktree layout

This repository is used as **several git worktrees**, each on its own branch.
On the new cluster they live under `$HOME` = **`/mnt/afs`**:

| Directory (under `/mnt/afs`) | Branch | Line |
|---|---|---|
| `WorldAct-cosmos` | `cosmos` | the main checkout |
| `WorldAct-cosmos3-edge-droid-sft` | `cosmos3-edge-droid-sft` | **PointFlow / point cloud**; also owns the venv |
| `WorldAct-cosmos3-edge-droid-sft_mano` | same name | **the FK line — this document** |
| `WorldAct-cosmos3-edge-droid-sft_base` | same name | base; this doc set lives in its `docs/` |
| `WorldAct-cosmos3-edge-droid-sft-tac` | same name | tactile |
| `WorldAct-lingbot-va*` | same names | a different model line |

Everything those runs *read and write* — `raw_data`, `datasets`, `models`, `runs`,
`renders` — is on the data volume, **`/data/shichaojian/`**, not under `$HOME`.
Keeping the two apart is the point: source and venv on `$HOME`, bulk storage on the
data volume.

The two that matter here: **`_mano` (FK code and the design doc)** and
**`-sft` (the venv and the PointFlow reference)**.

```bash
git worktree list          # from any of them
```

### 4.1 Documents, by location

| Document | Path (relative to the `_mano` worktree root) |
|---|---|
| **Design of record** | `docs/fk_modality_design.md` |
| Problem summaries | `docs/fk_summary_1.md` |
| Implementation view of FK | `../WorldAct-cosmos3-edge-droid-sft_base/docs/action_policy_fk_singlerighthand_edge_posttrain.md` |
| This setup guide | `../WorldAct-cosmos3-edge-droid-sft_base/docs/fk_new_cluster_setup.md` |
| Repo-wide conventions | `AGENTS.md` |
| Training / inference / setup | `docs/training.md`, `docs/inference.md`, `docs/setup.md`, `docs/faq.md` |

### 4.2 Agent memory

Sessions keep notes under `~/.claude/projects/<mangled-worktree-path>/memory/` —
on the new cluster, `/mnt/afs/.claude/projects/…`. Those live in `$HOME`, **not**
in the repository, so they do **not** travel with the code. On a new cluster they
are gone. This document and the two FK docs above
are what replaces them; if you are an agent reading this on a new cluster, treat
`docs/fk_modality_design.md` plus the implementation doc as the memory you do not
have.

---

## 5. Data: what to copy, what to rebuild

### 5.1 What must exist

All under `/data/shichaojian/` on the new cluster.

| Item | Location | How to get it |
|---|---|---|
| **Raw episodes** | `raw_data/singlerighthand_sandwich_100/` | **copy** from the old cluster. ~101 episode directories, each with `videos/head.mp4`, `videos/right_wrist.mp4`, and state/action arrays |
| **FK annotations** | `raw_data/sandwich_fk21/` | **copy**. The 21-keypoint labels, one directory per episode. This is the modality's ground truth and cannot be regenerated from the videos alone |
| **Cosmos cache** | `datasets/singlerighthand-sandwich-100-cosmos-cache/` | **copy** if storage allows, else rebuild with `tools/prepare_singlerighthand_raw.py` |
| ↳ `manifest.json`, `video_manifest.json` | same | written by the prep tool; schema-checked on load |
| ↳ `episodes/*.npz` | same | state and action arrays |
| ↳ `video_frames/` | same | decoded pixels |
| ↳ `vae_window_latents/*.npy` | same | **the VAE latents the model actually trains on.** uint16 holding bfloat16 bits, `(n_windows, 48, 9, 46, 34)` per episode. Rebuilding needs a GPU and is slow |
| **Base DCP checkpoint** | `models/cosmos3-edge-droid-dcp/` | copy, or convert from the released checkpoint with `python -m cosmos_framework.scripts.convert_model_to_dcp` |
| **Wan2.2 VAE** | `models/cosmos3-edge-droid/vae/Wan2.2_VAE.pth` | copy or download |
| **Edge-DROID processor bundle** | `EDGE_DROID_MODEL_PATH` | copy. The launcher checks for `processor_config.json`, `preprocessor_config.json`, `video_preprocessor_config.json`, `tokenizer.json`, `chat_template.jinja` and refuses to start without them |
| **Episode allowlists** | `examples/pointflow_sandwich_10_episodes.txt`, `examples/singlerighthand_101_episodes.txt` | **in the repo**, travels with the code |
| **Training outputs** | `runs/cosmos/` | copy only if you need the existing checkpoints; otherwise regenerate |
| *(optional)* `wuji-mjlab/` | `wuji-mjlab/` | **copy if you may recalibrate** — it is self-contained (URDF + meshes + replay reader) and covers every tool without migrating `WorldAct-lingbot-va-mano`. Training and eval never touch it — see §2.2d |
| *(optional)* `mjlib/` | `mjlib/` | **do not copy** — rebuild with `pip install --target=… mujoco==3.12.0 lmdb`. See §2.2d |

**Copy the FK annotations before anything else.** Everything downstream is
derivable from them; they are not derivable from anything.

### 5.2 Rebuilding the cache

The cache is the expensive part. Build order:

```bash
# 1. state/action + manifests from the raw episodes
PYTHONPATH=. <venv>/bin/python tools/prepare_singlerighthand_raw.py ...

# 2. decoded video frames (optionally limited to an episode set)
PYTHONPATH=. <venv>/bin/python tools/prepare_singlerighthand_video_cache.py \
    --episode-allowlist examples/singlerighthand_101_episodes.txt

# 3. VAE window latents -- needs a GPU
```

The dataset **validates the manifest against its own settings** (fps,
chunk_length, sample_stride) and refuses a stale cache rather than misreading it.
If you change any of those, regenerate rather than patch.

> **The cache is not index-stable across a change of `sample_stride` or episode
> set.** A dataset whose length changes re-picks its `np.linspace` eval candidates,
> so `fixed_cases` selects different windows. That is why `fixed_cases.json` is
> written down and re-checked (§6.3) and why **two runs can disagree on `val_00`
> without either being wrong.**

### 5.3 Verify the annotation coverage

```bash
PYTHONPATH=. <venv>/bin/python tools/verify_fk21_allowlist.py
```

An episode that slipped out of the annotation set shows up here as itself, rather
than later as a silently low-motion eval case.

---

## 6. Configuration reference

### 6.1 Data and model paths (set these in the environment)

| Variable | Points at |
|---|---|
| `SINGLERIGHTHAND_RAW_ROOT` | the raw episode tree |
| `SINGLERIGHTHAND_CACHE_ROOT` | the cosmos cache |
| `SINGLERIGHTHAND_VAE_WINDOW_LATENT_ROOT` | the window latent cache |
| `FK_ANNOTATION_ROOT` | the FK annotation tree |
| `BASE_CHECKPOINT_PATH` | the base DCP checkpoint |
| `EDGE_DROID_MODEL_PATH` | the Edge-DROID processor bundle |
| `SINGLERIGHTHAND_EPISODE_ALLOWLIST` | which episodes this run uses |
| `IMAGINAIRE_OUTPUT_ROOT` | **where the run writes.** The trainer reads this one — see §6.4 |
| `SINGLERIGHTHAND_USE_VIDEO_CACHE` | `true`/`false` |
| `SINGLERIGHTHAND_VIDEO_DECODER` | frame decoder backend |
| `COSMOS_GPU_VIDEO_AUGMENTATION` | augmentation toggle |
| `WANDB_MODE` | `offline` on an air-gapped cluster |

### 6.2 Eval and rollout switches

| Variable | Default | Effect |
|---|---|---|
| `FK_EVAL_EVERY` | `1` | run the FK eval every N validation passes |
| `FK_EVAL_CASES` | `2` | cases per split |
| `FK_EVAL_STEPS` | `4` | sampler steps |
| `FK_EVAL_SAMPLER` / `FK_EVAL_COMPARISON_SAMPLER` | `unipc` / `euler` | the second solver exists so one eval answers "was the solver the constraint" |
| `FK_EVAL_VIDEO` | `true` | render `comparison.mp4` in-process |
| `FK_EVAL_FIGURES` | **`false`** | draw the matplotlib figures during training. Off because they cost ~0.35 s/step; render later with `tools/render_fk_case.py` |
| `FK_VAL_ON_START` | — | validate at the resume point |
| `FK_EVAL_JOINT` | `false` | the joint arm: video from noise, FK alongside |
| `FK_EVAL_JOINT_STEPS` | = `FK_EVAL_STEPS` | |
| `FK_EVAL_JOINT_ACTION` | `false` | **the deployed configuration**: action denoised too |

| Variable | Default | Effect |
|---|---|---|
| `FK_ROLLOUT` | `false` | tile a val episode's middle seconds and sample each window |
| `FK_ROLLOUT_SECONDS` | `16` | clip length; `0` = the whole episode (~40 s, ~2.5× the forwards) |
| `FK_ROLLOUT_EVERY` | `10` | in **validation passes**, not steps |
| `FK_ROLLOUT_GENERATED` | `false` | generate video per window (two-panel output) |
| `FK_ROLLOUT_JOINT_ACTION` | — | as above, for the rollout |
| `FK_ROLLOUT_STEPS` / `FK_ROLLOUT_EPISODES` | = `FK_EVAL_STEPS` / `2` | |

### 6.3 `scheduler.cycle_lengths` must track `max_iter`

**Not optional.** `LambdaWarmUpCosineScheduler.find_in_interval` returns `None`
past the last cycle, and `schedule` then indexes `lr_warm_up_steps[None]`:

```
omegaconf.errors.KeyValidationError: ListConfig indices must be integers or slices, not NoneType
```

Raising `max_iter` alone trains to the old bound and **then dies on the next step**.
A run of this recipe was lost at step 10001 to exactly that. `tools/run_fk_101.sh`
pins both from one variable for this reason; keep that.

`fixed_cases.json` is also written by the first eval and **re-checked on every
resume**: a run that silently evaluated different windows would look like a
regression or a jump. Reusing an output directory for a *different* dataset is
guaranteed to abort with `Fixed eval identity changed` — start a new directory.

### 6.4 `IMAGINAIRE_OUTPUT_ROOT`, not `OUTPUT_ROOT`

`cosmos_framework/utils/config.py` reads **`IMAGINAIRE_OUTPUT_ROOT`**.
`_sft_launcher_common.sh` sets it from `OUTPUT_ROOT` **only if it is unset** —
`${IMAGINAIRE_OUTPUT_ROOT:-$OUTPUT_ROOT}`. If the ambient environment already
exports it, `OUTPUT_ROOT` is ignored and the run lands somewhere you did not
choose. This is how two runs once overwrote each other's checkpoints.

Both launchers set both variables. Keep that.

---

## 7. Architecture in one page

```
PackedSequence
├── vision      [B, C, T, H, W]                 ─┐
├── action                                       ├──> one denoise() pass,
├── fk_data     labels + anchor geometry          │    one attention
├── fk          learned FK tokens                 │
└── fk_noised   noisy state + sigma              ─┘
             ↓
      denoise() -> preds_vision, preds_action, preds_fk
```

FK is a **sibling field**, not a second model. Modules:

| Concern | File |
|---|---|
| annotations → camera-frame metres | `cosmos_framework/data/generator/action/fk_source.py` |
| base→camera transform | `cosmos_framework/data/fk_camera_extrinsic.py` |
| ragged per-sample windows | `cosmos_framework/data/fk_window.py`, `fk_batch.py` |
| token layout | `cosmos_framework/model/generator/fk_sequence.py` |
| the branch | `cosmos_framework/model/generator/fk_branch.py` |
| noising / loss / ADE | `cosmos_framework/model/generator/fk_training.py` |
| sigma grid, joint layout | `cosmos_framework/model/generator/fk_sampling.py` |
| training + inference arms | `cosmos_framework/model/generator/omni_mot_model.py` |
| install into the network | `cosmos_framework/model/generator/mot/cosmos3_vfm_network.py` |
| eval | `cosmos_framework/callbacks/fk_{eval,eval_cases,visualize,rollout}.py` |
| recipe | `cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_singlerighthand_edge.py` |

**Read [`action_policy_fk_singlerighthand_edge_posttrain.md`](../docs/action_policy_fk_singlerighthand_edge_posttrain.md) before interpreting any number.**
The short version, which is the thing most easily got wrong:

> The default eval arm hands the model the **entire 33-frame video including the 32
> future frames**. It is a legitimate ceiling and it is **not the deployed
> configuration** — its numbers do not predict the robot. Only the joint arms
> (`FK_EVAL_JOINT`, and `FK_EVAL_JOINT_ACTION` for the full deployed form) generate
> the video and action instead.

---

## 8. Commands

All from the `_mano` worktree root.

```bash
# ---- data ----
PYTHONPATH=. <venv>/bin/python tools/verify_fk21_allowlist.py
PYTHONPATH=. <venv>/bin/python tools/scan_fk_displacement_scale.py \
    --episodes examples/singlerighthand_101_episodes.txt

# ---- train ----
bash tools/run_fk_10_2.sh       # 10 episodes
bash tools/run_fk_101.sh        # 101 episodes

# ---- evaluate an existing checkpoint, no training ----
JOINT_ACTION=1 \
SOURCE_RUN=fk-singlerighthand-edge-10 OUT=fk-joint-eval-10 ITER=9500 \
ALLOWLIST=$PWD/examples/pointflow_sandwich_10_episodes.txt \
bash tools/run_fk_joint_eval.sh

# ---- render (needs a GPU: the generated panel is a VAE decode) ----
ARM=joint_action ROOT=<eval root> bash tools/render_fk_joint.sh

# ---- draw the fixed-case figures offline, for any past step ----
PYTHONPATH=. <venv>/bin/python tools/render_fk_case.py \
    --eval-dir <run>/.../fk_eval --all --video
```

`tools/run_fk_joint_eval.sh` symlinks the source run's checkpoint into a fresh
output root and pins `max_iter` to its own iteration, so the start validation runs
and nothing else does. It **refuses to start on a `fk_displacement_scale`
mismatch** — the sampler multiplies by whatever the *config* says, not by what the
checkpoint trained under, so a mismatch silently rescales every prediction.
`SCALE=` is the override.

---

## 9. Verification checklist

Run these **in order** on the new cluster. Each one is cheap; skipping one moves the
failure somewhere more expensive.

**Step 0 — paths.**

```bash
grep -rn "/mnt/shared-storage-gpfs2" --include="*.py" --include="*.sh" \
    --include="*.toml" cosmos_framework/ tools/ examples/ docs/
# expect: no output
```

**Step 1 — the interpreter resolves where you think.**

```bash
PYTHONPATH=. <venv>/bin/python -c "import cosmos_framework, os; print(os.path.realpath(cosmos_framework.__file__))"
# expect: a path under THIS worktree, not the sibling
```

**Step 2 — the unit tests.** They are scripts, not pytest:

```bash
cd cosmos_framework/model/generator
for t in fk_joint_sampling_test fk_sequence_test fk_branch_test fk_independent_schedule_test; do
    PYTHONPATH=<repo root> <venv>/bin/python $t.py
done
cd -
PYTHONPATH=. <venv>/bin/python cosmos_framework/data/fk_batch_test.py
PYTHONPATH=. <venv>/bin/python cosmos_framework/callbacks/fk_eval_test.py
PYTHONPATH=. <venv>/bin/python tools/render_fk_projection_test.py
```

Every one must print `PASS` **and** its checks — a test that prints only `PASS`
after silently skipping a missing path is a failure, not a pass. The projection test
in particular prints a keypoint count (e.g. `672/672 GT keypoints inside 640x480`);
if that line is absent it skipped.

**Step 3 — the data loads.** One dataset construction, no GPU:

```bash
PYTHONPATH=. <venv>/bin/python -c "
from cosmos_framework.callbacks.fk_eval_cases import split_dataset
import os
os.environ.setdefault('SINGLERIGHTHAND_EPISODE_ALLOWLIST', 'examples/pointflow_sandwich_10_episodes.txt')
print('constructing datasets...')"
```
then the real test is the first eval of any run: it writes `fk_eval/fixed_cases.json`
and four case directories. **If `fixed_cases.json` exists but the case directories
are empty, the run died during the first forward** — check the log, not the eval.

**Step 4 — one forward on a GPU.** `tools/fk_gpu_smoke.py` covers the dtype/device
paths a CPU run never exercises:

```bash
PYTHONPATH=. <venv>/bin/python tools/fk_gpu_smoke.py
```

**Step 5 — a real training step.** Launch, confirm the log line

```
Iteration N: Total Loss: ... | Video Loss: ... | Action Loss: ... | FK Loss: ... | FK ADE: ...mm (zero ...mm)
```

`FK ADE` next to `zero` is the pair that matters; the FK *loss* number is not
comparable across runs.

**Step 6 — the eval and the render.**

```bash
bash tools/run_fk_joint_eval.sh          # with the env from §8
bash tools/render_fk_joint.sh
```

Open `projection.mp4` and **check the green skeleton on the LEFT panel first**. If
GT is not on the hand, the projection or the canvas crop is wrong and nothing else
in the picture means anything.

---

## 10. Cluster-specific gotchas

These are environment properties the code works around, not bugs. On a new cluster
re-check each one rather than assuming it carried over.

| Symptom | Cause | What to do |
|---|---|---|
| `torch._C` ImportError in a container | `LD_LIBRARY_PATH` inherited from the NGC image | `export LD_LIBRARY_PATH=''` before `python` |
| `pytest` says "no tests ran" | the root `conftest.py` needs `pytest-xdist`, absent from the venv | run the `*_test.py` files directly (§9) |
| `OSError: [Errno 19] No such device` on `np.load(..., mmap_mode="r")` | GPFS/virtiofs on the old mount does not support mmap | read the `.npy` **header** when only the shape is needed; the old cluster did this |
| `import torch` takes minutes | virtiofs latency, not a deadlock | wait; do not `grep -rn` the whole repo concurrently |
| Edits have no effect after a launch | the venv's editable `cosmos_framework` resolved to the sibling worktree | run the launcher from its own worktree root; the launcher now checks |
| A pasted command fails with `-bash: eval/step_...: No such file or directory` | the terminal wrapped a ~150-character path and bash read the tail as a command | use the launcher scripts — they take **no path arguments** for this reason |
| `Disk quota exceeded` mid-run | checkpoints are 17–30 GB each and there is no retention policy | prune old `checkpoints/iter_*` during the run; the launcher prints the command |
| `Fixed eval identity changed; use a new output directory` | the output directory was reused for a different dataset | start a new directory |

### 10.1 Disk

A checkpoint is **17–30 GB** (with optimizer state, more). Eval and render
artefacts are tens of MB. Plan for the checkpoints; budget the rest as noise.

`tools/run_fk_retrain.sh` prints the prune command but does not enforce a limit —
retention is the operator's call.

---

## Where to go next

| Question | Document |
|---|---|
| What is this modality and why is it shaped this way? | [`action_policy_fk_singlerighthand_edge_posttrain.md`](../docs/action_policy_fk_singlerighthand_edge_posttrain.md), and the design doc in the `_mano` worktree: `docs/fk_modality_design.md` |
| How do I train the base action policy (no FK)? | `docs/action_policy_singlerighthand_edge_posttrain.md` |
| How does the framework fit together? | `docs/code_structure.md`, `AGENTS.md` |
| Multi-node, parallelism, mixed precision | `docs/training.md` |
| Install, containers, checkpoints | `docs/setup.md` |
| Something is broken | `docs/faq.md` |
