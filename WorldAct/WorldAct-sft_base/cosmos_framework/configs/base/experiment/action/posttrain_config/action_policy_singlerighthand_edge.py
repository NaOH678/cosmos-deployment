# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cosmos3-Edge-DROID policy SFT for the single-right-hand sandwich data."""

import copy
import os

from hydra.core.config_store import ConfigStore

from cosmos_framework.callbacks.fk_eval import FKEvalCallback
from cosmos_framework.callbacks.fk_rollout import FKRolloutCallback
from cosmos_framework.callbacks.gpu_video_augmentation import GPUVideoAugmentation
from cosmos_framework.callbacks.pointflow_eval import PointFlowEvalCallback
from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_droid_nano import (
    action_policy_droid_nano,
)
from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import (
    get_action_singlerighthand_raw_sft_dataset,
)
from cosmos_framework.data.generator.joint_dataloader import limit_dataloader_worker_threads
from cosmos_framework.data.generator.processors import build_processor_lazy
from cosmos_framework.utils.lazy_config import LazyCall as L

action_policy_singlerighthand_edge = copy.deepcopy(action_policy_droid_nano)
action_policy_singlerighthand_edge["job"].update(
    project="cosmos3_action",
    group="action_sft",
    name="action_policy_singlerighthand_edge",
    wandb_mode="${oc.env:WANDB_MODE,disabled}",
)

model_config = copy.deepcopy(EDGE_MODEL_CONFIG)
# The TOML schema for [model.compile] only carries ``enabled`` and
# ``compile_dynamic``; compiled_region / max_autotune_pointwise are rejected as
# extra inputs.  EDGE_MODEL_CONFIG ships compiled_region="language" and
# max_autotune_pointwise=False, so tune them here rather than editing the shared
# edge model config that vision_sft_edge also deep-copies.
#
# compiled_region stays "language" (the EDGE_MODEL_CONFIG default).  Setting it
# to "all" -- which would additionally compile the five VFM head methods wrapped
# by parallelize_vfm_network.apply_compile -- does not finish on this recipe:
# inductor's tiling selection (simd.py get_first_compatible_tiling ->
# tiling_is_compatible -> _split_iteration_ranges) reduces to a sympy integer
# polynomial gcd over the head methods' index expressions, and that gcd recurses
# ~65 frames deep in dmp_zz_heu_gcd.  Measured: still inside that single call
# after 20 minutes, never reaching iteration 1 (baseline warmup: 52s).  The only
# knob that touches the path, config.triton.max_tiles=1, is global and would
# disable tiling for the 28 blocks that are 94% of the step and currently work.
# Don't retry without addressing the index expressions themselves.
#
# max_autotune_pointwise is also not worth setting here: in torch 2.10 it is a
# no-op for pointwise kernels.  config.triton.autotune_pointwise already defaults
# to True, so triton_heuristics takes its multi-config branch unconditionally;
# the flag only still changes the reduction heuristics, which this recipe already
# steers via TORCHINDUCTOR_PERSISTENT_REDUCTIONS=0 and
# TORCHINDUCTOR_MIX_ORDER_REDUCTION=0.  Measured: 6.51 s/iter vs a 6.45 s
# baseline over the same iterations, warmup 70s vs 62s.
model_config["tokenizer"]["encode_exact_durations"] = [33]
model_config["max_num_tokens_after_packing"] = -1
model_config["rectified_flow_training_config"]["loss_scale"] = 10.0

# PointFlow's clean sample is the point displacement in METRES, and the branch
# adds unit-variance noise to it (``pointflow_add_noise``: ``target =
# displacement / scale``, ``epsilon = randn(...)``).  At the inherited scale of
# 1.0 the measured displacement std is 0.0839 m against a noise std of 1.0 -- a
# 1:12 signal-to-noise ratio.  The branch can then drive ~99% of the
# flow-matching loss to zero by re-reading the noise off its own input, and the
# sampled trajectory degenerates into a scaled copy of the initial per-point
# noise, which renders as a fan of straight lines radiating from the point
# cluster.  Loss and ADE move in opposite directions as a result: train
# pointflow loss fell 0.237 -> 0.012 while the eval ADE ratio to the
# "predict no motion" baseline rose 1.64x -> 3.75x.
#
# Every other modality in this pipeline is already unit-scale -- the video VAE
# latent measures std 0.872, and the action is normalised by ``quantile_rot`` --
# so this restores the convention rather than introducing a special case.
#
# The scale must follow the DATA and the SELECTION, because faster-moving points
# displace more.  Pooled per-element std of the valid displacement over the
# training set (tools/scan_pointflow_selection.py with the same PointFlowSource
# seed derivation training uses):
#   legacy 10-episode dense export, select_top_n=300, min_voxel_members=3:
#     std 0.0432 m, n=3,764,532
#     (2026-09-20, pointflow_outputs/selection_scan_top300_20260920.json)
#   labeled 29-episode delivery (hand/object only, no background), same selection:
#     std 0.0751 m, n=5,641,299  (select_min_valid_steps=0)
#     std 0.0740 m, n=6,056,661  (select_min_valid_steps=16)
#     (2026-09-21, pointflow_outputs/selection_scan_labeled29_top300_mvs{0,16}_20260921.json)
#   The labeled std is ~1.7x the legacy one: every candidate is a hand/object
#   point, so the top-300 motion ranking no longer dilutes itself with a static
#   background.  --top-n 300 --min-voxel-members 3 [--min-valid-steps 16]
# A std rather than a q01/q99 span, because the noise is unit-variance Gaussian
# and the distribution measures std/robust <= 1, i.e. not tail-inflated, so
# extreme tracking outliers do not inflate it.
#
# Checkpoints trained with a different scale cannot be resumed: both the target
# and the scale the branch learned changed.  After a scale change the pointflow
# loss READS differently and its "predict zero" baseline moves with it -- judge
# the branch by ADE against the zero-motion baseline, never by the loss value.
# 0.0740 pairs with manifest_sandwich_labeled_20260921.json + top300 + mvs16;
# the legacy manifest needs 0.0432.
# 0.075255 m = pooled per-element std of the displacement this run's selection
# produces: manifest sandwich_924_20260928 + select_top_n=300, with
# min_voxel_members / select_min_valid_steps at 0, over n=22,595,703 elements
# (recomputed from pointflow_outputs/sandwich_924_20260928/scale_scan_top300.json).
# This replaced 0.0740, which belonged to the older labeled-29 delivery -- that
# manifest's pointflow_source entries are now null, so it can no longer be run.
# tools/run_fk_point_101.sh refuses to launch if this value and its own expectation
# disagree.  If the manifest or any selection knob changes, RE-MEASURE: the scale is
# a property of the selection, and a wrong one produces no shape error at all --
# only predictions that are uniformly scaled.
model_config["rectified_flow_training_config"]["pointflow_displacement_scale"] = 0.075255

# PointFlow must see "clean video + noisy points" in training: the conditional-fit
# evaluation (and any deployment observing the current video) presents exactly that
# combination, but the legacy shared schedule always pairs noisy points with equally
# noisy video.  Measured on the scale-0.0432 per-point run (2026-09-20): eval UniPC-4
# ADE/zero rose monotonically (1.0-2.6 @step100 -> 1.5-6.8 @step2500) while the
# training one-step ratio fell to ~0.3 -- a train/eval condition mismatch, not a
# fitting failure.  The independent draw keeps the sigma marginal identical and only
# removes the coupling.  Model shapes are unchanged, so runs resume across this flag.
# See docs/pointflow_per_point_tokens_20260919.md.
model_config["rectified_flow_training_config"]["independent_pointflow_schedule"] = False
# ^ 2026-09-29 决定：训练回到原生 Cosmos 的【全共享 σ】——四模态在同一个
#   σ 下加噪，与四模态联合去噪推理的部署条件一致。独立 σ 当初是为了让
#   『条件 arm』（干净视频 + 噪声模态）在训练分布里存在；那个条件只在诊断
#   arm 里出现，不在部署形态里。代价：条件 arm 重新变成分布外，其数字不再
#   可比（它本来就是上限、不是部署性能）。


# FK's clean target must be unit-scale, for the same reason PointFlow's must be:
# the RF forward process is ``x_sigma = sigma*eps + (1-sigma)*target`` with
# ``eps`` unit-variance Gaussian, so a target 13x smaller than the noise leaves
# the branch training almost entirely on the noise term.  Every other modality
# here is already unit-scale -- the video VAE latent measures std 0.872, the
# action is normalised by ``quantile_rot`` -- so this restores the convention
# rather than introducing a special case.
#
# 0.083745 is the pooled per-element std of the window displacement over the 101
# training episodes, n=202,138,272, measured in the camera frame FK actually
# labels in.  Reproduce:
#     python tools/scan_fk_displacement_scale.py \
#         --episodes examples/singlerighthand_101_episodes.txt
#
# The 10-episode allowlist this recipe started on measured 0.074450 (n=26,522,496).
# The two are 12% apart because the wider set carries larger hand motion -- the
# docstring's point, that this is a property of the *selection*: change the
# allowlist and it has to be re-measured, and a checkpoint trained under one value
# cannot be resumed onto the other.
#
# Dropping the default 1.0 is a real change to the branch's problem, not a
# rescaling of the report: checkpoints trained under 1.0 cannot be resumed onto
# this.  After it the FK loss will READ HIGHER and its "predict zero" baseline
# grows -- judge the branch by ADE against that baseline, never by the loss
# value.
#
# Env override FK_DISPLACEMENT_SCALE switches the dataset (the value is a property
# of the data selection, not the model): dropper 101-episode measures 0.040438
# (n=204,097,824; tools/scan_fk_displacement_scale.py --root .../dropper_fk21
# --episodes examples/singlerighthand_dropper_101_episodes.txt).
model_config["rectified_flow_training_config"]["fk_displacement_scale"] = float(
    os.environ.get("FK_DISPLACEMENT_SCALE", "0.083745")
)

# Decouple FK's noise level from the video's.  Sharing one sigma per sample lets the
# model learn "the video is clean => sigma ~ 0 => the answer is the input unchanged",
# and the eval's condition (clean video + FK from pure noise) is precisely what fires
# that shortcut: the sampled trajectory collapses onto its own initial noise, which is
# the "scattered, no hand shape" failure.  Drawing FK's sigma independently keeps the
# marginal over sigma identical and puts that condition back in the training
# distribution.  The PointFlow branch hit the identical bug and fixed it this way
# (independent_pointflow_schedule, docs/pointflow_per_point_tokens_20260919.md).
model_config["rectified_flow_training_config"]["independent_fk_schedule"] = False
# ^ 2026-09-29 决定：训练回到原生 Cosmos 的【全共享 σ】——四模态在同一个
#   σ 下加噪，与四模态联合去噪推理的部署条件一致。独立 σ 当初是为了让
#   『条件 arm』（干净视频 + 噪声模态）在训练分布里存在；那个条件只在诊断
#   arm 里出现，不在部署形态里。代价：条件 arm 重新变成分布外，其数字不再
#   可比（它本来就是上限、不是部署性能）。
model_config["vlm_config"]["tokenizer"] = L(build_processor_lazy)(tokenizer_type="${oc.env:EDGE_DROID_MODEL_PATH}")
model_config["vlm_config"]["pretrained_weights"].update(
    enabled=False,
    backbone_path="",
    credentials_path="",
    enable_gcs_patch_in_boto3=False,
)
action_policy_singlerighthand_edge["model"]["config"] = model_config

# Edge uses an additional K-normalization parameter on the generation pathway.
optimizer_keys = action_policy_singlerighthand_edge["optimizer"]["keys_to_select"]
if "k_norm_und_for_gen" not in optimizer_keys:
    optimizer_keys.append("k_norm_und_for_gen")

# PointFlow trains end to end: the Sonata geometry encoder, the point codec and
# its token/sigma embeddings all live under ``net.pointflow_branch``.  The
# optimizer allowlist freezes any parameter whose name matches no key here, so
# without this entry the whole branch receives no gradient.
if "pointflow_branch" not in optimizer_keys:
    optimizer_keys.append("pointflow_branch")

# The codec is initialized from scratch while every other trained tensor comes
# from a pretrained checkpoint, so the base LR (2e-5) only moves its weights by
# ~lr per step and the branch needs tens of thousands of steps to emit the
# unit-scale velocity.  Give it the same treatment the Nano recipe gives its
# action heads, scaled for that codec being new rather than fine-tuned.
# Lower this to 10.0 if Video Loss starts degrading: the PointFlow gradient also
# flows into the shared trunk.
action_policy_singlerighthand_edge["optimizer"]["lr_multipliers"]["pointflow_branch.codec"] = 25.0

# Edge-DROID already contains trained action heads. Preserve all of them when
# loading the converted DCP; only EMA is warm-started from the regular network.
action_policy_singlerighthand_edge["checkpoint"]["keys_to_skip_loading"] = ["net_ema."]
action_policy_singlerighthand_edge["checkpoint"]["enable_gcs_patch_in_boto3"] = False

# Which components a resume may load.  Empty -- the default -- means all of them,
# which is what a run resuming its own latest checkpoint wants and what training
# always uses.
#
# tools/run_fk_joint_eval.sh sets this when it evaluates a checkpoint that has been
# pruned to ``model/`` alone.  These runs delete the training state of older
# iterations as they go, keeping the weights; a resume, though, asks for all of
# CHECKPOINT_KEYS unconditionally (utils/checkpoint/dcp.py:737) and a component that
# is not in the directory does not produce a message naming it -- the reader returns
# no metadata and torch reports ``AssertionError: metadata is None``, with the real
# reason ("Global metadata is not found") swallowed by a bare ``except Exception``
# (torch/distributed/checkpoint/state_dict_loader.py:229-244).
#
# Unlike the three flags above, this one applies on the RESUME path as well
# (dcp.py:777 sits outside the load_path branch), so it narrows the key set without
# switching to a warm start -- which matters, because a warm start would also apply
# ``keys_to_skip_loading`` and so evaluate weights other than the checkpointed ones.
#
# The cost is the iteration: ``trainer`` is the only component that carries it
# (dcp.py:920), so dropping it makes the load report 0.  The eval pins max_iter to 0
# to match; the loop tests ``iteration >= max_iter`` AFTER the start validation
# (trainer/__init__.py:244 then :288), so the eval still runs and no step does.
action_policy_singlerighthand_edge["checkpoint"]["keys_not_to_resume"] = [
    key for key in os.environ.get("FK_SKIP_RESUME_KEYS", "").split(",") if key
]

# FK trains end to end and has no pretrained part at all: the index embeddings,
# the position MLP and the decoder are all new.  The optimizer allowlist freezes
# any parameter whose name matches no key here, so without this entry the branch
# receives no gradient -- and the failure is silent, since the rest of the model
# keeps training normally.
_fk_optimizer_keys = action_policy_singlerighthand_edge["optimizer"]["keys_to_select"]
if "fk_branch" not in _fk_optimizer_keys:
    _fk_optimizer_keys.append("fk_branch")

# At the base LR (2e-5) freshly-initialized weights move by ~lr per step and need
# tens of thousands of steps to emit a unit-scale velocity, so the whole branch
# gets the boost the PointFlow codec gets.  Lower this to 10.0 if Video Loss
# starts degrading: the FK gradient also flows into the shared trunk.
action_policy_singlerighthand_edge["optimizer"]["lr_multipliers"]["fk_branch"] = 25.0

action_policy_singlerighthand_edge["trainer"]["callbacks"]["gpu_video_augmentation"] = L(GPUVideoAugmentation)(
    enabled="${oc.env:COSMOS_GPU_VIDEO_AUGMENTATION,false}",
    crop_ratio=0.95,
    brightness=0.3,
    contrast=0.4,
    saturation=0.5,
    hue=0.08,
    chunk_size=8,
)
action_policy_singlerighthand_edge["trainer"]["callbacks"]["pointflow_eval"] = L(PointFlowEvalCallback)(
    every_n="${oc.env:POINTFLOW_EVAL_EVERY,1}",
    max_points=512,
    # 4 steps, the same solver (UniPC) and the same shift as the video/action path --
    # the point tokens are denoised inside that loop in the deployed system, so the
    # diagnostic must not integrate them differently.  comparison_steps gives a second
    # UniPC trajectory to see whether the step count is even the binding constraint.
    max_cases=2,
    sampling_steps=4,
    comparison_steps=16,
    seed=42,
    sampler="${oc.env:POINTFLOW_EVAL_SAMPLER,unipc}",
    comparison_sampler="${oc.env:POINTFLOW_EVAL_COMPARISON_SAMPLER,euler}",
    # Diagnostic: sample each case again with one conditioning channel neutralized.
    # The eval hands the model the ground-truth future video/action, so a prediction
    # that does not move when a channel is blanked is not reading it.  Modes:
    # action (zero GT action), video (black out all frames), first_frame (freeze the
    # window to frame 0 -- tests whether FUTURE video frames are read at all; in wam
    # training they are noisy targets, never clean context).
    ablate_action="${oc.env:POINTFLOW_ABLATE_ACTION,false}",
    ablate_modes="${oc.env:POINTFLOW_ABLATE_MODES,}",
    # Route B: also evaluate the deployment-shaped joint rollout (point tokens
    # denoise inside the main video/action loop; first frame + state action clean,
    # future video/action/point jointly sampled).  Writes <case>_joint/ outputs.
    joint_rollout="${oc.env:POINTFLOW_EVAL_JOINT,false}",
    # Diagnostic: sweep the one-step clean estimate x0_hat = x_sigma - sigma*v_hat over
    # a fixed sigma grid and log the ADE at each.  The sampled trajectory ends at
    # sigma = 0, so this shows where the direct estimate breaks -- one number at the
    # training sigma cannot.
    sigma_scan="${oc.env:POINTFLOW_SIGMA_SCAN,false}",
)
# Periodic FK fitting evaluation.  ``every_n`` counts *validation* calls, and the
# trainer runs validation every ``validation_iter`` (100) steps, so this produces a
# step_XXXXXXXXX/ directory every 100 steps.
action_policy_singlerighthand_edge["trainer"]["callbacks"]["fk_eval"] = L(FKEvalCallback)(
    # Main fk_eval artifacts now measure joint generation, at the same paths.
    # PointFlow joins automatically when the fixed case carries its payload.
    joint_only=True,
    every_n="${oc.env:FK_EVAL_EVERY,1}",
    max_cases="${oc.env:FK_EVAL_CASES,2}",
    # 4 steps, the same solver (UniPC) and the same shift as the video/action path --
    # the FK tokens are denoised inside that loop in the deployed system, so the
    # diagnostic must not integrate them differently.  comparison_steps gives a second
    # trajectory to see whether the step count is even the binding constraint.
    sampling_steps="${oc.env:FK_EVAL_STEPS,4}",
    comparison_steps=16,
    seed=42,
    sampler="${oc.env:FK_EVAL_SAMPLER,unipc}",
    # joint_only uses UniPC for the reference too; this setting is conditional-only.
    comparison_sampler="${oc.env:FK_EVAL_COMPARISON_SAMPLER,euler}",
    make_video="${oc.env:FK_EVAL_VIDEO,true}",
    # Figures and MP4 are matplotlib, and the only CPU-heavy part of the eval:
    # ~128 renders per validation pass, ~0.4 s/step at validation_iter=100 on a
    # single GPU.  Off by default -- the eval still writes prediction.npz and
    # metrics.json (kilobytes, milliseconds), and tools/render_fk_case.py draws
    # the identical figures offline, for any past step.  Set FK_EVAL_FIGURES=true
    # to have them during training as before.
    make_figures="${oc.env:FK_EVAL_FIGURES,false}",
    # Legacy diagnostic-arm settings apply only with joint_only=False.
    # The primary joint mode above always generates future action and suppresses
    # these extra runs, keeping the existing root artifact paths.
    video_joint="${oc.env:FK_EVAL_JOINT,false}",
    video_joint_steps="${oc.env:FK_EVAL_JOINT_STEPS,${oc.env:FK_EVAL_STEPS,4}}",
    joint_action="${oc.env:FK_EVAL_JOINT_ACTION,false}",
)

# Whole-episode rollout of the val episodes, so the two figures below cover the
# whole recording rather than one window.  Registered always -- it returns
# immediately when disabled, so an ordinary run carries no cost -- but enabled only
# by FK_ROLLOUT=1, because it tiles an episode (~1200 frames) into ~37 windows and
# samples each: roughly 150 forwards per case against the fixed eval's 4.
#
# every_n counts VALIDATION EVENTS, not steps: the point of the gate is to spend the
# forwards rarely, and an iteration modulus would change cadence with validation_iter.
action_policy_singlerighthand_edge["trainer"]["callbacks"]["fk_rollout"] = L(FKRolloutCallback)(
    enabled="${oc.env:FK_ROLLOUT,false}",
    every_n="${oc.env:FK_ROLLOUT_EVERY,10}",
    sampling_steps="${oc.env:FK_ROLLOUT_STEPS,${oc.env:FK_EVAL_STEPS,4}}",
    sampler="${oc.env:FK_EVAL_SAMPLER,unipc}",
    seed=42,
    max_episodes="${oc.env:FK_ROLLOUT_EPISODES,2}",
    # Length of the clip, taken from the middle of the episode.  A whole episode is
    # ~40 s: long to watch, and ~2.5x the forwards (18 windows vs 8).  Set 0 for the
    # whole episode.  fps must match the dataset's -- see ``fps=15.0`` above; it is
    # what converts seconds into the model frames the windows are cut on.
    seconds="${oc.env:FK_ROLLOUT_SECONDS,16}",
    fps=15.0,
    # Joint arm per window: generate the video from noise and denoise FK alongside it,
    # instead of conditioning FK on the clean GT video.  This is what turns the
    # rollout into the two-panel form (left real, right generated) and what makes the
    # clip longer than one window -- one generated video per window, decoded offline
    # by tools/render_fk_rollout.py.  Sampling switch only; no retraining.
    generated_video="${oc.env:FK_ROLLOUT_GENERATED,false}",
    # Same extension as the fixed eval's: denoise the future action alongside the
    # video and FK, so each window's clip is the deployed configuration rather than
    # a clean-action approximation of it.  REQUIRES FK_ROLLOUT_GENERATED (it is
    # rejected on its own, like the eval's -- set both); writes to
    # ``fk_rollout_joint_action/`` beside the two-segment arm's ``fk_rollout/``,
    # because both are rollouts of the same episode and one would otherwise replace
    # the other.
    joint_action="${oc.env:FK_ROLLOUT_JOINT_ACTION,false}",
)

# One-off sampler diagnostic.  Off unless FK_SAMPLER_PROBE=1, so an ordinary run
# carries no extra callbacks: this samples and forwards the fixed case several
# extra times, which is worth it exactly once and not every validation.
if os.environ.get("FK_SAMPLER_PROBE", "").strip().lower() in {"1", "true", "yes", "on"}:
    from cosmos_framework.callbacks.fk_sampler_probe import FKSamplerProbeCallback

    action_policy_singlerighthand_edge["trainer"]["callbacks"]["fk_sampler_probe"] = L(FKSamplerProbeCallback)(
        every_n=1,
        max_cases=1,
        sampling_steps=int(os.environ.get("FK_SAMPLER_PROBE_STEPS", "4")),
        sampler=os.environ.get("FK_SAMPLER_PROBE_SAMPLER", "unipc"),
        grid_points=int(os.environ.get("FK_SAMPLER_PROBE_GRID", "13")),
    )

# Periodic validation is what makes a run readable while it is still going: without it
# the only signal is a training loss that can fall while the model learns nothing usable.
action_policy_singlerighthand_edge["trainer"].update(
    run_validation=True,
    # Validate before the first optimizer step.  With it off the earliest look at
    # the eval -- including the FK eval's rendered cases -- is `validation_iter`
    # steps in, so a defect anywhere in the eval path costs a full wait to surface
    # (it cost exactly that once: `validation_step` was a stub returning None and
    # the run died at step 100 with everything before it looking healthy).
    #
    # ``oc.decode`` because ``oc.env`` yields a STRING, and a bare
    # "${oc.env:FK_VAL_ON_START,false}" would be the truthy string "false" -- the
    # flag would read as on no matter what was passed.
    #
    # MERGE: the PointFlow and FK recipes each carried their own `trainer.update`
    # call; this is the single merged one.  `run_validation_on_start` was the one
    # key the two set to different values (PointFlow: False, FK: on by default).
    # It now defaults to False -- to validate at (re)start, including on resume,
    # pass either a tail override
    #   EXTRA_TAIL_OVERRIDES="trainer.run_validation_on_start=true"
    # or the environment variable below.
    run_validation_on_start="${oc.decode:${oc.env:FK_VAL_ON_START,false}}",
    validation_iter=100,
    max_val_iter=1,
)

action_policy_singlerighthand_edge["dataloader_train"]["dataset_name"] = "action_singlerighthand"
dataloader = action_policy_singlerighthand_edge["dataloader_train"]["dataloader"]
dataloader.update(
    batch_size=2,
    # Keep video/PointFlow decoding off the training critical path.  Two
    # persistent workers overlap CPU decode and host-to-device copies with
    # the GPU step; validation remains single-process below.
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    # Two batches in flight per worker (12 samples) already covers several steps
    # of consumption.  The earlier value of 4 kept 32 windows resident, and this
    # node's per-step allocation cost is dominated by page reclaim rather than by
    # arithmetic -- the same reason the FK batch buffers are recycled.
    prefetch_factor=2,
    # Each worker otherwise opens its own OpenCV/FFmpeg/torch thread pools and
    # oversubscribes this rank's CPU partition, which deschedules the trainer
    # during its per-step data marshalling (measured on this node: 2.5 ms of work
    # stretching to 230-265 ms of wall clock, with the GPU idle).
    worker_init_fn=limit_dataloader_worker_threads,
)
dataloader["datasets"] = {
    "singlerighthand": {
        "ratio": 1,
        "dataset": L(get_action_singlerighthand_raw_sft_dataset)(
            root="${oc.env:SINGLERIGHTHAND_RAW_ROOT}",
            cache_root="${oc.env:SINGLERIGHTHAND_CACHE_ROOT}",
            fps=15.0,
            chunk_length=32,
            split="train",
            split_seed=42,
            # 0.03 rounds to ZERO validation episodes on a 10-episode allowlist, which
            # leaves the val dataloader empty and the periodic eval with nothing to run.
            # 0.2 gives 2 of 10.
            split_val_ratio=0.2,
            sample_stride=1,
            mode="wam",
            use_state=True,
            iterable_shuffle=True,
            episode_shuffle_seed=42,
            shuffle_block_size=256,
            use_image_augmentation=False,
            use_precomputed_video="${oc.env:SINGLERIGHTHAND_USE_VIDEO_CACHE,false}",
            # torchcodec is not installed in this environment, so opencv is the default.
            video_decoder="${oc.env:SINGLERIGHTHAND_VIDEO_DECODER,opencv}",
            # Precomputed per-window VAE latents.  Without this every step re-runs
            # the frozen Wan2.2 VAE over every pixel frame of every window.  The
            # dataset validates the manifest's fps/chunk_length/sample_stride
            # against its own, so a stale cache is refused rather than misread.
            #
            # Only the window cache: the whole-episode cache is not equivalent
            # (see SingleRightHandRawDataset._read_window_latent).
            vae_window_latent_root="${oc.env:SINGLERIGHTHAND_VAE_WINDOW_LATENT_ROOT,}",
            # Legacy whole-episode cache: a whole-episode encode at source FPS that is
            # resampled onto the window's 15 Hz lattice.  Measured to differ from a
            # fresh encode by about one frame of window shift, and it changes the
            # identity of the conditioning latent (index 0).  Used only when the
            # window cache above is unset; prefer the window cache.
            vae_latent_root="${oc.env:SINGLERIGHTHAND_VAE_LATENT_ROOT,}",
            # Episode allowlist.  Two exist: the small-sample ten the PointFlow
            # recipe uses (examples/pointflow_sandwich_10_episodes.txt) and the
            # full set (examples/singlerighthand_101_episodes.txt, a superset).
            # Which one is in force is decided by the launcher's default and the
            # SINGLERIGHTHAND_EPISODE_ALLOWLIST environment, NOT here -- and it
            # has to match fk_displacement_scale above and the window-latent cache
            # below, or the branch trains on a different distribution than the one
            # its scale was measured on.
            #
            # Both of these resolve to the EMPTY STRING when unset, and the dataset
            # reads that as "off".  Do NOT append `or None`: it binds to the string
            # literal rather than to the substituted value, so it never fires and
            # only makes the line look like it handles the case.
            episode_allowlist="${oc.env:SINGLERIGHTHAND_EPISODE_ALLOWLIST,}",
            # FK annotations live in their own tree (raw_data/sandwich_fk21), not
            # beside the raw episodes.
            fk_root="${oc.env:FK_ANNOTATION_ROOT,}",
            fk_camera_profile="${oc.env:FK_CAMERA_PROFILE,legacy}",
            fk_steps_per_token=4,
            # Point tracks are sampled every 4th frame, matching the VAE's temporal
            # compression: one PointFlow block per video latent.
            video_temporal_downsample=4,
            pointflow_manifest="${oc.env:POINTFLOW_MANIFEST,/mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow/pointflow_outputs/sandwich_924_20260928/manifest.json}",
            pointflow_max_points=8192,
            pointflow_voxel_size=0.02,
            pointflow_seed=42,
            # GT-ranked restriction to the most-moving voxels. Diagnostic only: the
            # ranking reads future labels, so it cannot be reproduced at inference.
            # 0 keeps the full cloud.
            pointflow_select_motion_fraction="${oc.env:POINTFLOW_SELECT_MOTION_FRACTION,0}",
            # Keep exactly N points by the same ground-truth motion ranking, with NO
            # floor (the fraction above floors at 3).  Set to 1 to collapse the cloud to
            # a single point, which makes "one cluster token has to describe many points"
            # disappear and leaves the rest of the pipeline testable on its own.  Takes
            # precedence over the fraction.  0 disables it.
            pointflow_select_top_n="${oc.env:POINTFLOW_SELECT_TOP_N,0}",
            # Guard for every motion ranking: a point whose 2 cm voxel holds fewer than
            # this many points is ranked last.  A lost track is always alone -- measured
            # on this data, the ten fastest points of a window all sit in single-member
            # voxels and "move" up to 1.6 m, while the fastest point in a voxel of three
            # or more moves 0.15 m.  0 = off.
            pointflow_min_voxel_members="${oc.env:POINTFLOW_MIN_VOXEL_MEMBERS,0}",
            # Supervise only N points, taken from the most-moving voxel that satisfies
            # the guard, and mask every other point out of the loss.  The cloud is NOT
            # reduced, so the geometry encoder still sees a full surface: this separates
            # "the decode path cannot emit one point's motion" from "the geometry input
            # was degenerate".  0 supervises everything.
            pointflow_supervise_cluster_n="${oc.env:POINTFLOW_SUPERVISE_CLUSTER_N,0}",
            # Semantic region restriction for anchor candidates, read from the labeled
            # delivery's region_labels.npy (1=fingertip, 2=hand, 3+=objects; sandwich
            # uses 3=lettuce, 4=bread/cheese).  Anchor-time information only, so unlike
            # the motion ranking this stays a deployable conditioning rule.  Comma-
            # separated; empty keeps every region.
            pointflow_select_regions="${oc.env:POINTFLOW_SELECT_REGIONS,}",
            pointflow_select_region_quotas="${oc.env:POINTFLOW_SELECT_REGION_QUOTAS,}",
            # Demote points whose future label is valid for fewer than this many steps
            # in motion-ranked selection: a track that dies mid-window (late-stage
            # points leave the frame) cannot anchor a full-window prediction.  Reads
            # labels, so it is training-time only.  0 = off.
            pointflow_select_min_valid_steps="${oc.env:POINTFLOW_SELECT_MIN_VALID_STEPS,0}",
            pointflow_select_phantom_guard="${oc.env:POINTFLOW_SELECT_PHANTOM_GUARD,false}",
            pointflow_window_cache_root="${oc.env:POINTFLOW_WINDOW_CACHE_ROOT,}",
            viewpoint="concat_view",
            resolution="480",
            max_action_dim="${model.config.max_action_dim}",
            cfg_dropout_rate=0.1,
            tokenizer_config="${model.config.vlm_config.tokenizer}",
            format_prompt_as_json=True,
        ),
    }
}
# A separate validation loader.  batch_size=1 and num_workers=0 keep the eval
# deterministic and cheap: the eval callbacks re-instantiate this loader's dataset
# themselves to build their fixed cases, so anything stochastic here would make the
# reported numbers move for reasons unrelated to the model.
#
# MERGE: the PointFlow and FK recipes each carried an identical copy of this block
# under a different local name (`val_dataloader` / `_val_dataloader`); both assigned
# the same key, so the second was dead.  This is the single surviving copy -- if a
# later edit changes it, check that both `pointflow_eval` and `fk_eval` still get the
# deterministic loader they assume.
_val_dataloader = copy.deepcopy(action_policy_singlerighthand_edge["dataloader_train"])
_val_dataloader["dataset_name"] = "action_singlerighthand_val"
_val_dataloader["max_samples_per_batch"] = 2
_val_dataloader["dataloader"].update(batch_size=1, num_workers=0, persistent_workers=False, prefetch_factor=None)
_val_dataloader["dataloader"]["datasets"]["singlerighthand"]["dataset"]["split"] = "val"
_val_dataloader["dataloader"]["datasets"]["singlerighthand"]["dataset"]["iterable_shuffle"] = False
action_policy_singlerighthand_edge["dataloader_val"] = _val_dataloader

ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name="action_policy_singlerighthand_edge",
    node=action_policy_singlerighthand_edge,
)


__all__ = ["action_policy_singlerighthand_edge"]
