"""Periodic FK fitting evaluation: sample the future, report mm, render it.

The FK counterpart of ``pointflow_eval.py``.  Runs on every rank -- the sampler
needs a full FSDP forward, so gating it to rank zero would hang -- and only rank
zero writes.

The number this reports is ``all_ade_mm`` against ``zero_all_ade_mm``, the
stationary-trajectory baseline.  The masked MSE in the training loss cannot
answer the question the design poses: its scale is dominated by the
unit-variance noise term, so a prediction of zero already scores ~1.0 and a
falling loss says nothing about whether the motion is right.
"""

import copy
import json
import logging
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.callbacks import fk_visualize
from cosmos_framework.callbacks.fk_eval_cases import evaluation_rng, fixed_cases
from cosmos_framework.utils import callback, distributed, log, misc

logger = logging.getLogger(__name__)

DEFAULT_SAMPLER = "unipc"


def _numpy(value):
    """Host array from anything the sampler or the batch can hand back.

    The two sides of this callback live on different devices: ``cpu_batch`` is
    built on the host by ``fixed_cases``, while ``prediction`` and ``reference``
    come out of ``model.sample_fk`` on CUDA.  A bare ``np.asarray`` works for one
    and raises "can't convert cuda:0 device type tensor to numpy" for the other,
    so every value goes through here instead of assuming.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _as_bool(value):
    """Parse a flag that may arrive as a bool or as the string ``oc.env`` yields.

    ``bool("false")`` is True, so a plain ``bool()`` would turn every env-gated
    switch *on* the moment the variable is set to anything at all -- including
    ``false``.  Only the spelled-out truthy words count.
    """
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


class FKEvalCallback(callback.Callback):
    """Fixed cases; joint_only replaces conditional outputs at the existing paths.

    All ranks participate in sampling. Joint mode generates future video/action
    and any PointFlow payload together with FK, retaining the first-frame/state
    conditions. The optional conditional/extra-arm mode remains for other recipes.
    """

    def __init__(
        self,
        every_n=1,
        max_cases=2,
        sampling_steps=4,
        comparison_steps=16,
        sampler=DEFAULT_SAMPLER,
        comparison_sampler="euler",
        seed=42,
        make_video=True,
        make_figures=True,
        video_joint=False,
        video_joint_steps=None,
        joint_action=False,
        joint_only=False,
        config=None,
        trainer=None,
    ):
        super().__init__(config, trainer)
        self.every_n, self.max_cases = int(every_n), int(max_cases)
        self.sampling_steps, self.comparison_steps = int(sampling_steps), int(comparison_steps)
        self.seed = int(seed)
        # Match the deployed solver; retain the legacy Euler comparison only for
        # conditional evaluation.
        self.joint_only = _as_bool(joint_only)
        self.sampler = str(sampler)
        # The joint path supports UniPC only. Keep the longer reference rollout,
        # but compare step counts under the same joint conditions, not clean GT.
        self.comparison_sampler = "unipc" if self.joint_only else str(comparison_sampler)
        if self.joint_only and self.sampler != "unipc":
            raise ValueError("joint_only FK eval requires sampler='unipc'")
        self.make_video = _as_bool(make_video)
        # Figures are matplotlib and the only CPU-heavy part of the eval: ~128 of
        # them per validation pass.  Off means the eval writes npz + metrics.json
        # (kilobytes, milliseconds) and tools/render_fk_case.py draws the same
        # figures offline, for any step, later.
        self.make_figures = _as_bool(make_figures)
        # Second arm: generate the video from noise (i2v) and denoise FK alongside
        # it, instead of conditioning on the clean GT video.  Answers "how much of
        # the FK accuracy was being propped up by being handed the answer".
        # oc.env yields a string, so "false" is truthy -- hence _as_bool.
        self.video_joint = _as_bool(video_joint)
        self.video_joint_steps = self.sampling_steps if video_joint_steps is None else int(video_joint_steps)
        # Fourth arm, and the only one that matches deployment: the future ACTION is
        # denoised alongside the video and FK instead of read clean.  It is a *mode*
        # of the joint arm rather than a third sampler run -- the arm's cost is one
        # extra pass per case, and running both joint layouts would double that to
        # answer a question one subtraction answers.  Read against the two-segment
        # arm at the same case, seed and step count: the difference is what the
        # clean action was contributing.
        self.joint_action = _as_bool(joint_action)
        if self.joint_action and not self.video_joint and not self.joint_only:
            raise ValueError(
                "joint_action requires video_joint: it is the joint arm with the action segment added, "
                "so the joint arm has to be on"
            )
        if min(self.every_n, self.max_cases, self.sampling_steps, self.comparison_steps) <= 0:
            raise ValueError("FK eval limits must be positive")
        self._fixed = None
        self._done = False
        self._compared = False

    def on_validation_start(self, model, dataloader, iteration=0):
        self._done = False

    def on_validation_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        if iteration % self.every_n or self._done:
            return
        value = output_batch.get("flow_matching_loss_fk")
        if value is not None and distributed.is_rank0():
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log({"fk/val_loss": float(value.detach().cpu())}, step=iteration)
            except ImportError:
                pass
        # Called inside the trainer's eval/EMA context, on EVERY rank.  FSDP
        # collectives are still live here, so a rank-zero-only forward would hang.
        with evaluation_rng(self.seed):
            if self._fixed is None:
                self._fixed = fixed_cases(self.config, count=self.max_cases, seed=self.seed)
                if distributed.is_rank0():
                    root = Path(self.config.job.path_local) / "fk_eval"
                    root.mkdir(parents=True, exist_ok=True)
                    temporary = root / "fixed_cases.tmp.json"
                    temporary.write_text(json.dumps(self._fixed[0], indent=2) + "\n")
                    temporary.replace(root / "fixed_cases.json")
            for identity, cpu_batch in zip(*self._fixed, strict=True):
                case_id, seed = identity["case_id"], identity["seed"]
                logger.info(f"FK eval {case_id}: {self.sampler} {self.sampling_steps} steps, seed={seed}")
                sampling_kwargs = {"generate_video": True, "joint_action": True} if self.joint_only else {}
                vision = None
                with evaluation_rng(seed):
                    batch = misc.to(copy.deepcopy(cpu_batch), device="cuda")
                    prediction = model.sample_fk(
                        batch, steps=self.sampling_steps, seed=seed, sampler=self.sampler, **sampling_kwargs
                    )
                    if self.joint_only:
                        if not prediction.finite:
                            logger.warning(f"FK eval {case_id}: joint sampling produced non-finite latents")
                        vision, prediction = prediction.vision, prediction.fk
                reference = None
                if not self._compared:
                    with evaluation_rng(seed):
                        reference = model.sample_fk(
                            misc.to(copy.deepcopy(cpu_batch), device="cuda"),
                            steps=self.comparison_steps,
                            seed=seed,
                            sampler=self.comparison_sampler,
                            **sampling_kwargs,
                        )
                        if self.joint_only:
                            reference = reference.fk
                # Third arm -- the same case and the same FK seed, but with the
                # video generated rather than handed over clean.  Same seed means
                # the two arms start from identical FK noise, so any difference is
                # attributable to the video conditioning and nothing else.
                # Every rank must run it: FSDP collectives are live in here.
                joint = None
                if self.video_joint and not self.joint_only:
                    with evaluation_rng(seed):
                        joint = model.sample_fk(
                            misc.to(copy.deepcopy(cpu_batch), device="cuda"),
                            steps=self.video_joint_steps,
                            seed=seed,
                            sampler=self.sampler,
                            generate_video=True,
                            joint_action=self.joint_action,
                        )
                        if not joint.finite:
                            # Reported, never raised: an early exit here would give
                            # the ranks different forward counts and deadlock.
                            logger.warning(
                                f"FK eval {case_id}: joint arm produced non-finite latents; "
                                "the video trajectory diverged and FK read it"
                            )
                if distributed.is_rank0():
                    self._save_case(
                        identity, cpu_batch, prediction, reference, iteration,
                        vision=vision,
                        subtitle="joint video + action + available PointFlow + FK; first frame/state conditioned",
                    )
                    if joint is not None:
                        self._save_case(
                            identity,
                            cpu_batch,
                            joint.fk,
                            None,
                            iteration,
                            subdir="joint_action" if self.joint_action else "joint",
                            key_prefix="joint_action_" if self.joint_action else "joint_",
                            vision=joint.vision,
                            subtitle=(
                                "video + action GENERATED (i2v), not given"
                                if self.joint_action
                                else "video GENERATED (i2v), not given"
                            ),
                        )
        self._done = True
        self._compared = True

    # ------------------------------------------------------------------
    def _record(self, identity, cpu_batch, prediction, reference, iteration, *, vision=None):
        # ``cpu_batch`` is on the host (fixed_cases builds it from a CPU loader),
        # but ``prediction``/``reference`` come straight out of ``model.sample_fk``
        # and are therefore CUDA tensors.  ``np.asarray`` on one raises
        # "can't convert cuda:0 device type tensor to numpy", so go through
        # ``_numpy`` rather than assuming either input is already an array.
        sample = cpu_batch["fk"][0]
        target = _numpy(sample["targets"]["displacement"])[None][0]
        valid = np.asarray(sample["targets"]["valid"], bool)
        anchor = _numpy(sample["inputs"]["anchor_xyz"])
        prediction = _numpy(prediction)
        if prediction.shape != target.shape:
            raise ValueError(f"FK prediction {tuple(prediction.shape)} vs target {tuple(target.shape)}")
        metrics = fk_visualize.trajectory_metrics(prediction, target, valid)
        if self.joint_only:
            metrics["sampling_mode"] = "joint"
            metrics["conditions"] = "first frame + state action; future video/action/available PointFlow/FK jointly denoised"
        # The baseline the verdict is a ratio against.  Keys are prefixed rather
        # than merged: "zero_ade_mm" and "all_ade_mm" must not be confusable.
        baseline = fk_visualize.trajectory_metrics(np.zeros_like(target), target, valid)
        metrics.update({f"zero_{key}": value for key, value in baseline.items() if not key.startswith("per_step")})
        metrics["zero_per_step_ade_mm"] = baseline["per_step_ade_mm"]
        metrics["ratio_to_zero"] = (
            metrics["all_ade_mm"] / metrics["zero_all_ade_mm"] if metrics["zero_all_ade_mm"] else float("nan")
        )
        metrics.update(fk_visualize.per_keypoint_error(prediction, target, valid))
        if reference is not None:
            # Same seed, so both trajectories start from the same noise; the
            # difference is the solver alone.
            reference = _numpy(reference)
            metrics.update(
                {
                    f"{self.comparison_sampler}{self.comparison_steps}_" + key: value
                    for key, value in fk_visualize.trajectory_metrics(reference, target, valid).items()
                    if not key.startswith("per_step")
                }
            )
            metrics["sampling_difference_mm"] = float(
                np.linalg.norm(reference - prediction, axis=-1)[valid].mean() * 1000.0
            )
        return {
            "case_id": identity["case_id"],
            "episode": identity["episode"],
            "frame_ids": identity["raw_frame_ids"],
            "anchor": anchor,
            "target": target,
            "valid": valid,
            "prediction": prediction,
            "reference": reference,
            # The joint arm's generated video, kept as a LATENT.  Decoding to pixels
            # costs a VAE pass and is not needed to score FK, so it is deferred to
            # tools/render_fk_projection.py, which runs offline.  ~1.4 MB per case
            # as float16, so keeping it on every eval is cheap.
            "vision": (
                None
                if vision is None
                else _numpy(torch.stack([v.detach().float().cpu() for v in vision])).astype(np.float16)
            ),
            "metrics": metrics,
        }

    def _save_case(
        self,
        identity,
        cpu_batch,
        prediction,
        reference,
        iteration,
        *,
        subdir="",
        key_prefix="",
        vision=None,
        subtitle="video GENERATED (i2v), not given",
    ):
        """Write one arm's artifacts.  ``subdir`` keeps the joint arm beside the
        clean-video one rather than on top of it; ``key_prefix`` keeps the two
        apart in wandb, where they should be read against each other.
        ``subtitle`` says which conditioning produced the panel, because the two
        joint layouts write to different directories and a figure signed with the
        wrong one is worse than an unsigned one."""
        record = self._record(identity, cpu_batch, prediction, reference, iteration, vision=vision)
        directory = Path(self.config.job.path_local) / "fk_eval" / f"step_{iteration:07d}" / identity["case_id"]
        if subdir:
            directory = directory / subdir
        title = f"{identity['case_id']} ({identity['episode']})"
        if subdir:
            title += f" — {subdir}: {subtitle}"
        elif self.joint_only:
            title += f" — {subtitle}"
        paths = fk_visualize.write_case(record, directory, title=title, draw=self.make_figures)
        if self.make_video and self.make_figures:
            # The mp4 is the figures again, frame by frame -- 32 more matplotlib
            # renders per case.  Gating it on make_figures keeps the two together:
            # "no figures during training" has to mean no video either, or the
            # expensive half survives the switch.
            paths["comparison.mp4"] = fk_visualize.write_video(record, directory)
        log.info(
            f"FK eval {identity['case_id']}{'/' + subdir if subdir else ''}: "
            f"ADE {record['metrics']['all_ade_mm']:.2f} mm "
            f"(zero {record['metrics']['zero_all_ade_mm']:.2f} mm, ratio {record['metrics']['ratio_to_zero']:.2f})"
        )
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(
                    {
                        f"fk/{identity['case_id']}/{key_prefix}ade_mm": record["metrics"]["all_ade_mm"],
                        f"fk/{identity['case_id']}/{key_prefix}ratio_to_zero": record["metrics"]["ratio_to_zero"],
                    },
                    step=iteration,
                )
        except ImportError:
            pass
        return paths
