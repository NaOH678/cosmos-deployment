"""Whole-episode FK rollout: tile a val episode with one-window-spaced windows, sample each.

The fixed-case eval renders **one** window's 32 steps.  That answers "is this
trajectory right" but not "does it stay right for a minute", because the two are
different questions: a window is conditioned on its own anchor frame, and nothing
tests what happens when the anchor advances.

The tiling is by one full window span (``fk_eval_cases.window_step``), so no
predicted frame comes from two windows -- there is no "which chunk wins" to decide,
and the stitched trajectory is a genuine 1:1 function of the episode's frames.  The
single frame two windows share is the later window's anchor, which is ground truth
and is therefore not emitted twice.

Only the **middle** ``seconds`` of each episode is sampled (default 16 s).  A whole
episode is ~40 s, which is long to watch and ~2.5x the forwards; the middle is where
the hand is doing the task rather than reaching in or settling.  The cut lands on
window boundaries, so it is within one window of the ask.

Cost is the reason this is a separate callback and not another arm of ``fk_eval``.
A window covers 32 model frames, and 16 s at the dataset's 15 Hz is 240 frames, so 8
windows -- ~32 forwards per case at ``sampling_steps=4``, against the fixed eval's 4.
(A whole episode would be ~18 windows.)  It is gated by ``every_n`` in validation
**passes** and off unless enabled.

Every rank samples: FSDP collectives are live inside the eval context, so a
rank-zero-only forward would hang.  Only rank zero writes.
"""

import copy
import json
import logging
from pathlib import Path

import numpy as np

from cosmos_framework.callbacks import fk_visualize
from cosmos_framework.callbacks.fk_eval import _as_bool, _numpy
from cosmos_framework.callbacks.fk_eval_cases import evaluation_rng, fixed_cases, rollout_windows
from cosmos_framework.utils import callback, distributed, log, misc

logger = logging.getLogger(__name__)


class FKRolloutCallback(callback.Callback):
    """One contiguous rollout per val episode; all ranks participate in sampling."""

    def __init__(
        self,
        enabled=False,
        every_n=10,
        sampling_steps=4,
        sampler="unipc",
        seed=42,
        max_episodes=2,
        seconds=16.0,
        fps=15.0,
        generated_video=False,
        joint_action=False,
        config=None,
        trainer=None,
    ):
        super().__init__(config, trainer)
        self.enabled = _as_bool(enabled)
        self.every_n = int(every_n)
        self.sampling_steps = int(sampling_steps)
        self.sampler = str(sampler)
        self.seed = int(seed)
        self.max_episodes = int(max_episodes)
        # Middle-of-episode clip length.  0 or None means the whole episode, which
        # is ~40 s: long to watch and ~2.5x the forwards.
        self.seconds = None if seconds in (None, "", 0, "0") else float(seconds)
        self.fps = float(fps)
        if min(self.every_n, self.sampling_steps, self.max_episodes) <= 0:
            raise ValueError("FK rollout limits must be positive")
        if self.seconds is not None and self.seconds <= 0:
            raise ValueError("FK rollout seconds must be positive")
        # Joint arm per window: generate the video from noise and denoise FK
        # alongside it, instead of conditioning FK on the clean GT video.  Turns the
        # rollout from an FK-only overlay into the two-panel form the PointFlow stages
        # use -- and it is a sampling switch, not a training one.
        self.generated_video = _as_bool(generated_video)
        # Deployed configuration, not a variant: the future action is denoised too,
        # so the only thing still handed over is the anchor frame.  REQUIRES the
        # joint arm, and says so rather than implying it -- on its own it is a
        # mistake worth naming, not a setting that quietly does nothing
        # (fk_joint_sampling_test asserts exactly that).  The config comment used to
        # say "Implies", which is what made a launcher set only this switch and die
        # here at startup (2026-09-24 02:50); the comment is fixed, not the guard.
        self.joint_action = _as_bool(joint_action)
        if self.joint_action and not self.generated_video:
            raise ValueError("joint_action requires generated_video: it is the joint arm with the action added")
        self._passes = 0
        self._fired_this_pass = False
        self._val = None

    def on_validation_start(self, model, dataloader, iteration=0):
        # Gate on PASSES, counted here.  ``on_validation_step_end`` fires once per
        # validation *batch*, so counting there and resetting on each pass made
        # ``(events - 1) % every_n == 0`` true at the first batch of every pass:
        # every_n=10 fired every pass -- every 100 steps, not every 1000 -- and the
        # rollout's ~144 forwards then roughly doubled the step time.  Caught on the
        # first GPU run, whose fk_rollout/ held step_0000000/100/200/300.
        self._passes += 1
        self._fired_this_pass = False

    def on_validation_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        if not self.enabled or self._fired_this_pass:
            return
        if (self._passes - 1) % self.every_n:
            return
        # One rollout per gated pass, on the first batch that offers a chance: the
        # hook is per-batch, and re-entering it would tile the same episode a dozen
        # times per validation.
        self._fired_this_pass = True
        if distributed.is_rank0():
            logger.info(f"FK rollout at iteration {iteration}: tiling val episodes")
        # The fixed cases are the authority on which val episodes exist; the rollout
        # extends them rather than re-deriving a split, so the two cannot disagree.
        # Cached: ``fixed_cases`` instantiates the dataset and packs a batch per
        # case -- seconds of work that cannot change between validation passes, and
        # that the eval callback already avoids paying more than once.
        if self._val is None:
            identities, _ = fixed_cases(self.config, count=2, seed=self.seed)
            self._val = [row for row in identities if row["split"] == "val"][: self.max_episodes]
        for identity in self._val:
            self._rollout(model, identity, iteration)

    def _rollout(self, model, identity, iteration):
        episode = identity["episode"]
        windows, batches = rollout_windows(
            self.config,
            split="val",
            base_index=identity["index"],
            seconds=self.seconds,
            fps=self.fps,
            seed=self.seed,
        )
        if distributed.is_rank0():
            logger.info(
                f"FK rollout {episode}: {len(windows)} windows x {self.sampling_steps} steps "
                f"= {len(windows) * windows[0]['step']} frames"
                f"{'' if self.seconds is None else f' (middle {self.seconds:g}s of the episode)'}"
                f" (index {windows[0]['index']}..{windows[-1]['index']}, step {windows[0]['step']})"
            )
        prediction, target, anchor, valid, frame_ids = [], [], [], [], []
        latents = []
        for case, cpu_batch in zip(windows, batches, strict=True):
            # Every window draws the same FK noise seed, which is what makes the
            # rollout comparable window-to-window: the differences you see are the
            # anchor moving, not the sampler re-rolling.  ``evaluation_rng`` keeps
            # this out of training's RNG streams.
            with evaluation_rng(self.seed):
                batch = misc.to(copy.deepcopy(cpu_batch), device="cuda")
                sample = cpu_batch["fk"][0]
                sampled = model.sample_fk(
                    batch,
                    steps=self.sampling_steps,
                    seed=self.seed,
                    sampler=self.sampler,
                    generate_video=self.generated_video,
                    joint_action=self.joint_action,
                )
            if self.generated_video:
                # ``generate_video`` changes the return to SimpleNamespace(fk, vision,
                # finite); ``vision`` is this window's generated video as a latent,
                # decoded offline.  One per window is what makes the clip longer than
                # a single window.
                predicted = sampled.fk
                if not sampled.finite:
                    logger.warning(
                        f"FK rollout {episode} window {case['case_id']}: joint arm produced "
                        "non-finite latents; the video trajectory diverged and FK read it"
                    )
                latent = _numpy(sampled.vision[0]).astype(np.float16)
                # [1,C,T,H,W] (the packed item carries its own batch axis) -> [C,T,H,W].
                while latent.ndim > 4 and latent.shape[0] == 1:
                    latent = latent[0]
                latents.append(latent)
            else:
                predicted = sampled
            # Both sides are made ABSOLUTE here, in camera-frame metres.  A window's
            # displacement is relative to its own anchor, so a rollout that kept them
            # relative would stack ~18 different origins on top of each other.  The
            # published intrinsics and the renderer both work on absolute positions.
            base = _numpy(sample["inputs"]["anchor_xyz"])[None]
            prediction.append(_numpy(predicted) + base)
            target.append(_numpy(sample["targets"]["displacement"]) + base)
            anchor.append(_numpy(sample["inputs"]["anchor_xyz"]))
            valid.append(np.asarray(sample["targets"]["valid"], bool))
            # Steps 0..N-1 map to frames 1..N: frame 0 is the anchor, which is the
            # previous window's last prediction and is emitted by that window.
            frame_ids.append(np.asarray(case["raw_frame_ids"], np.int64)[1:])
        if not distributed.is_rank0():
            return
        prediction = np.concatenate(prediction).astype(np.float32)
        target = np.concatenate(target).astype(np.float32)
        valid = np.concatenate(valid)
        frame_ids = np.concatenate(frame_ids)
        metrics = fk_visualize.trajectory_metrics(prediction, target, valid)
        baseline = fk_visualize.trajectory_metrics(np.zeros_like(target), target, valid)
        metrics.update({f"zero_{k}": v for k, v in baseline.items() if not k.startswith("per_step")})
        metrics["ratio_to_zero"] = (
            metrics["all_ade_mm"] / metrics["zero_all_ade_mm"] if metrics["zero_all_ade_mm"] else float("nan")
        )
        metrics["windows"] = len(windows)
        metrics["episode_frames"] = int(target.shape[0])

        # Two joint layouts mean two rollouts of the same episode can exist under
        # one run; a shared directory would leave the later one silently replacing
        # the earlier, and the two are read against each other by subtraction.
        root = "fk_rollout_joint_action" if self.joint_action else "fk_rollout"
        directory = Path(self.config.job.path_local) / root / f"step_{iteration:07d}" / episode
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "rollout.npz",
            prediction=prediction,
            target=target,
            # First window's anchor; per-window anchors are baked into the absolute
            # arrays above, so this is kept only for provenance.
            anchor=np.asarray(anchor[0], np.float32),
            valid=valid,
            frame_ids=frame_ids,
            keypoint_names=np.asarray(fk_visualize.KEYPOINT_NAMES),
            episode=np.asarray(episode),
            units=np.asarray("metre"),
            coordinate=np.asarray("camera_d435_real"),
            window_step=np.asarray(windows[0]["step"], np.int64),
            window_span=np.asarray(windows[0]["span"], np.int64),
            window_count=np.asarray(len(windows), np.int64),
            # One generated video per window, [n_windows, C, T, H, W] float16, or the
            # None placeholder for the FK-only arm -- exactly the shape convention
            # fk_visualize.write_case uses, so ``stored_latent`` reads both.
            vision=np.stack(latents) if latents else None,
            # Steps each window contributes, so the renderer can map a frame back to
            # (window, step).  NOT ``windows[0]["step"]``: that is the index stride
            # (64 raw frames), a different number -- writing it here is what made the
            # first render die on "8 vision windows x 64 steps != 256 predicted
            # frames".  The steps are one less than the window's observation count,
            # since frame 0 of each window is its anchor.
            window_frames=(
                np.asarray(len(windows[0]["raw_frame_ids"]) - 1, np.int64) if latents else np.asarray(0, np.int64)
            ),
        )
        (directory / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        log.info(
            f"FK rollout {episode}: {metrics['windows']} windows, {metrics['episode_frames']} frames, "
            f"ADE {metrics['all_ade_mm']:.2f} mm (zero {metrics['zero_all_ade_mm']:.2f} mm, "
            f"ratio {metrics['ratio_to_zero']:.2f})"
        )
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(
                    {
                        f"{root}/{episode}/ade_mm": metrics["all_ade_mm"],
                        f"{root}/{episode}/ratio_to_zero": metrics["ratio_to_zero"],
                    },
                    step=iteration,
                )
        except ImportError:
            pass
