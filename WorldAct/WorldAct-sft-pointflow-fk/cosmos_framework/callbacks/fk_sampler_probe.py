"""Why does a 4-step FK sample land near the noise instead of near the motion?

The FK eval reports ``all_ade_mm`` around 1028 mm against a 225 mm stationary
baseline, while the training-time one-step estimate is 18 mm.  Something in the
walk from sigma=1 to sigma=0 is not removing the noise, and reading the code has
not found it -- the sign convention, the sigma schedule, the timestep-to-sigma
conversion and the FKNoised plumbing all check out.

So this measures instead of arguing.  It answers three separate questions with
numbers, and they are separable on purpose:

1. **Did the checkpoint actually load?**  The warm-start branch of
   ``checkpointer.load`` is known to produce a dead frozen branch while reporting
   every key kept (see the PointFlow notes: pure noise, ``|pred| = 1631 mm``).
   A probe that ran on a dead model would "find" a broken sampler.  So the first
   thing written is a replay of the recorded eval for this exact case and step,
   which has to reproduce -- if it does not, every later number is meaningless
   and the report says so.

2. **Is the velocity field right where the sampler walks?**  The sampler's own
   trajectory is not a fair place to measure accuracy, because it has already
   drifted.  Instead the *true* path is constructed from the eval case's ground
   truth (``x_sigma = sigma*eps + (1-sigma)*target``) and the model is asked for
   its velocity at those points.  The reference velocity is constant along that
   path (``v = eps - target``), so per sigma the report gives the magnitude ratio
   and the cosine.  A small magnitude with a healthy cosine is a scale problem; a
   low cosine is a field that was never learned there.

3. **Where does the actual walk go wrong?**  The probe also records the state and
   velocity the sampler really used at each evaluation, so the two can be put
   side by side.

Run only when ``FK_SAMPLER_PROBE=1``; it is registered by the experiment config
under that flag, so an ordinary run is unaffected.
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


def _to_cuda(cpu_batch):
    """A fresh device copy per call.

    ``misc.to`` may move tensors in place, and this callback samples the same
    case more than once; sharing one batch across those calls would let the first
    move corrupt what the second sees.
    """
    return misc.to(copy.deepcopy(cpu_batch), device="cuda")


def _to_mm(value, scale):
    """Per-keypoint norm in millimetres from a **model-space** tensor.

    Model space is ``physical metres / scale``, so a model-space length is metres
    after multiplying by ``scale``, and millimetres after 1000 more.  A bare
    ``* 1000`` is correct only while ``scale == 1.0``; once the recipe sets the
    scale from the data (0.0745 for FK) it overstates every velocity by 1/scale --
    13.4x here.  States, velocities and their differences are all model-space, so
    they all go through this; raw label tensors are already metres and do not.
    """
    return float(value.float().norm(dim=-1).mean() * scale * 1000.0)


def _diff_packs(left, right, *, max_depth=3):
    """Field-by-field difference of two packed sequences, as ``(path, description)``.

    Walking both objects generically rather than comparing a hand-listed set of
    fields is the point: the field that differs is the one nobody thought to
    list.  Tensor leaves report ``max|diff|``; everything else reports its shape,
    length or value so a structural difference is visible even when the numbers
    are not.
    """
    import dataclasses

    rows: list[tuple[str, str]] = []

    def leaves(obj):
        if dataclasses.is_dataclass(obj):
            return [(f.name, getattr(obj, f.name, None)) for f in dataclasses.fields(obj)]
        if hasattr(obj, "__dict__"):
            return [(k, v) for k, v in vars(obj).items() if not k.startswith("_")]
        return []

    def show(value):
        if isinstance(value, torch.Tensor):
            return f"tensor{tuple(value.shape)} {value.dtype}"
        if isinstance(value, (list, tuple)):
            return f"{type(value).__name__}[{len(value)}]"
        return f"{value!r}"[:70]

    def walk(path, a, b, depth):
        if depth > max_depth:
            return
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if a.shape != b.shape:
                rows.append((path, f"SHAPE {show(a)} vs {show(b)}"))
            elif a.numel():
                delta = (a.float() - b.float().to(a.device)).abs().max().item()
                if delta > 0:
                    rows.append((path, f"max|diff| {delta:.6g}  ({show(a)})"))
            return
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                rows.append((path, f"LEN {len(a)} vs {len(b)}"))
                return
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                walk(f"{path}[{i}]", x, y, depth + 1)
            return
        a_leaves = leaves(a)
        if not a_leaves:
            if a != b:
                rows.append((path, f"{show(a)} vs {show(b)}"))
            return
        b_leaves = dict(leaves(b))
        for name, value in a_leaves:
            walk(f"{path}.{name}", value, b_leaves.get(name), depth + 1)

    walk("pack", left, right, 0)
    return rows


def _fk_eval_settings(config):
    """``fk_eval``'s own case count, seed, steps and sampler.

    The probe's whole point is to reproduce a number ``fk_eval`` already
    recorded, so it has to measure the *same* windows through the *same* solver.
    Asking for anything else is not merely wasteful: ``fixed_cases`` refuses a
    count that disagrees with the manifest it finds, and the per-case seed is
    derived from the top-level seed, so a mismatched seed silently measures a
    different trajectory and the replay check fails for a reason that has nothing
    to do with the model.

    Read from the live config rather than duplicated as defaults, so the two
    cannot drift apart.
    """
    callbacks = getattr(config.trainer, "callbacks", None)
    entry = callbacks.get("fk_eval") if hasattr(callbacks, "get") else None
    if entry is None:
        return None
    read = entry.get if hasattr(entry, "get") else (lambda key, default=None: getattr(entry, key, default))
    return {
        "count": int(read("max_cases", 2)),
        "seed": int(read("seed", 42)),
        "steps": int(read("sampling_steps", 4)),
        "sampler": str(read("sampler", "unipc")),
    }


class _Recorder:
    """Passed as ``probe`` to ``sample_fk``; records what the sampler really did."""

    def __init__(self, scale):
        self.scale = float(scale)
        self.sigmas: list[float] = []
        self.state_norm: list[float] = []
        self.velocity_norm: list[float] = []

    def __call__(self, *, sigma, state, velocity):
        self.sigmas.append(float(sigma.reshape(-1)[0]))
        self.state_norm.append(_to_mm(state, self.scale))
        self.velocity_norm.append(_to_mm(velocity, self.scale))

    def rows(self):
        return [
            {"sigma": s, "state_norm_mm": n, "velocity_norm_mm": v}
            for s, n, v in zip(self.sigmas, self.state_norm, self.velocity_norm, strict=True)
        ]


class FKSamplerProbeCallback(callback.Callback):
    """One-off diagnostic on the first validation of the resumed run."""

    def __init__(
        self,
        every_n=1,
        max_cases=1,
        sampling_steps=4,
        sampler="unipc",
        seed=42,
        grid_points=13,
        sweep_shifts=(5.0, 3.0, 2.0, 1.0),
        sweep_steps=(4, 8, 16),
        config=None,
        trainer=None,
    ):
        super().__init__(config, trainer)
        self.every_n, self.max_cases = int(every_n), int(max_cases)
        self.sampling_steps, self.sampler, self.seed = int(sampling_steps), str(sampler), int(seed)
        self.grid_points = int(grid_points)
        # The shift x steps cross product, for the sampler-config table.  Defaults
        # bracket the deployed shift=5 on one axis and the deployed 4 steps on the
        # other, so the current configuration is always one of the cells.
        self.sweep_shifts = tuple(float(s) for s in sweep_shifts)
        self.sweep_steps = tuple(int(s) for s in sweep_steps)
        self._done = False

    def on_validation_start(self, model, dataloader, iteration=0):
        self._done = False
        self._captured = {}
        # Stash the packed sequence the trainer actually feeds the network.  Every
        # measurement in this file rebuilds that pack by hand, and the two disagree
        # by 15x on the same batch (trainer 0.116 vs probe 1.80 in model units),
        # which no conditioning hypothesis has explained.  Capturing the real one
        # makes the difference a diff of two objects rather than a hypothesis.
        original = model.denoise

        def capturing(*args, **kwargs):
            out = original(*args, **kwargs)
            if "pack" not in self._captured:
                self._captured["pack"] = kwargs.get("data_batch_packed")
                self._captured["preds_fk"] = out.get("preds_fk")
            return out

        self._original_denoise = original
        model.denoise = capturing

    def on_validation_end(self, model, iteration=0):
        # Two arguments only: the trainer calls this one without a dataloader, and a
        # missing default here is a TypeError at the end of every validation.
        if getattr(self, "_original_denoise", None) is not None:
            model.denoise = self._original_denoise

    def on_validation_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        if self._done or iteration % self.every_n:
            return
        self._done = True
        with evaluation_rng(self.seed):
            if not distributed.is_rank0():
                # The sampler needs a full forward on every rank; only rank 0
                # writes, so the non-zero ranks run the identical path and drop
                # the result.
                self._run(model, iteration, data_batch, output_batch, write=False)
                return
            report = self._run(model, iteration, data_batch, output_batch, write=True)
        log.info(f"FK sampler probe: {report['headline']}")

    # ------------------------------------------------------------------
    def _run(self, model, iteration, data_batch, output_batch, *, write: bool):
        # Mirror fk_eval exactly; its recorded number is what the replay is
        # checked against.  The constructor's arguments are only a fallback for
        # the case where fk_eval is not registered at all.
        settings = _fk_eval_settings(self.config) or {
            "count": self.max_cases,
            "seed": self.seed,
            "steps": self.sampling_steps,
            "sampler": self.sampler,
        }
        cases = fixed_cases(self.config, count=settings["count"], seed=settings["seed"])
        identities, batches = cases
        identity, cpu_batch = identities[0], batches[0]
        case_id, seed = identity["case_id"], identity["seed"]

        target = np.asarray(cpu_batch["fk"][0]["targets"]["displacement"], np.float64)
        valid = np.asarray(cpu_batch["fk"][0]["targets"]["valid"], bool)
        scale = float(model.config.rectified_flow_training_config.fk_displacement_scale)

        report = {
            "case_id": case_id,
            "iteration": iteration,
            "sampler": settings["sampler"],
            "steps": settings["steps"],
            "seed": seed,
            "target_norm_mm": float(np.linalg.norm(target, axis=-1)[valid].mean() * 1000.0),
            "scale": scale,
        }

        # --- 1. replay the recorded eval -------------------------------------
        recorder = _Recorder(scale)
        prediction = model.sample_fk(
            _to_cuda(cpu_batch),
            steps=settings["steps"],
            seed=seed,
            sampler=settings["sampler"],
            probe=recorder,
        )
        prediction = np.asarray(prediction.detach().float().cpu(), np.float64)
        live = fk_visualize.trajectory_metrics(prediction, target, valid)
        baseline = fk_visualize.trajectory_metrics(np.zeros_like(target), target, valid)
        report["replay"] = {
            "all_ade_mm": live["all_ade_mm"],
            "zero_all_ade_mm": baseline["all_ade_mm"],
            "ratio_to_zero": live["all_ade_mm"] / baseline["all_ade_mm"] if baseline["all_ade_mm"] else float("nan"),
            "per_step_ade_mm": live["per_step_ade_mm"],
        }
        report["sampler_evaluations"] = recorder.rows()

        # A dead checkpoint returns essentially the starting noise.  The start is
        # unit-variance, so its per-keypoint magnitude is what a no-op sampler
        # would score; comparing against that is what makes "dead model" and
        # "broken sampler" distinguishable at all.
        unit_noise = float(
            np.linalg.norm(np.random.default_rng(0).normal(size=target.shape), axis=-1).mean() * scale * 1000.0
        )
        report["unit_noise_mm"] = unit_noise
        report["looks_like_pure_noise"] = bool(live["all_ade_mm"] > 0.8 * unit_noise)

        # --- 2. the velocity field on the TRUE path --------------------------
        # eps is redrawn with the sampler's own seed, so the true path and the
        # sampled path share a starting point.
        device = "cuda"
        generator = torch.Generator(device=device).manual_seed(int(seed))
        horizon, num_points = target.shape[0], target.shape[1]
        epsilon = torch.randn((horizon, num_points, 3), generator=generator, device=device, dtype=torch.float32)
        reference_velocity = (epsilon - torch.from_numpy(target / scale).float().to(device)).detach()
        report["reference_velocity_norm_mm"] = _to_mm(reference_velocity, scale)

        truth = torch.from_numpy(target / scale).float().to(device)
        grid = torch.linspace(1.0, 0.0, self.grid_points, device=device)
        states = [sigma * epsilon + (1 - sigma) * truth for sigma in grid]
        sigmas = [sigma.reshape(1) for sigma in grid]
        velocities = model.fk_velocity_field(_to_cuda(cpu_batch), states, sigmas)
        reference_norm = reference_velocity.norm(dim=-1).mean()
        field = []
        for sigma_value, velocity in zip(grid, velocities, strict=True):
            value = velocity.float()
            field.append(
                {
                    "sigma": float(sigma_value),
                    "model_velocity_norm_mm": _to_mm(value, scale),
                    "reference_velocity_norm_mm": float(reference_norm * scale * 1000.0),
                    "magnitude_ratio": float(value.norm(dim=-1).mean() / reference_norm),
                    "cosine_to_reference": float(
                        torch.nn.functional.cosine_similarity(value.reshape(-1), reference_velocity.reshape(-1), dim=0)
                    ),
                }
            )
        report["velocity_field_on_true_path"] = field

        # --- 2b. the trainer's own loss, on the batch it just ran -------------
        # This is the comparison with no remaining degree of freedom.  Every other
        # number here was produced by this probe: it built its own tensors, called
        # fk_loss itself, and read ~1000mm while the trainer logs ~45mm for the
        # same branch.  ``validate()`` calls ``training_step()``, so ``output_batch``
        # already carries the FK loss the trainer computed through the real training
        # path on ``data_batch`` -- the very batch whose prediction the sampler
        # above was fed.  Nothing here is reconstructed, so if these two disagree
        # the difference is in the forward, not in how the probe measures it.
        trainer_values = {}
        for key in ("flow_matching_loss_fk", "fk_ade_mm", "fk_zero_ade_mm", "fk_loss"):
            value = (output_batch or {}).get(key)
            if value is not None:
                try:
                    trainer_values[key] = float(value.detach().float().mean())
                except Exception:  # noqa: BLE001
                    pass
        report["trainer_on_this_batch"] = trainer_values

        # --- 2b-bis. is the video conditioning the difference? -----------------
        try:
            report["vision_conditioning_ab"] = self._vision_conditioning_ab(model, cpu_batch, sigma_value=0.8333)
        except Exception as error:  # noqa: BLE001
            report["vision_conditioning_ab"] = {"error": f"{type(error).__name__}: {error}"}

        # --- 2e. does the magnitude compression depend on sigma? --------------
        try:
            report["shrinkage_vs_sigma"] = self._shrinkage(model, identities, batches, scale)
        except Exception as error:  # noqa: BLE001
            report["shrinkage_vs_sigma"] = {"error": f"{type(error).__name__}: {error}"}

        # --- 2b-ter. the trainer's own pack, diffed against the rebuild -------
        try:
            report["pack_diff"] = self._pack_diff(model, data_batch, scale)
        except Exception as error:  # noqa: BLE001
            report["pack_diff"] = {"error": f"{type(error).__name__}: {error}"}

        # --- 2c. EMA weights vs training weights ------------------------------
        # Everything above runs inside the trainer's ``ema_scope``, so it measures
        # the EMA copy; ``training_step`` -- and therefore the ``fk_loss`` that
        # gets logged -- measures the live weights.  The two differ by only a few
        # percent per tensor, which looks harmless, but this branch's output is
        # not obviously Lipschitz in its weights and the discrepancy being chased
        # is a factor of ~40 in RMS.  So measure the same point under both.
        report["ema_vs_raw"] = self._ema_vs_raw(model, cpu_batch, sigma_value=0.8333)

        # --- 2d. a training-sized pack ----------------------------------------
        try:
            report["training_pack"] = self._training_pack_control(model, sigma_value=0.8333)
        except Exception as error:  # noqa: BLE001
            report["training_pack"] = {"error": f"{type(error).__name__}: {error}"}
        pack_batch = getattr(self, "_last_training_pack_batch", None)
        if pack_batch is not None:
            try:
                report["loss_crosscheck"] = self._fk_loss_crosscheck(model, pack_batch)
            except Exception as error:  # noqa: BLE001
                report["loss_crosscheck"] = {"error": f"{type(error).__name__}: {error}"}
            try:
                # Dense near 0.83 (the sampler's third node and the training
                # median), sparse elsewhere.
                grid = [
                    0.50,
                    0.60,
                    0.70,
                    0.75,
                    0.78,
                    0.80,
                    0.81,
                    0.82,
                    0.825,
                    0.83,
                    0.8333,
                    0.835,
                    0.84,
                    0.85,
                    0.86,
                    0.88,
                    0.90,
                    0.9375,
                    0.97,
                    1.0,
                ]
                report["sigma_sweep"] = self._sigma_sweep(model, pack_batch, sigma_values=grid)
            except Exception as error:  # noqa: BLE001
                report["sigma_sweep"] = {"error": f"{type(error).__name__}: {error}"}

        if data_batch is not None:
            try:
                report["control_batch"] = self._measure_on_batch(model, data_batch, sigma_value=0.8333)
            except Exception as error:  # noqa: BLE001 - a control, must not sink the report
                report["control_batch"] = {"error": f"{type(error).__name__}: {error}"}

        # --- 3. separate the solver from the step count -----------------------
        # The main sampler (unipc, 4 steps) and the comparison sampler (euler, 16
        # steps) differ in *two* things, so the fact that they disagreed by over a
        # metre on a trained model does not say which one is responsible.  Crossing
        # both factors on the same case and seed does: the four cells below move
        # one at a time from the deployed configuration.
        sweep = []
        for sampler, steps in (("unipc", 4), ("unipc", 16), ("euler", 4), ("euler", 16)):
            with evaluation_rng(seed):
                attempt = model.sample_fk(_to_cuda(cpu_batch), steps=steps, seed=seed, sampler=sampler)
            values = fk_visualize.trajectory_metrics(
                np.asarray(attempt.detach().float().cpu(), np.float64), target, valid
            )
            sweep.append(
                {
                    "sampler": sampler,
                    "steps": steps,
                    "all_ade_mm": values["all_ade_mm"],
                    "final_ade_mm": values["final_ade_mm"],
                    "ratio_to_zero": values["all_ade_mm"] / baseline["all_ade_mm"],
                }
            )
        report["sampler_sweep"] = sweep

        # --- 3b. the sampler's sigma grid, swept ---------------------------------
        # Section 3 varies the integrator and the step count but not the grid, so
        # it cannot see the thing the compression measurement points at: the
        # deployed grid puts three of four nodes above sigma=0.6, which is where
        # the branch is worst.  This crosses the grid's warp with the step count.
        try:
            report["shift_steps_sweep"] = self._shift_steps_sweep(
                model,
                cpu_batch,
                target=target,
                valid=valid,
                seed=seed,
                base_ade=baseline["all_ade_mm"],
            )
        except Exception as error:  # noqa: BLE001
            report["shift_steps_sweep"] = {"error": f"{type(error).__name__}: {error}"}

        report["headline"] = (
            f"{case_id} @{iteration}: replay ADE {live['all_ade_mm']:.1f}mm "
            f"(zero {baseline['all_ade_mm']:.1f}mm, unit-noise {unit_noise:.1f}mm), "
            f"field cosine {field[0]['cosine_to_reference']:+.3f}..{field[-1]['cosine_to_reference']:+.3f}, "
            f"mag ratio {min(f['magnitude_ratio'] for f in field):.2f}..{max(f['magnitude_ratio'] for f in field):.2f}"
        )

        if write:
            root = Path(self.config.job.path_local) / "fk_sampler_probe"
            root.mkdir(parents=True, exist_ok=True)
            path = root / f"step_{iteration:07d}_{case_id}.json"
            path.write_text(json.dumps(report, indent=2, default=float) + "\n")
            log.info(f"FK sampler probe written to {path}")
            self._print(report)
        return report

    def _sigma_sweep(self, model, data_batch, *, sigma_values):
        """Velocity error against sigma, sampled finely near the sampler's nodes.

        ``fk_loss`` agrees with a hand computation to the last digit, and its own
        ``fk_ade`` reports ~1000mm on tensors the probe builds -- so the function
        is not the problem.  What is left is *which sigma* it is asked about: the
        logged per-sigma bin covering (0.8, 0.9) implies a 49mm error, while the
        probe measures 1204mm at 0.8333, inside that same bin.

        Those can both be true if the error is a sharp function of sigma.  The
        sigma features are ``cos/sin(sigma * freq * 1000)`` with ``freq`` from 1
        down to 1e-4, so at sigma~0.83 the highest frequency turns by 17 radians
        per 0.017 of sigma: the encoding is aliased, the branch can only be smooth
        where sigma is densely sampled, and a sampler whose nodes are four fixed
        values can land on the gaps.  A sweep is the direct test.
        """
        from cosmos_framework.data.fk_batch import FKNoised

        context = model._fk_fitting_context(_to_cuda(data_batch))  # noqa: SLF001
        truth = context.data.displacement.float()
        scale = float(context.scale)
        generator = torch.Generator(device=context.device).manual_seed(1234)
        epsilon = torch.randn(truth.shape, generator=generator, device=context.device, dtype=torch.float32)
        reference = epsilon - truth / scale
        reference_norm = reference.norm(dim=-1).mean()
        rows = []
        for value in sigma_values:
            sigma = torch.full((context.batch_size,), float(value), device=context.device)
            per_point = sigma[context.data.inputs["point_batch"]].reshape(1, -1, 1)
            state = per_point * epsilon + (1 - per_point) * (truth / scale)
            context.packed.fk_noised = FKNoised(state, state, state, sigma)
            velocity = model.denoise(data_batch_packed=context.packed)["preds_fk"].float()
            rows.append(
                {
                    "sigma": float(value),
                    "velocity_error_mm": _to_mm(velocity - reference, scale),
                    "magnitude_ratio": float(velocity.norm(dim=-1).mean() / reference_norm),
                    "cosine": float(
                        torch.nn.functional.cosine_similarity(velocity.reshape(-1), reference.reshape(-1), dim=0)
                    ),
                }
            )
        return rows

    def _fk_loss_crosscheck(self, model, data_batch, *, sigma_value=0.8333):
        """Call ``fk_loss`` itself on tensors this probe also scores by hand.

        Five independent measurements now agree that the velocity error at
        sigma=0.833 is ~1200mm -- on a 32-sample pack, on a validation batch, on
        the fixed case, under EMA weights and under the live ones.  The logged
        ``fk_loss`` implies 49mm for the same quantity at the same sigma, and its
        per-sigma bin covering 0.833 is the one it reports.  Masking is not the
        explanation (``valid`` is 100% true here), nor is the pack size, nor the
        weights.

        That leaves the arithmetic, so run both on the *same* tensors: the
        library's ``fk_loss`` and this probe's own norm, side by side.  If they
        disagree, the difference is a line of code rather than a hypothesis.
        """
        from cosmos_framework.data.fk_batch import FKNoised
        from cosmos_framework.model.generator.fk_training import fk_loss

        context = model._fk_fitting_context(_to_cuda(data_batch))  # noqa: SLF001
        truth = context.data.displacement.float()
        scale = float(context.scale)
        generator = torch.Generator(device=context.device).manual_seed(1234)
        epsilon = torch.randn(truth.shape, generator=generator, device=context.device, dtype=torch.float32)
        sigma = torch.full((context.batch_size,), float(sigma_value), device=context.device)
        per_point = sigma[context.data.inputs["point_batch"]].reshape(1, -1, 1)
        xt = per_point * epsilon + (1 - per_point) * (truth / scale)
        noised = FKNoised(xt, epsilon, epsilon - truth / scale, sigma)
        context.packed.fk_noised = noised
        prediction = model.denoise(data_batch_packed=context.packed)["preds_fk"]

        library, metrics = fk_loss(prediction, noised, context.data, scale)
        by_hand = (prediction.float() - (epsilon - truth / scale)).square().mean()
        return {
            "sigma": float(sigma_value),
            "n_points": int(truth.shape[1]),
            "valid_fraction": float(context.data.valid.float().mean()),
            "fk_loss_mse": float(library),
            "by_hand_mse": float(by_hand),
            "ratio": float(library) / max(float(by_hand), 1e-30),
            "fk_ade_mm_reported": float(metrics.get("fk_ade_mm", float("nan"))),
            "by_hand_norm_mm": _to_mm(prediction - (epsilon - truth / scale), scale),
        }

    def _pack_diff(self, model, data_batch, scale):
        """The trainer's own packed sequence, diffed against the probe's rebuild.

        The two agree on the batch (both report ``target |x| = 59.8mm``) yet the
        trainer measures |dv| = 0.116 in model units where the probe measures 1.80
        -- 15x -- and no conditioning hypothesis has moved that number.  So stop
        hypothesising: take the pack the trainer really fed the network, score the
        trainer's own ``preds_fk`` against its own target (which must reproduce the
        logged value), then re-run the same state through the probe's rebuild.
        With the FK state held fixed, whatever still differs is the pack, and the
        field diff names it.
        """
        captured = getattr(self, "_captured", {}).get("pack")
        preds = getattr(self, "_captured", {}).get("preds_fk")
        if captured is None or preds is None:
            return {"error": "no denoise call captured during validation"}
        noised = getattr(captured, "fk_noised", None)
        if noised is None:
            return {"error": "captured pack carries no FK state"}
        preds = preds.float()
        target = noised.velocity_target.float()

        out = {
            "trainer_pack_error_mm": _to_mm(preds - target, scale),
            "trainer_pack_mse": float((preds - target).square().mean()),
        }

        again = model.denoise(data_batch_packed=copy.deepcopy(captured)).get("preds_fk")
        if again is not None:
            out["redenoise_max_abs_diff"] = float((again.float() - preds).abs().max())

        probe_pack = model._fk_fitting_context(_to_cuda(data_batch)).packed  # noqa: SLF001
        probe_pack.fk_noised = noised
        probe_preds = model.denoise(data_batch_packed=copy.deepcopy(probe_pack)).get("preds_fk")
        if probe_preds is None or probe_preds.shape != preds.shape:
            out["probe_pack_error_mm"] = None
            shape = None if probe_preds is None else tuple(probe_preds.shape)
            out["shape_mismatch"] = f"{shape} vs {tuple(preds.shape)}"
        else:
            out["probe_pack_error_mm"] = _to_mm(probe_preds.float() - target, scale)
            out["probe_pack_mse"] = float((probe_preds.float() - target).square().mean())
            out["preds_max_abs_diff"] = float((probe_preds.float() - preds).abs().max())
        out["field_diff"] = _diff_packs(captured, probe_pack)
        return out

    def _shrinkage(self, model, identities, batches, scale):
        """Pool `_shrinkage_vs_sigma` over every fixed case, and report it by bin."""
        import numpy as _np

        sigmas = (0.20, 0.40, 0.60, 0.75, 0.85, 0.95, 1.00)
        pooled = {}
        for identity, batch in zip(identities, batches, strict=True):
            for value, estimate, gt in self._shrinkage_vs_sigma(model, batch, sigmas=sigmas, scale=scale):
                e, g = estimate.reshape(-1).cpu().numpy(), gt.reshape(-1).cpu().numpy()
                keep = g > 1e-6
                pooled.setdefault(value, []).append((g[keep], e[keep] / g[keep]))
        if not pooled:
            return {"error": "no cases"}
        gs = _np.concatenate([x[0] for v in pooled.values() for x in v])
        edges = _np.percentile(gs, [0, 10, 25, 50, 75, 90, 100])
        out = {"n": int(gs.size), "bin_edges_mm": [float(x) for x in edges], "rows": []}
        for value in sorted(pooled):
            gg = _np.concatenate([x[0] for x in pooled[value]])
            rr = _np.concatenate([x[1] for x in pooled[value]])
            bins = []
            for i in range(len(edges) - 1):
                m = (gg >= edges[i]) & (gg <= edges[i + 1])
                bins.append(float(_np.median(rr[m])) if m.sum() else float("nan"))
            out["rows"].append({"sigma": value, "overall": float(_np.median(rr)), "bins": bins})
        return out

    def _shrinkage_vs_sigma(self, model, cpu_batch, *, sigmas, scale):
        """``|x0_hat| / |target|`` at each sigma, binned by how big the target is.

        The whole eval error decomposes as ~87% "the hand is displaced as a block,
        hand shape intact", and that block offset is the magnitude compression seen
        in absolute terms.  `pred/gt` is not a constant: it runs 1.19 -> 0.83 as the
        true displacement grows, inside every case -- regression to the mean, which
        a constant multiplier cannot fix.

        Whether it is *intrinsic* or *under-using the conditioning* is decidable by
        where it lives in sigma.  At sigma~1 the video is pure noise and the answer
        genuinely is the conditional mean, so shrinkage there is correct; at low
        sigma the video is legible and the shrinkage should vanish.  If instead it
        is flat in sigma, it is the model's real limit and only more data moves it.
        """
        context = model._fk_fitting_context(_to_cuda(cpu_batch))  # noqa: SLF001
        target = context.data.displacement.float() / scale  # model units
        generator = torch.Generator(device=context.device).manual_seed(int(self.seed))
        epsilon = torch.randn(target.shape, generator=generator, device=context.device, dtype=torch.float32)
        gt = (target * scale * 1000.0).norm(dim=-1)  # [T, N] mm
        rows = []
        with torch.no_grad():
            for value in sigmas:
                sigma = torch.full((context.batch_size,), float(value), device=context.device)
                per_point = sigma[context.data.inputs["point_batch"]].reshape(1, -1, 1)
                state = per_point * epsilon + (1 - per_point) * target
                velocity = context.velocity(state, sigma).float()
                estimate = ((state - per_point * velocity) * scale * 1000.0).norm(dim=-1)  # [T,N] mm
                rows.append((float(value), estimate, gt))
        return rows

    def _shift_steps_sweep(self, model, cpu_batch, *, target, valid, seed, base_ade):
        """Cross the sampler's two free knobs on one case and one draw.

        ``shift`` and ``num_steps`` are independent, and both are usually set by
        habit: shift warps the sigma grid (``fm_solvers_unipc.py:188``) and the
        step count decides how finely that warped grid is walked.  Neither has
        been chosen for FK -- the grid came from video, where the branch was also
        *trained* on it.

        The reason to sweep rather than reason: the probe already measured that
        the branch is accurate at low sigma (pred/gt ~ 1.00 at sigma=0.20) and
        compressed at high sigma (0.80 at sigma=1.00), and the deployed grid
        spends three of its four nodes above 0.6.  So moving the nodes down looks
        free.  It is not *known* to be free: training draws sigma from the shifted
        marginal too, so a low-shift grid walks where the branch saw few samples.
        Both effects are real and their sum is not predictable from either one --
        only a sweep on the same case separates them.

        Every cell reuses the same seed, so all of them start from the identical
        noise and differ only in the grid and the step count.
        """
        from cosmos_framework.model.generator.fk_sampling import shifted_sigmas

        # One set of bin edges for the whole table, taken from the target -- which
        # is the same array in every cell.  Per-cell edges would make the columns
        # incomparable, which is the only thing this table exists to be.
        gt = np.linalg.norm(target, axis=-1) * 1000.0  # [H,N] mm
        edges = np.percentile(gt[valid], [0, 25, 50, 75, 90, 100])

        rows = []
        for shift in self.sweep_shifts:
            for steps in self.sweep_steps:
                try:
                    with evaluation_rng(seed):
                        attempt = model.sample_fk(
                            _to_cuda(cpu_batch), steps=steps, seed=seed, sampler="unipc", shift=shift
                        )
                    row = self._score_shift(
                        np.asarray(attempt.detach().float().cpu(), np.float64),
                        target,
                        valid,
                        gt,
                        edges,
                        shift=shift,
                        steps=steps,
                        base_ade=base_ade,
                    )
                except Exception as error:  # noqa: BLE001 - one bad cell must not sink the table
                    row = {"shift": float(shift), "steps": int(steps), "error": f"{type(error).__name__}: {error}"}
                # abs() around the whole diff, not ``-diff.max()``: the nodes
                # descend, so every diff is negative and ``.max()`` returns the
                # *smallest* jump.  That version printed 0.0625 for the grid whose
                # largest jump is 0.625 -- the min, labelled max.
                nodes = shifted_sigmas(steps, shift, torch.device("cpu")).numpy()
                row["max_jump"] = float(np.abs(np.diff(nodes)).max())
                row["nodes"] = [float(x) for x in nodes]
                rows.append(row)
        return {"bin_edges_mm": [float(x) for x in edges], "rows": rows}

    @staticmethod
    def _score_shift(prediction, target, valid, gt, edges, *, shift, steps, base_ade):
        """ADE, split into "the whole hand moved" and "the hand shape broke".

        That split is the whole reason this table is readable.  A uniform
        translation of every keypoint leaves any ratio or cosine metric near 1.0,
        so those cannot see it -- and it is ~87% of the error here.  ``offset`` is
        the magnitude of the per-frame mean error vector (the block displacement),
        ``scatter`` the mean residual after removing it.  ``offset + scatter``
        overestimates ADE only when the residuals point in the same direction as
        the offset, so ``offset/ADE`` is a share, not an identity.
        """
        error = (prediction - target) * 1000.0  # [H,N,3] mm
        mask = valid[:, :, None].astype(np.float64)
        count = np.maximum(mask.sum(axis=1, keepdims=True), 1.0)  # [H,1,1]
        common = (error * mask).sum(axis=1, keepdims=True) / count  # [H,1,3]

        present = valid.any(axis=1)
        step_offset = np.linalg.norm(common[:, 0, :], axis=-1)  # [H]
        residual = np.linalg.norm(error - common, axis=-1)  # [H,N]
        step_scatter = np.array([residual[h][valid[h]].mean() if valid[h].any() else np.nan for h in range(len(valid))])
        ade = fk_visualize.trajectory_metrics(prediction, target, valid)["all_ade_mm"]
        offset = float(np.nanmean(step_offset[present]))
        scatter = float(np.nanmean(step_scatter[present]))

        predicted = np.linalg.norm(prediction, axis=-1) * 1000.0
        g, p = gt[valid], predicted[valid]
        bins = []
        for index in range(len(edges) - 1):
            inside = (g >= edges[index]) & (g <= edges[index + 1])
            bins.append(float(np.median(p[inside] / np.maximum(g[inside], 1e-9))) if inside.sum() else float("nan"))
        return {
            "shift": float(shift),
            "steps": int(steps),
            "all_ade_mm": float(ade),
            "ratio_to_zero": float(ade / base_ade) if base_ade else float("nan"),
            "offset_mm": offset,
            "scatter_mm": scatter,
            "offset_share": float(offset / ade) if ade else float("nan"),
            "pred_over_gt_bins": bins,
        }

    def _training_pack_control(self, model, *, sigma_value, pack_size=32):
        """The same field measurement inside a training-sized pack.

        Everything else has been ruled out -- weights (EMA and live agree), the
        batch (an ordinary validation batch is as bad as the fixed case), the
        sigma schedule, the sign convention.  What remains is the only structural
        difference between ``training_step`` and this probe: the number of samples
        sharing one packed sequence.  ``training_step`` packs 32; the fixed case
        packs 1 and a validation batch packs 2.

        That matters because the mRoPE temporal axis accumulates across a pack
        (``mrope.py``: after each segment the offset becomes ``max(positions)+1``),
        and ``fk_positions`` reads each sample's origin out of the packed sequence.
        RoPE is relative, so a uniform shift should cancel -- but that is exactly
        the assumption to test rather than assert, and the ``fk_loss`` that looks
        healthy is only ever computed inside a 32-sample pack.
        """
        import copy as _copy

        from omegaconf import OmegaConf

        from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader, custom_collate_fn
        from cosmos_framework.utils.lazy_config import instantiate

        loader = _copy.deepcopy(self.config.dataloader_train)
        loader_cfg = OmegaConf.to_container(loader, resolve=True) if OmegaConf.is_config(loader) else loader
        dataset_cfg = _copy.deepcopy(next(iter(loader_cfg["dataloader"]["datasets"].values()))["dataset"])
        dataset_cfg.update(iterable_shuffle=False, cfg_dropout_rate=0.0, use_image_augmentation=False)
        dataset = instantiate(dataset_cfg)
        samples = [dataset[i] for i in range(min(pack_size, len(dataset)))]
        pack_cfg = {k: v for k, v in loader_cfg.items() if k not in ("_target_", "dataloader")}
        # No override: use the recipe's own packing, which is the point.
        inner = torch.utils.data.DataLoader(samples, batch_size=1, num_workers=0, collate_fn=custom_collate_fn)
        batch = next(iter(PackingDataLoader(dataloader=inner, **pack_cfg)))
        measured = self._measure_on_batch(model, batch, sigma_value=sigma_value)
        self._last_training_pack_batch = batch
        measured["pack_size_requested"] = pack_size
        # The packer may drop or carry samples, so report what actually got packed.
        measured["sample_lens"] = (
            [int(n) for n in getattr(batch.packed_sequence, "sample_lens", [])][:8]
            if hasattr(batch, "packed_sequence")
            else None
        )
        return measured

    def _ema_vs_raw(self, model, cpu_batch, *, sigma_value):
        """Velocity error at one sigma under the EMA weights and under the live ones.

        Leaves the model exactly as ema_scope had it: the raw values re-cached and
        the EMA values re-copied, so the scope's own ``restore`` on exit still
        finds what it put there.  Getting that wrong aborts the run at the end of
        validation, so the two calls mirror the scope's pair in the same order.
        """
        worker = getattr(model, "net_ema_worker", None)
        if worker is None or not getattr(model.config.ema, "enabled", False):
            return {"error": "EMA not enabled"}
        with evaluation_rng(self.seed):
            ema_value = self._field_error(model, cpu_batch, sigma_value=sigma_value)
            # Back to the live weights for the duration of one measurement.
            worker.restore(model.net.parameters())
            try:
                raw_value = self._field_error(model, cpu_batch, sigma_value=sigma_value)
            finally:
                # Re-establish the scope exactly as ema_scope left it.
                worker.cache(model.net.parameters())
                worker.copy_to(src_model=model.net_ema, tgt_model=model.net)
        return {"sigma": float(sigma_value), "ema": ema_value, "raw": raw_value}

    def _field_error(self, model, cpu_batch, *, sigma_value, vision_sigma=None):
        """``|v_pred - v_ref|`` at one sigma on the fixed case's own true path."""
        target = np.asarray(cpu_batch["fk"][0]["targets"]["displacement"], np.float64)
        context = model._fk_fitting_context(_to_cuda(cpu_batch), vision_sigma=vision_sigma)  # noqa: SLF001
        truth = context.data.displacement.float()  # metres, straight off the labels
        scale = float(context.scale)  # metres per model unit
        generator = torch.Generator(device=context.device).manual_seed(int(self.seed))
        epsilon = torch.randn(truth.shape, generator=generator, device=context.device, dtype=torch.float32)
        sigma = torch.full((context.batch_size,), float(sigma_value), device=context.device)
        per_point = sigma[context.data.inputs["point_batch"]].reshape(1, -1, 1)
        state = per_point * epsilon + (1 - per_point) * (truth / scale)
        reference = epsilon - truth / scale
        velocity = context.velocity(state, sigma).float()
        return {
            "velocity_error_mm": _to_mm(velocity - reference, scale),
            "magnitude_ratio": float(velocity.norm(dim=-1).mean() / reference.norm(dim=-1).mean()),
            "target_norm_mm": float(np.linalg.norm(target, axis=-1).mean() * 1000.0),
        }

    def _vision_conditioning_ab(self, model, cpu_batch, *, sigma_value):
        """The same field, same sigma, same state -- video clean vs video noised.

        Everything else has been excluded by measurement: the weights (live and
        EMA agree), the sigma (the sweep is flat), the loss arithmetic (fk_loss
        matches a hand computation to the last digit), the target, the pack size,
        and the token positions (``input_timestep`` only ever reaches
        ``vision.timesteps``, never ``vision_mrope_ids``).  What remains is the
        one thing ``_fk_fitting_context`` does differently from ``training_step``:
        it packs the video **clean** at timestep 0, while training noises the
        non-conditioning frames to the *same* sigma the FK state is at, and the
        FK tokens attend to exactly those frames.

        The consequence is checkable in the logged history: the training-time
        per-sigma velocity error in the (0.8, 0.9) bin is ~27mm, while this probe
        measures ~1200mm at 0.8333 -- same bin, same weights.  If the noised arm
        lands near the logged value and the clean arm stays near 1200mm, the
        conditioning is the whole difference and the branch itself is healthy.

        The two arms are seeded identically so they draw the *same* vision
        epsilon, leaving the conditioning as the only moving part.
        """
        rows = {}
        for label, vision_sigma in (("video_clean", None), ("video_noised", sigma_value)):
            torch.manual_seed(int(self.seed))
            try:
                rows[label] = self._field_error(model, cpu_batch, sigma_value=sigma_value, vision_sigma=vision_sigma)
            except Exception as error:  # noqa: BLE001 - one arm failing still reports the other
                rows[label] = {"error": f"{type(error).__name__}: {error}"}
        clean, noised = rows.get("video_clean", {}), rows.get("video_noised", {})
        if "velocity_error_mm" in clean and "velocity_error_mm" in noised:
            rows["ratio_clean_over_noised"] = clean["velocity_error_mm"] / max(noised["velocity_error_mm"], 1e-9)
        return rows

    def _measure_on_batch(self, model, data_batch, *, sigma_value):
        """Velocity error at one sigma on an arbitrary batch, training-style.

        Same arithmetic as ``fk_add_noise`` and the same context builder the
        sampler uses, so the number is directly comparable to ``fk_loss``'s
        per-sigma bins and to the field measured on the fixed case.
        """
        context = model._fk_fitting_context(_to_cuda(data_batch))  # noqa: SLF001 - same module family
        truth = context.data.displacement.float()
        scale = float(context.scale)
        generator = torch.Generator(device=context.device).manual_seed(1234)
        epsilon = torch.randn(truth.shape, generator=generator, device=context.device, dtype=torch.float32)
        sigma = torch.full((context.batch_size,), float(sigma_value), device=context.device)
        per_point = sigma[context.data.inputs["point_batch"]].reshape(1, -1, 1)
        state = per_point * epsilon + (1 - per_point) * (truth / scale)
        reference = epsilon - truth / scale
        velocity = context.velocity(state, sigma)
        delta = _to_mm(velocity - reference, scale)
        return {
            "sigma": float(sigma_value),
            "n_points": int(truth.shape[1]),
            "batch_size": int(context.batch_size),
            "target_norm_mm": float(truth.norm(dim=-1).mean() * 1000.0),
            "reference_velocity_norm_mm": _to_mm(reference, scale),
            "model_velocity_norm_mm": _to_mm(velocity, scale),
            "velocity_error_mm": float(delta),
            "magnitude_ratio": float(velocity.float().norm(dim=-1).mean() / reference.norm(dim=-1).mean()),
        }

    def _print(self, report):
        log.info(f"--- FK sampler probe: {report['case_id']} @ {report['iteration']} ---")
        log.info(
            f"  replay all_ade {report['replay']['all_ade_mm']:.1f}mm   "
            f"zero {report['replay']['zero_all_ade_mm']:.1f}mm   "
            f"unit-noise {report['unit_noise_mm']:.1f}mm   "
            f"pure-noise? {report['looks_like_pure_noise']}"
        )
        log.info(f"  reference |v| = {report['reference_velocity_norm_mm']:.1f}mm")
        cross = report.get("loss_crosscheck") or {}
        if "fk_loss_mse" in cross:
            log.info(
                f"  LOSS CROSSCHECK (n={cross['n_points']}, valid={cross['valid_fraction']:.0%}, "
                f"sigma={cross['sigma']:.3f}): fk_loss gives MSE {cross['fk_loss_mse']:.6g} "
                f"(-> |d| {cross['by_hand_norm_mm']:.1f}mm by its own prediction), "
                f"by hand {cross['by_hand_mse']:.6g}  ratio {cross['ratio']:.4g}"
            )
            log.info(f"  LOSS CROSSCHECK: fk_loss's own fk_ade says {cross['fk_ade_mm_reported']:.1f}mm")
        elif cross:
            log.info(f"  LOSS CROSSCHECK unavailable: {cross.get('error')}")
        trainer_vals = report.get("trainer_on_this_batch") or {}
        cross = report.get("loss_crosscheck") or {}
        if trainer_vals:
            log.info("  TRAINER ON THIS BATCH (same data_batch, its own sigma):")
            for k in ("flow_matching_loss_fk", "fk_ade_mm", "fk_zero_ade_mm"):
                if k in trainer_vals:
                    log.info(f"      {k:24s} = {trainer_vals[k]:.6g}")
            if "fk_loss_mse" in cross:
                log.info(f"      probe fk_loss on a batch it built = {cross['fk_loss_mse']:.6g}")
                if "flow_matching_loss_fk" in trainer_vals:
                    ratio = trainer_vals["flow_matching_loss_fk"] / max(cross["fk_loss_mse"], 1e-30)
                    log.info(f"      -> ratio trainer/probe = {ratio:.4g}")
                    if ratio < 0.1:
                        log.info("      -> the trainer's FK loss is far smaller than this probe's on")
                        log.info("         equivalent tensors: its preds_fk is not what the sampler gets.")
        sh = report.get("shrinkage_vs_sigma") or {}
        if sh and "error" not in sh:
            log.info("  MAGNITUDE COMPRESSION vs SIGMA  (|x0_hat|/|target|, median)")
            log.info(
                f"      n={sh['n']} 真值分档(mm): "
                + "  ".join(
                    f"{sh['bin_edges_mm'][i]:.0f}-{sh['bin_edges_mm'][i + 1]:.0f}"
                    for i in range(len(sh["bin_edges_mm"]) - 1)
                )
            )
            for row in sh["rows"]:
                log.info(
                    f"      sigma {row['sigma']:.2f}  整体 {row['overall']:.3f} | "
                    + "  ".join(f"{b:.3f}" if b == b else "  -  " for b in row["bins"])
                )
            log.info("      -> 每行内部若 Q1>Q4 => 该 sigma 下仍是回归到均值;")
            log.info("      -> 若低 sigma 的行趋于平坦且接近 1 => 压缩由高 sigma 主导,采样起点可改。")
        elif sh:
            log.info(f"  MAGNITUDE COMPRESSION vs SIGMA unavailable: {sh.get('error')}")

        pd = report.get("pack_diff") or {}
        if pd and "error" not in pd:
            log.info("  TRAINER PACK vs PROBE REBUILD (same FK state, only the pack moves):")
            log.info(
                f"      trainer pack : err {pd.get('trainer_pack_error_mm'):.1f}mm  mse {pd.get('trainer_pack_mse'):.6g}"
            )
            if pd.get("probe_pack_error_mm") is not None:
                log.info(f"      probe rebuild: err {pd['probe_pack_error_mm']:.1f}mm  mse {pd['probe_pack_mse']:.6g}")
                log.info(f"      preds max|diff| = {pd.get('preds_max_abs_diff'):.6g}")
            else:
                log.info(f"      probe rebuild: {pd.get('shape_mismatch')}")
            if pd.get("redenoise_max_abs_diff") is not None:
                log.info(f"      re-denoise determinism: max|diff| = {pd['redenoise_max_abs_diff']:.6g}")
            diff = pd.get("field_diff") or []
            log.info(f"      FIELDS THAT DIFFER ({len(diff)}):")
            for path, why in diff[:25]:
                log.info(f"        {path:52s} {why}")
        elif pd:
            log.info(f"  TRAINER PACK vs PROBE REBUILD unavailable: {pd.get('error')}")

        ab = report.get("vision_conditioning_ab") or {}
        if "video_clean" in ab:
            log.info("  VISION CONDITIONING A/B (same state, sigma, and vision epsilon):")
            for label, arm in (
                ("video clean  (eval path)", ab.get("video_clean")),
                ("video noised (training path)", ab.get("video_noised")),
            ):
                if arm and "velocity_error_mm" in arm:
                    log.info(
                        f"    {label:30s} err {arm['velocity_error_mm']:8.1f}mm  ratio {arm['magnitude_ratio']:5.3f}"
                    )
                elif arm:
                    log.info(f"    {label:30s} {arm.get('error')}")
            if "ratio_clean_over_noised" in ab:
                log.info(f"    -> clean/noised = {ab['ratio_clean_over_noised']:.1f}x")
                log.info("    -> near 1 means the conditioning is not it; large means the eval")
                log.info("       asks a (video, state) pair training never produced.")

        sweep = report.get("sigma_sweep") or []
        if sweep:
            log.info("  SIGMA SWEEP (training-scale pack, same eps every row):")
            for row in sweep:
                bar = "#" * min(int(row["velocity_error_mm"] / 60), 40)
                log.info(
                    f"    sigma {row['sigma']:.4f}  err {row['velocity_error_mm']:8.1f}mm  "
                    f"ratio {row['magnitude_ratio']:5.3f}  cos {row['cosine']:+.3f}  {bar}"
                )
        pack = report.get("training_pack") or {}
        if "velocity_error_mm" in pack:
            log.info(
                f"  TRAINING PACK (n={pack['n_points']} pts, batch={pack['batch_size']}, "
                f"sigma={pack['sigma']:.3f}): velocity error {pack['velocity_error_mm']:.1f}mm, "
                f"mag ratio {pack['magnitude_ratio']:.3f}, target |x| {pack['target_norm_mm']:.1f}mm"
            )
            log.info("  -> compare with the 1-sample fixed case above: if this is the small")
            log.info("     one, the pack size is the difference and fk_loss never saw eval's.")
        elif pack:
            log.info(f"  TRAINING PACK unavailable: {pack.get('error')}")
        pair = report.get("ema_vs_raw") or {}
        if "ema" in pair:
            e, r = pair["ema"], pair["raw"]
            log.info(
                f"  EMA vs RAW (fixed case, sigma={pair['sigma']:.3f}): "
                f"ema error {e['velocity_error_mm']:.1f}mm (ratio {e['magnitude_ratio']:.3f})  |  "
                f"raw error {r['velocity_error_mm']:.1f}mm (ratio {r['magnitude_ratio']:.3f})"
            )
            if r["velocity_error_mm"] < 0.25 * e["velocity_error_mm"]:
                log.info("  -> the LIVE weights are accurate and the EMA copy is not:")
                log.info("     that is what fk_loss measured, and the eval never saw it.")
        elif pair:
            log.info(f"  EMA vs RAW unavailable: {pair.get('error')}")
        control = report.get("control_batch") or {}
        if "velocity_error_mm" in control:
            log.info(
                f"  CONTROL (ordinary val batch, sigma={control['sigma']:.3f}): "
                f"velocity error {control['velocity_error_mm']:.1f}mm, "
                f"mag ratio {control['magnitude_ratio']:.3f}, "
                f"target |x| {control['target_norm_mm']:.1f}mm, n={control['n_points']}"
            )
            fixed = report["velocity_field_on_true_path"]
            near = min(fixed, key=lambda r: abs(r["sigma"] - control["sigma"]))
            err = (
                abs(
                    near["model_velocity_norm_mm"] ** 2
                    + near["reference_velocity_norm_mm"] ** 2
                    - 2
                    * near["model_velocity_norm_mm"]
                    * near["reference_velocity_norm_mm"]
                    * near["cosine_to_reference"]
                )
                ** 0.5
            )
            log.info(
                f"  CONTROL (fixed case, same sigma): velocity error {err:.1f}mm, "
                f"mag ratio {near['magnitude_ratio']:.3f}, target |x| "
                f"{report['target_norm_mm']:.1f}mm, n=21"
            )
            log.info("  -> if these two differ by ~20x, the branch is fine on ordinary")
            log.info("     batches and the fixed case is the anomaly, and the logged")
            log.info("     fk_loss was never measuring the windows the verdict uses.")
        elif control:
            log.info(f"  CONTROL unavailable: {control.get('error')}")
        log.info(f"  {'sampler':>8} {'steps':>6} {'all_ade':>10} {'final_ade':>10} {'ratio':>7}")
        for row in report["sampler_sweep"]:
            log.info(
                f"  {row['sampler']:>8} {row['steps']:>6} {row['all_ade_mm']:>10.1f} "
                f"{row['final_ade_mm']:>10.1f} {row['ratio_to_zero']:>7.3f}"
            )
        grid = report.get("shift_steps_sweep") or {}
        if grid and "error" not in grid:
            edges = grid["bin_edges_mm"]
            log.info("  SHIFT x STEPS (same case, same seed; unipc):")
            log.info(
                "      shift steps    ADE   ratio   offset  scatter  off%   "
                + "  ".join(f"{edges[i]:.0f}-{edges[i + 1]:.0f}" for i in range(len(edges) - 1))
                + "   maxjump"
            )
            for row in grid["rows"]:
                if "error" in row:
                    log.info(f"      {row['shift']:5.1f} {row['steps']:5d}   {row['error']}")
                    continue
                log.info(
                    f"      {row['shift']:5.1f} {row['steps']:5d} "
                    f"{row['all_ade_mm']:7.1f} {row['ratio_to_zero']:7.3f} "
                    f"{row['offset_mm']:8.1f} {row['scatter_mm']:8.1f} "
                    f"{row['offset_share']:5.0%}   "
                    + "  ".join(f"{b:9.3f}" if b == b else "     -   " for b in row["pred_over_gt_bins"])
                    + f"   {row['max_jump']:.4f}"
                )
            log.info(f"      -> 真值分档(mm): {'  '.join(f'{x:.0f}' for x in edges)}")
            log.info("      -> ratio<1 = 比'预测不动'好; off% 高 = 误差是整只手平移而非手型坏;")
            log.info("      -> pred/gt 列越接近 1 越好,<1 就是幅度被压缩; maxjump 是采样器最大单步跨度。")
        elif grid:
            log.info(f"  SHIFT x STEPS unavailable: {grid.get('error')}")

        log.info(f"  {'sigma':>7} {'|v_model|':>10} {'ratio':>7} {'cosine':>8}")
        for row in report["velocity_field_on_true_path"]:
            log.info(
                f"  {row['sigma']:>7.3f} {row['model_velocity_norm_mm']:>10.1f} "
                f"{row['magnitude_ratio']:>7.3f} {row['cosine_to_reference']:>+8.3f}"
            )
