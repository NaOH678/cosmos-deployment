"""The independent FK sigma schedule must actually reach ``fk_add_noise``.

Why this needs a guard rather than a comment.  FK's sigma used to be the video's
sigma, unconditionally.  That coupling is invisible in every training metric --
the loss falls, the per-sigma bins fall -- and only shows up as the sampled
trajectory collapsing onto its own initial noise.  So the failure mode of a
broken *fix* is equally quiet: if the new sigma is computed and then dropped on
the floor, or the config flag is misspelled, training still runs and still looks
healthy, and the only symptom is that nothing got better.

The check is static (AST) on purpose: it needs no GPU and no checkpoint, so it can
run before a multi-hour launch, which is the only time it is worth anything.

    PYTHONPATH=. python cosmos_framework/model/generator/fk_independent_schedule_test.py
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
# Running `python <this file>` puts *this file's directory* on sys.path[0], and that
# directory holds a ``tokenizers/`` package which then shadows the installed
# ``tokenizers`` for every later import (transformers needs the real one).  Drop it
# before importing anything from the repo, then put the repo root first.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(REPO))

MODEL = REPO / "cosmos_framework/model/generator/omni_mot_model.py"
DEFAULTS = {
    "EDGE_DROID_MODEL_PATH": "/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/models/cosmos3-edge-droid",
    "BASE_CHECKPOINT_PATH": "/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/models/cosmos3-edge-droid-dcp",
    "WAN_VAE_PATH": "/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/models/cosmos3-edge-droid/vae/Wan2.2_VAE.pth",
    "SINGLERIGHTHAND_RAW_ROOT": "/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/raw_data/singlerighthand_sandwich_100",
    "SINGLERIGHTHAND_CACHE_ROOT": "/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/datasets/singlerighthand-sandwich-100-cache",
    "FK_ANNOTATION_ROOT": "/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/raw_data/sandwich_fk21",
}


def _functions(tree):
    return {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}


def _text(node) -> str:
    return ast.unparse(node)


def main() -> int:
    tree = ast.parse(MODEL.read_text())
    functions = _functions(tree)
    failures = []

    def check(name, condition, detail):
        print(f"  {'✅' if condition else '❌'} {name}")
        if not condition:
            failures.append(f"{name}: {detail}")

    # 1. training_step computes the independent sigma, gated on the config flag.
    step = functions.get("training_step")
    check("training_step exists", step is not None, "not found")
    body = _text(step) if step else ""
    check(
        "training_step gates on independent_fk_schedule",
        "independent_fk_schedule" in body,
        "the flag never reaches training_step",
    )
    check(
        "training_step calls _get_train_noise_level_fk",
        "_get_train_noise_level_fk" in body,
        "the sigma is never drawn",
    )
    check(
        "training_step forwards sigmas_fk",
        "sigmas_fk=sigmas_fk" in body or "sigmas_fk=" in body,
        "the drawn sigma is not passed to _add_noise_to_input",
    )

    # 2. the sampler exists and draws from the vision RF sampler (same marginal).
    sampler = functions.get("_get_train_noise_level_fk")
    check("_get_train_noise_level_fk exists", sampler is not None, "not found")
    if sampler:
        sampler_body = _text(sampler)
        check(
            "_get_train_noise_level_fk uses rectified_flow_video",
            "rectified_flow_video" in sampler_body and "sample_train_time" in sampler_body,
            "must draw from the vision sampler to keep the marginal identical",
        )
        check("_get_train_noise_level_fk applies the shift", "shift" in sampler_body, "shift not applied")

    # 3. _add_noise_to_input accepts it and uses it *in the FK branch*.
    noise = functions.get("_add_noise_to_input")
    check("_add_noise_to_input exists", noise is not None, "not found")
    if noise:
        args = [a.arg for a in noise.args.args]
        check("_add_noise_to_input takes sigmas_fk", "sigmas_fk" in args, f"args = {args}")
        noise_body = _text(noise)
        check(
            "_add_noise_to_input selects sigmas_fk over sigmas",
            "sigmas if sigmas_fk is None else sigmas_fk" in noise_body,
            "the override is accepted but never used",
        )

    # 4. the flag exists in the defaults, and the recipe turns it on.
    from cosmos_framework.configs.base.defaults.model_config import RectifiedFlowTrainingConfig

    check(
        "default independent_fk_schedule exists and is off",
        getattr(RectifiedFlowTrainingConfig(), "independent_fk_schedule", None) is False,
        "the default must stay False so other experiments are unaffected",
    )

    os.environ["FK_ENCODER_CHECKPOINT"] = "1"
    for key, value in DEFAULTS.items():
        os.environ.setdefault(key, value)
    os.environ.setdefault(
        "SINGLERIGHTHAND_EPISODE_ALLOWLIST", str(REPO / "examples/pointflow_sandwich_10_episodes.txt")
    )
    os.environ.setdefault(
        "SINGLERIGHTHAND_VAE_WINDOW_LATENT_ROOT",
        os.path.join(DEFAULTS["SINGLERIGHTHAND_CACHE_ROOT"], "vae_window_latents"),
    )
    from omegaconf import OmegaConf

    from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_singlerighthand_edge import (
        action_policy_singlerighthand_edge as cfg,
    )

    rf = OmegaConf.to_container(cfg["model"]["config"], resolve=True)["rectified_flow_training_config"]

    # 5. The sigma schedule is ONE decision, not four.
    #
    # Every modality in a sample either draws its own sigma or shares the video's, and
    # the two settings answer different questions:
    #
    #   all True  -- merge doc §2's fix.  Every modality's sigma is independent, so
    #               "clean video + noisy modality" is inside the training distribution
    #               and the CONDITIONING arm can be read as a real number.
    #   all False -- merge doc §D4 (the current setting).  All five modalities share
    #               one sigma per sample, which matches the deployment condition where
    #               video/action/pointflow/FK are denoised in one loop.  The cost is
    #               §2's pathology coming back: the conditioning arm is out of
    #               distribution and its ratios are NOT comparable -- so it is a
    #               ceiling, not deployable performance.
    #
    # What must never happen is a MIX.  One modality on its own stream and the rest
    # shared is a schedule neither §2 nor §D4 describes, it is silent (training runs,
    # loss falls), and a single flipped switch is exactly how it would appear.  The
    # two checks below close both ends: agreement catches the mix in either direction,
    # and the value check catches a uniform flip away from the documented decision.
    SWITCHES = (
        "independent_action_schedule",
        "independent_sound_schedule",
        "independent_pointflow_schedule",
        "independent_fk_schedule",
    )
    values = {key: rf[key] for key in SWITCHES}
    check(
        "all four independent_*_schedule switches agree",
        len(set(values.values())) == 1,
        f"a MIXED schedule is a configuration neither §2 nor §D4 describes: {values}",
    )
    check(
        "the FK recipe matches §D4 (shared sigma, so all four are False)",
        values["independent_fk_schedule"] is False,
        f"independent_fk_schedule resolved to {values['independent_fk_schedule']!r} "
        f"(all four: {values}).  If this flip was deliberate, it is a §D4-level decision, "
        "not a wiring fix: True restores §2's fix and makes the conditioning arm "
        "readable again, False aligns with the fused inference path and makes it "
        "incomparable.  Update this check together with the decision.",
    )

    print()
    if failures:
        print("❌ 接线断了：")
        for item in failures:
            print(f"   - {item}")
        return 1
    print("✅ 独立 FK σ 调度接线完整")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
