"""Sampling boundary units: meters in/out, model units inside the sampler."""

from types import SimpleNamespace as NS

import torch

from cosmos_framework.model.generator.fk_sampling import joint_layout
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel


def test_clean_fk_context_uses_model_units():
    model = NS(config=NS(rectified_flow_training_config=NS(fk_displacement_scale=0.1)))
    packed = NS(fk_data=NS(inputs={"point_ids": torch.arange(3)}, displacement=torch.full((2, 3, 3), 0.2)))
    OmniMoTModel._attach_clean_fk_state(model, packed, NS(batch_size=1))
    torch.testing.assert_close(packed.fk_noised.xt, torch.full((2, 3, 3), 2.0))
    assert packed.fk_noised.sigma.item() == 0


def test_joint_outputs_restore_each_modality_own_scale(monkeypatch):
    monkeypatch.delenv("POINTFLOW_DISPLACEMENT_FRAME_SCALES", raising=False)
    monkeypatch.delenv("POINTFLOW_DISPLACEMENT_FRAME_SCALES_FILE", raising=False)
    vision = torch.full((1, 2, 2, 2), 3.0)
    action = torch.full((2, 4), 4.0)
    pf = torch.full((2, 5, 3), 5.0)
    fk = torch.full((2, 3, 3), 6.0)
    context = NS(
        packed=NS(vision=NS(tokens=[vision]), action=NS(tokens=[action])),
        layout=joint_layout([vision.shape], [(0, 3)], 2, [action.shape], [pf.shape]),
        joint_action=True,
        seeded_pointflow=[pf],
        horizon=2,
        joint_velocity=None,
        shift=5.0,
        scale=0.1,
    )
    model = NS(
        config=NS(rectified_flow_training_config=NS(pointflow_displacement_scale=0.2)),
        sampler=lambda velocity, state, **kwargs: state,
    )
    result = OmniMoTModel._sample_joint(model, context, fk, steps=1, seed=17)
    torch.testing.assert_close(result.vision[0], vision)
    torch.testing.assert_close(result.action[0], action)
    torch.testing.assert_close(result.pointflow[0], pf * 0.2)
    torch.testing.assert_close(result.fk, fk * 0.1)
