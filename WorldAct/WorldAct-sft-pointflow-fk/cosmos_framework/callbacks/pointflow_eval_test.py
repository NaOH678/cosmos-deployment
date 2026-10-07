"""Action-ablation wiring: the dataloader's nested action form must be fully zeroed."""

import pytest
import torch

from cosmos_framework.callbacks.pointflow_eval import zero_actions_


def test_zeroes_every_tensor_in_the_nested_form():
    """`[[tensor], [None], [tensor]]` is what the joint dataloader hands the model."""
    first = torch.ones(4, 7)
    third = torch.full((4, 7), 3.0)
    batch = {"action": [[first], [None], [third]], "video": [torch.ones(2)]}

    zero_actions_(batch)

    assert not first.any() and not third.any()
    assert batch["action"][1] == [None], "None placeholders must survive"
    assert len(batch["action"]) == 3, "the per-sample structure must not collapse"
    assert batch["video"][0].any(), "only the action is ablated"


def test_handles_a_bare_tensor_and_a_dict():
    bare = torch.ones(2, 3)
    zero_actions_({"action": bare})
    assert not bare.any()

    nested = torch.ones(2, 3)
    zero_actions_({"action": {"inner": [nested]}})
    assert not nested.any()


def test_refuses_a_batch_without_an_action():
    """Silently doing nothing would read as 'the branch ignores the action'."""
    with pytest.raises(ValueError, match="no 'action'"):
        zero_actions_({"video": [torch.ones(1)]})
    with pytest.raises(ValueError, match="no 'action'"):
        zero_actions_({"action": None})
