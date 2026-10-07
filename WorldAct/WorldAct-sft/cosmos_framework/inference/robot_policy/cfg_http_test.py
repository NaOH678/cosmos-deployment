"""CPU-only transport contract tests for the isolated experimental service."""
import numpy as np
import pytest

from cosmos_framework.inference.robot_policy.cfg_http import action_digest, validate_ack


def test_ack_requires_exact_sequence_and_output():
    actions = np.zeros((32, 27), dtype=np.float32)
    ack = {"kind": "complete", "sequence": 7, "action_sha256": action_digest(actions)}
    validate_ack(ack, 7, actions)
    with pytest.raises(RuntimeError, match="acknowledgement"):
        validate_ack(ack, 8, actions)
    actions[0, 0] = 1
    with pytest.raises(RuntimeError, match="different actions"):
        validate_ack(ack, 7, actions)
    with pytest.raises(RuntimeError):
        validate_ack(None, 7, actions)
