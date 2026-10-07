"""FKBranch structure and dtype contract.

Both tests here pin a bug the GPU smoke test found that the float32 CPU path
could not: the training dtype is bf16 while the noised displacement arrives
float32 by design, and the mismatch only surfaces at the first Linear.
"""

import pytest
import torch

from cosmos_framework.data.fk_batch import build_fk_batch
from cosmos_framework.data.fk_window import FKTiming
from cosmos_framework.model.generator.fk_branch import FKBranch, motion_blocks, restore_motion

TIMING = FKTiming()
KEYPOINTS = 21
HIDDEN = 32


def fk_sample(n=KEYPOINTS, *, timing=TIMING):
    return {
        "inputs": {
            "anchor_xyz": torch.linspace(0.1, 0.9, n * 3).reshape(n, 3).float(),
            "point_ids": torch.arange(n),
        },
        "targets": {
            "displacement": torch.zeros(timing.steps, n, 3),
            "valid": torch.ones(timing.steps, n, dtype=torch.bool),
        },
        "metadata": {"timing": timing},
    }


def branch(dtype=torch.float32):
    return FKBranch(timing=TIMING, hidden_dim=HIDDEN, index_dim=8, content_dim=16, decoder_dim=16).to(dtype=dtype)


def test_motion_blocks_round_trip_preserves_chronology():
    displacement = torch.arange(TIMING.steps * 4 * 3, dtype=torch.float32).reshape(TIMING.steps, 4, 3)
    blocks = motion_blocks(displacement, TIMING)
    assert blocks.shape == (TIMING.blocks, 4, 3 * TIMING.steps_per_token)
    assert torch.equal(restore_motion(blocks, TIMING), displacement)
    # Block b must hold steps [b*q, b*q+q), in order.
    assert torch.equal(blocks[1, 0], displacement[TIMING.steps_per_token : 2 * TIMING.steps_per_token, 0].flatten())


def test_branch_encodes_and_decodes_in_bfloat16():
    """The training dtype.  ``fk_add_noise`` builds xt in float32 on purpose
    (metric geometry stays out of the model dtype), so the branch has to cast it
    -- and a CPU float32 run never exercises that."""
    batch = build_fk_batch([fk_sample(), fk_sample()], batch_size=2)
    net = branch(torch.bfloat16)

    # Exactly what training hands over: float32 state, bf16 parameters.
    noisy = torch.zeros(TIMING.steps, 2 * KEYPOINTS, 3, dtype=torch.float32)
    sigma = torch.tensor([0.3, 0.7], dtype=torch.float32)

    encoded = net.encode(batch, noisy, sigma)
    assert encoded["anchor_tokens"].dtype == torch.bfloat16
    assert encoded["noisy_tokens"].shape == (TIMING.blocks, 2 * KEYPOINTS, HIDDEN)
    assert torch.isfinite(encoded["noisy_tokens"]).all()

    hidden = torch.randn(TIMING.blocks, 2 * KEYPOINTS, HIDDEN, dtype=torch.bfloat16)
    velocity = net.decode(hidden, noisy, sigma, batch.inputs["point_batch"], 2)
    assert velocity.shape == (TIMING.steps, 2 * KEYPOINTS, 3)
    assert torch.isfinite(velocity).all()


def test_decoder_input_width_matches_what_encode_concatenates():
    """sigma is `hidden_dim` wide, not a scalar; the decoder's first Linear has to
    be sized for backbone-hidden + block + sigma."""
    net = branch()
    first = net.decoder[0]
    assert first.in_features == 2 * HIDDEN + 3 * TIMING.steps_per_token


def test_encoder_adds_identity_and_position_rather_than_concatenating():
    """``W_idx . e_idx(i) + MLP_xyz(p)``: two separately-encoded halves in
    different subspaces, mirroring the PointFlow codec.  If this were a
    concatenation the position half could not be ablated on its own."""
    net = branch()
    xyz = torch.zeros(KEYPOINTS, 3)
    index = torch.arange(KEYPOINTS)
    everything = net.encoder(xyz, index)
    # Zeroing the xyz must change the output only through the position branch, and
    # leave the identity branch intact.
    net.encoder.xyz_encoder[0].weight.data.zero_()
    net.encoder.xyz_encoder[0].bias.data.zero_()
    net.encoder.xyz_encoder[2].weight.data.zero_()
    net.encoder.xyz_encoder[2].bias.data.zero_()
    without_position = net.encoder(xyz, index)
    assert not torch.allclose(everything, without_position)
    assert torch.allclose(without_position, net.encoder.index_projection(net.encoder.index_embedding(index))), (
        "identity branch is not independent of the position branch"
    )


def test_encoder_rejects_mismatched_shapes():
    net = branch()
    with pytest.raises(ValueError, match="disagree"):
        net.encoder(torch.zeros(KEYPOINTS, 3), torch.arange(KEYPOINTS - 1))
    with pytest.raises(ValueError, match=r"expected \[\.\.\., 3\]"):
        net.encoder(torch.zeros(KEYPOINTS, 4), torch.arange(KEYPOINTS))


def test_encode_rejects_a_wrongly_sized_noisy_state():
    batch = build_fk_batch([fk_sample()], batch_size=1)
    with pytest.raises(ValueError, match="cover every keypoint"):
        branch().encode(batch, torch.zeros(TIMING.steps, KEYPOINTS + 1, 3), torch.tensor([0.5]))


def test_decode_rejects_hidden_of_the_wrong_width():
    batch = build_fk_batch([fk_sample()], batch_size=1)
    noisy = torch.zeros(TIMING.steps, KEYPOINTS, 3)
    with pytest.raises(ValueError, match="Expected FK point hidden"):
        branch().decode(
            torch.zeros(TIMING.blocks, KEYPOINTS, HIDDEN + 1),
            noisy,
            torch.tensor([0.5]),
            batch.inputs["point_batch"],
            1,
        )
