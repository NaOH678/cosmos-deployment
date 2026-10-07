"""FK token placement and mRoPE positions: the stage-3 gate.

Three things must hold, and none of them raises on its own downstream:

1. the token count is 21 anchor + 8 blocks x 21 = 189 per sample;
2. the temporal axis sits on the *video's* lattice -- block ``b`` shares its
   timestamp with video latent ``b``.  A small drift here trains happily and
   learns nothing, and the loss curve looks the same;
3. the spatial axes are zero, produced the same way action's are.
"""

import torch

from cosmos_framework.data.fk_window import FKTiming
from cosmos_framework.data.generator.sequence_packing.modality import ModalityData, ModalitySpan
from cosmos_framework.data.generator.sequence_packing.mrope import get_3d_mrope_ids_vae_tokens
from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequence
from cosmos_framework.model.generator.fk_branch import FKBranch
from cosmos_framework.model.generator.fk_sequence import attach_fk_tokens, fk_positions

TIMING = FKTiming()
KEYPOINTS = 21
TEXT_LEN = 2
GRID_T, GRID_H, GRID_W = 9, 2, 2
VISION_LEN = GRID_T * GRID_H * GRID_W
ORIGIN = 2.0  # the temporal offset the text tokens push the video to


def build_sequence(samples=1, *, with_action=False):
    """A minimal two-way packed sequence: [causal text | full vision (+action)]."""
    text_ids = torch.arange(TEXT_LEN, dtype=torch.float32)  # t = 0, 1
    video, _ = get_3d_mrope_ids_vae_tokens(
        grid_t=GRID_T,
        grid_h=GRID_H,
        grid_w=GRID_W,
        temporal_offset=ORIGIN,
        fps=TIMING.fps,
        base_fps=24.0,
        temporal_compression_factor=TIMING.steps_per_token,
    )
    video = video.float()

    action = None
    full_extra = 0
    if with_action:
        # Built exactly the way Cosmos builds action positions: a degenerate 1x1
        # spatial grid, so its h/w are zero too.
        action_pos, _ = get_3d_mrope_ids_vae_tokens(
            grid_t=GRID_T,
            grid_h=1,
            grid_w=1,
            temporal_offset=ORIGIN,
            fps=TIMING.fps,
            base_fps=24.0,
            temporal_compression_factor=1,
        )
        action_pos = action_pos.float()
        action = ModalityData(
            sequence_indexes=torch.arange(TEXT_LEN + VISION_LEN, TEXT_LEN + VISION_LEN + GRID_T),
            mse_loss_indexes=torch.arange(TEXT_LEN + VISION_LEN, TEXT_LEN + VISION_LEN + GRID_T),
            spans=[
                ModalitySpan(
                    sequence_start=TEXT_LEN + VISION_LEN,
                    sequence_len=GRID_T,
                    payload_index=0,
                    payload_start=0,
                    payload_len=GRID_T,
                    payload_shape=(GRID_T,),
                )
            ],
        )
        full_extra = GRID_T

    sample_len = TEXT_LEN + VISION_LEN + full_extra

    # Per-sample slices: a two-sample pack repeats the whole layout, it does not
    # lengthen sample 0's vision span.
    def at(sample, offset, length):
        return torch.arange(sample * sample_len + offset, sample * sample_len + offset + length)

    text_indexes = torch.cat([at(s, 0, TEXT_LEN) for s in range(samples)])
    vision_indexes = torch.cat([at(s, TEXT_LEN, VISION_LEN) for s in range(samples)])
    position_ids = torch.cat([torch.cat((text_ids[None].expand(3, -1), video), dim=1) for _ in range(samples)], dim=1)
    if with_action:
        position_ids = torch.cat([position_ids, action_pos], dim=1)

    sequence = PackedSequence(
        sample_lens=[sample_len] * samples,
        split_lens=[TEXT_LEN, VISION_LEN + full_extra] * samples,
        attn_modes=["causal", "full"] * samples,
        sequence_length=sample_len * samples,
        text_ids=text_ids,
        text_indexes=text_indexes,
        position_ids=position_ids,
        vision=ModalityData(
            sequence_indexes=vision_indexes,
            mse_loss_indexes=vision_indexes,
            spans=[
                ModalitySpan(
                    sequence_start=int(vision_indexes[s * VISION_LEN]),
                    sequence_len=VISION_LEN,
                    payload_index=0,
                    payload_start=0,
                    payload_len=VISION_LEN,
                    payload_shape=(GRID_T, GRID_H, GRID_W),
                )
                for s in range(samples)
            ],
        ),
        action=action,
    )
    return sequence, video


def encode_fixture(sequence, keypoints=KEYPOINTS):
    """A stand-in for FKBranch.encode's output, with recognisable constants."""
    samples = len(sequence.sample_lens)
    n = keypoints * samples
    blocks = TIMING.blocks
    return {
        "anchor_tokens": torch.full((n, 8), 0.25),
        "noisy_tokens": torch.full((blocks, n, 8), 0.75),
        # Cumulative EXCLUSIVE ends, one per sample -- the same convention
        # build_fk_batch writes (``bounds[1:]``), not [0, 21, 42, ...].
        "point_offsets": torch.arange(keypoints, n + 1, keypoints, dtype=torch.long),
        "point_batch": torch.repeat_interleave(torch.arange(samples), keypoints),
    }


def test_token_count_is_189_per_sample():
    sequence, _ = build_sequence()
    packed = attach_fk_tokens(sequence, encode_fixture(sequence), _positions(sequence))
    assert packed.fk.sequence_indexes.numel() == 189
    assert packed.sequence_length == sequence.sequence_length + 189
    assert packed.sample_lens == [sequence.sample_lens[0] + 189]
    assert packed.split_lens == [TEXT_LEN, VISION_LEN + 189]
    assert packed.fk.noisy_indexes.shape == (TIMING.blocks, KEYPOINTS)
    assert packed.fk.tokens.shape == (189, 8)


def _positions(sequence):
    """FK positions for one encoded batch: ``point_batch`` names each keypoint's sample."""
    point_batch = torch.repeat_interleave(torch.arange(len(sequence.sample_lens)), KEYPOINTS)
    return fk_positions(sequence, point_batch, TIMING)


def fk_time(packed, sample=0, block=0):
    """Temporal position of sample ``sample``'s FK token for 1-based ``block``.

    ``block=0`` is the anchor; block ``b`` is the b-th noisy block, whose
    positions row is ``b`` -- the anchor occupies row 0.
    """
    fk = packed.fk.sequence_indexes
    per_sample = KEYPOINTS * (1 + TIMING.blocks)
    within = KEYPOINTS * block
    return float(packed.position_ids[0, fk[sample * per_sample + within]])


def video_time(packed, latent):
    """Temporal position of the sample's video latent ``latent``."""
    vision = packed.vision.sequence_indexes[:VISION_LEN]
    return float(packed.position_ids[0, vision[latent * GRID_H * GRID_W]])


def test_fk_time_lands_on_the_video_lattice():
    sequence, _ = build_sequence()
    packed = attach_fk_tokens(sequence, encode_fixture(sequence), _positions(sequence))

    # The anchor shares the first video latent's timestamp...
    assert fk_time(packed, block=0) == video_time(packed, 0)
    # ...and every block shares the timestamp of the video latent at the same index.
    for block in range(1, TIMING.blocks + 1):
        assert fk_time(packed, block=block) == video_time(packed, block), f"block {block} drifted"


def test_spatial_axes_are_zero_and_match_the_action_rule():
    sequence, video = build_sequence(with_action=True)
    packed = attach_fk_tokens(sequence, encode_fixture(sequence), _positions(sequence))
    fk = packed.position_ids[:, packed.fk.sequence_indexes]
    assert torch.all(fk[1] == 0) and torch.all(fk[2] == 0), "FK took a spatial position"

    # Action is the reference implementation of "a modality with no pixel
    # position": both use grid_h = grid_w = 1, so both come out zero.
    action = packed.position_ids[:, packed.action.sequence_indexes]
    assert torch.all(action[1] == 0) and torch.all(action[2] == 0)


def test_existing_modalities_are_remapped_not_moved_in_content():
    sequence, _ = build_sequence(with_action=True)
    packed = attach_fk_tokens(sequence, encode_fixture(sequence), _positions(sequence))
    # FK is appended after every existing token, so nothing before it shifts.
    assert torch.equal(packed.text_indexes, sequence.text_indexes)
    assert torch.equal(packed.vision.sequence_indexes, sequence.vision.sequence_indexes)
    assert int(packed.fk.sequence_indexes.min()) == sequence.sequence_length


def test_hidden_gathers_the_noisy_tokens():
    sequence, _ = build_sequence()
    packed = attach_fk_tokens(sequence, encode_fixture(sequence), _positions(sequence))
    hidden = torch.arange(packed.sequence_length, dtype=torch.float32)[:, None].expand(-1, 8)
    gathered = packed.fk.hidden(hidden)
    assert gathered.shape == (TIMING.blocks, KEYPOINTS, 8)
    # hidden() gathers the *sequence* rows named by noisy_indexes.  The anchor
    # occupies the first KEYPOINTS rows, so noisy block 0 starts right after it,
    # and its gathered value is that sequence position, not the token content.
    first_noisy = int(packed.fk.sequence_indexes[KEYPOINTS + 0])
    assert float(gathered[0, 0, 0]) == float(first_noisy)
    assert torch.equal(gathered[0], hidden[packed.fk.noisy_indexes[0]])


def test_attaching_twice_is_rejected():
    sequence, _ = build_sequence()
    packed = attach_fk_tokens(sequence, encode_fixture(sequence), _positions(sequence))
    try:
        attach_fk_tokens(packed, encode_fixture(packed), _positions(packed))
    except ValueError as error:
        assert "already attached" in str(error)
    else:
        raise AssertionError("re-attaching FK tokens should fail")


def test_branch_encode_produces_the_expected_token_shapes():
    from cosmos_framework.data.fk_batch import build_fk_batch
    from cosmos_framework.model.generator.fk_sequence import video_temporal_origins

    branch = FKBranch(timing=TIMING, hidden_dim=32, index_dim=8, content_dim=16, decoder_dim=16)
    sample = {
        "inputs": {"anchor_xyz": torch.zeros(KEYPOINTS, 3), "point_ids": torch.arange(KEYPOINTS)},
        "targets": {
            "displacement": torch.zeros(TIMING.steps, KEYPOINTS, 3),
            "valid": torch.ones(TIMING.steps, KEYPOINTS, dtype=torch.bool),
        },
        "metadata": {"timing": TIMING},
    }
    batch = build_fk_batch([sample, sample], batch_size=2)
    noisy = torch.zeros(TIMING.steps, 2 * KEYPOINTS, 3)
    encoded = branch.encode(batch, noisy, torch.tensor([0.5, 0.5]))
    assert encoded["anchor_tokens"].shape == (2 * KEYPOINTS, 32)
    assert encoded["noisy_tokens"].shape == (TIMING.blocks, 2 * KEYPOINTS, 32)

    sequence, _ = build_sequence(samples=2)
    assert torch.equal(video_temporal_origins(sequence), torch.full((2,), ORIGIN))
