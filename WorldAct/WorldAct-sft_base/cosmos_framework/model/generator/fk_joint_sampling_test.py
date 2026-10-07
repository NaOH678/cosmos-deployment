"""The joint state's per-sample arithmetic, without a GPU or a model.

``sample_fk(..., generate_video=True)`` threads one flat vector per sample holding
``[vision_i | action_i? | FK_i]`` through a single sampler, and ``joint_action`` is
what fills in the middle segment.  Three things there are easy to get silently
wrong and none needs a model to check:

1. **The reassembly.**  Sample ``i`` owns a contiguous run of the FK tensor's
   *keypoint* axis (``point_spans[i]``), and its flat layout is step-major.  So the
   pieces must be recombined with ``torch.cat(..., dim=1)``; flattening the
   concatenation and reshaping interleaves steps across samples and produces a
   tensor of the right *shape* carrying the wrong numbers.  The teeth below pin
   that: the wrong recombination is computed too, and asserted to differ.

2. **The middle segment.**  It shifts the FK slice by the action's length, and a
   flat vector's length is not self-describing -- ``torch.cat`` accepts any lengths,
   and the split that reads the result back slices at the layout's boundaries, so a
   mis-sized segment silently shifts every segment after it.  ``flatten_pieces`` is
   what turns that into an error, and the teeth assert it fires.

3. **The flag parsing.**  The callback's switches arrive from ``oc.env`` as strings,
   and ``bool("false")`` is True -- so a naive bool would turn the joint arm on
   whenever the variable is set to anything, including "false".

Not covered here: anything requiring the network, the sampler or CUDA.  The
condition-frame invariance under UniPC and the end-to-end numbers are GPU checks
(see docs/fk_modality_design.md).

Run:  PYTHONPATH=. <venv>/bin/python cosmos_framework/model/generator/fk_joint_sampling_test.py
"""

import sys
from pathlib import Path

import torch

# Running this file directly puts *this directory* on sys.path[0], where a
# ``tokenizers/`` package shadows the installed one for anything that imports it.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]

from cosmos_framework.callbacks.fk_eval import FKEvalCallback, _as_bool  # noqa: E402
from cosmos_framework.model.generator.fk_sampling import (  # noqa: E402
    flatten_pieces,
    joint_layout,
    reassemble_fk,
)

HORIZON = 4
VISION_SHAPES = [(2, 3, 4, 5)] * 3  # 120 elements each
SPANS = [(0, 21), (21, 42), (42, 47)]  # ragged: 21, 21, 5 keypoints
ACTION_SHAPES = [(33, 8)] * 3  # 264 elements each


def pieces_for(spans):
    """Per-sample FK pieces with recognisable, sample-specific values."""
    out = []
    for i, (start, end) in enumerate(spans):
        # value = sample_index * 1000 + step * 100 + keypoint, so any step/sample
        # transposition shows up as a wrong number rather than a wrong shape.
        step = torch.arange(HORIZON).view(HORIZON, 1, 1) * 100
        point = torch.arange(end - start).view(1, -1, 1)
        out.append(torch.full((HORIZON, end - start, 3), float(i * 1000)) + step + point)
    return out


def flat_and_split(layout, pieces, with_action):
    """Build each sample's flat vector from recognisable segments, then read it back.

    This is the model's two halves in miniature: ``flatten_pieces`` on the way in,
    layout-ordered slicing on the way out.
    """
    flat, expected = [], []
    for i in range(len(layout)):
        n_vision, n_action, _n_pointflow, n_fk = layout.sizes(i)
        segments = [torch.full((n_vision,), float(i))]
        if with_action:
            segments.append(torch.full((n_action,), float(i * 1000 + 500)))
        segments.append(pieces[i])
        expected.append((n_vision, n_action, n_fk))
        flat.append(torch.cat(flatten_pieces(segments, layout.segment_sizes(i, with_action=with_action), what="s")))
    rebuilt = []
    for i, vector in enumerate(flat):
        n_vision, n_action = expected[i][0], expected[i][1]
        cursor = n_vision
        assert torch.equal(vector[:n_vision], torch.full((n_vision,), float(i))), f"sample {i} vision"
        if with_action:
            assert torch.equal(vector[cursor : cursor + n_action], torch.full((n_action,), float(i * 1000 + 500))), (
                f"sample {i} action"
            )
            cursor += n_action
        start, end = layout.spans[i]
        rebuilt.append(vector[cursor:].view(HORIZON, end - start, 3))
    return rebuilt


def main():
    # --- 1. layout: ragged spans, FK last, sizes are element counts ------------
    two = joint_layout(VISION_SHAPES, SPANS, HORIZON)
    three = joint_layout(VISION_SHAPES, SPANS, HORIZON, ACTION_SHAPES)
    for layout in (two, three):
        assert layout.vision_sizes == [120, 120, 120], layout.vision_sizes
        assert layout.spans == SPANS, layout.spans
        assert len(layout) == 3
        for i, (start, end) in enumerate(SPANS):
            assert layout.fk_sizes[i] == (end - start) * 3 * HORIZON, layout.fk_sizes
    assert two.action_sizes == [0, 0, 0], two.action_sizes
    assert three.action_sizes == [264, 264, 264], three.action_sizes
    assert two.segment_sizes(0, with_action=False) == (120, two.fk_sizes[0])
    assert three.segment_sizes(0, with_action=True) == (120, 264, three.fk_sizes[0])
    # The fused four-modality layout: PointFlow sits between action and FK, and FK
    # is still last.  This is the ordering every split in the joint path relies on.
    POINTFLOW_SHAPES = [(HORIZON, 7, 3), (HORIZON, 5, 3), (HORIZON, 6, 3)]
    four = joint_layout(VISION_SHAPES, SPANS, HORIZON, ACTION_SHAPES, POINTFLOW_SHAPES)
    assert four.pointflow_sizes == [HORIZON * 7 * 3, HORIZON * 5 * 3, HORIZON * 6 * 3], four.pointflow_sizes
    assert four.segment_sizes(0, with_action=True, with_pointflow=True) == (
        120,
        264,
        four.pointflow_sizes[0],
        four.fk_sizes[0],
    )
    # With the action left clean the pointflow segment still has to appear, before FK.
    assert four.segment_sizes(0, with_action=False, with_pointflow=True) == (
        120,
        four.pointflow_sizes[0],
        four.fk_sizes[0],
    )
    # An arm that leaves PointFlow out lists no pointflow size at all -- otherwise
    # flatten_pieces' zip(strict=True) would reject the piece count.
    assert three.segment_sizes(0, with_action=True, with_pointflow=False) == (120, 264, three.fk_sizes[0])
    print(f"✅ four-modality layout: pointflow_sizes={four.pointflow_sizes}, FK last")
    print(f"✅ layout: two-segment action_sizes={two.action_sizes}; three-segment {three.action_sizes}")

    # --- 2. the round trip, ragged, both layouts ------------------------------
    pieces = pieces_for(SPANS)
    for layout, with_action, label in ((two, False, "two-segment"), (three, True, "three-segment")):
        rebuilt = flat_and_split(layout, pieces, with_action)
        for i, piece in enumerate(pieces):
            assert torch.equal(rebuilt[i], piece), f"{label} sample {i} FK round trip changed"
        print(f"✅ round trip ({label}): ragged {[e - s for s, e in SPANS]} -> columns land correctly")

    # --- 3. teeth: the wrong recombination is wrong --------------------------
    # This is the bug the test exists for, and it is invisible in the shape.
    joined = reassemble_fk(flat_and_split(three, pieces, True), HORIZON)
    assert joined.shape == (HORIZON, 47, 3), joined.shape
    for i, (start, end) in enumerate(SPANS):
        assert torch.equal(joined[:, start:end, :], pieces[i]), f"sample {i} landed in the wrong columns"
    wrong = torch.cat([p.reshape(-1) for p in flat_and_split(three, pieces, True)]).view(HORIZON, 47, 3)
    assert wrong.shape == joined.shape, "the wrong recombination must at least produce the right shape"
    assert not torch.equal(wrong, joined), "flatten-and-reshape agreed with cat(dim=1); no teeth"
    n_bad = int((wrong != joined).any(dim=-1).sum())
    print(f"🔬 对照：flatten+reshape 形状相同 {tuple(wrong.shape)}，但有 {n_bad}/{HORIZON * 47} 个位置数值不同")

    # --- 4. teeth: a mis-sized segment is caught, not silently concatenated ---
    # Without the numel check ``torch.cat`` accepts this and every later segment
    # shifts -- which is exactly the failure the action segment introduces.  The
    # vision and FK pieces are the RIGHT size here so the bad one is unambiguously
    # the middle segment: an earlier version of this test passed its pieces in the
    # wrong order and fired on segment 0, which proves the check runs but not that
    # it catches the segment the action added.
    vision_size, action_size, fk_size = three.segment_sizes(0, with_action=True)
    try:
        flatten_pieces(
            [torch.zeros(vision_size), torch.zeros(action_size - 1), torch.zeros(fk_size)],
            (vision_size, action_size, fk_size),
            what="teeth",
        )
    except ValueError as error:
        assert "segment 1" in str(error), f"the action segment is segment 1, got: {error}"
        print(f"✅ 拒绝错长度的 action 段: {str(error)[:70]}")
    else:
        raise AssertionError("a short action segment must be rejected before the cat")

    # A too-LONG piece is the same silent shift in the other direction, and a
    # correctly-sized one must still pass -- otherwise the check above could be
    # firing on something else entirely.
    try:
        flatten_pieces(
            [torch.zeros(vision_size), torch.zeros(action_size + 1), torch.zeros(fk_size)],
            (vision_size, action_size, fk_size),
            what="teeth",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("an over-long action segment must be rejected too")
    good = flatten_pieces(
        [torch.zeros(vision_size), torch.zeros(action_size), torch.zeros(fk_size)],
        (vision_size, action_size, fk_size),
        what="teeth",
    )
    assert [int(p.numel()) for p in good] == [vision_size, action_size, fk_size]
    print("✅ 长度正确时通过；多一个/少一个元素都被拒绝")

    # --- 5. layout rejects degenerate input ----------------------------------
    for args, why in (
        (([(0, 21)], [(0, 0)], HORIZON), "empty span"),
        (([(0, 21)], [(0, 21), (21, 42)], HORIZON), "vision/span count mismatch"),
        (([(0, 21)], [(0, 21)], HORIZON, [(33, 8), (33, 8)]), "action/vision count mismatch"),
        (([(0, 21)], [(0, 21)], 0), "non-positive horizon"),
    ):
        try:
            joint_layout(*args)
        except ValueError as error:
            print(f"✅ 拒绝 {why}: {str(error)[:52]}")
        else:
            raise AssertionError(f"{why} should have been rejected")

    # --- 6. env flags: "false" must be False ---------------------------------
    truths = {
        "true": True,
        "1": True,
        "yes": True,
        "on": True,
        True: True,
        "false": False,
        "0": False,
        "no": False,
        "": False,
        "False": False,
        False: False,
    }
    for raw, want in truths.items():
        got = _as_bool(raw)
        assert got is want, f"_as_bool({raw!r}) = {got}, want {want}"
    assert _as_bool("false") is False, "bool('false') is True -- this is the trap"
    print(f"✅ _as_bool: {len(truths)} 个取值全部正确（含字符串 'false' -> False）")

    # --- 7. the callback defaults the new arms OFF and gates the pairing ------
    default = FKEvalCallback(sampling_steps=4)
    assert default.video_joint is False, "the joint arm must be opt-in"
    assert default.video_joint_steps == 4, default.video_joint_steps
    assert default.joint_action is False, "the action segment must be opt-in"
    on = FKEvalCallback(sampling_steps=4, video_joint="true", video_joint_steps=8, joint_action="1")
    assert on.video_joint is True and on.video_joint_steps == 8 and on.joint_action is True
    off = FKEvalCallback(sampling_steps=4, video_joint="false")
    assert off.video_joint is False, "string 'false' must not enable the arm"
    inherited = FKEvalCallback(sampling_steps=6, video_joint="true")
    assert inherited.video_joint_steps == 6, "steps should default to sampling_steps"
    # ``joint_action`` is a mode of the joint arm, so on its own it is a mistake
    # worth naming rather than a setting that quietly does nothing.
    try:
        FKEvalCallback(sampling_steps=4, joint_action="true")
    except ValueError as error:
        print(f"✅ joint_action 单独出现被拒绝: {str(error)[:56]}")
    else:
        raise AssertionError("joint_action without video_joint must be rejected")
    print("✅ callback: 默认关闭；'false' 不启用；steps 默认继承 sampling_steps")

    print("\nPASS")


if __name__ == "__main__":
    main()
