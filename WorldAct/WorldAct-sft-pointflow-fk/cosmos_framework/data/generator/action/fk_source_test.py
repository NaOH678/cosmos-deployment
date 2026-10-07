"""FK window construction: frame alignment, camera conversion, annotation checks."""

import numpy as np
import pytest

from cosmos_framework.data.fk_camera_extrinsic import base_to_camera
from cosmos_framework.data.fk_window import FKTiming
from cosmos_framework.data.generator.action.fk_source import FKSource

KEYPOINTS = 21
STEPS = 32
TIMING = FKTiming()


def write_annotation(root, episode, frames=80, *, positions=None, **fields):
    """A minimal but well-formed ``wuji_fk21.npz`` under ``root/<episode>/annotations``."""
    if positions is None:
        rng = np.random.default_rng(0)
        # Distinct per frame and keypoint so a windowing error cannot cancel out.
        right = (
            np.array([0.5, 0.0, 0.8])
            + rng.normal(scale=0.01, size=(frames, KEYPOINTS, 3))
            + np.arange(frames, dtype=np.float64)[:, None, None] * np.array([0.001, 0.0, 0.0])
        )
        # The left hand is deliberately 40 cm away, so a source that resolved the
        # wrong side of ``side_is_observed`` would still produce plausible shapes.
        left = right + np.array([0.0, 0.4, 0.0])
        positions = np.stack([left, right], axis=1)
    document = {
        "positions": np.asarray(positions, dtype=np.float32),
        "coordinate_frame": "Link_Base",
        "units": "metre",
        "side_is_observed": np.array([False, True]),
        "qpos_was_clipped": np.array(False),
    }
    document.update(fields)
    path = root / episode / "annotations"
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / "wuji_fk21.npz", **document)
    return np.asarray(positions, dtype=np.float64)


@pytest.fixture
def root(tmp_path):
    return tmp_path


def test_window_uses_the_requested_frames_and_differences_the_anchor(root):
    frames = 80
    base = write_annotation(root, "ep", frames=frames)
    source = FKSource(root, timing=TIMING)

    frame_ids = np.arange(0, 2 * (STEPS + 1), 2)  # 33 source frames at stride 2
    sample = source.load("ep", frame_ids)

    anchor = sample["inputs"]["anchor_xyz"]
    assert anchor.shape == (KEYPOINTS, 3)
    assert sample["targets"]["displacement"].shape == (STEPS, KEYPOINTS, 3)
    assert sample["targets"]["valid"].shape == (STEPS, KEYPOINTS)

    # The anchor must be the *first requested* frame, in the camera frame.  The
    # fixture's right hand is the observed one, so this also pins the side choice.
    expected_anchor = base_to_camera(base[frame_ids[0], 1])
    assert np.allclose(anchor, expected_anchor, atol=1e-5)
    assert not np.allclose(anchor, base_to_camera(base[frame_ids[0], 0]), atol=1e-3), "picked the wrong hand"
    # ...and the labels the difference against it, still in the camera frame.
    expected = base_to_camera(base[frame_ids[1:], 1]) - expected_anchor
    assert np.allclose(sample["targets"]["displacement"], expected, atol=1e-5)

    # A different window must move the anchor: a stale cache would keep it fixed.
    other = source.load("ep", frame_ids + 10)
    assert not np.allclose(other["inputs"]["anchor_xyz"], anchor)


def test_points_land_in_front_of_the_camera(root):
    base = write_annotation(root, "ep")
    source = FKSource(root, timing=TIMING)
    camera = source.load("ep", np.arange(0, 66, 2))["inputs"]["anchor_xyz"].astype(np.float64)
    # The head camera looks at the workspace; anything behind it means the roll or
    # the optical-centre offset was applied in the wrong direction, which still
    # produces plausible-looking numbers on paper.
    assert (camera[:, 2] > 0).all(), f"keypoints behind the camera: {camera[:, 2]}"


def test_point_ids_are_the_anatomical_index(root):
    write_annotation(root, "ep")
    source = FKSource(root, timing=TIMING)
    ids = source.load("ep", np.arange(0, 66, 2))["inputs"]["point_ids"]
    assert ids.tolist() == list(range(KEYPOINTS))


@pytest.mark.parametrize(
    "field, value, match",
    [
        ("coordinate_frame", "camera", "frame is"),
        ("units", "millimetre", "units are"),
        ("side_is_observed", np.array([True, False]), "not marked observed"),
        ("qpos_was_clipped", np.array(True), "clipped to joint limits"),
    ],
)
def test_bad_annotations_are_rejected(root, field, value, match):
    write_annotation(root, "ep", **{field: value})
    with pytest.raises(ValueError, match=match):
        FKSource(root, timing=TIMING).positions("ep")


def test_window_shape_and_ordering_are_enforced(root):
    write_annotation(root, "ep")
    source = FKSource(root, timing=TIMING)
    with pytest.raises(ValueError, match="expected 33 integer frame IDs"):
        source.load("ep", np.arange(STEPS))
    with pytest.raises(ValueError, match="must strictly increase"):
        source.load("ep", np.r_[np.arange(0, 64, 2), 62])
    with pytest.raises(ValueError, match="only .* are annotated"):
        source.load("ep", np.arange(0, 66, 2) + 1000)


def test_missing_episode_names_the_path(root):
    source = FKSource(root, timing=TIMING)
    with pytest.raises(FileNotFoundError, match="missing FK annotation"):
        source.positions("absent")


def test_timing_uses_the_cosmos_configuration(root):
    timing = FKTiming.from_cosmos({"fps": 15.0, "chunk_length": 32}, {"temporal_compression_factor": 4})
    assert (timing.fps, timing.steps, timing.steps_per_token) == (15.0, 32, 4)
    assert timing.blocks == 8
    # A duration mismatch means the video and the labels cover different lengths.
    with pytest.raises(ValueError, match="encode_exact_durations"):
        FKTiming.from_cosmos(
            {"fps": 15.0, "chunk_length": 32},
            {"temporal_compression_factor": 4, "encode_exact_durations": [17]},
        )


def test_existing_projection_composes_pixel_center_affine():
    from cosmos_framework.data.generator.action.fk_source import attach_fk_projection
    from tools.verify_fk_camera_projection import IMG_H, IMG_W, project

    xyz = np.array([[0.10, -0.05, 0.80], [-0.20, 0.03, 1.50]], np.float32)
    frames = np.arange(33) * 2
    fk = {"inputs": {"anchor_xyz": xyz}, "metadata": {"raw_frame_ids": frames}}
    pf = {"inputs": {"image_size_wh": np.array([640, 448])}, "metadata": {"raw_frame_ids": frames}}
    attach_fk_projection(fk, pf)
    uv, _ = project(xyz)
    # Existing tracker->composite affine undoes resize and puts head below wrist.
    tracker = fk["inputs"]["anchor_uv"]
    composite = (tracker + 0.5) * [IMG_W / 640, IMG_H / 448] - 0.5 + [0, 362]
    np.testing.assert_allclose(composite, uv + [0, 362], atol=3e-5)
    # Width resize to 544, then patch stride 16; origin is added in the branch.
    via_tracker = ((composite + 0.5) * 0.85) / 16 - 0.5
    direct = ((uv + [0, 362] + 0.5) * 0.85) / 16 - 0.5
    np.testing.assert_allclose(via_tracker, direct, atol=3e-6)
    pf["metadata"]["raw_frame_ids"] = frames + 1
    with pytest.raises(ValueError, match="identical source frames"):
        attach_fk_projection(fk, pf)
