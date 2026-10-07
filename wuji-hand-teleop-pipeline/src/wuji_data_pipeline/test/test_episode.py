import json
import pickle

import numpy as np
import pytest

from wuji_data_pipeline.auxiliary_camera import load_auxiliary_depth
from wuji_data_pipeline.episode import EpisodeWriter, load_episode
from wuji_data_pipeline.inspect_episode import main as inspect_episode_main
from wuji_data_pipeline.schema import RobotLayout
from wuji_data_pipeline.teleop_diagnostics import (
    empty_optional_frame,
    teleop_diagnostic_specs,
)


def _frame(layout: RobotLayout, value: float = 0.0):
    return {
        "action": np.full(layout.action_dim, value, dtype=np.float32),
        "action_eef": np.full(layout.total_eef_dim, value, dtype=np.float32),
        "action_bases": np.zeros(6, dtype=np.float32),
        "qpos": np.full(layout.state_dim, value, dtype=np.float32),
        "qvel": np.zeros(layout.state_dim, dtype=np.float32),
        "effort": np.zeros(layout.state_dim, dtype=np.float32),
        "eef": np.full(layout.total_eef_dim, value, dtype=np.float32),
        "robot_base": np.zeros(6, dtype=np.float32),
        "hand_joint_deg": np.zeros(layout.hand_dof * layout.n_sides, dtype=np.float32),
        "commanded_eef": np.full(layout.total_eef_dim, value, dtype=np.float32),
        "arm_joint_command": np.zeros(layout.arm_dof * layout.n_sides, dtype=np.float32),
        "zsp": np.zeros(3 * layout.n_sides, dtype=np.float32),
    }


def test_no_camera_lmdb_episode_round_trip(tmp_path):
    layout = RobotLayout()
    writer = EpisodeWriter(tmp_path, layout, camera_names=[], frame_rate=30.0)
    writer.append(_frame(layout, 1.0), {}, {"anchor": 10.0})
    writer.append(_frame(layout, 2.0), {}, {"anchor": 10.0 + 1.0 / 30.0})

    episode_dir = writer.finalize({"test_marker": True})
    arrays, metadata = load_episode(episode_dir)

    assert not episode_dir.name.endswith(".inprogress")
    assert arrays["action"].shape == (2, 54)
    assert arrays["qpos"].shape == (2, 54)
    assert arrays["eef"].shape == (2, 14)
    assert np.allclose(arrays["action"][:, 0], [1.0, 2.0])
    assert metadata["robot_layout"]["hand_dof_per_side"] == 20
    assert metadata["camera_names"] == []
    assert metadata["test_marker"] is True

    with (episode_dir / "meta_info.pkl").open("rb") as handle:
        disk_metadata = pickle.load(handle)
    assert disk_metadata["keys"]["scalar_data"][0] == b"action"


def test_invalid_scalar_shape_never_advances_episode(tmp_path):
    layout = RobotLayout()
    writer = EpisodeWriter(tmp_path, layout, camera_names=[])
    frame = _frame(layout)
    frame["qpos"] = np.zeros(53, dtype=np.float32)

    with pytest.raises(ValueError, match="qpos must have shape"):
        writer.append(frame, {}, {"anchor": 1.0})

    assert writer.step_count == 0
    writer.close_incomplete()


def test_discard_removes_only_unfinalized_episode(tmp_path):
    layout = RobotLayout()
    writer = EpisodeWriter(tmp_path, layout, camera_names=[])
    inprogress = writer.episode_dir

    discarded = writer.discard()

    assert discarded == inprogress
    assert not inprogress.exists()


def test_discard_never_removes_finalized_episode(tmp_path):
    layout = RobotLayout()
    writer = EpisodeWriter(tmp_path, layout, camera_names=[])
    writer.append(_frame(layout), {}, {"anchor": 1.0})
    episode_dir = writer.finalize()

    returned = writer.discard()

    assert returned == episode_dir
    assert episode_dir.is_dir()
    assert (episode_dir / "meta_info.pkl").is_file()


def test_optional_teleop_arrays_are_aligned_and_old_scalar_schema_is_unchanged(
    tmp_path, capsys,
):
    layout = RobotLayout()
    specs = teleop_diagnostic_specs()
    writer = EpisodeWriter(
        tmp_path,
        layout,
        camera_names=[],
        optional_specs=specs,
        metadata={"teleop_diagnostics": {"tracker_roles": ["test"]}},
    )
    diagnostics = empty_optional_frame(specs)
    diagnostics["teleop_tracker_available"][:] = 1
    diagnostics["teleop_tracker_valid"][:] = [1, 1, 0, 1, 0]
    diagnostics["teleop_manus_right_glove_id"][:] = 42
    diagnostics["teleop_manus_right_available"][:] = 1

    writer.append(
        _frame(layout, 1.0),
        {},
        {"anchor": 10.0},
        optional_frame=diagnostics,
    )
    # A missing optional frame must still append the original training data.
    writer.append(_frame(layout, 1.01), {}, {"anchor": 10.1})
    episode_dir = writer.finalize()
    arrays, metadata = load_episode(episode_dir)

    assert arrays["action"].shape == (2, 54)
    assert arrays["teleop_tracker_raw_pose"].shape == (2, 5, 7)
    assert arrays["teleop_manus_right_keypoints_21"].shape == (2, 21, 3)
    assert arrays["teleop_manus_right_chain_type_code"].dtype == np.int16
    assert arrays["teleop_tracker_available"][:, 0].tolist() == [1, 0]
    assert arrays["teleop_manus_right_glove_id"][:, 0].tolist() == [42, -1]
    assert metadata["schema_version"] == 1
    assert metadata["teleop_diagnostics"]["schema_version"] == 1
    assert metadata["teleop_diagnostics"]["tracker_roles"] == ["test"]
    assert b"/teleop/tracker/raw_pose" in metadata["keys"]["teleop_data"]

    inspect_episode_main([str(episode_dir)])
    summary = json.loads(capsys.readouterr().out)
    assert summary["steps"] == 2
    assert summary["teleop_diagnostics"]["tracker_available_rate"] == 0.5
    assert summary["teleop_diagnostics"]["manus"]["right"]["available_rate"] == 0.5


def test_auxiliary_depth_does_not_change_original_training_schema(
    tmp_path, capsys
):
    layout = RobotLayout()
    writer = EpisodeWriter(
        tmp_path,
        layout,
        camera_names=[],
        auxiliary_camera={
            "depth_stream_names": (
                "head_depth",
                "left_wrist_depth",
                "right_wrist_depth",
            ),
            "infrared_stream_names": (),
            "map_size": 1 << 26,
        },
    )
    head_depth = np.arange(48, dtype=np.uint16).reshape(6, 8)
    writer.append(
        _frame(layout, 3.0),
        {},
        {"anchor": 10.0},
        auxiliary_images={"head_depth": head_depth},
        auxiliary_timestamps={"head_depth": 10.002},
        auxiliary_sequences={"head_depth": 7},
    )

    episode_dir = writer.finalize()
    arrays, metadata = load_episode(episode_dir)

    assert arrays["action"].shape == (1, 54)
    assert metadata["schema_version"] == 1
    assert metadata["camera_names"] == []
    assert "auxiliary_camera" not in metadata["keys"]
    assert metadata["auxiliary_camera"]["best_effort"] is True
    assert np.array_equal(
        load_auxiliary_depth(episode_dir, "head", 0),
        head_depth,
    )
    inspect_episode_main([str(episode_dir)])
    summary = json.loads(capsys.readouterr().out)
    assert summary["auxiliary_camera"]["depth_saved_frames"] == {
        "head_depth": 1
    }
    assert summary["auxiliary_camera"]["infrared_saved_frames"] == {}
