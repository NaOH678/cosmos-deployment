import copy

import pytest

from wuji_data_pipeline.deployment_splice_analysis import analyze_records


def records():
    action = {"arm_action_right": {"ee_pos": [0.16, 0, 0],
                                   "ee_quat": [0, 0, 0, 1]},
              "hand_action_right": [0] * 20}
    rows = [
        {"event": "policy_action_chunk", "generation": 4, "request_id": 9,
         "stage": "server_output", "actions": [copy.deepcopy(action) for _ in range(5)]},
        {"event": "command_publish", "published_at": 1.0,
         "arm_eef": {"right": [0, 0, 0, 0, 0, 0, 1]}, "hand_deg": {"right": [0] * 20}},
        {"event": "pending_chunk_activate", "generation": 4, "request_id": 9,
         "chunk_id": 2, "skipped_actions": 1, "installed_actions": 4,
         "blend_steps": 4, "blend_method": "smoothstep", "action_smoothing_method": "none",
         "raw_position_jump_m": {"right": 0.16}},
    ]
    # Analytic smoothstep weights for four steps: 5/32, 1/2, 27/32, 1.
    for index, x in enumerate([0.025, 0.08, 0.135, 0.16]):
        rows.append({"event": "action_dispatch", "chunk_id": 2,
                     "chunk_action_index": index,
                     "arm_eef": {"right": [x, 0, 0, 0, 0, 0, 1]}})
    return rows


def test_replay_distinguishes_smooth_chunk_from_large_splice():
    source = records()
    before = copy.deepcopy(source)
    chunk = analyze_records(source)["chunks"][0]
    assert chunk["server_internal_max_step_cm"] == 0
    assert chunk["aligned_internal_max_step_cm"] == 0
    assert chunk["executed_internal_max_step_cm"] == pytest.approx(5.5)
    assert chunk["replay_max_position_error_m"] < 1e-7
    assert chunk["fixed_input_alternative_max_step_cm"]["1"] == pytest.approx(16)
    assert source == before


def test_missing_raw_is_unavailable_not_zero():
    chunk = analyze_records(records()[1:])["chunks"][0]
    assert "unavailable_reason" in chunk
    assert "server_internal_max_step_cm" not in chunk


def test_missing_filtered_snapshot_does_not_replay_raw_as_filtered():
    source = records()
    source[2]["action_smoothing_method"] = "butterworth"
    chunk = analyze_records(source)["chunks"][0]
    assert chunk["unavailable_reason"] == "missing post_smoothing snapshot"


def test_generation_is_part_of_request_identity():
    source = records()
    source[0]["generation"] = 3
    assert "unavailable_reason" in analyze_records(source)["chunks"][0]


def test_stream_clear_prevents_reusing_previous_session_anchor():
    source = records()
    source.insert(2, {"event": "action_stream_clear"})
    assert "anchor" in analyze_records(source)["chunks"][0]["unavailable_reason"]


def test_partial_dispatch_is_compared_only_at_recorded_indices():
    source = records()
    source.pop(3)
    chunk = analyze_records(source)["chunks"][0]
    assert chunk["observed_dispatch_count"] == 3
    assert chunk["replay_max_position_error_m"] < 1e-7


def test_missing_middle_dispatch_is_not_treated_as_one_large_step():
    source = records()
    source.pop(4)
    chunk = analyze_records(source)["chunks"][0]
    assert chunk["observed_adjacent_step_count"] == 1
    assert chunk["executed_internal_max_step_cm"] == pytest.approx(2.5)


def test_early_splice_is_not_misreported_as_ordinary_boundary_replay():
    source = records()
    source[2].update(early_splice=True, early_bridge_steps=2, blend_steps=0,
                     blend_method="none")
    chunk = analyze_records(source)["chunks"][0]
    assert "early splice" in chunk["unavailable_reason"]
    assert "replay_max_position_error_m" not in chunk
    assert chunk["executed_internal_max_step_cm"] == pytest.approx(5.5)
