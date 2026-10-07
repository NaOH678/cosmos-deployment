import copy

import pytest

from wuji_data_pipeline.paper_observation_clock import observation_origin


def observation():
    return {
        "images": {"head": {}, "right_wrist": {}},
        "source_timestamps": {
            "camera_head": 1000.8,
            "camera_right_wrist": 1000.9,
            "arm_eef_right": 1000.95,
            "hand_state_right": 1000.94,
        },
        "source_age_ms": {
            "camera_head": -999
        },  # Do not trust unrelated log-age samples.
    }


def test_oldest_camera_origin_and_all_source_metadata_preserved():
    data = observation()
    original = copy.deepcopy(data)
    origin, audit = observation_origin(
        data, source_clock_ros=1001, captured_monotonic=51
    )
    assert origin == pytest.approx(50.8)
    assert audit["camera_skew_s"] == pytest.approx(0.1)
    assert audit["camera_origin_monotonic"]["camera_right_wrist"] == pytest.approx(50.9)
    assert audit["source_timestamps"] == data["source_timestamps"]
    assert audit["reference_camera"] == "camera_head"
    assert data == original


@pytest.mark.parametrize(
    "stamp", [0, -1, float("nan"), float("inf"), True, 1001.01, 999.9]
)
def test_missing_invalid_future_stale_camera_rejected(stamp):
    data = observation()
    data["source_timestamps"]["camera_head"] = stamp
    with pytest.raises(ValueError):
        observation_origin(data, source_clock_ros=1001, captured_monotonic=51)


def test_missing_declared_camera_rejected():
    data = observation()
    del data["source_timestamps"]["camera_right_wrist"]
    with pytest.raises(ValueError, match="Missing camera"):
        observation_origin(data, source_clock_ros=1001, captured_monotonic=51)


def test_pair_tracking_and_caller_chosen_jump_bound():
    _, audit = observation_origin(
        observation(), source_clock_ros=1001, captured_monotonic=51
    )
    data = observation()
    _, next_audit = observation_origin(
        data,
        source_clock_ros=1001.1,
        captured_monotonic=51.1,
        previous_clock_pair=audit["clock_pair"],
        max_clock_offset_change_s=0.01,
    )
    assert next_audit["clock_offset_change_s"] == pytest.approx(0)
    with pytest.raises(ValueError, match="offset jumped"):
        observation_origin(
            data,
            source_clock_ros=1001.2,
            captured_monotonic=51.1,
            previous_clock_pair=audit["clock_pair"],
            max_clock_offset_change_s=0.01,
        )
    with pytest.raises(ValueError, match="backwards"):
        observation_origin(
            data,
            source_clock_ros=1000.99,
            captured_monotonic=51.1,
            previous_clock_pair=audit["clock_pair"],
        )


def test_wrong_rate_is_detected_with_explicit_jump_bound():
    with pytest.raises(ValueError, match="offset jumped"):
        observation_origin(
            observation(),
            source_clock_ros=1001,
            captured_monotonic=51,
            previous_clock_pair={
                "source_clock_ros": 1000.8,
                "captured_monotonic": 50.9,
            },
            max_clock_offset_change_s=0.01,
        )
