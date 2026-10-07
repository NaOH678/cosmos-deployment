import numpy as np

from wuji_data_pipeline.sync import FiniteDifferenceVelocity, TimedRingBuffer


def test_nearest_honours_tolerance():
    buffer = TimedRingBuffer(maxlen=10, max_age_s=5.0)
    buffer.append(1.00, "a")
    buffer.append(1.04, "b")

    assert buffer.nearest(1.03, 0.02).value == "b"
    assert buffer.nearest(1.20, 0.02) is None


def test_clock_jump_resets_source_order():
    buffer = TimedRingBuffer(maxlen=10, max_age_s=5.0)
    buffer.append(10.0, "old")
    buffer.append(2.0, "new-clock")

    assert len(buffer) == 1
    assert buffer.latest().value == "new-clock"


def test_missing_velocity_is_derived_in_the_callers_native_unit():
    estimator = FiniteDifferenceVelocity()

    assert np.allclose(estimator.measure("arm_left", 1.0, [10.0, 20.0]), [0.0, 0.0])
    assert np.allclose(estimator.measure("arm_left", 1.5, [11.0, 19.0]), [2.0, -2.0])


def test_reported_velocity_takes_precedence_and_clock_gaps_reset_derivative():
    estimator = FiniteDifferenceVelocity(max_dt_s=0.5)
    estimator.measure("hand_left", 1.0, [0.0], [0.25])

    assert np.allclose(estimator.measure("hand_left", 1.1, [1.0], [0.5]), [0.5])
    assert np.allclose(estimator.measure("hand_left", 2.0, [2.0]), [0.0])
