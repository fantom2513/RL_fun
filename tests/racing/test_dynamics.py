import math

import pytest

from rl_fun.racing.dynamics import KinematicBicycle, VehicleState, make_dynamics


def test_throttle_accelerates_along_heading():
    # Arrange
    model = KinematicBicycle()
    state = VehicleState(0.0, 0.0, 0.0)

    # Act
    after = model.step(state, steer=0.0, throttle=1.0, dt=0.1)

    # Assert
    assert after.v_long == pytest.approx(model.acceleration * 0.1)
    assert after.x == pytest.approx(after.v_long * 0.1)
    assert after.y == pytest.approx(0.0)


def test_speed_is_capped():
    model = KinematicBicycle(max_speed=5.0)
    state = VehicleState(0.0, 0.0, 0.0, v_long=4.99)

    after = model.step(state, 0.0, 1.0, dt=1.0)

    assert after.v_long == pytest.approx(5.0)


def test_braking_stops_without_reversing():
    model = KinematicBicycle()
    state = VehicleState(0.0, 0.0, 0.0, v_long=1.0)

    after = model.step(state, 0.0, -1.0, dt=1.0)

    assert after.v_long == 0.0


def test_positive_steer_turns_left():
    model = KinematicBicycle()
    state = VehicleState(0.0, 0.0, 0.0, v_long=10.0)

    after = model.step(state, steer=1.0, throttle=0.0, dt=0.1)

    expected_rate = 10.0 / model.wheelbase * math.tan(model.max_steer)
    assert after.yaw_rate == pytest.approx(expected_rate)
    assert after.heading == pytest.approx(expected_rate * 0.1)
    assert after.y > 0.0


def test_kinematic_model_has_no_lateral_velocity():
    model = KinematicBicycle()
    state = VehicleState(0.0, 0.0, 0.0, v_long=10.0)

    assert model.step(state, 1.0, 1.0, dt=0.1).v_lat == 0.0


def test_out_of_range_inputs_are_clipped():
    model = KinematicBicycle()
    state = VehicleState(0.0, 0.0, 0.0, v_long=10.0)

    clipped = model.step(state, 5.0, 5.0, dt=0.1)
    limit = model.step(state, 1.0, 1.0, dt=0.1)

    assert clipped == limit


def test_max_yaw_rate_matches_full_lock_at_top_speed():
    model = KinematicBicycle()

    assert model.max_yaw_rate == pytest.approx(
        model.max_speed / model.wheelbase * math.tan(model.max_steer)
    )


def test_make_dynamics_creates_kinematic_model():
    assert isinstance(make_dynamics("kinematic"), KinematicBicycle)


def test_make_dynamics_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown dynamics"):
        make_dynamics("hovercraft")
