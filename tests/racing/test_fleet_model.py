import numpy as np
import pytest

from rl_fun.racing.fleet import RacingFleet
from rl_fun.racing.model_spec import ModelSpec


def run(fleet: RacingFleet, action: list[float], steps: int) -> np.ndarray:
    observation = fleet.reset()
    for _ in range(steps):
        observation, _ = fleet.step(np.tile(action, (fleet.n_cars, 1)))
    return observation


def test_default_fleet_uses_default_model():
    fleet = RacingFleet(2)

    assert fleet.model == ModelSpec()
    assert fleet.action_size == 2


def test_model_and_ray_angles_are_mutually_exclusive():
    with pytest.raises(ValueError):
        RacingFleet(2, model=ModelSpec(), ray_angles_deg=(0,))


def test_observation_follows_input_order():
    fleet = RacingFleet(2, model=ModelSpec(inputs=("speed", "ray:0", "yaw_rate")))
    default = RacingFleet(2)

    observation = fleet.reset()
    default_observation = default.reset()

    assert observation.shape == (2, 3)
    assert observation[:, 0] == pytest.approx([0.0, 0.0])
    assert observation[:, 1] == pytest.approx(default_observation[:, 2])


def test_selected_columns_match_the_full_default_observation():
    spec = ModelSpec(inputs=("ray:30", "speed", "yaw_rate"))
    selected = run(RacingFleet(2, model=spec), [0.2, 1.0], 8)
    full = run(RacingFleet(2), [0.2, 1.0], 8)

    assert selected[:, 0] == pytest.approx(full[:, 3])
    assert selected[:, 1] == pytest.approx(full[:, 5])
    assert selected[:, 2] == pytest.approx(full[:, 7])


def test_acceleration_input_is_positive_when_speeding_up_and_negative_when_braking():
    spec = ModelSpec(inputs=("acceleration",))
    fleet = RacingFleet(1, model=spec)
    fleet.reset()
    for _ in range(10):
        accelerating, _ = fleet.step(np.array([[0.0, 1.0]]))
    for _ in range(2):
        braking, _ = fleet.step(np.array([[0.0, -1.0]]))

    assert accelerating[0, 0] > 0
    assert braking[0, 0] < 0


def test_steering_angle_input_reports_the_last_steer_command():
    spec = ModelSpec(inputs=("steering_angle",))
    fleet = RacingFleet(1, model=spec)
    assert fleet.reset()[0, 0] == pytest.approx(0.0)

    observation, _ = fleet.step(np.array([[0.5, 0.2]]))

    assert observation[0, 0] == pytest.approx(0.5)


def test_action_size_follows_outputs_and_width_is_checked():
    spec = ModelSpec(outputs=("steer", "accelerate", "brake", "boost"))
    fleet = RacingFleet(2, model=spec)
    fleet.reset()

    assert fleet.action_size == 4
    fleet.step(np.zeros((2, 4)))
    with pytest.raises(ValueError):
        fleet.step(np.zeros((2, 2)))


def test_brake_output_slows_the_car():
    spec = ModelSpec(outputs=("steer", "accelerate", "brake"))
    fleet = RacingFleet(1, model=spec)
    fleet.reset()
    for _ in range(15):
        fleet.step(np.array([[0.0, 1.0, -1.0]]))
    speed_before = fleet.v_long[0]

    for _ in range(5):
        fleet.step(np.array([[0.0, -1.0, 1.0]]))

    assert fleet.v_long[0] < speed_before


def test_boost_output_makes_the_car_faster():
    spec = ModelSpec(outputs=("steer", "throttle", "boost"))
    boosted, plain = RacingFleet(1, model=spec), RacingFleet(1, model=spec)
    boosted.reset()
    plain.reset()

    for _ in range(30):
        boosted.step(np.array([[0.0, 1.0, 1.0]]))
        plain.step(np.array([[0.0, 1.0, -1.0]]))

    assert boosted.v_long[0] > plain.v_long[0]


def test_stats_start_at_zero_and_track_driving_quality():
    fleet = RacingFleet(2)
    fleet.reset()
    assert fleet.mean_speed.tolist() == [0.0, 0.0]

    for step in range(30):
        steer = 0.0 if step % 2 == 0 else 0.6
        fleet.step(np.array([[0.0, 1.0], [steer, 1.0]]))

    assert fleet.mean_speed[0] > 0
    assert fleet.centering[0] > 0.9
    assert fleet.smoothness[0] == pytest.approx(1.0)
    assert fleet.smoothness[1] < fleet.smoothness[0]
    assert np.all((0.0 <= fleet.centering) & (fleet.centering <= 1.0))
