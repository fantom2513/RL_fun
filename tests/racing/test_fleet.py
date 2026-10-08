import math

import numpy as np
import pytest

from rl_fun.racing.env import RacingEnv
from rl_fun.racing.fleet import RacingFleet

FORWARD = np.array([0.0, 1.0])
IDLE = np.array([0.0, 0.0])


def repeat(action: np.ndarray, count: int) -> np.ndarray:
    return np.tile(action, (count, 1))


def test_reset_returns_observation_per_car():
    fleet = RacingFleet(4)

    observation = fleet.reset()

    assert observation.shape == (4, 8)
    assert observation.dtype == np.float32
    assert fleet.observation_size == 8
    assert fleet.alive.all()
    assert fleet.progress == pytest.approx(np.zeros(4))
    assert not fleet.done


def test_ray_angles_define_observation_size():
    fleet = RacingFleet(2, ray_angles_deg=(-90, 0, 90))

    assert fleet.reset().shape == (2, 6)


def test_single_car_matches_racing_env():
    fleet = RacingFleet(1)
    env = RacingEnv()
    observation_fleet = fleet.reset()
    observation_env, _ = env.reset(seed=0)
    assert observation_fleet[0] == pytest.approx(observation_env, abs=1e-5)

    for step in range(40):
        action = np.array([0.3 * math.sin(step / 5), 1.0], dtype=np.float32)
        observation_fleet, rewards = fleet.step(action[None, :])
        observation_env, reward, terminated, _, info = env.step(action)

        assert not terminated
        assert observation_fleet[0] == pytest.approx(observation_env, abs=1e-5)
        assert rewards[0] == pytest.approx(reward, abs=1e-6)
        assert fleet.progress[0] == pytest.approx(info["progress"], abs=1e-6)
    env.close()


def test_cars_with_different_actions_diverge():
    fleet = RacingFleet(2)
    fleet.reset()

    for _ in range(10):
        fleet.step(np.array([[0.0, 1.0], [0.5, 1.0]]))

    assert fleet.heading[0] != fleet.heading[1]


def test_crashed_car_dies_and_freezes_while_others_continue():
    fleet = RacingFleet(2)
    fleet.reset()

    for _ in range(600):
        fleet.step(np.array([FORWARD, IDLE]))
        if not fleet.alive[0]:
            break
    frozen_x, frozen_y = fleet.x[0], fleet.y[0]
    _, rewards = fleet.step(np.array([FORWARD, IDLE]))

    assert not fleet.alive[0]
    assert fleet.alive[1]
    assert (fleet.x[0], fleet.y[0]) == (frozen_x, frozen_y)
    assert rewards[0] == 0.0
    assert fleet.steps_alive[0] < fleet.steps_alive[1]
    assert not fleet.done


def test_done_after_max_steps():
    fleet = RacingFleet(2, max_steps=3)
    fleet.reset()

    for _ in range(3):
        fleet.step(repeat(IDLE, 2))

    assert fleet.done
    assert fleet.alive.all()


def test_done_when_every_car_has_crashed():
    fleet = RacingFleet(2)
    fleet.reset()

    for _ in range(600):
        fleet.step(repeat(FORWARD, 2))
        if fleet.done:
            break

    assert fleet.done
    assert not fleet.alive.any()


def test_finishing_the_lap_marks_car_finished():
    fleet = RacingFleet(1)
    fleet.reset()
    fleet._travelled[:] = fleet.track.length - 0.001

    fleet.step(repeat(FORWARD, 1))

    assert fleet.finished[0]
    assert not fleet.alive[0]
    assert fleet.progress[0] >= 1.0
    assert fleet.lap_steps[0] == 1.0


def test_lap_steps_is_nan_for_unfinished_cars():
    fleet = RacingFleet(2)
    fleet.reset()
    fleet.step(repeat(FORWARD, 2))

    assert np.isnan(fleet.lap_steps).all()


def test_actions_are_clipped_and_shape_checked():
    clipped, limit = RacingFleet(2), RacingFleet(2)
    clipped.reset()
    limit.reset()

    observation_a, _ = clipped.step(np.full((2, 2), 5.0))
    observation_b, _ = limit.step(np.full((2, 2), 1.0))

    assert np.array_equal(observation_a, observation_b)
    with pytest.raises(ValueError):
        clipped.step(np.zeros((3, 2)))


@pytest.mark.parametrize(
    "kwargs",
    [{"n_cars": 0}, {"n_cars": 2, "ray_angles_deg": ()}, {"n_cars": 2, "max_steps": 0},
     {"n_cars": 2, "track": "missing"}, {"n_cars": 2, "dt": 0.0}],
)
def test_invalid_arguments_raise(kwargs: dict):
    with pytest.raises(ValueError):
        RacingFleet(**kwargs)


def test_step_before_reset_raises():
    with pytest.raises(RuntimeError, match="reset"):
        RacingFleet(2).step(repeat(IDLE, 2))
