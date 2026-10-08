from types import SimpleNamespace

import numpy as np
import pytest

from rl_fun.learners.rewards import (
    LAP_BONUS_SCALE,
    PROGRESS_SCALE,
    STEP_TERM_SCALE,
    StepReward,
)
from rl_fun.racing.fitness import FitnessSpec
from rl_fun.racing.fleet import RacingFleet

DT = 1 / 30
WIDTH = 10.0
MAX_STEPS = 100


def make_fleet(count: int = 1, **overrides) -> SimpleNamespace:
    """A stand-in fleet in the state right after one step; every car moved during the step."""
    state = {
        "n_cars": count,
        "alive": np.ones(count, dtype=bool),
        "just_crashed": np.zeros(count, dtype=bool),
        "just_finished": np.zeros(count, dtype=bool),
        "progress_delta": np.zeros(count),
        "v_long": np.zeros(count),
        "lateral_offset": np.zeros(count),
        "steer_change": np.zeros(count),
        "lap_steps": np.full(count, np.nan),
        "dt": DT,
        "max_steps": MAX_STEPS,
        "dynamics": SimpleNamespace(max_speed=20.0),
        "track": SimpleNamespace(width=WIDTH),
    }
    state.update(overrides)
    return SimpleNamespace(**state)


def reward(weights: dict[str, float], fleet, crash_penalty: float = 0.0) -> np.ndarray:
    return StepReward.from_fitness(FitnessSpec(weights), DT, MAX_STEPS, crash_penalty).compute(
        fleet
    )


def test_scales_match_the_design():
    assert PROGRESS_SCALE == 10.0
    assert STEP_TERM_SCALE == 0.1
    assert LAP_BONUS_SCALE == 5.0


def test_progress_is_paid_per_lap_gained_and_negative_when_reversing():
    # Arrange
    fleet = make_fleet(2, progress_delta=np.array([0.01, -0.01]))

    # Act
    result = reward({"progress": 1.0}, fleet)

    # Assert
    assert result == pytest.approx([0.1, -0.1])


def test_speed_term_is_fraction_of_max_speed_times_dt_times_scale():
    # Arrange
    fleet = make_fleet(2, v_long=np.array([20.0, 10.0]))

    # Act
    result = reward({"speed": 1.0}, fleet)

    # Assert
    assert result == pytest.approx([DT * 0.1, DT * 0.05])


def test_centering_term_is_one_on_the_centerline_and_zero_at_the_edge():
    # Arrange
    fleet = make_fleet(3, lateral_offset=np.array([0.0, WIDTH / 4, WIDTH]))

    # Act
    result = reward({"centering": 1.0}, fleet)

    # Assert
    assert result == pytest.approx([DT * 0.1, DT * 0.05, 0.0])


def test_smoothness_term_falls_with_steer_change():
    # Arrange
    fleet = make_fleet(3, steer_change=np.array([0.0, 1.0, 2.0]))

    # Act
    result = reward({"smoothness": 1.0}, fleet)

    # Assert
    assert result == pytest.approx([DT * 0.1, DT * 0.05, 0.0])


def test_survival_pays_a_constant_per_living_step():
    # Arrange
    fleet = make_fleet(2)

    # Act
    result = reward({"survival": 1.0}, fleet)

    # Assert
    assert result == pytest.approx([DT * 0.1, DT * 0.1])


def test_lap_bonus_is_paid_only_on_the_finish_step_and_shrinks_with_time():
    # Arrange
    fleet = make_fleet(
        3,
        alive=np.array([True, False, False]),
        just_finished=np.array([False, True, True]),
        lap_steps=np.array([np.nan, 40.0, 100.0]),
    )

    # Act
    result = reward({"lap_bonus": 1.0}, fleet)

    # Assert
    assert result == pytest.approx([0.0, 0.6 * LAP_BONUS_SCALE, 0.0])


def test_crash_penalty_is_applied_on_the_crash_step_only():
    # Arrange
    fleet = make_fleet(
        2, alive=np.array([True, False]), just_crashed=np.array([False, True])
    )

    # Act
    result = reward({"progress": 1.0}, fleet, crash_penalty=2.5)

    # Assert
    assert result == pytest.approx([0.0, -2.5])


def test_zero_weight_and_absent_term_are_disabled():
    # Arrange
    fleet = make_fleet(
        1,
        progress_delta=np.array([0.05]),
        v_long=np.array([20.0]),
        steer_change=np.array([0.0]),
    )

    # Act
    result = reward({"progress": 0.0, "speed": 0.0}, fleet)

    # Assert
    assert result == pytest.approx([0.0])


def test_weights_multiply_their_terms():
    # Arrange
    fleet = make_fleet(1, progress_delta=np.array([0.01]), v_long=np.array([20.0]))

    # Act
    result = reward({"progress": 2.0, "speed": 3.0}, fleet)

    # Assert
    assert result == pytest.approx([2.0 * 0.1 + 3.0 * DT * 0.1])


def test_cars_that_were_not_moving_get_zero_even_with_stale_values():
    # Arrange
    fleet = make_fleet(
        2,
        alive=np.array([True, False]),
        progress_delta=np.array([0.01, 0.01]),
        v_long=np.array([20.0, 20.0]),
    )

    # Act
    result = reward({"progress": 1.0, "speed": 1.0, "survival": 1.0}, fleet, crash_penalty=1.0)

    # Assert
    assert result[1] == 0.0
    assert result[0] > 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dt": 0.0}, "dt"),
        ({"max_steps": 0}, "max_steps"),
        ({"crash_penalty": -1.0}, "crash_penalty"),
    ],
)
def test_invalid_arguments_raise_value_error(kwargs, message):
    # Arrange
    arguments = {"dt": DT, "max_steps": MAX_STEPS, "crash_penalty": 0.0}
    arguments.update(kwargs)

    # Act / Assert
    with pytest.raises(ValueError, match=message):
        StepReward.from_fitness(FitnessSpec({"progress": 1.0}), **arguments)


def test_episode_sum_for_straight_driving_matches_progress_and_survival():
    # Arrange
    fleet = RacingFleet(1, track="oval", max_steps=200)
    fleet.reset()
    step_reward = StepReward.from_fitness(
        FitnessSpec({"progress": 1.0, "survival": 1.0}), fleet.dt, fleet.max_steps, 1.0
    )

    # Act
    total = 0.0
    steps = 40
    for _ in range(steps):
        fleet.step(np.array([[0.0, 1.0]]))
        total += float(step_reward.compute(fleet)[0])

    # Assert
    assert fleet.alive[0]
    expected = PROGRESS_SCALE * float(fleet.progress[0]) + steps * fleet.dt * STEP_TERM_SCALE
    assert total == pytest.approx(expected)
    assert fleet.progress[0] > 0.0


def test_real_fleet_crash_step_gets_penalty_and_next_step_is_zero():
    # Arrange
    fleet = RacingFleet(1, track="oval", max_steps=900)
    fleet.reset()
    step_reward = StepReward.from_fitness(FitnessSpec({"survival": 1.0}), fleet.dt, 900, 3.0)

    # Act
    crash_reward = None
    for _ in range(900):
        fleet.step(np.array([[0.0, 1.0]]))
        value = float(step_reward.compute(fleet)[0])
        if fleet.just_crashed[0]:
            crash_reward = value
            break
    fleet.step(np.array([[0.0, 1.0]]))
    after = float(step_reward.compute(fleet)[0])

    # Assert
    assert crash_reward == pytest.approx(fleet.dt * STEP_TERM_SCALE - 3.0)
    assert after == 0.0


def test_fleet_exposes_progress_delta_in_laps():
    # Arrange
    fleet = RacingFleet(2)
    fleet.reset()
    assert fleet.progress_delta == pytest.approx(np.zeros(2))

    # Act
    _, rewards = fleet.step(np.array([[0.0, 1.0], [0.0, 0.0]]))

    # Assert
    assert fleet.progress_delta == pytest.approx(rewards)
    assert fleet.progress_delta[0] > 0.0
