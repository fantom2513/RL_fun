import numpy as np
import pytest

from rl_fun.racing.fleet import RacingFleet

FORWARD = np.array([0.0, 1.0])


def repeat(action: np.ndarray, count: int) -> np.ndarray:
    return np.tile(action, (count, 1))


def _crash(fleet: RacingFleet, index: int = 0) -> None:
    for _ in range(900):
        actions = repeat(np.zeros(2), fleet.n_cars)
        actions[index] = FORWARD
        fleet.step(actions)
        if not fleet.alive[index]:
            return
    raise AssertionError("car did not crash")


def test_respawn_returns_only_masked_cars_to_the_start():
    # Arrange
    fleet = RacingFleet(3)
    fleet.reset()
    start_x, start_y = fleet.x[0], fleet.y[0]
    for _ in range(30):
        fleet.step(repeat(FORWARD, 3))
    before_x = fleet.x.copy()
    before_progress = fleet.progress.copy()
    before_steps = fleet.steps_alive.copy()

    # Act
    fleet.respawn(np.array([True, False, True]))

    # Assert
    for index in (0, 2):
        assert (fleet.x[index], fleet.y[index]) == (start_x, start_y)
        assert fleet.alive[index]
        assert not fleet.finished[index]
        assert fleet.progress[index] == 0.0
        assert fleet.steps_alive[index] == 0
        assert fleet.v_long[index] == 0.0
        assert fleet.mean_speed[index] == 0.0
        assert fleet.centering[index] == 0.0
        assert np.isnan(fleet.lap_steps[index])
    assert fleet.x[1] == before_x[1]
    assert fleet.progress[1] == before_progress[1]
    assert fleet.steps_alive[1] == before_steps[1]


def test_respawn_resets_a_crashed_or_finished_car():
    # Arrange
    fleet = RacingFleet(2)
    fleet.reset()
    _crash(fleet)
    fleet._travelled[1] = fleet.track.length - 0.001
    fleet.step(repeat(FORWARD, 2))
    assert fleet.finished[1]

    # Act
    fleet.respawn(np.array([True, True]))

    # Assert
    assert fleet.alive.all()
    assert not fleet.finished.any()
    assert not fleet.just_crashed.any()
    assert not fleet.just_finished.any()


def test_respawned_car_behaves_like_a_fresh_one():
    # Arrange
    fleet = RacingFleet(2)
    fresh = RacingFleet(1)
    fleet.reset()
    fresh.reset()
    for _ in range(25):
        fleet.step(np.array([[0.4, 1.0], [-0.2, 1.0]]))
    fleet.respawn(np.array([False, True]))

    # Act
    for step in range(40):
        action = np.array([0.3 if step % 2 else -0.1, 1.0])
        fleet.step(np.array([[0.0, 0.0], action]))
        fresh.step(action[None, :])

    # Assert
    assert fleet.x[1] == pytest.approx(fresh.x[0])
    assert fleet.y[1] == pytest.approx(fresh.y[0])
    assert fleet.heading[1] == pytest.approx(fresh.heading[0])
    assert fleet.progress[1] == pytest.approx(fresh.progress[0])
    assert fleet.mean_speed[1] == pytest.approx(fresh.mean_speed[0])
    assert fleet.smoothness[1] == pytest.approx(fresh.smoothness[0])
    assert fleet.distances[1] == pytest.approx(fresh.distances[0])


def test_respawn_returns_observations_of_all_cars():
    # Arrange
    fleet = RacingFleet(2)
    fleet.reset()
    for _ in range(10):
        fleet.step(repeat(FORWARD, 2))

    # Act
    observation = fleet.respawn(np.array([True, False]))

    # Assert
    assert observation.shape == (2, fleet.observation_size)
    assert observation[0] == pytest.approx(fleet.reset()[0])


def test_respawn_with_empty_mask_changes_nothing():
    # Arrange
    fleet = RacingFleet(2)
    fleet.reset()
    for _ in range(10):
        fleet.step(repeat(FORWARD, 2))
    x, progress = fleet.x.copy(), fleet.progress.copy()

    # Act
    fleet.respawn(np.zeros(2, dtype=bool))

    # Assert
    assert (fleet.x == x).all()
    assert (fleet.progress == progress).all()


def test_respawn_with_full_mask_matches_reset_state():
    # Arrange
    fleet = RacingFleet(3)
    fleet.reset()
    for _ in range(10):
        fleet.step(repeat(FORWARD, 3))

    # Act
    fleet.respawn(np.ones(3, dtype=bool))

    # Assert
    fresh = RacingFleet(3)
    fresh.reset()
    assert (fleet.x == fresh.x).all()
    assert (fleet.heading == fresh.heading).all()
    assert (fleet.steps_alive == 0).all()
    assert fleet.distances == pytest.approx(fresh.distances)


@pytest.mark.parametrize("mask", [np.ones(2, dtype=bool), np.ones((3, 1), dtype=bool)])
def test_respawn_rejects_a_mask_of_the_wrong_shape(mask):
    # Arrange
    fleet = RacingFleet(3)
    fleet.reset()

    # Act / Assert
    with pytest.raises(ValueError):
        fleet.respawn(mask)


def test_respawn_before_reset_is_an_error():
    # Arrange
    fleet = RacingFleet(2)

    # Act / Assert
    with pytest.raises(RuntimeError):
        fleet.respawn(np.ones(2, dtype=bool))


def test_just_crashed_is_true_only_on_the_crash_step():
    # Arrange
    fleet = RacingFleet(2)
    fleet.reset()
    assert not fleet.just_crashed.any()

    # Act
    flags = []
    for _ in range(900):
        was_alive = fleet.alive.copy()
        fleet.step(np.array([FORWARD, [0.0, 0.0]]))
        flags.append((was_alive[0], fleet.just_crashed.copy()))
        if not fleet.alive[0]:
            break
    crash_step_flags = fleet.just_crashed.copy()
    fleet.step(np.array([FORWARD, [0.0, 0.0]]))

    # Assert
    assert crash_step_flags.tolist() == [True, False]
    assert not fleet.just_crashed.any()
    assert sum(1 for _, flag in flags if flag[0]) == 1
    assert not fleet.just_finished.any()


def test_just_finished_is_true_only_on_the_finish_step():
    # Arrange
    fleet = RacingFleet(2)
    fleet.reset()
    fleet._travelled[0] = fleet.track.length - 0.001

    # Act
    fleet.step(repeat(FORWARD, 2))
    on_finish = fleet.just_finished.copy()
    crashed_on_finish = fleet.just_crashed.copy()
    fleet.step(repeat(FORWARD, 2))

    # Assert
    assert on_finish.tolist() == [True, False]
    assert not crashed_on_finish.any()
    assert not fleet.just_finished.any()


def test_lateral_offset_is_zero_on_the_centerline_and_grows_when_drifting():
    # Arrange
    fleet = RacingFleet(2)
    fleet.reset()
    assert fleet.lateral_offset == pytest.approx(np.zeros(2), abs=1e-9)

    # Act
    offsets = []
    for _ in range(45):
        fleet.step(np.array([[0.0, 1.0], [1.0, 1.0]]))
        offsets.append(fleet.lateral_offset.copy())

    # Assert
    assert offsets[-1][1] > 2 * offsets[-1][0]
    assert offsets[-1][1] > offsets[0][1]
    assert offsets[-1][1] > 0.5
    assert fleet.lateral_offset.shape == (2,)


def test_steer_change_is_the_absolute_difference_of_consecutive_steering():
    # Arrange
    fleet = RacingFleet(2)
    fleet.reset()
    assert fleet.steer_change == pytest.approx(np.zeros(2))

    # Act
    fleet.step(np.array([[0.5, 1.0], [-0.25, 1.0]]))
    first = fleet.steer_change.copy()
    fleet.step(np.array([[-0.5, 1.0], [-0.25, 1.0]]))
    second = fleet.steer_change.copy()

    # Assert
    assert first == pytest.approx([0.5, 0.25])
    assert second == pytest.approx([1.0, 0.0])
