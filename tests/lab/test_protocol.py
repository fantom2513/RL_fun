import json
import math

import numpy as np
import pytest
from rl_fun.lab.protocol import build_frame, build_gen_message, notice_message, status_message

from rl_fun.racing.fleet import RacingFleet
from rl_fun.racing.generation import GenerationResult

PLAIN_TYPES = (int, float, str, bool, type(None))


def assert_plain(value: object) -> None:
    """Every node is a JSON-friendly Python value, not a numpy scalar or array."""
    if isinstance(value, dict):
        for key, item in value.items():
            assert isinstance(key, str)
            assert_plain(item)
    elif isinstance(value, list):
        for item in value:
            assert_plain(item)
    else:
        assert type(value) in PLAIN_TYPES, f"non-plain value {value!r} of type {type(value)}"


def _fleet(count: int = 5) -> RacingFleet:
    fleet = RacingFleet(count, track="oval", max_steps=200)
    fleet.reset()
    for _ in range(3):
        fleet.step(np.zeros((count, fleet.action_size)))
    return fleet


def _crash_first_car(fleet: RacingFleet) -> None:
    """Drive car 0 straight until it leaves the road; the other cars stay parked."""
    actions = np.zeros((fleet.n_cars, fleet.action_size))
    actions[0] = [0.0, 1.0]
    for _ in range(2000):
        if not fleet.alive[0]:
            return
        fleet.step(actions)
    raise AssertionError("car 0 did not crash while driving straight")


def _finish_last_car(fleet: RacingFleet) -> None:
    """Put the last car one lap short of the line and take one step to cross it."""
    fleet._travelled[-1] = fleet.track.length
    fleet.step(np.zeros((fleet.n_cars, fleet.action_size)))


def test_frame_has_protocol_keys_and_lengths():
    # Arrange
    fleet = _fleet(5)

    # Act
    frame = build_frame(fleet, generation=3)

    # Assert
    assert frame["t"] == "frame"
    assert frame["gen"] == 3
    assert frame["step"] == fleet.steps
    assert frame["n"] == 5
    assert frame["alive"] == 5
    assert len(frame["cars"]) == 5
    assert all(len(car) == 4 for car in frame["cars"])
    assert frame["net"] is None
    assert set(frame["hud"]) == {"speed", "progress"}


def test_car_fields_are_rounded_to_two_decimals():
    # Arrange
    fleet = _fleet(5)

    # Act
    frame = build_frame(fleet, generation=0)

    # Assert
    for index, car in enumerate(frame["cars"]):
        assert car[0] == round(float(fleet.x[index]), 2)
        assert car[1] == round(float(fleet.y[index]), 2)
        assert car[2] == round(float(fleet.heading[index]), 2)


def test_car_states_mark_alive_crashed_and_finished():
    # Arrange
    fleet = _fleet(5)
    _crash_first_car(fleet)
    _finish_last_car(fleet)

    # Act
    frame = build_frame(fleet, generation=0)

    # Assert
    states = [car[3] for car in frame["cars"]]
    assert states == [0, 1, 1, 1, 2]
    assert frame["alive"] == 3


def test_leader_is_the_living_car_with_most_progress():
    # Arrange
    fleet = _fleet(4)
    fleet._travelled[0] = 0.9 * fleet.track.length
    fleet.alive[0] = False
    fleet._travelled[2] = 0.2 * fleet.track.length
    fleet._travelled[3] = 0.5 * fleet.track.length

    # Act
    frame = build_frame(fleet, generation=0)

    # Assert
    assert frame["leader"] == 3
    assert fleet.alive[frame["leader"]]


def test_leader_falls_back_to_most_progress_when_nobody_lives():
    # Arrange
    fleet = _fleet(3)
    fleet.alive[:] = False
    fleet._travelled[:] = np.array([0.1, 0.4, 0.2]) * fleet.track.length

    # Act
    frame = build_frame(fleet, generation=0)

    # Assert
    assert frame["leader"] == 1


def test_rays_of_leader_match_distances_and_angles():
    # Arrange
    fleet = _fleet(5)

    # Act
    frame = build_frame(fleet, generation=0)
    leader = frame["leader"]

    # Assert
    assert len(frame["rays"]) == len(fleet.ray_angles)
    for k, angle in enumerate(fleet.ray_angles):
        distance = fleet.distances[leader, k]
        expected_x = fleet.x[leader] + math.cos(fleet.heading[leader] + angle) * distance
        expected_y = fleet.y[leader] + math.sin(fleet.heading[leader] + angle) * distance
        assert frame["rays"][k] == [round(float(expected_x), 2), round(float(expected_y), 2)]


def test_hud_reports_leader_speed_and_progress():
    # Arrange
    fleet = _fleet(4)
    leader = build_frame(fleet, generation=0)["leader"]

    # Act
    frame = build_frame(fleet, generation=0)

    # Assert
    assert frame["hud"]["speed"] == round(float(fleet.v_long[leader]), 2)
    assert frame["hud"]["progress"] == round(float(fleet.progress[leader]), 2)


def test_network_is_passed_as_plain_lists():
    # Arrange
    fleet = _fleet(3)
    matrices = [np.ones((2, 3)), np.full((1, 2), 0.5)]
    activations = [np.zeros(3), np.array([0.1, -0.2]), np.array([0.3])]

    # Act
    frame = build_frame(fleet, generation=0, network=(matrices, activations))

    # Assert
    assert frame["net"]["matrices"] == [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], [[0.5, 0.5]]]
    assert frame["net"]["activations"] == [[0.0, 0.0, 0.0], [0.1, -0.2], [0.3]]


def test_frame_survives_json_round_trip_with_plain_types():
    # Arrange
    fleet = _fleet(5)
    _crash_first_car(fleet)
    network = ([np.ones((2, 3))], [np.zeros(3), np.zeros(2)])

    # Act
    frame = build_frame(fleet, generation=2, network=network)
    encoded = json.dumps(frame)

    # Assert
    assert_plain(frame)
    assert json.loads(encoded) == frame


def _result(
    progress: list[float], finished: list[bool], lap_steps: list[float]
) -> GenerationResult:
    return GenerationResult(
        progress=np.array(progress),
        finished=np.array(finished),
        steps_alive=np.arange(1, len(progress) + 1),
        lap_steps=np.array(lap_steps),
    )


def test_gen_message_summarizes_generation():
    # Arrange
    result = _result(
        [0.2, 0.5, 1.1, 0.3],
        [False, False, True, False],
        [math.nan, math.nan, 120.0, math.nan],
    )
    scores = np.array([1.0, 2.5, 3.0, -0.5])
    params = {"mutation_rate": 0.15, "mutation_scale": 0.3, "elite": 6, "population": 60}

    # Act
    message = build_gen_message(4, result, scores, params)

    # Assert
    assert message["t"] == "gen"
    assert message["gen"] == 4
    assert message["best"] == pytest.approx(1.1)
    assert message["mean"] == pytest.approx(0.525)
    assert message["finished"] == 1
    assert message["best_fitness"] == pytest.approx(3.0)
    assert message["best_lap_steps"] == 120
    assert message["params"] == {"mutation_rate": 0.15, "mutation_scale": 0.3, "elite": 6}


def test_gen_message_takes_fastest_lap_among_finishers():
    # Arrange
    result = _result(
        [1.2, 1.3, 0.4],
        [True, True, False],
        [150.0, 120.0, math.nan],
    )

    # Act
    message = build_gen_message(0, result, np.array([1.0, 2.0, 0.0]), {"elite": 2})

    # Assert
    assert message["finished"] == 2
    assert message["best_lap_steps"] == 120


def test_gen_message_has_no_lap_steps_without_finishers():
    # Arrange
    result = _result([0.1, 0.2], [False, False], [math.nan, math.nan])

    # Act
    message = build_gen_message(1, result, np.array([0.1, 0.2]), {})

    # Assert
    assert message["finished"] == 0
    assert message["best_lap_steps"] is None
    assert message["params"] == {}


def test_gen_message_is_json_compatible_with_plain_types():
    # Arrange
    result = _result([0.2, 1.1], [True, False], [99.0, math.nan])

    # Act
    message = build_gen_message(2, result, np.array([0.5, 2.0]), {"elite": 1})

    # Assert
    assert_plain(message)
    assert json.loads(json.dumps(message)) == message


def test_status_message_without_text():
    # Arrange
    # Act
    message = status_message("running")

    # Assert
    assert message == {"t": "status", "status": "running"}


def test_status_message_with_text():
    # Arrange
    # Act
    message = status_message("error", "boom")

    # Assert
    assert message == {"t": "status", "status": "error", "message": "boom"}


def test_status_message_rejects_unknown_status():
    # Arrange
    # Act
    with pytest.raises(ValueError) as error:
        status_message("sleeping")

    # Assert
    assert "sleeping" in str(error.value)


def test_notice_message_carries_text():
    # Arrange
    # Act
    message = notice_message("elite должен быть меньше популяции")

    # Assert
    assert message == {"t": "notice", "text": "elite должен быть меньше популяции"}
