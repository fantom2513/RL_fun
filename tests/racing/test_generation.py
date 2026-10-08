import numpy as np
import pytest

from rl_fun.racing.fleet import RacingFleet
from rl_fun.racing.generation import GenerationResult, run_generation


def forward_straight(weights: np.ndarray, observation: np.ndarray) -> np.ndarray:
    return np.array([0.0, 1.0])


def forward_idle(weights: np.ndarray, observation: np.ndarray) -> np.ndarray:
    return np.array([0.0, 0.0])


def test_returns_one_result_per_car():
    fleet = RacingFleet(3, max_steps=50)

    result = run_generation(fleet, np.zeros((3, 4)), forward_straight)

    assert isinstance(result, GenerationResult)
    for values in (result.progress, result.finished, result.steps_alive, result.lap_steps):
        assert values.shape == (3,)
    assert not result.finished.any()
    assert np.isnan(result.lap_steps).all()


def test_forward_receives_weight_row_and_observation():
    seen: list[tuple[np.ndarray, tuple[int, ...]]] = []

    def forward(weights: np.ndarray, observation: np.ndarray) -> np.ndarray:
        seen.append((weights.copy(), observation.shape))
        return np.array([0.0, 0.0])

    population = np.arange(6, dtype=float).reshape(2, 3)

    run_generation(RacingFleet(2, max_steps=2), population, forward)

    assert len(seen) == 4
    assert seen[0][1] == (8,)
    assert np.array_equal(seen[0][0], population[0])
    assert np.array_equal(seen[1][0], population[1])


def test_dead_cars_are_not_queried():
    calls = [0, 0]

    def forward(weights: np.ndarray, observation: np.ndarray) -> np.ndarray:
        calls[int(weights[0])] += 1
        return np.array([0.0, 1.0]) if weights[0] == 0 else np.array([0.0, 0.0])

    run_generation(RacingFleet(2, max_steps=800), np.array([[0.0], [1.0]]), forward)

    assert calls[1] == 800
    assert calls[0] < calls[1]


def test_run_resets_the_fleet_each_time():
    fleet = RacingFleet(2, max_steps=20)
    population = np.zeros((2, 1))

    first = run_generation(fleet, population, forward_straight)
    second = run_generation(fleet, population, forward_straight)

    assert first.progress == pytest.approx(second.progress)


def test_population_size_must_match_fleet():
    with pytest.raises(ValueError, match="population"):
        run_generation(RacingFleet(3), np.zeros((2, 4)), forward_straight)


def test_driving_beats_standing_still():
    fleet = RacingFleet(2, max_steps=60)
    population = np.array([[0.0], [1.0]])

    def forward(weights: np.ndarray, observation: np.ndarray) -> np.ndarray:
        return np.array([0.0, 1.0 - weights[0]])

    result = run_generation(fleet, population, forward)

    assert result.progress[0] > result.progress[1]
    assert result.best_index == 0
