import math

import numpy as np
import pytest

from rl_fun.racing.fitness import PRESETS, FitnessSpec, available_terms
from rl_fun.racing.generation import GenerationResult


def make_result() -> GenerationResult:
    return GenerationResult(
        progress=np.array([0.5, 1.0]),
        finished=np.array([False, True]),
        steps_alive=np.array([100, 200]),
        lap_steps=np.array([np.nan, 150.0]),
        mean_speed=np.array([0.4, 0.8]),
        centering=np.array([0.9, 0.5]),
        smoothness=np.array([0.7, 0.9]),
        max_steps=400,
    )


def test_progress_only_matches_progress():
    assert FitnessSpec({"progress": 1.0}).score(make_result()) == pytest.approx([0.5, 1.0])


def test_weighted_sum_of_terms():
    spec = FitnessSpec({"progress": 1.0, "speed": 0.5})

    assert spec.score(make_result()) == pytest.approx([0.7, 1.4])


def test_lap_bonus_rewards_finished_cars_only():
    score = FitnessSpec({"lap_bonus": 1.0}).score(make_result())

    assert score == pytest.approx([0.0, 1 - 150 / 400])


def test_survival_term():
    assert FitnessSpec({"survival": 1.0}).score(make_result()) == pytest.approx([0.25, 0.5])


def test_negative_weight_acts_as_penalty():
    assert FitnessSpec({"speed": -1.0}).score(make_result()) == pytest.approx([-0.4, -0.8])


@pytest.mark.parametrize(
    "weights", [{}, {"warp": 1.0}, {"progress": math.nan}, {"progress": math.inf}]
)
def test_invalid_weights_raise(weights: dict):
    with pytest.raises(ValueError):
        FitnessSpec(weights)


def test_presets_are_valid_and_cover_known_terms():
    assert {"racer", "careful", "balanced"} <= set(PRESETS)
    for spec in PRESETS.values():
        assert set(spec.weights) <= set(available_terms())
        assert spec.score(make_result()).shape == (2,)


def test_dict_round_trip():
    spec = FitnessSpec({"progress": 1.0, "centering": 0.25})

    assert FitnessSpec.from_dict(spec.to_dict()) == spec


def test_result_without_stats_cannot_score_stat_terms():
    bare = GenerationResult(
        progress=np.zeros(2), finished=np.zeros(2, bool),
        steps_alive=np.zeros(2, int), lap_steps=np.full(2, np.nan),
    )

    assert FitnessSpec({"progress": 1.0}).score(bare) == pytest.approx([0.0, 0.0])
    with pytest.raises(ValueError):
        FitnessSpec({"speed": 1.0}).score(bare)
