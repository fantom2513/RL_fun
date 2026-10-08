import numpy as np
import pytest

from rl_fun.racing import reference as ref


def test_layer_sizes_and_weight_count():
    sizes = ref.layer_sizes(8, (6, 5))

    assert sizes == [8, 6, 5, 2]
    assert ref.weight_count(sizes) == (8 + 1) * 6 + (6 + 1) * 5 + (5 + 1) * 2


def test_init_population_shape_and_determinism():
    sizes = ref.layer_sizes(8, (6, 5))

    a = ref.init_population(np.random.default_rng(1), 10, sizes, 1.0)
    b = ref.init_population(np.random.default_rng(1), 10, sizes, 1.0)

    assert a.shape == (10, ref.weight_count(sizes))
    assert np.array_equal(a, b)


def test_forward_returns_bounded_action_that_depends_on_weights():
    sizes = ref.layer_sizes(8, (6, 5))
    rng = np.random.default_rng(2)
    first, second = ref.init_population(rng, 2, sizes, 1.0)
    observation = np.linspace(0.0, 1.0, 8)

    action = ref.forward(first, observation, sizes)

    assert action.shape == (2,)
    assert np.all(np.abs(action) <= 1.0)
    assert not np.allclose(action, ref.forward(second, observation, sizes))


def test_inspect_matches_forward():
    sizes = ref.layer_sizes(8, (6, 5))
    weights = ref.init_population(np.random.default_rng(3), 1, sizes, 1.0)[0]
    observation = np.linspace(0.0, 1.0, 8)

    matrices, activations = ref.inspect(weights, observation, sizes)

    assert [m.shape for m in matrices] == [(6, 8), (5, 6), (2, 5)]
    assert [a.shape[0] for a in activations] == [8, 6, 5, 2]
    assert activations[-1] == pytest.approx(ref.forward(weights, observation, sizes))


def test_next_generation_keeps_elite_best_first():
    params = ref.EvolutionParams(population=10, elite=2)
    population = np.random.default_rng(4).normal(size=(10, 20))
    fitness = np.arange(10, dtype=float)

    new = ref.next_generation(population, fitness, np.random.default_rng(5), params)

    assert new.shape == population.shape
    assert np.array_equal(new[0], population[9])
    assert np.array_equal(new[1], population[8])


def test_next_generation_mutates_children_and_is_deterministic():
    params = ref.EvolutionParams(population=10, elite=2, mutation_rate=1.0, mutation_scale=0.5)
    population = np.random.default_rng(4).normal(size=(10, 20))
    fitness = np.arange(10, dtype=float)

    a = ref.next_generation(population, fitness, np.random.default_rng(5), params)
    b = ref.next_generation(population, fitness, np.random.default_rng(5), params)

    assert np.array_equal(a, b)
    for child in a[2:]:
        assert not any(np.array_equal(child, parent) for parent in population)


def test_reference_improves_on_oval():
    params = ref.EvolutionParams(population=24, elite=4)

    run = ref.train("oval", seed=7, generations=15, params=params, fleet_kwargs={"max_steps": 600})

    bests = [entry["best"] for entry in run.history]
    assert len(run.history) == 15
    assert max(bests) > bests[0]
    assert run.best_weights.shape == (ref.weight_count(run.sizes),)
