import numpy as np
import pytest

from rl_fun.lab.config import RunConfig
from rl_fun.lab.protocol import build_gen_message
from rl_fun.learners.base import IterationResult
from rl_fun.learners.evolution import EvolutionLearner
from rl_fun.learners.registry import available_learners, get_learner
from rl_fun.racing import reference
from rl_fun.racing.fleet import RacingFleet
from rl_fun.racing.generation import GenerationResult, run_generation


def _config(**overrides) -> RunConfig:
    values = {"track": "oval", "population": 8, "elite": 2, "max_steps": 60, "seed": 3}
    values.update(overrides)
    return RunConfig(**values)


def _learner(config: RunConfig) -> EvolutionLearner:
    fleet = RacingFleet(
        config.population,
        track=config.track,
        model=config.model,
        max_steps=config.max_steps,
        ray_range=config.ray_range,
    )
    learner = EvolutionLearner()
    learner.setup(config, fleet, np.random.default_rng(config.seed), None)
    return learner


def _legacy(config: RunConfig, generations: int) -> tuple[list[tuple], np.ndarray, float]:
    """The evolution loop as it was in worker.py before the Learner abstraction."""
    sizes = config.model.layer_sizes
    activation = config.model.activation
    fleet = RacingFleet(
        config.population,
        track=config.track,
        model=config.model,
        max_steps=config.max_steps,
        ray_range=config.ray_range,
    )
    rng = np.random.default_rng(config.seed)
    population = reference.init_population(rng, config.population, sizes, config.init_scale)
    rows = []
    best_weights = population[0].copy()
    best_fitness = -np.inf
    for _ in range(generations):
        result = run_generation(
            fleet,
            population,
            lambda w, o: reference.forward(w, o, sizes, activation),
        )
        scores = config.fitness.score(result)
        message = build_gen_message(0, result, scores, {})
        rows.append(
            (message["best"], message["mean"], message["finished"], message["best_fitness"])
        )
        top = int(np.argmax(scores))
        if scores[top] > best_fitness:
            best_fitness = float(scores[top])
            best_weights = population[top].copy()
        population = reference.next_generation(
            population,
            scores,
            rng,
            reference.EvolutionParams(
                population=config.population,
                elite=config.elite,
                mutation_rate=config.mutation_rate,
                mutation_scale=config.mutation_scale,
                init_scale=config.init_scale,
            ),
        )
    return rows, best_weights, best_fitness


def test_iterations_reproduce_the_legacy_evolution_loop_for_one_seed():
    # Arrange
    config = _config()
    expected_rows, expected_weights, expected_fitness = _legacy(config, 3)
    learner = _learner(config)

    # Act
    results = [learner.run_iteration(index) for index in range(3)]
    snapshot = learner.best_snapshot()

    # Assert
    rows = [(r.best, r.mean, r.finished, r.best_fitness) for r in results]
    assert rows == expected_rows
    assert snapshot is not None
    assert snapshot["weights"] == expected_weights.tolist()
    assert snapshot["fitness"] == expected_fitness


def test_iteration_result_carries_summary_params_and_no_extra():
    # Arrange
    learner = _learner(_config())

    # Act
    result = learner.run_iteration(0)

    # Assert
    assert isinstance(result, IterationResult)
    assert result.extra == {}
    assert result.params == {"mutation_rate": 0.15, "mutation_scale": 0.3, "elite": 2}
    assert result.best_lap_steps is None or isinstance(result.best_lap_steps, int)
    assert isinstance(result.finished, int)


def test_best_snapshot_uses_the_shared_weights_format():
    # Arrange
    config = _config()
    learner = _learner(config)
    assert learner.best_snapshot() is None

    # Act
    learner.run_iteration(0)
    learner.run_iteration(1)
    snapshot = learner.best_snapshot()

    # Assert
    assert snapshot is not None
    assert set(snapshot) == {"sizes", "activation", "generation", "fitness", "weights"}
    assert snapshot["sizes"] == config.model.layer_sizes
    assert snapshot["activation"] == config.model.activation
    assert snapshot["generation"] == 1
    assert len(snapshot["weights"]) == config.model.weight_count


def test_apply_update_changes_live_parameters_for_the_next_iteration():
    # Arrange
    learner = _learner(_config())
    learner.run_iteration(0)

    # Act
    notices = learner.apply_update({"mutation_rate": 0.5, "elite": 3})
    result = learner.run_iteration(1)

    # Assert
    assert notices == []
    assert result.params["mutation_rate"] == 0.5
    assert result.params["elite"] == 3


def test_apply_update_reports_rejected_and_unknown_parameters_in_russian():
    # Arrange
    learner = _learner(_config())

    # Act
    notices = learner.apply_update({"elite": 99, "population": 10, "ppo": {"gamma": 0.5}})

    # Assert
    assert len(notices) == 3
    assert "elite" in notices[0] and "не применён" in notices[0]
    assert "population" in notices[1] and "нельзя менять" in notices[1]
    assert "ppo" in notices[2]
    assert learner.run_iteration(0).params["elite"] == 2


def test_apply_update_changes_the_fitness_profile():
    # Arrange
    learner = _learner(_config())

    # Act
    notices = learner.apply_update({"fitness": {"weights": {"progress": 1.0}}})

    # Assert
    assert notices == []


def test_schema_is_the_evolution_schema():
    # Arrange / Act
    schema = EvolutionLearner.schema()

    # Assert
    assert schema["id"] == "evolution"


def test_registry_creates_the_evolution_learner():
    # Arrange / Act / Assert
    assert isinstance(get_learner("evolution"), EvolutionLearner)
    assert available_learners() == ("evolution", "ppo")


def test_registry_creates_the_ppo_learner():
    # Arrange
    from rl_fun.learners.ppo import PPOLearner

    # Act
    learner = get_learner("ppo")

    # Assert
    assert isinstance(learner, PPOLearner)


def test_registry_rejects_unknown_names():
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="sac"):
        get_learner("sac")


def test_gen_message_gains_optional_extra():
    # Arrange
    result = GenerationResult(
        progress=np.array([0.1, 0.2]),
        finished=np.array([False, False]),
        steps_alive=np.array([3, 4]),
        lap_steps=np.array([np.nan, np.nan]),
    )
    scores = np.array([0.1, 0.2])

    # Act
    plain = build_gen_message(0, result, scores, {})
    with_extra = build_gen_message(0, result, scores, {}, extra={"entropy": 1.5})

    # Assert
    assert "extra" not in plain
    assert with_extra["extra"] == {"entropy": 1.5}
