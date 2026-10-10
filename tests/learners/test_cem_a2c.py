"""The cross-entropy method and A2C learners, next to evolution and PPO."""

import numpy as np
import pytest

from rl_fun.lab.config import RunConfig
from rl_fun.learners.base import IterationResult
from rl_fun.learners.cem import CEMLearner, cem_update
from rl_fun.learners.registry import available_learners, get_learner
from rl_fun.learners.schemas import A2C_SCHEMA, CEM_SCHEMA, PPO_SCHEMA
from rl_fun.racing.fleet import RacingFleet


def _config(learner: str, **overrides) -> RunConfig:
    values = {
        "track": "oval",
        "population": 8,
        "elite": 3,
        "max_steps": 50,
        "seed": 4,
        "learner": learner,
        "ppo": {"rollout_steps": 16, "minibatch_size": 32, "epochs": 4},
    }
    values.update(overrides)
    return RunConfig.from_dict(values)


def _start(learner_id: str, **overrides):
    config = _config(learner_id, **overrides)
    fleet = RacingFleet(
        config.population,
        track=config.track,
        model=config.model,
        max_steps=config.max_steps,
        ray_range=config.ray_range,
    )
    learner = get_learner(learner_id)
    learner.setup(config, fleet, np.random.default_rng(config.seed), None)
    return learner, config


# ---- registry and schemas --------------------------------------------------------------------


def test_the_registry_knows_all_four_learners():
    assert available_learners() == ("evolution", "ppo", "cem", "a2c")
    assert isinstance(get_learner("cem"), CEMLearner)


def test_run_config_accepts_the_new_learners():
    assert _config("cem").learner == "cem"
    assert _config("a2c").learner == "a2c"
    with pytest.raises(ValueError, match="learner"):
        _config("sac")


def test_catalog_schemas_of_the_new_learners_use_real_config_fields():
    for schema in (CEM_SCHEMA, A2C_SCHEMA):
        keys = [param["key"] for group in schema["groups"] for param in group["params"]]
        assert "population" in keys and "max_steps" in keys and len(set(keys)) == len(keys)
        assert schema["label"] and schema["description"]
    a2c = {p["key"] for g in A2C_SCHEMA["groups"] for p in g["params"]}
    ppo = {p["key"] for g in PPO_SCHEMA["groups"] for p in g["params"]}
    assert a2c < ppo and {"ppo.clip_epsilon", "ppo.epochs", "ppo.minibatch_size"} == ppo - a2c


# ---- CEM -------------------------------------------------------------------------------------


def test_the_cem_update_moves_the_mean_and_spread_to_the_elite_samples():
    samples = np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 5.0], [9.0, 9.0]])
    scores = np.array([0.1, 0.9, 0.8, 0.0])  # the best two are rows 1 and 2

    mean, std = cem_update(samples, scores, elite=2, noise=0.5)

    assert mean == pytest.approx([2.0, 3.0])
    assert std == pytest.approx([1.0 + 0.5, 2.0 + 0.5])


def test_the_cem_update_rejects_an_elite_that_does_not_fit():
    with pytest.raises(ValueError, match="elite"):
        cem_update(np.zeros((3, 2)), np.zeros(3), elite=3, noise=0.1)


def test_a_cem_iteration_returns_a_summary_and_a_best_snapshot_in_the_shared_format():
    learner, config = _start("cem")

    result = learner.run_iteration(0)
    snapshot = learner.best_snapshot()

    assert isinstance(result, IterationResult) and result.extra == {}
    assert result.params["elite"] == 3 and "mutation_scale" in result.params
    assert snapshot is not None and snapshot["sizes"] == config.model.layer_sizes
    assert len(snapshot["weights"]) == config.model.weight_count and snapshot["generation"] == 0


def test_cem_is_deterministic_for_one_seed():
    first, _ = _start("cem")
    second, _ = _start("cem")

    a = [first.run_iteration(i).best for i in range(3)]
    b = [second.run_iteration(i).best for i in range(3)]

    assert a == b


def test_cem_narrows_its_spread_when_the_elite_agree():
    learner, config = _start("cem", mutation_scale=0.01)
    before = learner.spread()

    for iteration in range(3):
        learner.run_iteration(iteration)

    assert learner.spread() < before


def test_cem_applies_live_parameters_and_rejects_the_others():
    learner, _ = _start("cem")

    notices = learner.apply_update({"elite": 2, "mutation_scale": 0.05, "population": 99})
    result = learner.run_iteration(0)

    assert len(notices) == 1 and "population" in notices[0]
    assert result.params["elite"] == 2 and result.params["mutation_scale"] == 0.05


# ---- A2C -------------------------------------------------------------------------------------


def test_an_a2c_iteration_is_one_gradient_step_per_rollout_with_ppo_metrics():
    learner, config = _start("a2c")

    result = learner.run_iteration(0)

    assert {"policy_loss", "value_loss", "entropy", "std", "kl", "mean_return"} <= set(result.extra)
    assert learner.config.ppo.epochs == 1
    assert learner.config.ppo.minibatch_size >= config.ppo.rollout_steps * config.population
    assert result.extra["kl"] == pytest.approx(0.0, abs=1e-6) or result.extra["kl"] >= 0


def test_a2c_applies_the_hot_ppo_parameters_and_keeps_its_single_step():
    learner, _ = _start("a2c")

    notices = learner.apply_update({"ppo": {"learning_rate": 0.001, "epochs": 5}})

    assert len(notices) == 1 and "epochs" in notices[0]
    assert learner.config.ppo.learning_rate == 0.001 and learner.config.ppo.epochs == 1


def test_a2c_best_snapshot_is_in_the_shared_format():
    learner, config = _start("a2c")
    learner.run_iteration(0)

    snapshot = learner.best_snapshot()

    assert snapshot is not None and len(snapshot["weights"]) == config.model.weight_count


# ---- through the worker ----------------------------------------------------------------------


@pytest.mark.parametrize("learner", ["cem", "a2c"])
def test_the_worker_runs_the_new_learners_and_saves_a_network(learner, tmp_path):
    import json

    from rl_fun.lab.worker import run_worker
    from .test_worker_ppo import FakeConnection

    conn = FakeConnection()
    config = _config(learner, generations=2).to_dict()

    run_worker(config, conn, tmp_path, speed="max")

    assert conn.sent[-1] == {"t": "status", "status": "finished"}
    assert [message["gen"] for message in conn.of_type("gen")] == [0, 1]
    [folder] = [path for path in tmp_path.iterdir() if path.is_dir()]
    snapshot = json.loads((folder / "best_weights.json").read_text("utf-8"))
    assert len(snapshot["weights"]) == RunConfig().model.weight_count
