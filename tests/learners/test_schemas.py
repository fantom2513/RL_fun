from typing import Any

import pytest

from rl_fun.lab.catalog import build_catalog
from rl_fun.lab.config import PPOParams, RunConfig
from rl_fun.learners.schemas import LEARNER_SCHEMAS

PARAM_KEYS = {"key", "label", "type", "min", "max", "step", "default", "log", "live", "hint"}


def _params(schema: dict[str, Any]) -> list[dict[str, Any]]:
    return [param for group in schema["groups"] for param in group["params"]]


def _lookup(config: RunConfig, key: str) -> Any:
    value: Any = config
    for part in key.split("."):
        value = getattr(value, part)
    return value


@pytest.fixture(scope="module")
def schemas() -> list[dict[str, Any]]:
    return build_catalog(["oval"])["learners"]


def test_catalog_lists_evolution_and_ppo(schemas):
    # Arrange / Act
    ids = [schema["id"] for schema in schemas]

    # Assert
    assert ids == ["evolution", "ppo", "cem", "a2c"]
    assert all(schema["label"] and schema["description"] for schema in schemas)


def test_catalog_learners_equal_module_data(schemas):
    # Arrange / Act / Assert
    assert schemas == LEARNER_SCHEMAS


def test_every_param_has_all_fields(schemas):
    # Arrange / Act / Assert
    for schema in schemas:
        assert schema["groups"]
        for group in schema["groups"]:
            assert group["label"]
        for param in _params(schema):
            assert set(param) == PARAM_KEYS
            assert param["type"] in ("int", "float", "choice")
            assert isinstance(param["live"], bool)
            assert isinstance(param["log"], bool)
            assert param["label"]
            assert param["hint"]


def test_keys_exist_in_run_config_and_defaults_match(schemas):
    # Arrange
    config = RunConfig()

    # Act / Assert
    for schema in schemas:
        for param in _params(schema):
            if schema["id"] == "a2c" and param["key"] in A2C_OWN_DEFAULTS:
                continue  # A2C recommends other values than the API defaults, see its schema
            assert _lookup(config, param["key"]) == param["default"], param["key"]


A2C_OWN_DEFAULTS = {"ppo.learning_rate", "ppo.rollout_steps"}


def test_a2c_recommends_short_rollouts_and_a_bigger_step_unlike_ppo(schemas):
    a2c = {p["key"]: p["default"] for p in _params(next(s for s in schemas if s["id"] == "a2c"))}

    assert (a2c["ppo.rollout_steps"], a2c["ppo.learning_rate"]) == (16, 0.003)


def test_default_lies_between_min_and_max(schemas):
    # Arrange / Act / Assert
    for schema in schemas:
        for param in _params(schema):
            assert param["min"] <= param["default"] <= param["max"], param["key"]
            assert param["step"] > 0


def test_range_ends_and_defaults_are_accepted_by_config_validation(schemas):
    # Arrange
    ppo_fields = set(PPOParams().to_dict())

    # Act / Assert
    for param in _params(next(item for item in schemas if item["id"] == "ppo")):
        if not param["key"].startswith("ppo."):
            continue
        name = param["key"].removeprefix("ppo.")
        assert name in ppo_fields
        PPOParams(**{name: param["default"]})


def test_ppo_schema_covers_every_ppo_field(schemas):
    # Arrange
    ppo = next(item for item in schemas if item["id"] == "ppo")

    # Act
    keys = {param["key"] for param in _params(ppo)}

    # Assert
    assert {f"ppo.{name}" for name in PPOParams().to_dict()} <= keys
    assert {"population", "max_steps"} <= keys


def test_evolution_schema_covers_the_flat_fields(schemas):
    # Arrange
    evolution = next(item for item in schemas if item["id"] == "evolution")

    # Act
    keys = {param["key"] for param in _params(evolution)}

    # Assert
    assert {"population", "elite", "mutation_rate", "mutation_scale", "max_steps"} <= keys
    assert not any(key.startswith("ppo.") for key in keys)


def test_live_flags_follow_the_plan(schemas):
    # Arrange
    evolution = {p["key"]: p["live"] for p in _params(schemas[0])}
    ppo = {p["key"]: p["live"] for p in _params(schemas[1])}

    # Act / Assert
    assert evolution["elite"] and evolution["mutation_rate"] and evolution["mutation_scale"]
    assert not evolution["population"] and not evolution["max_steps"]
    for name in (
        "learning_rate",
        "entropy_coef",
        "clip_epsilon",
        "gamma",
        "gae_lambda",
        "value_coef",
        "crash_penalty",
    ):
        assert ppo[f"ppo.{name}"], name
    for name in ("rollout_steps", "epochs", "minibatch_size", "initial_std", "max_grad_norm"):
        assert not ppo[f"ppo.{name}"], name
    assert not ppo["population"]


def test_learning_rate_is_logarithmic_with_a_beginner_hint(schemas):
    # Arrange
    rate = next(p for p in _params(schemas[1]) if p["key"] == "ppo.learning_rate")

    # Act / Assert
    assert rate["log"] is True
    assert rate["type"] == "float"
    assert len(rate["hint"]) > 40


def test_texts_are_russian(schemas):
    # Arrange / Act / Assert
    for schema in schemas:
        for param in _params(schema):
            assert any("а" <= char.lower() <= "я" for char in param["label"]), param["key"]
            assert any("а" <= char.lower() <= "я" for char in param["hint"]), param["key"]
