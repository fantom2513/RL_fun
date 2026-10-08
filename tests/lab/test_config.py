import json
import re

import pytest

from rl_fun.lab.config import RunConfig
from rl_fun.racing.fitness import PRESETS, FitnessSpec
from rl_fun.racing.model_spec import ModelSpec

CYRILLIC = re.compile(r"[А-Яа-яЁё]")


def _custom_config() -> RunConfig:
    return RunConfig(
        name="контроль",
        track="oval",
        model=ModelSpec(
            inputs=("ray:-45", "ray:0", "ray:45", "speed"),
            hidden=(4, 3, 2),
            outputs=("steer", "accelerate", "brake", "boost"),
            activation="relu",
        ),
        fitness=FitnessSpec({"progress": 1, "centering": 0.3}),
        population=20,
        elite=3,
        mutation_rate=0.4,
        mutation_scale=0.1,
        init_scale=0.5,
        max_steps=300,
        ray_range=25.0,
        generations=5,
        seed=42,
    )


def test_defaults_match_plan():
    # Arrange
    # Act
    config = RunConfig()

    # Assert
    assert config.name == "run"
    assert config.track == "circuit"
    assert config.population == 60
    assert config.elite == 6
    assert config.mutation_rate == 0.15
    assert config.mutation_scale == 0.3
    assert config.init_scale == 1.0
    assert config.max_steps == 1500
    assert config.ray_range == 40.0
    assert config.generations is None
    assert config.seed == 7
    assert config.model == ModelSpec()
    assert config.fitness == PRESETS["balanced"]


def test_round_trip_keeps_custom_config_equal():
    # Arrange
    config = _custom_config()

    # Act
    restored = RunConfig.from_dict(config.to_dict())

    # Assert
    assert restored == config


def test_to_dict_is_json_compatible():
    # Arrange
    config = _custom_config()

    # Act
    encoded = json.dumps(config.to_dict())

    # Assert
    assert json.loads(encoded) == config.to_dict()


def test_from_dict_accepts_nested_json_shapes():
    # Arrange
    data = {
        "name": "json",
        "model": {"inputs": ["ray:0", "speed"], "hidden": [3], "outputs": ["steer", "throttle"]},
        "fitness": {"weights": {"progress": 1.0, "lap_bonus": 0.5}},
    }

    # Act
    config = RunConfig.from_dict(data)

    # Assert
    assert config.model == ModelSpec(inputs=("ray:0", "speed"), hidden=(3,))
    assert config.fitness == FitnessSpec({"progress": 1.0, "lap_bonus": 0.5})
    assert config.population == 60


def test_from_dict_with_partial_keys_takes_defaults():
    # Arrange
    data = {"population": 10, "elite": 2}

    # Act
    config = RunConfig.from_dict(data)

    # Assert
    assert config.population == 10
    assert config.elite == 2
    assert config.seed == 7
    assert config.track == "circuit"
    assert config.model == ModelSpec()


def test_unknown_top_level_key_is_rejected_by_name():
    # Arrange
    data = {"population": 10, "turbo": True}

    # Act
    with pytest.raises(ValueError) as error:
        RunConfig.from_dict(data)

    # Assert
    assert "turbo" in str(error.value)
    assert CYRILLIC.search(str(error.value))


@pytest.mark.parametrize(
    ("data", "field"),
    [
        ({"population": 1}, "population"),
        ({"population": 201}, "population"),
        ({"elite": 0}, "elite"),
        ({"population": 10, "elite": 10}, "elite"),
        ({"mutation_rate": 0.0}, "mutation_rate"),
        ({"mutation_rate": 1.5}, "mutation_rate"),
        ({"mutation_scale": 0.0}, "mutation_scale"),
        ({"init_scale": -1.0}, "init_scale"),
        ({"ray_range": 0.0}, "ray_range"),
        ({"max_steps": 49}, "max_steps"),
        ({"max_steps": 5001}, "max_steps"),
        ({"generations": 0}, "generations"),
        ({"name": ""}, "name"),
        ({"name": "я" * 41}, "name"),
        ({"track": "nope"}, "track"),
    ],
)
def test_out_of_range_values_raise_value_error_naming_field(data, field):
    # Arrange
    # Act
    with pytest.raises(ValueError) as error:
        RunConfig.from_dict(data)

    # Assert
    assert field in str(error.value)
    assert CYRILLIC.search(str(error.value))


def test_boundary_values_are_accepted():
    # Arrange
    low = {"population": 2, "elite": 1, "mutation_rate": 1.0, "max_steps": 50}
    high = {"population": 200, "elite": 199, "max_steps": 5000, "name": "я" * 40}

    # Act
    low_config = RunConfig.from_dict(low)
    high_config = RunConfig.from_dict(high)

    # Assert
    assert low_config.population == 2
    assert low_config.mutation_rate == 1.0
    assert high_config.population == 200
    assert high_config.elite == 199
    assert high_config.name == "я" * 40


def test_invalid_nested_model_raises_value_error_with_original_reason():
    # Arrange
    data = {"model": {"inputs": []}}

    # Act
    with pytest.raises(ValueError) as error:
        RunConfig.from_dict(data)

    # Assert
    assert "model" in str(error.value)
    assert "inputs must not be empty" in str(error.value)
    assert CYRILLIC.search(str(error.value))
