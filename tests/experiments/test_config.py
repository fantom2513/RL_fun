import json

import pytest

from rl_fun.experiments.config import ExperimentConfig


def test_config_loads_explicit_seeds(tmp_path):
    # Arrange
    path = tmp_path / "experiment.json"
    path.write_text(
        json.dumps(
            {
                "name": "bandit-demo",
                "algorithm": "epsilon_greedy",
                "seeds": [11, 22],
                "steps": 100,
                "workers": 2,
                "output_root": "runs",
                "parameters": {"arms": 10, "epsilon": 0.1},
            }
        ),
        encoding="utf-8",
    )

    # Act
    config = ExperimentConfig.from_json(path)

    # Assert
    assert config.seeds == (11, 22)


@pytest.mark.parametrize("steps", [0, -1])
def test_config_rejects_non_positive_steps(steps):
    # Arrange
    config = ExperimentConfig(
        name="invalid",
        algorithm="random",
        seeds=(1,),
        steps=steps,
        workers=1,
        output_root="runs",
        parameters={},
    )

    # Act / Assert
    with pytest.raises(ValueError, match="steps must be positive"):
        config.validate()


def test_config_round_trip_preserves_values():
    # Arrange
    config = ExperimentConfig(
        name="bandit-demo",
        algorithm="greedy",
        seeds=(3, 5),
        steps=50,
        workers=1,
        output_root="runs",
        parameters={"arms": 4},
    )

    # Act
    restored = ExperimentConfig.from_dict(config.to_dict())

    # Assert
    assert restored == config
