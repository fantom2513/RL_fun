import pytest

from rl_fun.experiments.config import ExperimentConfig


def valid_values() -> dict[str, object]:
    return {
        "name": "cartpole-random",
        "environment": {"id": "CartPole-v1", "kwargs": {}},
        "algorithm": {"id": "random", "kwargs": {}},
        "run": {
            "seeds": [11, 22],
            "total_steps": 100,
            "workers": 2,
            "output_root": "runs",
        },
        "evaluation": {"episodes": 5, "max_episode_steps": 500},
    }


def test_config_v2_round_trip_preserves_sections():
    config = ExperimentConfig.from_dict(valid_values())

    assert ExperimentConfig.from_dict(config.to_dict()) == config


def test_old_flat_schema_has_actionable_error():
    with pytest.raises(ValueError, match="configuration version 2"):
        ExperimentConfig.from_dict({"name": "old", "algorithm": "random"})


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value["run"].update(total_steps=0), "total_steps must be positive"),
        (lambda value: value["run"].update(seeds=[1, 1]), "seeds must be unique"),
        (lambda value: value["algorithm"].update(id=""), "algorithm.id must not be empty"),
        (lambda value: value.update(extra=True), "unknown top-level keys"),
    ],
)
def test_config_rejects_invalid_values(mutate, message):
    values = valid_values()
    mutate(values)

    with pytest.raises(ValueError, match=message):
        ExperimentConfig.from_dict(values)
