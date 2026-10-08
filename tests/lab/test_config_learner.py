import pytest

from rl_fun.lab.config import PPOParams, RunConfig


def test_defaults_choose_evolution_and_default_ppo_params():
    # Arrange / Act
    config = RunConfig()

    # Assert
    assert config.learner == "evolution"
    assert config.ppo == PPOParams()


def test_round_trip_keeps_learner_and_ppo_params():
    # Arrange
    config = RunConfig(learner="ppo", ppo=PPOParams(learning_rate=1e-3, epochs=6), population=12)

    # Act
    data = config.to_dict()
    restored = RunConfig.from_dict(data)

    # Assert
    assert data["learner"] == "ppo"
    assert data["ppo"]["learning_rate"] == 1e-3
    assert restored == config


def test_from_dict_accepts_a_nested_ppo_dict_with_partial_keys():
    # Arrange / Act
    config = RunConfig.from_dict({"learner": "ppo", "ppo": {"epochs": 9}})

    # Assert
    assert config.ppo == PPOParams(epochs=9)


def test_constructor_accepts_a_ppo_dict():
    # Arrange / Act
    config = RunConfig(ppo={"gamma": 0.9})  # type: ignore[arg-type]

    # Assert
    assert config.ppo == PPOParams(gamma=0.9)


def test_unknown_learner_raises_value_error_naming_the_field():
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="learner.*sac"):
        RunConfig(learner="sac")


def test_invalid_ppo_value_raises_value_error_naming_the_field():
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="ppo.gamma"):
        RunConfig.from_dict({"ppo": {"gamma": 2}})


def test_ppo_that_is_not_an_object_is_rejected():
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="ppo"):
        RunConfig.from_dict({"ppo": 3})


def test_evolution_flat_fields_are_unchanged_for_evolution():
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="elite"):
        RunConfig(learner="evolution", population=4, elite=4)


def test_ppo_ignores_the_evolution_elite_bound():
    # Arrange / Act
    config = RunConfig(learner="ppo", population=4)

    # Assert
    assert config.elite == 6
    assert config.population == 4
