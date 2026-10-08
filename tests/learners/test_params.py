import dataclasses
import math

import pytest

from rl_fun.lab.config import PPOParams

FIELDS = (
    "learning_rate",
    "gamma",
    "gae_lambda",
    "clip_epsilon",
    "entropy_coef",
    "value_coef",
    "rollout_steps",
    "epochs",
    "minibatch_size",
    "initial_std",
    "max_grad_norm",
    "crash_penalty",
)


def test_defaults_match_the_plan():
    # Arrange / Act
    params = PPOParams()

    # Assert
    assert params.learning_rate == 3e-4
    assert params.gamma == 0.99
    assert params.gae_lambda == 0.95
    assert params.clip_epsilon == 0.2
    assert params.entropy_coef == 0.005
    assert params.value_coef == 0.5
    assert params.rollout_steps == 128
    assert params.epochs == 4
    assert params.minibatch_size == 256
    assert params.initial_std == 0.6
    assert params.max_grad_norm == 0.5
    assert params.crash_penalty == 1.0


def test_has_exactly_the_planned_fields_and_is_frozen():
    # Arrange
    params = PPOParams()

    # Act / Assert
    assert tuple(item.name for item in dataclasses.fields(PPOParams)) == FIELDS
    with pytest.raises(dataclasses.FrozenInstanceError):
        params.gamma = 0.5  # type: ignore[misc]


def test_round_trip_through_dict():
    # Arrange
    params = PPOParams(learning_rate=1e-3, gamma=0.9, epochs=7, minibatch_size=64)

    # Act
    restored = PPOParams.from_dict(params.to_dict())

    # Assert
    assert restored == params
    assert set(params.to_dict()) == set(FIELDS)


def test_from_dict_takes_defaults_for_missing_keys():
    # Arrange / Act
    params = PPOParams.from_dict({"epochs": 3})

    # Assert
    assert params == PPOParams(epochs=3)


def test_from_dict_rejects_unknown_keys_in_russian():
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="неизвестн.*lr"):
        PPOParams.from_dict({"lr": 0.1})


@pytest.mark.parametrize(
    ("name", "good", "bad"),
    [
        ("learning_rate", [1e-6, 0.5], [0.0, -1e-3, math.nan, math.inf]),
        ("gamma", [0.0, 0.999], [1.0, -0.1, 1.5]),
        ("gae_lambda", [0.0, 1.0], [1.01, -0.1]),
        ("clip_epsilon", [0.01, 0.99], [0.0, 1.0, -0.2]),
        ("entropy_coef", [0.0, 0.1], [-0.001]),
        ("value_coef", [0.0, 2.0], [-1.0]),
        ("rollout_steps", [16, 1024], [15, 1025, 0]),
        ("epochs", [1, 20], [0, 21]),
        ("minibatch_size", [16, 8192], [15, 8193]),
        ("initial_std", [0.06, 2.0], [0.05, 0.0, 2.01]),
        ("max_grad_norm", [0.01, 10.0], [0.0, -1.0]),
        ("crash_penalty", [0.0, 5.0], [-0.1]),
    ],
)
def test_bounds_accept_good_and_reject_bad_values_naming_the_field(name, good, bad):
    # Arrange / Act / Assert
    for value in good:
        assert getattr(PPOParams(**{name: value}), name) == value
    for value in bad:
        with pytest.raises(ValueError, match=f"ppo.{name}"):
            PPOParams(**{name: value})


@pytest.mark.parametrize("name", ["rollout_steps", "epochs", "minibatch_size"])
@pytest.mark.parametrize("value", [16.5, "20", True])
def test_integer_fields_reject_non_integers(name, value):
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="целое"):
        PPOParams(**{name: value})


def test_float_fields_reject_non_numbers_and_accept_ints():
    # Arrange / Act / Assert
    assert PPOParams(crash_penalty=2).crash_penalty == 2.0
    with pytest.raises(ValueError, match="ppo.gamma"):
        PPOParams(gamma="0.9")  # type: ignore[arg-type]


def test_error_messages_are_in_russian():
    # Arrange / Act
    with pytest.raises(ValueError) as caught:
        PPOParams(epochs=0)

    # Assert
    assert "поле ppo.epochs" in str(caught.value)
