import pytest

from rl_fun.environments import factory
from rl_fun.environments.factory import make_environment
from rl_fun.experiments.config import EnvironmentConfig


def test_factory_creates_cartpole_without_render_mode():
    # Arrange / Act
    env = make_environment(EnvironmentConfig(id="CartPole-v1"))

    try:
        # Assert
        assert env.spec is not None and env.spec.id == "CartPole-v1"
    finally:
        env.close()


def test_factory_omits_none_render_mode(monkeypatch):
    # Arrange
    received: dict[str, object] = {}

    def fake_make(environment_id: str, **kwargs: object) -> object:
        received["id"] = environment_id
        received["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(factory.gym, "make", fake_make)

    # Act
    environment = make_environment(EnvironmentConfig(id="CartPole-v1"))

    # Assert
    assert environment is not None
    assert received == {"id": "CartPole-v1", "kwargs": {}}


def test_factory_passes_requested_render_mode(monkeypatch):
    # Arrange
    received: dict[str, object] = {}

    def fake_make(environment_id: str, **kwargs: object) -> object:
        received["id"] = environment_id
        received["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(factory.gym, "make", fake_make)

    # Act
    environment = make_environment(EnvironmentConfig(id="CartPole-v1"), render_mode="human")

    # Assert
    assert environment is not None
    assert received == {"id": "CartPole-v1", "kwargs": {"render_mode": "human"}}


def test_factory_error_contains_environment_id():
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="Missing-v0"):
        make_environment(EnvironmentConfig(id="Missing-v0"))


def test_factory_error_lists_sorted_constructor_kwargs():
    # Arrange
    config = EnvironmentConfig(id="CartPole-v1", kwargs={"zeta": 1, "alpha": 2})

    # Act / Assert
    with pytest.raises(ValueError, match="CartPole-v1.*alpha.*zeta") as error:
        make_environment(config)

    assert isinstance(error.value.__cause__, TypeError)
