import gymnasium as gym
import numpy as np
import pytest
from gymnasium.envs.registration import EnvSpec
from gymnasium.utils.env_checker import check_env

from rl_fun.environments import register_environments
from rl_fun.environments.bandit import StationaryBanditEnv


def test_bandit_passes_gymnasium_contract():
    # Arrange
    env = StationaryBanditEnv(arms=4, horizon=20, reward_std=1.0)

    # Act / Assert
    check_env(env)


def test_project_bandit_is_registered_and_checked():
    # Arrange
    register_environments()
    register_environments()
    env = gym.make("RLFun/StationaryBandit-v0", arms=3, horizon=5, reward_std=0.0)

    try:
        # Act / Assert
        check_env(env.unwrapped)
    finally:
        env.close()


def test_register_environments_rejects_conflicting_entry_point(monkeypatch):
    # Arrange
    environment_id = "RLFun/StationaryBandit-v0"
    original = gym.registry[environment_id]
    conflicting_spec = EnvSpec(
        id=environment_id,
        entry_point="rl_fun.environments.bandit:ConflictingBanditEnv",
    )

    # Act / Assert
    with monkeypatch.context() as patch:
        patch.setitem(gym.registry, environment_id, conflicting_spec)
        with pytest.raises(RuntimeError, match="StationaryBandit-v0.*ConflictingBanditEnv"):
            register_environments()

    assert gym.registry[environment_id] is original


def test_same_seed_reproduces_arm_means():
    # Arrange
    first = StationaryBanditEnv(arms=3, horizon=5)
    second = StationaryBanditEnv(arms=3, horizon=5)

    # Act
    first.reset(seed=42)
    second.reset(seed=42)

    # Assert
    assert np.array_equal(first.arm_means, second.arm_means)


def test_bandit_truncates_at_horizon():
    # Arrange
    env = StationaryBanditEnv(arms=2, horizon=2)
    env.reset(seed=1)

    # Act
    env.step(0)
    _, _, _, truncated, _ = env.step(0)

    # Assert
    assert truncated is True
