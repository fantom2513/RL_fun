import numpy as np
from gymnasium.utils.env_checker import check_env

from rl_fun.environments.bandit import StationaryBanditEnv


def test_bandit_passes_gymnasium_contract():
    # Arrange
    env = StationaryBanditEnv(arms=4, horizon=20, reward_std=1.0)

    # Act / Assert
    check_env(env)


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
