import numpy as np

from rl_fun.algorithms.bandits.epsilon_greedy import run_epsilon_greedy
from rl_fun.algorithms.bandits.greedy import run_greedy
from rl_fun.algorithms.bandits.random_agent import run_random
from rl_fun.environments.bandit import StationaryBanditEnv
from rl_fun.tracking.metrics import MemoryMetricSink


def test_random_agent_takes_requested_number_of_steps():
    # Arrange
    env = StationaryBanditEnv(arms=3, horizon=12, reward_std=0.0)
    env.reset(seed=4)

    # Act
    outcome = run_random(env, 12, np.random.default_rng(4), MemoryMetricSink())

    # Assert
    assert int(outcome.counts.sum()) == 12


def test_greedy_updates_incremental_sample_mean():
    # Arrange
    env = StationaryBanditEnv(arms=2, horizon=4, reward_std=0.0)
    env.reset(seed=8)

    # Act
    outcome = run_greedy(env, 4, np.random.default_rng(8), MemoryMetricSink())

    # Assert
    assert np.allclose(outcome.estimates[outcome.counts > 0], env.arm_means[outcome.counts > 0])


def test_epsilon_one_explores_more_than_one_arm():
    # Arrange
    env = StationaryBanditEnv(arms=4, horizon=100, reward_std=0.0)
    env.reset(seed=2)

    # Act
    outcome = run_epsilon_greedy(
        env,
        100,
        np.random.default_rng(2),
        MemoryMetricSink(),
        epsilon=1.0,
    )

    # Assert
    assert np.count_nonzero(outcome.counts) > 1
