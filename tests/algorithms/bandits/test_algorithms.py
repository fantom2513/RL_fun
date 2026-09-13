import numpy as np
import pytest

from rl_fun.algorithms.bandits.epsilon_greedy import run_epsilon_greedy
from rl_fun.algorithms.bandits.greedy import run_greedy
from rl_fun.algorithms.bandits.random_agent import run_random
from rl_fun.environments.bandit import StationaryBanditEnv
from rl_fun.tracking.metrics import MemoryMetricSink


class SeedRecordingBanditEnv(StationaryBanditEnv):
    """Record reset seeds without changing the stationary-bandit contract."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.reset_seeds: list[int | None] = []

    def reset(
        self, *, seed: int | None = None, options: dict[str, object] | None = None
    ) -> tuple[np.ndarray, dict[str, int]]:
        self.reset_seeds.append(seed)
        return super().reset(seed=seed, options=options)


def test_random_agent_takes_requested_number_of_steps():
    # Arrange
    env = StationaryBanditEnv(arms=3, horizon=12, reward_std=0.0)
    env.reset(seed=4)

    # Act
    metrics = MemoryMetricSink()
    outcome = run_random(env, 12, np.random.default_rng(4), metrics)

    # Assert
    assert int(outcome.counts.sum()) == 12
    assert set(metrics.events[-1].metrics) >= {
        "train/cumulative_reward",
        "train/cumulative_regret",
        "train/optimal_action_rate",
    }


def test_greedy_updates_incremental_sample_mean():
    # Arrange
    env = StationaryBanditEnv(arms=2, horizon=4, reward_std=0.0)
    env.reset(seed=8)

    # Act
    metrics = MemoryMetricSink()
    outcome = run_greedy(env, 4, np.random.default_rng(8), metrics)

    # Assert
    assert np.allclose(outcome.estimates[outcome.counts > 0], env.arm_means[outcome.counts > 0])
    assert set(metrics.events[-1].metrics) >= {
        "train/cumulative_reward",
        "train/cumulative_regret",
        "train/optimal_action_rate",
    }


def test_epsilon_one_explores_more_than_one_arm():
    # Arrange
    env = StationaryBanditEnv(arms=4, horizon=100, reward_std=0.0)
    env.reset(seed=2)

    # Act
    metrics = MemoryMetricSink()
    outcome = run_epsilon_greedy(
        env,
        100,
        np.random.default_rng(2),
        metrics,
        epsilon=1.0,
    )

    # Assert
    assert np.count_nonzero(outcome.counts) > 1
    assert set(metrics.events[-1].metrics) >= {
        "train/cumulative_reward",
        "train/cumulative_regret",
        "train/optimal_action_rate",
    }


@pytest.mark.parametrize(
    ("algorithm_id", "parameters"),
    [
        ("bandit_greedy", {}),
        ("bandit_epsilon", {"epsilon": 0.1}),
    ],
)
def test_bandit_adapters_seed_their_first_reset(
    algorithm_id: str, parameters: dict[str, float]
):
    # Arrange
    from rl_fun.algorithms.registry import get_algorithm

    env = SeedRecordingBanditEnv(arms=3, horizon=4, reward_std=0.0)

    # Act
    get_algorithm(algorithm_id).run(env, 4, 29, np.random.default_rng(29), MemoryMetricSink(), parameters)

    # Assert
    assert env.reset_seeds == [29]
