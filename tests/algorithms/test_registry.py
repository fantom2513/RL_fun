import gymnasium as gym
import numpy as np
import pytest

from rl_fun.algorithms.registry import get_algorithm
from rl_fun.environments.bandit import StationaryBanditEnv
from rl_fun.tracking.metrics import MemoryMetricSink


def test_unknown_algorithm_lists_supported_ids():
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="bandit_epsilon.*bandit_greedy.*random"):
        get_algorithm("missing")


def test_bandit_algorithm_rejects_cartpole():
    # Arrange
    env = gym.make("CartPole-v1")

    try:
        # Act / Assert
        definition = get_algorithm("bandit_greedy")
        with pytest.raises(ValueError, match="bandit_greedy.*CartPole-v1"):
            definition.validate(env, "CartPole-v1")
    finally:
        env.close()


def test_random_algorithm_finishes_requested_steps():
    # Arrange
    env = gym.make("CartPole-v1")
    sink = MemoryMetricSink()

    try:
        # Act
        result = get_algorithm("random").run(env, 25, 7, np.random.default_rng(7), sink, {})

        # Assert
        assert result.metrics["train/steps"] == 25.0
    finally:
        env.close()


def test_random_algorithm_omits_safety_limited_rollout_from_episode_metrics():
    # Arrange
    env = gym.make("CartPole-v1")
    sink = MemoryMetricSink()

    try:
        # Act
        result = get_algorithm("random").run(env, 1, 7, np.random.default_rng(7), sink, {})

        # Assert
        assert result.metrics["train/episodes"] == 0.0
        assert sink.events == []
    finally:
        env.close()


@pytest.mark.parametrize(
    ("algorithm_id", "parameters"),
    [
        ("bandit_greedy", {}),
        ("bandit_epsilon", {"epsilon": 0.1}),
    ],
)
def test_bandit_adapters_step_through_time_limit_wrapper(algorithm_id, parameters):
    # Arrange
    env = gym.wrappers.TimeLimit(
        StationaryBanditEnv(arms=3, horizon=4, reward_std=0.0), max_episode_steps=1
    )
    sink = MemoryMetricSink()

    try:
        # Act
        result = get_algorithm(algorithm_id).run(
            env,
            4,
            7,
            np.random.default_rng(7),
            sink,
            parameters,
        )

        # Assert
        assert result.metrics["train/steps"] == 1.0
    finally:
        env.close()


@pytest.mark.parametrize("algorithm_id", ["random", "bandit_greedy", "bandit_epsilon"])
def test_algorithm_rejects_unknown_parameters(algorithm_id: str):
    # Arrange
    env = StationaryBanditEnv(arms=3, horizon=4, reward_std=0.0)

    # Act / Assert
    with pytest.raises(ValueError, match="unknown parameters"):
        get_algorithm(algorithm_id).run(
            env,
            4,
            7,
            np.random.default_rng(7),
            MemoryMetricSink(),
            {"unexpected": 1},
        )


def test_epsilon_adapter_requires_epsilon_parameter():
    # Arrange
    env = StationaryBanditEnv(arms=3, horizon=4, reward_std=0.0)

    # Act / Assert
    with pytest.raises(ValueError, match="epsilon"):
        get_algorithm("bandit_epsilon").run(
            env,
            4,
            7,
            np.random.default_rng(7),
            MemoryMetricSink(),
            {},
        )


def test_random_algorithm_includes_partial_episode_reward_in_training_total():
    # Arrange
    env = gym.make("CartPole-v1")
    try:
        # Act
        result = get_algorithm("random").run(
            env,
            1,
            7,
            np.random.default_rng(7),
            MemoryMetricSink(),
            {},
        )
        # Assert: CartPole returns one reward per step, even without an episode end.
        assert result.metrics["train/cumulative_reward"] == 1.0
    finally:
        env.close()


@pytest.mark.parametrize("algorithm_id", ["bandit_greedy", "bandit_epsilon"])
@pytest.mark.parametrize("entry_point", ["validate", "run"])
@pytest.mark.parametrize("wrapper", ["transform", "identity", "space"])
def test_bandit_rejects_incompatible_actions_before_environment_mutation(
    algorithm_id: str,
    entry_point: str,
    wrapper: str,
) -> None:
    # Arrange
    base = StationaryBanditEnv(arms=3, horizon=4, reward_std=0.0)
    if wrapper == "space":
        env = gym.Wrapper(base)
        env.action_space = gym.spaces.Discrete(2)
    else:
        env = gym.wrappers.TransformAction(
            base,
            (lambda action: action + 1) if wrapper == "transform" else lambda action: action,
            gym.spaces.Discrete(2 if wrapper == "transform" else 3),
        )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=2)
    definition = get_algorithm(algorithm_id)
    sink = MemoryMetricSink()
    try:
        # Act / Assert
        with pytest.raises(ValueError, match=f"{algorithm_id}.*environment"):
            if entry_point == "validate":
                definition.validate(env, "wrapped-bandit")
            else:
                definition.run(
                    env,
                    4,
                    7,
                    np.random.default_rng(7),
                    sink,
                    {"epsilon": 0.1} if algorithm_id == "bandit_epsilon" else {},
                )
        assert np.array_equal(base.arm_means, np.zeros(3)) and sink.events == []
    finally:
        env.close()
