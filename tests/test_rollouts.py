import gymnasium as gym
import pytest

from rl_fun.environments.bandit import StationaryBanditEnv
from rl_fun.policies import RandomPolicy
from rl_fun.rollouts import rollout_episode


def test_random_policy_repeats_actions_for_same_seed():
    # Arrange
    first = RandomPolicy()
    second = RandomPolicy()
    space_a = gym.spaces.Discrete(4)
    space_b = gym.spaces.Discrete(4)
    first.reset(17, space_a)
    second.reset(17, space_b)

    # Act
    first_actions = [first.act(None, False) for _ in range(8)]
    second_actions = [second.act(None, False) for _ in range(8)]

    # Assert
    assert first_actions == second_actions


def test_cartpole_rollout_is_reproducible():
    # Arrange
    first_env = gym.make("CartPole-v1")
    second_env = gym.make("CartPole-v1")

    try:
        # Act
        first = rollout_episode(first_env, RandomPolicy(), seed=23, max_episode_steps=20)
        second = rollout_episode(second_env, RandomPolicy(), seed=23, max_episode_steps=20)

        # Assert
        assert first == second
    finally:
        first_env.close()
        second_env.close()


def test_rollout_obeys_safety_limit():
    # Arrange
    env = gym.make("CartPole-v1")

    try:
        # Act
        result = rollout_episode(env, RandomPolicy(), seed=5, max_episode_steps=1)

        # Assert
        assert result.length == 1
    finally:
        env.close()


@pytest.mark.parametrize(
    ("ending", "expected"),
    [
        ("terminated", (True, False, False)),
        ("truncated", (False, True, False)),
        ("safety", (False, False, True)),
    ],
)
def test_rollout_reports_distinct_end_flags(ending: str, expected: tuple[bool, bool, bool]):
    # Arrange
    env = StationaryBanditEnv(arms=2, horizon=1 if ending == "truncated" else 4)
    if ending == "terminated":

        class TerminatingBandit(StationaryBanditEnv):
            def step(self, action):
                observation, reward, _, _, info = super().step(action)
                return observation, reward, True, False, info

        env = TerminatingBandit(arms=2, horizon=4)
    try:
        # Act
        result = rollout_episode(env, RandomPolicy(), seed=5, max_episode_steps=1)
        # Assert: an environment end on the budget boundary takes precedence.
        assert (result.terminated, result.truncated, result.reached_safety_limit) == expected
    finally:
        env.close()
