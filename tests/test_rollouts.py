import gymnasium as gym

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
