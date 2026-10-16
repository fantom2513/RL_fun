import copy

import numpy as np
import torch

from rl_fun.snake.config import SnakeConfig
from rl_fun.snake.env import TURNS, SnakeEnv
from rl_fun.snake.on_policy import OnPolicyTrainer
from rl_fun.snake.policy import Policy, evaluate


def test_egocentric_observation_is_identical_after_board_rotation():
    env = SnakeEnv(16, observation_mode="egocentric")
    env.reset(seed=42)
    original = env.observation().copy()
    assert original.shape == (5 * 31 * 31,)
    channels = original.reshape(5, 31, 31)
    assert channels[0, 15, 15] == channels[1, 15, 15] == 1
    for _ in range(4):
        env.snake = [(y, 15 - x) for x, y in env.snake]
        x, y = env.food
        env.food = (y, 15 - x)
        env.direction = (env.direction - 1) % 4
        np.testing.assert_array_equal(original, env.observation())
    assert env.observation_space.contains(original)


def test_long_start_is_connected_and_evaluation_keeps_standard_start():
    env = SnakeEnv(16, observation_mode="egocentric")
    env.reset(seed=9, options={"initial_length": 230})
    assert len(env.snake) == len(set(env.snake)) == 230
    assert all(
        abs(x - a) + abs(y - b) == 1
        for (x, y), (a, b) in zip(env.snake, env.snake[1:], strict=False)
    )
    assert env.food not in env.snake and env.score == 0
    for seed in range(100):
        env.reset(seed=seed, options={"initial_length": 243})
        assert any(not env._danger(env._next((env.direction + turn) % 4)) for turn in TURNS)
    snapshot = copy.deepcopy(env.state_dict())
    restored = SnakeEnv(16, observation_mode="egocentric")
    restored.load_state_dict(snapshot)
    np.testing.assert_array_equal(env.observation(), restored.observation())
    env.reset(seed=9)
    assert len(env.snake) == 3
    model = Policy(16, "egocentric")
    result = evaluate(model, 16, [1000000], 1)
    assert result["occupancy"] <= 4 / 256 and not result["wins"]


def test_curriculum_resume_reproduces_updates_and_environment():
    config = SnakeConfig(
        algorithm="ppo",
        observation_mode="egocentric",
        size=16,
        curriculum_steps=100,
        episodes=100000,
        rollout_steps=8,
        batch_size=4,
    )
    first = OnPolicyTrainer(config)
    first.run(lambda: first.steps < 24, lambda _: None)
    saved = first.state_dict()
    first.run(lambda: first.steps < 48, lambda _: None)
    final = first.state_dict()
    second = OnPolicyTrainer(config)
    second.load_state_dict(saved)
    second.run(lambda: second.steps < 48, lambda _: None)
    assert all(torch.equal(v, second.model.state_dict()[k]) for k, v in final["model"].items())
    np.testing.assert_array_equal(first.env.observation(), second.env.observation())
