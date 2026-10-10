import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import rl_fun.environments  # noqa: F401
from rl_fun.snake.env import SnakeEnv, relative_action


def test_seeded_reset_and_gym_contract():
    env = SnakeEnv(size=8)
    observation, _ = env.reset(seed=42)
    first = env.snapshot()
    env.reset(seed=42)
    assert env.snapshot() == first
    assert observation.shape == (11,)
    assert observation.dtype == np.float32
    check_env(env, skip_render_check=True)
    registered = gym.make("RLFun/Snake-v0", size=8)
    assert registered.reset(seed=42)[0].shape == (11,)
    registered.close()


def test_food_grows_snake_and_scores():
    env = SnakeEnv(size=8)
    env.reset(seed=1)
    x, y = env.snake[0]
    env.food = (x + 1, y)
    _, reward, terminated, truncated, info = env.step(0)
    assert reward == 10
    assert len(env.snake) == 4
    assert info["score"] == 1
    assert not terminated and not truncated
    assert env.food not in env.snake


def test_wall_collision_and_terminal_step():
    env = SnakeEnv(size=8)
    env.reset(seed=0)
    env.snake = [(7, 4), (6, 4), (5, 4)]
    assert env.observation()[0] == 1
    _, reward, done, truncated, _ = env.step(0)
    assert done and not truncated and reward == -10
    with pytest.raises(RuntimeError):
        env.step(0)


def test_tail_is_safe_when_it_vacates_but_body_is_not():
    env = SnakeEnv(size=8)
    env.reset(seed=0)
    env.snake = [(3, 3), (3, 4), (4, 4), (4, 3)]
    env.food = (0, 0)
    assert env.observation()[0] == 0
    assert not env.step(0)[2]
    env.reset(seed=0)
    env.snake = [(3, 3), (4, 3), (4, 4), (3, 4), (2, 4)]
    env.food = (0, 0)
    assert env.observation()[0] == 1
    assert env.step(0)[2]


def test_starvation_and_board_completion():
    env = SnakeEnv(size=8, starvation_factor=1)
    env.reset(seed=0)
    env.food = (0, 0)
    env.since_food = 63
    assert env.step(0)[3]
    env.reset(seed=0)
    env.snake = [(6, 0)] + [
        (x, y) for y in range(8) for x in range(8) if (x, y) not in [(6, 0), (7, 0)]
    ]
    env.food = (7, 0)
    _, reward, done, _, info = env.step(0)
    assert done and info["reason"] == "win" and reward == 10
    assert env.food is None


def test_direction_translation_and_invalid_actions():
    assert relative_action(0, 1) == 2
    assert relative_action(0, 3) == 1
    assert relative_action(0, 2) == 0
    env = SnakeEnv(size=8)
    env.reset()
    for action in (-1, 3, True, 1.5):
        with pytest.raises(ValueError):
            env.step(action)
