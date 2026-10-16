import numpy as np
import pytest

from rl_fun.snake.env import SnakeEnv


def test_cycle_mode_observation_has_six_planes_of_the_window():
    env = SnakeEnv(8, observation_mode="egocentric_cycle")

    observation, _ = env.reset(seed=0)

    assert env.observation_space.shape == (6 * 15 * 15,)
    assert observation.shape == env.observation_space.shape


def test_cycle_mode_keeps_the_first_five_planes_of_the_egocentric_mode():
    plain = SnakeEnv(8, observation_mode="egocentric")
    cycle = SnakeEnv(8, observation_mode="egocentric_cycle")
    a, _ = plain.reset(seed=3)
    b, _ = cycle.reset(seed=3)

    assert np.array_equal(a.reshape(5, -1), b.reshape(6, -1)[:5])


def test_cycle_plane_is_zero_at_the_head():
    env = SnakeEnv(8, observation_mode="egocentric_cycle")
    observation, _ = env.reset(seed=0)

    assert observation.reshape(6, 15, 15)[5, 7, 7] == 0


def test_cycle_mode_needs_an_even_board():
    with pytest.raises(ValueError):
        SnakeEnv(9, observation_mode="egocentric_cycle")
