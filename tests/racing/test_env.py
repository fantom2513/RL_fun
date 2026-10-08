import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rl_fun.environments import register_environments
from rl_fun.racing.env import RacingEnv

FORWARD = np.array([0.0, 1.0], dtype=np.float32)
IDLE = np.array([0.0, 0.0], dtype=np.float32)


@pytest.fixture
def env():
    environment = RacingEnv()
    yield environment
    environment.close()


def test_spaces_match_contract(env: RacingEnv):
    assert env.action_space.shape == (2,)
    assert env.observation_space.shape == (5 + 3,)


def test_environment_is_registered_in_gymnasium():
    register_environments()

    created = gym.make("RLFun/Racing-v0", track="oval")

    try:
        assert isinstance(created.unwrapped, RacingEnv)
    finally:
        created.close()


def test_passes_gymnasium_env_checker():
    environment = RacingEnv()
    try:
        check_env(environment, skip_render_check=True)
    finally:
        environment.close()


def test_reset_returns_valid_observation(env: RacingEnv):
    observation, info = env.reset(seed=1)

    assert env.observation_space.contains(observation)
    assert info["progress"] == 0.0
    assert info["collided"] is False


def test_initial_side_rays_see_edges_and_speeds_are_zero(env: RacingEnv):
    observation, _ = env.reset(seed=1)

    assert observation[0] == pytest.approx(5.0 / env.ray_range, abs=0.02)
    assert observation[4] == pytest.approx(5.0 / env.ray_range, abs=0.02)
    assert observation[-3:] == pytest.approx([0.0, 0.0, 0.0])


def test_reset_is_deterministic_for_same_seed():
    first, second = RacingEnv(), RacingEnv()
    try:
        a, _ = first.reset(seed=3)
        b, _ = second.reset(seed=3)
    finally:
        first.close()
        second.close()

    assert np.array_equal(a, b)


def test_driving_forward_earns_reward_equal_to_progress(env: RacingEnv):
    env.reset(seed=1)
    total = 0.0

    for _ in range(30):
        _, reward, terminated, truncated, info = env.step(FORWARD)
        total += reward

    assert total > 0.0
    assert total == pytest.approx(info["progress"])
    assert not (terminated or truncated)


def test_driving_straight_ends_with_collision(env: RacingEnv):
    env.reset(seed=1)

    terminated = False
    info: dict = {}
    for _ in range(600):
        _, _, terminated, _, info = env.step(FORWARD)
        if terminated:
            break

    assert terminated
    assert info["collided"] is True


def test_finishing_the_lap_terminates_without_collision(env: RacingEnv):
    env.reset(seed=1)
    env.unwrapped._progress = env.track.length - 0.001

    _, _, terminated, truncated, info = env.step(FORWARD)

    assert terminated and not truncated
    assert info["lap"] == 1
    assert info["collided"] is False


def test_truncates_at_max_steps():
    environment = RacingEnv(max_steps=3)
    try:
        environment.reset(seed=1)
        results = [environment.step(IDLE) for _ in range(3)]
    finally:
        environment.close()

    assert [result[2] for result in results] == [False, False, False]
    assert [result[3] for result in results] == [False, False, True]


def test_step_after_episode_end_requires_reset():
    environment = RacingEnv(max_steps=1)
    try:
        environment.reset(seed=1)
        environment.step(IDLE)
        with pytest.raises(RuntimeError, match="reset"):
            environment.step(IDLE)
    finally:
        environment.close()


def test_step_before_reset_raises(env: RacingEnv):
    with pytest.raises(RuntimeError, match="reset"):
        env.step(IDLE)


@pytest.mark.parametrize("action", [np.array([2.0, 0.0], dtype=np.float32), np.zeros(3)])
def test_invalid_action_raises(env: RacingEnv, action: np.ndarray):
    env.reset(seed=1)

    with pytest.raises(ValueError):
        env.step(action)


@pytest.mark.parametrize(
    "kwargs",
    [{"ray_angles_deg": []}, {"ray_range": 0.0}, {"dt": 0.0}, {"max_steps": 0}, {"laps": 0},
     {"track": "missing"}, {"dynamics": "missing"}, {"render_mode": "ansi"}],
)
def test_invalid_constructor_arguments_raise(kwargs: dict):
    with pytest.raises(ValueError):
        RacingEnv(**kwargs)


def test_rgb_array_render_returns_a_frame():
    environment = RacingEnv(render_mode="rgb_array")
    try:
        environment.reset(seed=1)
        frame = environment.render()
    finally:
        environment.close()

    assert frame.dtype == np.uint8
    assert frame.ndim == 3 and frame.shape[2] == 3
    assert frame.std() > 0


def test_human_render_mode_steps_without_error():
    environment = RacingEnv(render_mode="human")
    try:
        environment.reset(seed=1)
        environment.step(IDLE)
    finally:
        environment.close()
