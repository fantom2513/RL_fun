import numpy as np
import pytest
import torch

from rl_fun.snake.env import TURNS, SnakeEnv
from rl_fun.snake.gpu.vec_env import VecSnake

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def random_states(size, count, seed):
    """Snapshots of reference games after random moves, with a mix of headings."""
    rng = np.random.default_rng(seed)
    env = SnakeEnv(size, observation_mode="egocentric_cycle")
    states = []
    game = 0
    while len(states) < count:
        env.reset(seed=seed * 1000 + game)
        game += 1
        for _ in range(int(rng.integers(5, 80))):
            _, _, terminated, truncated, _ = env.step(int(rng.integers(3)))
            if terminated or truncated:
                break
            states.append(
                (
                    env.observation(),
                    [tuple(cell) for cell in env.snake],
                    env.food,
                    env.direction,
                    [not env._danger(env._next((env.direction + turn) % 4)) for turn in TURNS],
                )
            )
    return states[:count]


def load_batch(size, states, device):
    vec = VecSnake(size, len(states), device, seed=0)
    for index, (_, snake, food, direction, _) in enumerate(states):
        vec.load_snapshot(index, snake, food, direction)
    return vec


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("size", (8, 10, 12))
def test_observation_matches_the_reference_environment(size, device):
    states = random_states(size, 300, seed=size)
    vec = load_batch(size, states, device)
    expected = np.stack([observation for observation, *_ in states])

    actual = vec.observation().reshape(len(states), -1).cpu().numpy()

    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
    assert {state[3] for state in states} == {0, 1, 2, 3}


def test_observation_has_six_planes_of_the_head_centred_window():
    vec = VecSnake(8, 5, "cpu", seed=0)

    assert tuple(vec.observation().shape) == (5, 6, 15, 15)


def test_cycle_plane_is_zero_at_the_head_and_a_permutation_over_the_board():
    size = 8
    vec = VecSnake(size, 20, "cpu", seed=2)
    for _ in range(30):
        vec.step(torch.randint(3, (20,), generator=torch.Generator().manual_seed(1)))
    planes = vec.observation()

    for index in range(20):
        cycle = planes[index, 5]
        inside = planes[index, 4] == 0
        assert float(cycle[size - 1, size - 1]) == 0
        values = torch.sort(cycle[inside]).values
        assert torch.allclose(values, torch.arange(size**2) / size**2)


@pytest.mark.parametrize("device", DEVICES)
def test_action_mask_matches_the_reference_dangers(device):
    states = random_states(8, 300, seed=11)
    vec = load_batch(8, states, device)
    expected = torch.tensor([safe for *_, safe in states])

    mask = vec.action_mask().cpu()

    assert torch.equal(mask, expected)
    assert not expected.all() and expected.any()
