import pytest
import torch

from rl_fun.snake.gpu.cycle import hamiltonian_cycle
from rl_fun.snake.gpu.teacher import teacher_actions
from rl_fun.snake.gpu.vec_env import REASONS, VecSnake

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WIN = REASONS.index("win")


def first_outcomes(vec, policy, limit):
    """Play until every game has finished once; return (finished, reason, steps)."""
    count = vec.num_envs
    finished = torch.zeros(count, dtype=torch.bool, device=vec.device)
    reason = torch.zeros(count, dtype=torch.long, device=vec.device)
    steps = torch.zeros(count, dtype=torch.long, device=vec.device)
    for _ in range(limit):
        _, _, _, info = vec.step(policy(vec))
        fresh = info["done"] & ~finished
        reason[fresh] = info["reason"][fresh]
        steps[fresh] = info["steps"][fresh]
        finished |= fresh
        if bool(finished.all()):
            break
    return finished, reason, steps


def pure(vec):
    return teacher_actions(vec, shortcuts=False)


@pytest.mark.parametrize("size", (8, 10, 12, 14, 16, 18, 20))
def test_standard_start_follows_the_cycle_in_its_heading(size):
    vec = VecSnake(size, 1, "cpu", seed=0)
    order = hamiltonian_cycle(size)

    indexes = [int(order[y, x]) for x, y in vec.snake_cells(0)]  # head first

    assert all((a - b) % size**2 == 1 for a, b in zip(indexes, indexes[1:], strict=False))
    assert int(vec.direction[0]) == 0


@pytest.mark.parametrize(("size", "games"), ((8, 200), (12, 100), (16, 50)))
def test_the_pure_cycle_follower_wins_every_game(size, games):
    vec = VecSnake(size, games, DEVICE, seed=1)

    finished, reason, _ = first_outcomes(vec, pure, size**4)

    assert bool(finished.all()) and bool((reason == WIN).all())


@pytest.mark.parametrize(("size", "games"), ((8, 1000), (16, 200)))
def test_the_shortcut_teacher_wins_every_game(size, games):
    vec = VecSnake(size, games, DEVICE, seed=2)

    finished, reason, _ = first_outcomes(vec, teacher_actions, size**4)

    assert bool(finished.all()) and bool((reason == WIN).all())


def test_shortcuts_make_games_shorter_than_following_the_cycle():
    results = {}
    for name, policy in (("pure", pure), ("shortcut", teacher_actions)):
        vec = VecSnake(8, 300, DEVICE, seed=3)
        _, _, steps = first_outcomes(vec, policy, 8**4)
        results[name] = steps.float().mean()

    assert results["shortcut"] < results["pure"]


def test_the_teacher_never_picks_a_fatal_move():
    vec = VecSnake(12, 256, DEVICE, seed=4, start_mode="mixed", max_start_length=100)

    for _ in range(600):
        actions = teacher_actions(vec)
        safe = vec.action_mask().gather(1, actions[:, None])
        assert bool(safe.all())
        vec.step(actions)


@pytest.mark.parametrize("longest", (10, 32, 57))
def test_the_teacher_wins_from_cycle_segment_starts_of_every_length(longest):
    vec = VecSnake(8, 300, DEVICE, seed=5, start_mode="cycle_segment", max_start_length=longest)

    finished, reason, _ = first_outcomes(vec, teacher_actions, 8**4)

    assert bool(finished.all()) and bool((reason == WIN).all())


def test_a_shortcut_limit_turns_shortcuts_off_for_long_snakes():
    vec = VecSnake(8, 100, DEVICE, seed=6, start_mode="cycle_segment", max_start_length=57)
    long_snakes = vec.length > 8

    limited = teacher_actions(vec, shortcut_fraction=0.1)
    pure_actions = teacher_actions(vec, shortcuts=False)

    assert bool((limited == pure_actions)[long_snakes].all())
