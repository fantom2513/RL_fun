import pytest
import torch

from rl_fun.snake.gpu.imitation import clone
from rl_fun.snake.gpu.model import ConvActorCritic
from rl_fun.snake.gpu.schedule import learning_rate_at
from rl_fun.snake.gpu.selection import BestTracker
from rl_fun.snake.gpu.vec_env import VecSnake

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_the_learning_rate_is_constant_until_the_decay_starts():
    assert learning_rate_at(0, 100, 1e-3) == 1e-3
    assert learning_rate_at(69, 100, 1e-3, decay_start=0.7) == 1e-3


def test_the_learning_rate_ends_at_the_floor_fraction_of_the_base():
    assert learning_rate_at(100, 100, 1e-3, floor=0.05) == pytest.approx(5e-5)
    assert learning_rate_at(500, 100, 1e-3, floor=0.05) == pytest.approx(5e-5)


def test_the_learning_rate_falls_linearly_in_between():
    middle = learning_rate_at(85, 100, 1e-3, decay_start=0.7, floor=0.0)

    assert middle == pytest.approx(5e-4)


def test_the_learning_rate_never_increases():
    rates = [learning_rate_at(step, 200, 1e-3) for step in range(0, 201)]

    assert all(a >= b for a, b in zip(rates, rates[1:], strict=False))


@pytest.mark.parametrize(
    "arguments",
    [
        {"total": 0},
        {"total": 10, "decay_start": 1.5},
        {"total": 10, "floor": -0.1},
        {"total": 10, "floor": 1.5},
    ],
)
def test_invalid_schedule_parameters_are_rejected(arguments):
    with pytest.raises(ValueError):
        learning_rate_at(1, base=1e-3, **arguments)


def test_the_first_report_is_always_the_best():
    tracker = BestTracker()

    assert tracker.update({"win_rate": 0.1, "occupancy": 0.2}, 1000)
    assert tracker.iteration == 1000


def test_a_higher_win_rate_replaces_the_best_and_a_lower_one_does_not():
    tracker = BestTracker()
    tracker.update({"win_rate": 0.5, "occupancy": 0.6}, 1)

    assert tracker.update({"win_rate": 0.7, "occupancy": 0.5}, 2)
    assert not tracker.update({"win_rate": 0.6, "occupancy": 0.9}, 3)
    assert tracker.iteration == 2


def test_equal_win_rates_are_broken_by_occupancy():
    tracker = BestTracker()
    tracker.update({"win_rate": 0.5, "occupancy": 0.6}, 1)

    assert not tracker.update({"win_rate": 0.5, "occupancy": 0.6}, 2)
    assert tracker.update({"win_rate": 0.5, "occupancy": 0.7}, 3)


def test_clone_applies_the_schedule_with_global_iteration_numbers():
    vec = VecSnake(8, 32, DEVICE, seed=0)
    model = ConvActorCritic(8).to(DEVICE)

    history = clone(
        model,
        vec,
        3,
        rollout=1,
        start_iteration=10,
        learning_rate_schedule=lambda iteration: iteration * 1e-4,
    )

    assert [row["iteration"] for row in history] == [10, 11, 12]
    assert [row["lr"] for row in history] == pytest.approx([1.0e-3, 1.1e-3, 1.2e-3])
