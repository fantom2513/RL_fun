import torch

from rl_fun.snake.gpu.imitation import clone
from rl_fun.snake.gpu.model import ConvActorCritic
from rl_fun.snake.gpu.teacher import allowed_distances, teacher_actions
from rl_fun.snake.gpu.vec_env import VecSnake

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_allowed_distances_mark_unavailable_moves_with_minus_one():
    vec = VecSnake(8, 32, DEVICE, seed=0, start_mode="mixed", max_start_length=50)

    distances = allowed_distances(vec)

    assert distances.shape == (32, 3) and bool(((distances == -1) | (distances >= 1)).all())
    assert bool((distances.max(dim=1).values >= 1).all())
    assert torch.equal(distances.argmax(dim=1), teacher_actions(vec))


def test_cloning_a_tiny_board_reaches_high_accuracy():
    torch.manual_seed(0)
    vec = VecSnake(8, 256, DEVICE, seed=0, start_mode="mixed", max_start_length=50)
    model = ConvActorCritic(8).to(DEVICE)

    history = clone(model, vec, iterations=250, rollout=2, learning_rate=2e-3, explore=0.1)

    assert history[-1]["accuracy"] > 0.9


def test_cloning_reports_finite_losses_and_accuracy_per_iteration():
    vec = VecSnake(8, 64, DEVICE, seed=1)
    model = ConvActorCritic(8).to(DEVICE)

    history = clone(model, vec, iterations=5, rollout=1, learning_rate=1e-3)

    assert len(history) == 5
    assert all(0 <= row["accuracy"] <= 1 and row["loss"] == row["loss"] for row in history)


def test_cloning_does_not_touch_the_value_head():
    vec = VecSnake(8, 64, DEVICE, seed=1)
    model = ConvActorCritic(8).to(DEVICE)
    before = [parameter.clone() for parameter in model.value_head.parameters()]

    clone(model, vec, iterations=5, rollout=1, learning_rate=1e-3)

    assert all(
        torch.equal(a, b) for a, b in zip(before, model.value_head.parameters(), strict=True)
    )
