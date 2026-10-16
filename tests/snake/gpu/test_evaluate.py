import pytest
import torch

from rl_fun.snake.gpu.evaluate import evaluate_reference, evaluate_vec, greedy_policy
from rl_fun.snake.gpu.model import ConvActorCritic
from rl_fun.snake.gpu.teacher import teacher_actions

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_the_teacher_wins_every_evaluation_game():
    report = evaluate_vec(teacher_actions, size=8, games=100, seed=1, device=DEVICE)

    assert report["win_rate"] == 1.0 and report["wins"] == 100


def test_the_report_has_the_documented_fields():
    report = evaluate_vec(teacher_actions, size=8, games=20, seed=1, device=DEVICE)

    assert {"games", "wins", "win_rate", "occupancy", "steps", "reasons"} <= set(report)
    assert report["games"] == 20 and report["reasons"] == {"win": 20}


def test_the_same_seed_gives_the_same_report():
    first = evaluate_vec(teacher_actions, size=8, games=30, seed=7, device=DEVICE)
    second = evaluate_vec(teacher_actions, size=8, games=30, seed=7, device=DEVICE)

    assert first == second


def test_a_policy_that_never_eats_ends_by_starvation():
    def spinner(vec):
        return torch.ones(vec.num_envs, dtype=torch.long, device=vec.device)

    report = evaluate_vec(spinner, size=8, games=20, seed=2, device=DEVICE, max_steps=9000)

    assert report["wins"] == 0 and report["reasons"].get("starvation", 0) == 20


def test_games_cut_off_by_the_step_cap_are_reported_as_such():
    report = evaluate_vec(teacher_actions, size=8, games=10, seed=3, device=DEVICE, max_steps=20)

    assert report["reasons"] == {"max_steps": 10} and report["wins"] == 0


def test_a_model_policy_plays_on_the_batched_environment():
    model = ConvActorCritic(8).to(DEVICE).eval()

    report = evaluate_vec(
        greedy_policy(model), size=8, games=10, seed=1, device=DEVICE, max_steps=500
    )

    assert report["games"] == 10 and sum(report["reasons"].values()) == 10


def test_the_reference_backend_is_reproducible_and_reports_each_game():
    model = ConvActorCritic(8).eval()

    first = evaluate_reference(model, size=8, seeds=[1, 2, 3], max_steps=300)
    second = evaluate_reference(model, size=8, seeds=[1, 2, 3], max_steps=300)

    assert first == second
    assert first["games"] == 3 and len(first["outcomes"]) == 3
    assert {"seed", "score", "steps", "reason", "win"} <= set(first["outcomes"][0])


def test_the_reference_backend_needs_at_least_one_seed():
    with pytest.raises(ValueError):
        evaluate_reference(ConvActorCritic(8), size=8, seeds=[], max_steps=10)
