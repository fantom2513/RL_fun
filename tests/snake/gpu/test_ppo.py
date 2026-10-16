import pytest
import torch

from rl_fun.snake.gpu.model import ConvActorCritic
from rl_fun.snake.gpu.ppo import (
    PPOConfig,
    collect_rollout,
    compute_gae,
    finish_rollout,
    kl_divergence,
    masked_logits,
    policy_loss,
    ppo_update,
)
from rl_fun.snake.gpu.vec_env import VecSnake

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def column(*values):
    return torch.tensor(values, dtype=torch.float32).view(-1, 1)


def test_gae_matches_a_hand_computed_trajectory_with_a_terminal_step():
    rewards, values = column(1, 0, 2), column(0.5, 0.4, 0.3)
    dones = column(0, 0, 1).bool()

    advantages, returns = compute_gae(rewards, values, dones, torch.tensor([0.2]), 0.9, 0.8)

    assert torch.allclose(advantages, column(1.64768, 1.094, 1.7), atol=1e-5)
    assert torch.allclose(returns, column(2.14768, 1.494, 2.0), atol=1e-5)


def test_gae_bootstraps_from_the_last_value_when_the_rollout_is_cut():
    advantages, _ = compute_gae(
        column(1), column(0.5), column(0).bool(), torch.tensor([0.2]), 0.9, 0.8
    )

    assert torch.allclose(advantages, column(0.68), atol=1e-6)


def test_a_finished_game_stops_the_bootstrap_from_the_next_game():
    advantages, _ = compute_gae(
        column(1, 5), column(0.0, 0.0), column(1, 0).bool(), torch.tensor([0.0]), 0.9, 0.8
    )

    assert torch.allclose(advantages[0], torch.tensor([1.0]))


def test_policy_loss_is_zero_for_unchanged_policy_and_zero_advantage():
    log_prob = torch.log(torch.tensor([0.2, 0.5, 0.3]))

    loss = policy_loss(log_prob, log_prob, torch.zeros(3), clip=0.2)

    assert float(loss) == 0.0


def test_policy_loss_clips_a_ratio_that_grew_too_much():
    old = torch.log(torch.tensor([0.2]))
    new = torch.log(torch.tensor([0.3]))  # ratio 1.5

    loss = policy_loss(new, old, torch.ones(1), clip=0.2)

    assert float(loss) == pytest.approx(-1.2)


def test_kl_is_zero_for_identical_policies_and_positive_otherwise():
    logits = torch.randn(10, 3)

    assert float(kl_divergence(logits, logits)) == pytest.approx(0.0, abs=1e-7)
    assert float(kl_divergence(logits, logits + torch.randn(10, 3))) > 0


def test_masked_logits_remove_unsafe_moves_but_keep_dead_ends_uniform():
    logits = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    mask = torch.tensor([[True, False, True], [False, False, False]])

    probabilities = masked_logits(logits, mask).softmax(dim=1)

    assert float(probabilities[0, 1]) < 1e-6
    assert torch.allclose(probabilities[1], logits.softmax(dim=1)[1])


def make_rollout(use_mask=True, steps=8, envs=32, seed=0):
    torch.manual_seed(seed)
    model = ConvActorCritic(8).to(DEVICE)
    vec = VecSnake(8, envs, DEVICE, seed=seed, food_reward=1.0, death_penalty=1.0)
    config = PPOConfig(minibatch=64, epochs=1, use_mask=use_mask)
    rollout = collect_rollout(model, vec, steps, use_mask)
    return model, config, finish_rollout(rollout, config)


def test_a_rollout_has_one_entry_per_step_and_game():
    _, _, rollout = make_rollout()

    assert rollout["actions"].shape == (8, 32) and rollout["observations"].shape[:2] == (8, 32)
    assert rollout["advantages"].shape == (8, 32) and rollout["masks"].shape == (8, 32, 3)


def test_masked_collection_never_samples_a_fatal_move_when_a_safe_one_exists():
    model, _, rollout = make_rollout(use_mask=True, steps=20)
    picked_safe = rollout["masks"].gather(2, rollout["actions"][..., None])[..., 0]
    any_safe = rollout["masks"].any(dim=2)

    assert bool(picked_safe[any_safe].all())


def test_one_update_changes_the_parameters_and_keeps_them_finite():
    model, config, rollout = make_rollout()
    before = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    stats = ppo_update(model, optimizer, rollout, config)

    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert all(value == value for value in stats.values())


def test_the_critic_only_phase_leaves_the_policy_untouched():
    model, config, rollout = make_rollout()
    policy = [p for n, p in model.named_parameters() if not n.startswith("value_head")]
    before_policy = [p.detach().clone() for p in policy]
    before_value = [p.detach().clone() for p in model.value_head.parameters()]
    optimizer = torch.optim.Adam(model.value_head.parameters(), lr=1e-2)

    ppo_update(model, optimizer, rollout, config, critic_only=True)

    assert all(torch.equal(a, b) for a, b in zip(before_policy, policy, strict=True))
    assert any(
        not torch.equal(a, b)
        for a, b in zip(before_value, model.value_head.parameters(), strict=True)
    )


def test_updates_raise_the_probability_of_actions_with_positive_advantage():
    model, config, rollout = make_rollout(use_mask=False)
    rollout["advantages"] = torch.where(rollout["actions"] == 0, 1.0, -1.0)
    flat = rollout["observations"].flatten(0, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    with torch.no_grad():
        before = float(model(flat)[0].softmax(dim=1)[:, 0].mean())

    for _ in range(15):
        ppo_update(model, optimizer, rollout, config)

    with torch.no_grad():
        after = float(model(flat)[0].softmax(dim=1)[:, 0].mean())
    assert after > before


def test_the_kl_anchor_slows_the_drift_from_the_reference_policy():
    drift = {}
    for coef in (0.0, 50.0):
        model, config, rollout = make_rollout(use_mask=False)
        rollout["advantages"] = torch.where(rollout["actions"] == 0, 1.0, -1.0)
        config = PPOConfig(minibatch=64, epochs=1, use_mask=False, kl_coef=coef)
        reference = ConvActorCritic(8).to(DEVICE)
        reference.load_state_dict(model.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(15):
            ppo_update(model, optimizer, rollout, config, reference=reference)
        flat = rollout["observations"].flatten(0, 1)
        with torch.no_grad():
            drift[coef] = float(kl_divergence(model(flat)[0], reference(flat)[0]))

    assert drift[50.0] < drift[0.0]
