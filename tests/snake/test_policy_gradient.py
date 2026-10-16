import time
from dataclasses import replace

import pytest
import torch

from rl_fun.snake.checkpoint import load_checkpoint, save_checkpoint
from rl_fun.snake.config import SnakeConfig
from rl_fun.snake.on_policy import OnPolicyTrainer, generalized_advantages, ppo_objective


def test_gae_bootstraps_time_limit_but_not_collision_or_next_episode():
    advantage, returns = generalized_advantages(
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([4.0, 5.0, 6.0]),
        torch.tensor([5.0, 99.0, 8.0]),
        torch.tensor([False, True, False]),
        torch.tensor([False, True, True]),
        gamma=0.5,
        gae_lambda=1.0,
    )
    # Deltas: -0.5, -3, 1. Episode boundary stops carry from row 2 to row 1.
    torch.testing.assert_close(advantage, torch.tensor([-2.0, -3.0, 1.0]))
    torch.testing.assert_close(returns, torch.tensor([2.0, 2.0, 7.0]))


def test_ppo_clipping_limits_both_positive_and_negative_advantages():
    objective = ppo_objective(
        torch.log(torch.tensor([2.0, 0.5])),
        torch.zeros(2),
        torch.tensor([3.0, -4.0]),
        0.2,
    )
    torch.testing.assert_close(objective, torch.tensor([3.6, -3.2]))


@pytest.mark.parametrize("algorithm", ["reinforce", "a2c", "ppo"])
def test_on_policy_training_updates_actor_and_safe_checkpoint_plays(tmp_path, algorithm):
    config = SnakeConfig(
        algorithm=algorithm,
        size=16,
        observation_mode="board",
        episodes=3,
        max_steps=20,
        rollout_steps=8,
        evaluation_episodes=1,
        evaluation_interval=2,
    )
    trainer = OnPolicyTrainer(config)
    before = [p.detach().clone() for p in trainer.model.parameters()]
    metrics = []
    trainer.run(lambda: True, metrics.append)
    assert trainer.completed == 3 and trainer.updates > 0
    assert any(
        not torch.equal(a, b) for a, b in zip(before, trainer.model.parameters(), strict=True)
    )
    assert len(metrics) == 3 and all("entropy" in row for row in metrics)
    assert all(torch.isfinite(p).all() for p in trainer.model.parameters())
    path = tmp_path / "model.pt"
    save_checkpoint(path, trainer.state_dict())
    state = load_checkpoint(path)
    restored = OnPolicyTrainer(replace(config, episodes=4))
    restored.load_state_dict(state)
    restored.run(lambda: True, lambda _: None)
    assert restored.completed == 4 and restored.model.act(restored.env.observation()) in (0, 1, 2)


@pytest.mark.parametrize("algorithm", ["reinforce", "a2c", "ppo"])
def test_on_policy_mid_episode_resume_keeps_rollout_and_rng(tmp_path, algorithm):
    config = SnakeConfig(
        algorithm=algorithm,
        size=8,
        episodes=3,
        max_steps=20,
        rollout_steps=8,
        evaluation_episodes=1,
    )
    complete = OnPolicyTrainer(config)
    expected = []
    complete.run(lambda: True, expected.append)
    interrupted = OnPolicyTrainer(config)
    first = []
    interrupted.run(lambda: interrupted.steps < 5, first.append)
    path = tmp_path / "resume.pt"
    save_checkpoint(path, interrupted.state_dict())
    restored = OnPolicyTrainer(config)
    restored.load_state_dict(load_checkpoint(path))
    rest = []
    restored.run(lambda: True, rest.append)
    assert first + rest == expected
    assert restored.steps == complete.steps
    for a, b in zip(complete.model.parameters(), restored.model.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_on_policy_checkpoint_rejects_corrupt_critic(tmp_path):
    trainer = OnPolicyTrainer(SnakeConfig(algorithm="ppo"))
    state = trainer.state_dict()
    state["critic"]["0.weight"][0, 0] = float("nan")
    path = tmp_path / "bad.pt"
    save_checkpoint(path, state)
    with pytest.raises(ValueError):
        load_checkpoint(path)


def test_checkpoint_rejects_poisoned_adam_moments(tmp_path):
    trainer = OnPolicyTrainer(
        SnakeConfig(
            algorithm="ppo", episodes=1, max_steps=10, rollout_steps=8, evaluation_episodes=1
        )
    )
    trainer.run(lambda: True, lambda _: None)
    state = trainer.state_dict()
    next(iter(state["optimizer"]["state"].values()))["exp_avg"].flatten()[0] = float("nan")
    path = tmp_path / "poison.pt"
    save_checkpoint(path, state)
    with pytest.raises(ValueError):
        load_checkpoint(path)


def test_ppo_stops_inside_optimization_and_resumes_exactly(tmp_path):
    config = SnakeConfig(
        algorithm="ppo",
        size=8,
        episodes=2,
        max_steps=20,
        rollout_steps=8,
        batch_size=4,
        ppo_epochs=4,
        evaluation_episodes=1,
    )
    complete = OnPolicyTrainer(config)
    expected = []
    complete.run(lambda: True, expected.append)
    interrupted = OnPolicyTrainer(config)
    first = []
    interrupted.run(lambda: interrupted.updates < 2, first.append)
    assert interrupted.updates == 2 and interrupted.completed < 2
    path = tmp_path / "partial-update.pt"
    save_checkpoint(path, interrupted.state_dict())
    restored = OnPolicyTrainer(config)
    restored.load_state_dict(load_checkpoint(path))
    rest = []
    restored.run(lambda: True, rest.append)
    assert first + rest == expected
    for a, b in zip(complete.model.parameters(), restored.model.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("algorithm", ["reinforce", "a2c", "ppo", "td3"])
def test_spawned_on_policy_worker_saves_playable_actor_and_resumes(tmp_path, algorithm):
    from rl_fun.snake.manager import SnakeManager

    manager = SnakeManager(tmp_path)
    config = dict(
        algorithm=algorithm,
        size=16,
        observation_mode="board",
        episodes=2,
        max_steps=10,
        rollout_steps=8,
        batch_size=4,
        warmup=4,
        train_interval=1,
        evaluation_episodes=1,
    )

    def wait_finished():
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            run = manager.state()["run"]
            if run["status"] in ("finished", "error"):
                assert run["status"] == "finished", run
                return run
            time.sleep(0.02)
        pytest.fail("worker did not finish")

    try:
        run_id = manager.train(config, visualization={"enabled": False, "fps": 60})
        run = wait_finished()
        assert run["history"][-1]["updates"] > 0
        models = manager.models()
        assert any(row["id"] == run_id for row in models)
        if algorithm == "td3":
            path = tmp_path / "snake" / run_id / "model.pt"
            saved = load_checkpoint(path)
            for weights in (saved["model"], saved["best_model"]):
                for value in weights.values():
                    value.zero_()
                weights["4.bias"].copy_(torch.tensor([10.0, 20.0, 30.0]))
            save_checkpoint(path, saved)
        game = manager.create_game({"mode": "agent", "size": 16, "model": run_id})
        assert game["trained"]
        moved = manager.step_game(game["id"], {})
        assert moved["agent"]["steps"] == 1
        if algorithm == "td3":
            assert moved["agent"]["direction"] == 0
        manager.train({**config, "episodes": 1}, resume=run_id)
        continued = wait_finished()
        assert continued["history"][-1]["iteration"] == 3
    finally:
        manager.shutdown()
