import numpy as np
import pytest
import torch

from rl_fun.snake.checkpoint import load_checkpoint, save_checkpoint
from rl_fun.snake.config import SnakeConfig
from rl_fun.snake.td3 import TD3Trainer


def test_td3_uses_continuous_scores_and_delays_actor_updates():
    config = SnakeConfig(
        algorithm="td3",
        size=16,
        observation_mode="board",
        batch_size=4,
        warmup=4,
        train_interval=1,
        policy_delay=2,
    )
    trainer = TD3Trainer(config)
    observation, _ = trainer.env.reset(seed=1)
    for _ in range(4):
        action, scores = trainer.choose_action(observation)
        assert scores.shape == (3,) and (abs(scores) <= 1).all()
        assert action == int(scores.argmax())
        trainer.replay.append((observation, scores, 10.0, observation, True))
    before = [p.detach().clone() for p in trainer.model.parameters()]
    critics_before = [p.detach().clone() for p in trainer.critic1.parameters()]
    trainer.learn_batch()
    assert all(torch.equal(a, b) for a, b in zip(before, trainer.model.parameters(), strict=True))
    assert any(
        not torch.equal(a, b)
        for a, b in zip(critics_before, trainer.critic1.parameters(), strict=True)
    )
    trainer.learn_batch()
    assert any(
        not torch.equal(a, b) for a, b in zip(before, trainer.model.parameters(), strict=True)
    )


def test_td3_resume_keeps_continuous_replay_and_same_trajectory(tmp_path):
    config = SnakeConfig(
        algorithm="td3",
        size=8,
        episodes=3,
        max_steps=20,
        batch_size=4,
        warmup=4,
        train_interval=1,
        evaluation_episodes=1,
    )
    complete = TD3Trainer(config)
    expected = []
    complete.run(lambda: True, expected.append)
    interrupted = TD3Trainer(config)
    first = []
    interrupted.run(lambda: interrupted.steps < 5, first.append)
    path = tmp_path / "model.pt"
    save_checkpoint(path, interrupted.state_dict())
    state = load_checkpoint(path)
    assert state["replay"]["actions"].shape[1] == 3
    restored = TD3Trainer(config)
    restored.load_state_dict(state)
    rest = []
    restored.run(lambda: True, rest.append)
    assert first + rest == expected and restored.steps == complete.steps
    for a, b in zip(complete.model.parameters(), restored.model.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_td3_checkpoint_validates_both_target_critics(tmp_path):
    trainer = TD3Trainer(SnakeConfig(algorithm="td3"))
    state = trainer.state_dict()
    state["target_critic2"]["0.weight"][0, 0] = float("nan")
    path = tmp_path / "bad.pt"
    save_checkpoint(path, state)
    with pytest.raises(ValueError):
        load_checkpoint(path)


def test_td3_time_limit_does_not_make_replay_transition_terminal():
    trainer = TD3Trainer(
        SnakeConfig(
            algorithm="td3", size=16, episodes=1, max_steps=10, warmup=1000, evaluation_episodes=1
        )
    )
    # A three-cell snake can keep turning on a four-cell square safely.
    trainer.choose_action = lambda observation: (2, np.array([-1.0, -1.0, 1.0], dtype=np.float32))
    trainer.run(lambda: True, lambda _: None)
    assert trainer.env.reason == "max_steps"
    assert trainer.replay[-1][4] is False


def test_td3_greedy_playback_matches_adapter_even_when_scores_saturate():
    trainer = TD3Trainer(SnakeConfig(algorithm="td3", action_noise=0, warmup=4))
    observation, _ = trainer.env.reset(seed=1)
    with torch.no_grad():
        for parameter in trainer.model.parameters():
            parameter.zero_()
        trainer.model[-1].bias.copy_(torch.tensor([10.0, 20.0, 30.0]))
    trainer.steps = 4
    action, scores = trainer.choose_action(observation)
    assert action == 0 and scores.tolist() == [1.0, 1.0, 1.0]
    assert trainer.model.act(observation) == action
