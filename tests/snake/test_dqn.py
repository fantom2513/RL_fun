import numpy as np
import pytest
import torch

from rl_fun.snake.config import SnakeConfig
from rl_fun.snake.dqn import DQNTrainer, td_targets


def test_terminal_targets_do_not_bootstrap():
    result = td_targets(
        torch.tensor([10.0, -10.0, 1.0]),
        torch.tensor([False, True, False]),
        torch.tensor([4.0, 99.0, 2.0]),
        0.5,
    )
    torch.testing.assert_close(result, torch.tensor([12.0, -10.0, 2.0]))


def test_learning_changes_weights_and_bounds_replay():
    trainer = DQNTrainer(
        SnakeConfig(
            episodes=2, max_steps=20, batch_size=4, warmup=4, replay_capacity=8, train_interval=1
        )
    )
    before = [p.detach().clone() for p in trainer.model.parameters()]
    metrics = []
    trainer.run(lambda: True, metrics.append)
    assert len(metrics) == 2 and trainer.completed == 2
    assert len(trainer.replay) <= 8
    assert any(
        not torch.equal(a, b) for a, b in zip(before, trainer.model.parameters(), strict=True)
    )
    assert all(np.isfinite(m["reward"]) for m in metrics)
    assert trainer.epsilon <= trainer.config.epsilon_start


@pytest.mark.parametrize(
    "data",
    [
        {"size": 7},
        {"population": True},
        {"learning_rate": float("nan")},
        {"algorithm": "unknown"},
        {"unknown": 1},
        {"batch_size": 16, "replay_capacity": 8},
    ],
)
def test_config_rejects_invalid_input(data):
    with pytest.raises(ValueError):
        SnakeConfig.from_dict(data)


def test_warmup_must_fit_replay_capacity():
    with pytest.raises(ValueError):
        SnakeConfig.from_dict({"batch_size": 4, "replay_capacity": 8, "warmup": 256})


def test_mid_episode_checkpoint_continues_same_trajectory():
    config = SnakeConfig(episodes=3, size=8, max_steps=20, batch_size=4, warmup=4, train_interval=1)
    complete = DQNTrainer(config)
    expected = []
    complete.run(lambda: True, expected.append)
    interrupted = DQNTrainer(config)
    first = []
    interrupted.run(lambda: interrupted.steps < 5, first.append)
    restored = DQNTrainer(config)
    restored.load_state_dict(interrupted.state_dict())
    rest = []
    restored.run(lambda: True, rest.append)
    assert restored.steps == complete.steps
    assert first + rest == expected
    for a, b in zip(complete.model.parameters(), restored.model.parameters(), strict=True):
        torch.testing.assert_close(a, b)
