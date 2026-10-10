import pytest
import torch

from rl_fun.snake.checkpoint import load_checkpoint, save_checkpoint
from rl_fun.snake.config import SnakeConfig
from rl_fun.snake.dqn import DQNTrainer


def test_checkpoint_roundtrip_predictions_and_continuation(tmp_path):
    config = SnakeConfig(episodes=2, max_steps=12, batch_size=4, warmup=4, train_interval=1)
    trainer = DQNTrainer(config)
    trainer.run(lambda: True, lambda _: None)
    path = tmp_path / "model.pt"
    save_checkpoint(path, trainer.state_dict())
    restored = DQNTrainer(config)
    restored.load_state_dict(load_checkpoint(path))
    x = torch.zeros(11)
    torch.testing.assert_close(trainer.model(x), restored.model(x))
    assert restored.completed == 2
    assert len(restored.replay) == len(trainer.replay)
    assert restored.rng.random() == trainer.rng.random()
    assert restored.updates == trainer.updates


def test_checkpoint_rejects_wrong_version_and_corruption(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"version": 900}, path)
    with pytest.raises(ValueError):
        load_checkpoint(path)
    path.write_bytes(b"broken")
    with pytest.raises(ValueError):
        load_checkpoint(path)
