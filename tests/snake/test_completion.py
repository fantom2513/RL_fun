import numpy as np
import pytest
import torch

from rl_fun.snake.checkpoint import load_checkpoint, save_checkpoint
from rl_fun.snake.config import SnakeConfig
from rl_fun.snake.dqn import DQNTrainer
from rl_fun.snake.env import SnakeEnv
from rl_fun.snake.policy import evaluate


def test_board_observation_describes_order_and_food():
    env = SnakeEnv(16, observation_mode="board")
    observation, _ = env.reset(seed=42)
    assert observation.shape == (523,)
    body, food = observation[11:267].reshape(16, 16), observation[267:].reshape(16, 16)
    assert body[8, 8] == 1
    assert body[8, 7] == pytest.approx(2 / 3)
    assert body[8, 6] == pytest.approx(1 / 3)
    assert np.count_nonzero(body) == 3
    assert food[env.food[1], env.food[0]] == 1
    assert np.count_nonzero(food) == 1
    assert env.observation_space.contains(observation)


def test_win_bonus_and_occupancy_are_terminal():
    env = SnakeEnv(16, win_reward=100)
    env.reset(seed=1)
    env.snake = [(14, 0)] + [
        (x, y) for y in range(16) for x in range(16) if (x, y) not in [(14, 0), (15, 0)]
    ]
    env.food = (15, 0)
    _, reward, terminated, truncated, info = env.step(0)
    assert reward == 110 and terminated and not truncated
    assert info["occupancy"] == 1 and info["reason"] == "win"
    with pytest.raises(RuntimeError):
        env.step(0)


def test_evaluation_reports_failure_reason_and_initial_occupancy():
    class Straight:
        def act(self, observation):
            return 0

    result = evaluate(Straight(), 16, [42, 43], 1)
    assert result["wins"] == 0 and result["win_rate"] == 0
    assert result["episodes"] == 2
    assert result["occupancy"] == 3 / 256
    assert result["reasons"] == {"max_steps": 2}
    assert [row["seed"] for row in result["outcomes"]] == [42, 43]


def test_double_dqn_selects_online_action_and_target_value():
    from rl_fun.snake.dqn import next_q_values

    online = torch.tensor([[9.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
    target = torch.tensor([[3.0, 8.0, 7.0], [9.0, 4.0, 2.0]])
    torch.testing.assert_close(next_q_values(online, target, True), torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(next_q_values(online, target, False), torch.tensor([8.0, 9.0]))


def test_board_double_dqn_checkpoint_and_real_updates(tmp_path):
    config = SnakeConfig(
        algorithm="double_dqn",
        observation_mode="board",
        size=16,
        episodes=2,
        max_steps=12,
        batch_size=4,
        warmup=4,
        train_interval=1,
    )
    trainer = DQNTrainer(config)
    before = [p.detach().clone() for p in trainer.model.parameters()]
    trainer.run(lambda: True, lambda _: None)
    assert trainer.updates > 0
    assert any(
        not torch.equal(a, b) for a, b in zip(before, trainer.model.parameters(), strict=True)
    )
    path = tmp_path / "model.pt"
    save_checkpoint(path, trainer.state_dict())
    state = load_checkpoint(path)
    assert state["version"] == 2 and state["architecture"][0] == 523
    restored = DQNTrainer(config)
    restored.load_state_dict(state)
    torch.testing.assert_close(trainer.model(torch.zeros(523)), restored.model(torch.zeros(523)))


def test_disabled_frame_gate_does_not_construct_step_snapshots(monkeypatch):
    class Disabled:
        def enabled(self):
            return False

        def __call__(self, row):
            pytest.fail("disabled frame emitted")

    trainer = DQNTrainer(SnakeConfig(size=16, episodes=1, max_steps=10))

    def forbidden_snapshot():
        pytest.fail("training snapshot created while rendering disabled")

    monkeypatch.setattr(trainer.env, "snapshot", forbidden_snapshot)
    trainer.run(lambda: True, lambda _: None, Disabled())


def test_legacy_checkpoint_remains_loadable(tmp_path):
    trainer = DQNTrainer(SnakeConfig(episodes=1, max_steps=10))
    state = trainer.state_dict()
    state["version"] = 1
    for key in ("observation_mode", "win_reward", "evaluation_interval"):
        state["config"].pop(key)
    path = tmp_path / "legacy.pt"
    save_checkpoint(path, state)
    restored = DQNTrainer(SnakeConfig(episodes=1, max_steps=10))
    restored.load_state_dict(load_checkpoint(path))
    torch.testing.assert_close(trainer.model(torch.zeros(11)), restored.model(torch.zeros(11)))


def test_board_model_playback_and_live_replacement(tmp_path):
    from rl_fun.snake.manager import SnakeManager

    manager = SnakeManager(tmp_path)
    try:
        game = manager.create_game({"mode": "split", "size": 16, "follow_training": True})
        trainer = DQNTrainer(SnakeConfig(algorithm="double_dqn", observation_mode="board"))
        trainer.completed = trainer.updates = 1
        run_id = "b" * 32
        path = manager.root / run_id
        path.mkdir()
        save_checkpoint(path / "model.pt", trainer.state_dict())
        manager.run = {
            "id": run_id,
            "config": trainer.config.to_dict(),
            "saved": True,
            "saved_iteration": 1,
        }
        moved = manager.step_game(game["id"], {})
        assert moved["trained"] and moved["model"] == run_id
        assert manager.sessions[game["id"]].agent.observation().shape == (523,)
        fixed = manager.create_game({"mode": "agent", "size": 16, "model": run_id})
        assert fixed["trained"]
        assert manager.step_game(fixed["id"], {})["agent"]["steps"] == 1
    finally:
        manager.run = None
        manager.shutdown()


def test_resume_rejects_observation_mode_change():
    trainer = DQNTrainer(SnakeConfig())
    changed = DQNTrainer(SnakeConfig(observation_mode="board"))
    with pytest.raises(ValueError, match="наблюдение"):
        changed.load_state_dict(trainer.state_dict())


def test_headless_evolution_and_evaluation_skip_step_snapshots(monkeypatch):
    from rl_fun.snake.evolution import EvolutionTrainer

    class Disabled:
        def enabled(self):
            return False

        def __call__(self, row):
            pytest.fail("disabled frame emitted")

    original = SnakeEnv.snapshot
    calls = []

    def counted(env):
        calls.append(env.steps)
        return original(env)

    monkeypatch.setattr(SnakeEnv, "snapshot", counted)
    config = SnakeConfig(
        algorithm="evolution",
        size=16,
        generations=1,
        population=4,
        evaluation_episodes=1,
        max_steps=10,
    )
    trainer = EvolutionTrainer(config)
    trainer.run(lambda: True, lambda _: None, Disabled())
    # One final snapshot per evaluation episode; none for initial or step frames.
    assert len(calls) == 4 + 2


def test_preset_requires_sixteen_board_and_full_validation():
    from rl_fun.snake.presets import completion_preset, verification_status

    preset = completion_preset()
    assert preset["config"]["size"] == 16
    assert preset["config"]["algorithm"] == "double_dqn"
    assert not preset["visualization"]["enabled"]
    assert verification_status(16, 100, 90) == "verified"
    assert verification_status(8, 100, 100) == "experimental"
    assert verification_status(16, 10, 10) == "experimental"
    assert verification_status(16, 100, 89) == "experimental"


def test_validation_is_disjoint_from_training_and_tuning():
    from rl_fun.snake.presets import benchmark_seeds

    tuning, validation = benchmark_seeds()
    assert len(tuning) == 20 and len(validation) == 100
    assert not set(tuning) & set(validation)
    assert min(tuning) >= 1000000 and min(validation) > max(tuning)


def test_benchmark_saves_real_outcomes_and_checkpoint(tmp_path):
    import json

    from rl_fun.snake.benchmark import run_benchmark

    config = SnakeConfig(
        algorithm="double_dqn",
        observation_mode="board",
        size=16,
        episodes=2,
        max_steps=10,
        batch_size=4,
        warmup=4,
    )
    training = run_benchmark(config, tmp_path, train_seconds=10)
    assert not (training / "validation.json").exists()
    tuning = json.loads((training / "evaluation.json").read_text(encoding="utf-8"))
    assert tuning["episodes"] == 20 and tuning["phase"] == "tuning"
    directory = run_benchmark(config, tmp_path, resume=training / "model.pt", validate=True)
    report = json.loads((directory / "validation.json").read_text(encoding="utf-8"))
    assert report["episodes"] == 100 and len(report["outcomes"]) == 100
    assert report["status"] == "experimental" and report["wins"] == 0
    assert report["config"]["size"] == 16
    assert (directory / "record.json").is_file()
    state = load_checkpoint(directory / "model.pt")
    assert state["completed"] == 2


def test_checkpoint_rejects_replay_with_wrong_observation_size(tmp_path):
    trainer = DQNTrainer(
        SnakeConfig(observation_mode="board", batch_size=4, warmup=4, max_steps=10, episodes=1)
    )
    trainer.run(lambda: True, lambda _: None)
    state = trainer.state_dict()
    state["replay"]["observations"] = torch.zeros(len(trainer.replay), 11)
    path = tmp_path / "wrong-replay.pt"
    save_checkpoint(path, state)
    with pytest.raises(ValueError):
        load_checkpoint(path)


def test_frame_due_gate_skips_snapshot_before_callback(monkeypatch):
    from rl_fun.snake.frames import emit_frame

    class EverySecond:
        def __init__(self):
            self.calls = 0
            self.rows = []

        def should_emit(self, ended=False):
            self.calls += 1
            return ended or self.calls % 2 == 0

        def __call__(self, row):
            self.rows.append(row)

    env = SnakeEnv(16)
    env.reset(seed=42)
    callback = EverySecond()
    emit_frame(callback, env, ended=False)
    assert callback.rows == []
    emit_frame(callback, env, ended=False)
    emit_frame(callback, env, ended=True)
    assert len(callback.rows) == 2 and callback.rows[-1]["ended"]


def test_final_validation_cannot_change_game_rules(tmp_path):
    from dataclasses import replace

    from rl_fun.snake.benchmark import run_benchmark

    config = SnakeConfig(observation_mode="board", episodes=1, max_steps=10)
    trainer = DQNTrainer(config)
    path = tmp_path / "frozen.pt"
    save_checkpoint(path, trainer.state_dict())
    with pytest.raises(ValueError, match="проверки"):
        run_benchmark(replace(config, win_reward=100), tmp_path, resume=path, validate=True)


def test_resume_keeps_original_training_seed():
    trainer = DQNTrainer(SnakeConfig(seed=77, observation_mode="board"))
    restored = DQNTrainer(SnakeConfig(seed=42, observation_mode="board"))
    restored.load_state_dict(trainer.state_dict())
    assert restored.state_dict()["training_seed"] == 77


def test_won_agent_board_stays_visible_until_restart(tmp_path):
    from rl_fun.snake.manager import SnakeManager

    manager = SnakeManager(tmp_path)
    try:
        game = manager.create_game({"mode": "agent", "size": 16})
        session = manager.sessions[game["id"]]
        session.agent.snake = [(x, y) for y in range(16) for x in range(16)]
        session.agent.food = None
        session.agent.done, session.agent.reason = True, "win"
        session.agent.score = 253
        won = manager.step_game(game["id"], {})
        assert won["agent"]["reason"] == "win" and won["agent"]["occupancy"] == 1
        assert won["agent_episode"] == 1
    finally:
        manager.shutdown()
