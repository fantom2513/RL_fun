import json

import pytest
import torch

from rl_fun.snake.benchmark import run_benchmark
from rl_fun.snake.checkpoint import load_checkpoint, playback_weights
from rl_fun.snake.config import SnakeConfig
from rl_fun.snake.manager import SnakeManager


def test_benchmark_bounds_environment_steps_and_records_actual_work(tmp_path):
    config = SnakeConfig(
        algorithm="ppo",
        size=16,
        observation_mode="board",
        episodes=100000,
        max_steps=100000,
        rollout_steps=8,
        batch_size=4,
        evaluation_episodes=1,
        evaluation_interval=1000,
    )
    directory = run_benchmark(
        config, tmp_path, train_seconds=30, train_steps=24, progress=lambda _: None
    )
    state = load_checkpoint(directory / "model.pt")
    report = json.loads((directory / "evaluation.json").read_text(encoding="utf-8"))
    assert state["steps"] == report["steps"] == 24
    assert state["updates"] == report["updates"] and report["updates"] > 0
    assert report["training_steps_budget"] == 24
    assert report["phase"] == "tuning" and report["episodes"] == 20
    assert not report["validation_seeds"]
    assert report["latest_tuning"]["episodes"] == 20


@pytest.mark.parametrize("steps", [0, -1, True, 2.5])
def test_search_rejects_invalid_step_budgets(tmp_path, steps):
    with pytest.raises(ValueError):
        run_benchmark(
            SnakeConfig(observation_mode="board"),
            tmp_path,
            train_steps=steps,
            progress=lambda _: None,
        )


def test_early_stop_saves_the_policy_that_was_evaluated(tmp_path):
    config = SnakeConfig(
        algorithm="ppo",
        size=16,
        observation_mode="board",
        episodes=100000,
        rollout_steps=4,
        batch_size=4,
        evaluation_interval=1000,
    )
    directory = run_benchmark(
        config, tmp_path, train_seconds=30, train_steps=4, progress=lambda _: None
    )
    state = load_checkpoint(directory / "model.pt")
    assert state["completed"] == 0 and state["updates"] > 0
    assert all(
        torch.equal(value, playback_weights(state)[key]) for key, value in state["model"].items()
    )
    manager = SnakeManager(tmp_path)
    try:
        measured = [p for p in manager.presets() if p["id"] == "measured_candidate"]
        assert len(measured) == 1
        assert measured[0]["model"] == directory.name
        assert measured[0]["status"] == "experimental"
        assert measured[0]["config"] == config.to_dict()
        assert "0/20" in measured[0]["description"]
        record = json.loads((directory / "record.json").read_text(encoding="utf-8"))
        for index in range(100):
            newer = directory.parent / f"{index:032x}"
            newer.mkdir()
            (newer / "model.pt").write_bytes(b"unused model listing fixture")
            (newer / "record.json").write_text(
                json.dumps({**record, "created": record["created"] + index + 1}),
                encoding="utf-8",
            )
        assert directory.name not in {model["id"] for model in manager.models()}
        assert next(p for p in manager.presets() if p["id"] == "measured_candidate")[
            "model"
        ] == directory.name
        import os

        checkpoint = directory / "model.pt"
        stat = checkpoint.stat()
        os.utime(checkpoint, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
        assert not any(p["id"] == "measured_candidate" for p in manager.presets())
    finally:
        manager.shutdown()
