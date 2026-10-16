import importlib.util
import json
from pathlib import Path

import pytest

from rl_fun.snake.benchmark import run_benchmark
from rl_fun.snake.config import SnakeConfig


def loop_module():
    path = Path(__file__).parents[2] / "scripts" / "snake_training_loop.py"
    spec = importlib.util.spec_from_file_location("snake_training_loop", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_loop_variants_preserve_replay_rewards_and_validate_parameters():
    loop = loop_module()
    base = SnakeConfig(algorithm="double_dqn", observation_mode="board").to_dict()
    for cycle in range(12):
        config = loop.trial_config(base, cycle)
        assert config.food_reward == base["food_reward"]
        assert config.death_penalty == base["death_penalty"]
        assert config.size == 16
    ppo = {**base, "algorithm": "ppo"}
    variants = [loop.trial_config(ppo, cycle).to_dict() for cycle in range(6)]
    assert len({str(v) for v in variants}) == 6


@pytest.mark.parametrize("algorithm", ["ppo", "dqn", "double_dqn", "a2c", "reinforce", "td3"])
def test_all_scheduled_algorithms_have_distinct_valid_trials(algorithm):
    loop = loop_module()
    assert algorithm in loop.ALGORITHM_SCHEDULE
    base = SnakeConfig(algorithm=algorithm, observation_mode="board").to_dict()
    trials = [loop.trial_config(base, variant).to_dict() for variant in range(6)]
    assert len({str(trial) for trial in trials}) == 6
    if algorithm in ("dqn", "double_dqn", "td3"):
        for trial in trials:
            for key in ("food_reward", "death_penalty", "distance_reward", "step_penalty"):
                assert trial[key] == base[key]


def test_second_loop_cannot_take_live_lock(tmp_path):
    loop = loop_module()
    with loop.exclusive_loop(tmp_path / "loop.lock"):
        with pytest.raises(OSError):
            with loop.exclusive_loop(tmp_path / "loop.lock"):
                pytest.fail("duplicate acquired lock")


def test_stop_request_saves_checkpoint_without_training(tmp_path):
    config = SnakeConfig(algorithm="ppo", observation_mode="board", max_steps=10)
    directory = run_benchmark(
        config, tmp_path, stop_requested=lambda: True, progress=lambda _: None
    )
    import json

    report = json.loads((directory / "evaluation.json").read_text())
    assert report["steps_added"] == report["updates"] == 0
    assert (directory / "model.pt").is_file()


def test_loop_does_not_reuse_independent_validation(tmp_path):
    root = tmp_path / "snake"
    root.mkdir()
    state_path = root / "training-loop.json"
    state_path.write_text(json.dumps({"validation": "held-out-report"}))
    with pytest.raises(RuntimeError, match="validation already consumed"):
        loop_module().run_loop(tmp_path)
    assert json.loads(state_path.read_text())["validation"] == "held-out-report"


@pytest.mark.parametrize("occupancy,expected_cycles", [(0.2, 2), (0.55, 1)])
def test_loop_continues_latest_branch_and_validates_occupancy_target(
    tmp_path, monkeypatch, occupancy, expected_cycles
):
    loop = loop_module()
    config = SnakeConfig(algorithm="a2c", observation_mode="board").to_dict()
    monkeypatch.setattr(loop, "ALGORITHM_SCHEDULE", ("a2c",))
    monkeypatch.setattr(loop, "load_checkpoint", lambda _: {"config": config, "completed": 1})

    class Manager:
        def __init__(self, _):
            pass

        def presets(self):
            return [{"id": "best_a2c", "model": "champion", "config": config}]

        def shutdown(self):
            pass

    monkeypatch.setattr(loop, "SnakeManager", Manager)
    calls = []

    def benchmark(config, runs_dir, train_seconds=60, resume=None, **kwargs):
        calls.append((resume, kwargs.get("validate", False)))
        directory = Path(runs_dir) / "snake" / f"trial-{len(calls)}"
        directory.mkdir()
        validation = kwargs.get("validate", False)
        report = {
            "wins": 0,
            "episodes": 100 if validation else 20,
            "occupancy": occupancy,
            "steps_added": 0 if validation else 100,
            "config": config.to_dict(),
        }
        (directory / ("validation.json" if validation else "evaluation.json")).write_text(
            json.dumps(report)
        )
        return directory

    monkeypatch.setattr(loop, "run_benchmark", benchmark)
    state = loop.run_loop(tmp_path, max_cycles=2, target_occupancy=0.5)
    assert state["cycle"] == expected_cycles
    assert calls[1][0].parent.name == "trial-1"
    if occupancy >= 0.5:
        assert calls[1][1]
        assert state["status"] == "verified_occupancy_target"
    else:
        assert not calls[1][1]
        assert state["latest_models"]["a2c"] == "trial-2"
