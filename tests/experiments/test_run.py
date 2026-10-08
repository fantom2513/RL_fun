import json
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import pytest

from rl_fun.algorithms.registry import AlgorithmDefinition
from rl_fun.experiments import run as run_module
from rl_fun.experiments.config import AlgorithmConfig, EnvironmentConfig, ExperimentConfig
from rl_fun.experiments.result import RunSummary
from rl_fun.experiments.run import run_once


def bandit_config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig.from_dict(
        {
            "name": "bandit-smoke",
            "environment": {
                "id": "RLFun/StationaryBandit-v0",
                "kwargs": {"arms": 3, "horizon": 10, "reward_std": 0.0},
            },
            "algorithm": {"id": "bandit_epsilon", "kwargs": {"epsilon": 0.1}},
            "run": {
                "seeds": [17],
                "total_steps": 10,
                "workers": 1,
                "output_root": str(tmp_path),
            },
            "evaluation": {"episodes": 2, "max_episode_steps": 10},
        }
    )


def test_run_once_writes_generic_success_summary(tmp_path: Path) -> None:
    summary = run_once(bandit_config(tmp_path), 17, run_id="fixed")
    written = json.loads((summary.run_dir / "summary.json").read_text("utf-8"))
    assert written == {
        "status": "success",
        "seed": 17,
        "run_dir": str(tmp_path / "bandit-smoke" / "fixed"),
        "metrics": summary.metrics,
        "elapsed_seconds": summary.elapsed_seconds,
        "error": None,
    }


def test_run_once_persists_config_metadata_and_metrics(tmp_path: Path) -> None:
    config = bandit_config(tmp_path)
    summary = run_once(config, 17, "artifacts")
    written = json.loads((summary.run_dir / "config.json").read_text("utf-8"))
    metadata = json.loads((summary.run_dir / "metadata.json").read_text("utf-8"))
    events = [
        json.loads(line)
        for line in (summary.run_dir / "metrics.jsonl").read_text("utf-8").splitlines()
    ]
    assert (written, metadata["seed"], events[-1]["step"], summary.metrics["train/steps"]) == (
        config.to_dict(),
        17,
        10,
        10.0,
    )


def test_run_once_is_reproducible(tmp_path: Path) -> None:
    config = bandit_config(tmp_path)
    assert run_once(config, 17, "one").metrics == run_once(config, 17, "two").metrics


def test_run_once_runs_cartpole_random(tmp_path: Path) -> None:
    config = replace(
        bandit_config(tmp_path),
        environment=EnvironmentConfig("CartPole-v1"),
        algorithm=AlgorithmConfig("random"),
    )
    summary = run_once(config, 17, "cartpole")
    assert (summary.status, summary.metrics["train/steps"]) == ("success", 10.0)


def test_run_once_records_environment_failure(tmp_path: Path) -> None:
    config = replace(bandit_config(tmp_path), environment=EnvironmentConfig("Missing-v0"))
    summary = run_once(config, 17, "failed")
    error = (summary.run_dir / "error.txt").read_text("utf-8")
    written = json.loads((summary.run_dir / "summary.json").read_text("utf-8"))
    assert (
        summary.status == "failure"
        and "Missing-v0" in error
        and summary.error == error
        and summary.metrics == {}
        and written == summary.to_dict()
    )


def test_run_once_rejects_incompatible_pair_before_training(tmp_path: Path) -> None:
    config = replace(bandit_config(tmp_path), environment=EnvironmentConfig("CartPole-v1"))
    summary = run_once(config, 17, "incompatible")
    metric_path = summary.run_dir / "metrics.jsonl"
    assert (
        summary.status == "failure"
        and "bandit_epsilon" in (summary.error or "")
        and "CartPole-v1" in (summary.error or "")
        and (not metric_path.exists() or metric_path.read_text("utf-8") == "")
    )


@pytest.mark.parametrize("seed", [18, True, 17.0])
def test_run_once_rejects_invalid_seed_without_artifacts(tmp_path: Path, seed: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        run_once(bandit_config(tmp_path), seed, "invalid")
    assert list(tmp_path.iterdir()) == []


def test_run_once_revalidates_typed_config_before_artifacts(tmp_path: Path) -> None:
    config = bandit_config(tmp_path)
    config = replace(config, run=replace(config.run, total_steps=0))
    with pytest.raises(ValueError, match="total_steps"):
        run_once(config, 17, "invalid")
    assert list(tmp_path.iterdir()) == []


def test_run_once_unknown_algorithm_fails_before_artifacts(tmp_path: Path) -> None:
    config = replace(bandit_config(tmp_path), algorithm=AlgorithmConfig("missing"))
    with pytest.raises(ValueError, match="missing.*supported IDs"):
        run_once(config, 17, "invalid")
    assert list(tmp_path.iterdir()) == []


def test_run_once_default_ids_isolate_same_seed(tmp_path: Path) -> None:
    config = bandit_config(tmp_path)
    first = run_once(config, 17)
    second = run_once(config, 17)
    assert first.run_dir != second.run_dir and all(
        summary.run_dir.name.startswith("seed-17-") for summary in (first, second)
    )


def test_run_once_does_not_overwrite_existing_run(tmp_path: Path) -> None:
    config = bandit_config(tmp_path)
    first = run_once(config, 17, "fixed")
    original = (first.run_dir / "summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_once(config, 17, "fixed")
    assert (first.run_dir / "summary.json").read_bytes() == original


class LifecycleProbe(gym.Wrapper):
    def __init__(self) -> None:
        super().__init__(gym.make("CartPole-v1"))
        self.closed = False
        self.reset_seeds: list[int | None] = []

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.reset_seeds.append(seed)
        return super().reset(seed=seed, options=options)

    def close(self) -> None:
        self.closed = True
        super().close()


@pytest.mark.parametrize("outcome", ["success", "failure", "interrupt"])
def test_run_once_closes_environment_and_leaves_reset_to_algorithm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    env = LifecycleProbe()
    monkeypatch.setattr(run_module, "make_environment", lambda config: env)
    config = replace(
        bandit_config(tmp_path),
        environment=EnvironmentConfig("CartPole-v1"),
        algorithm=AlgorithmConfig("random"),
    )
    if outcome != "success":

        def fail(env, total_steps, seed, rng, metrics, parameters):
            if outcome == "interrupt":
                raise KeyboardInterrupt
            raise RuntimeError("algorithm failed")

        monkeypatch.setattr(
            run_module,
            "get_algorithm",
            lambda algorithm_id: AlgorithmDefinition(
                run=fail,
                validate=lambda env, environment_id: None,
            ),
        )
    if outcome == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            run_once(config, 17, outcome)
        assert env.closed and env.reset_seeds == []
    else:
        summary = run_once(config, 17, outcome)
        assert (summary.status, env.closed, env.reset_seeds) == (
            outcome,
            True,
            [17] if outcome == "success" else [],
        )


def test_generic_summary_serializes_arbitrary_metrics(tmp_path: Path) -> None:
    summary = RunSummary("success", 7, tmp_path, {"score": 3.5}, 0.25)
    assert json.loads(json.dumps(summary.to_dict())) == {
        "status": "success",
        "seed": 7,
        "run_dir": str(tmp_path),
        "metrics": {"score": 3.5},
        "elapsed_seconds": 0.25,
        "error": None,
    }


class FailingCloseProbe(LifecycleProbe):
    def close(self) -> None:
        super().close()
        raise RuntimeError("cleanup failed")


@pytest.mark.parametrize("outcome", ["success", "failure", "interrupt"])
def test_run_once_records_cleanup_failure_and_preserves_training_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    # Arrange
    env = FailingCloseProbe()
    monkeypatch.setattr(run_module, "make_environment", lambda config: env)
    config = replace(
        bandit_config(tmp_path),
        environment=EnvironmentConfig("CartPole-v1"),
        algorithm=AlgorithmConfig("random"),
    )
    interruption = KeyboardInterrupt("training interrupted")
    if outcome != "success":

        def fail(env, total_steps, seed, rng, metrics, parameters):
            if outcome == "interrupt":
                raise interruption
            raise RuntimeError("training failed")

        monkeypatch.setattr(
            run_module,
            "get_algorithm",
            lambda algorithm_id: AlgorithmDefinition(
                run=fail,
                validate=lambda env, environment_id: None,
            ),
        )
    # Act
    if outcome == "interrupt":
        with pytest.raises(KeyboardInterrupt) as caught:
            run_once(config, 17, outcome)
        assert caught.value is interruption
    else:
        summary = run_once(config, 17, outcome)
        assert summary.status == "failure"
    # Assert: artifacts retain both diagnostics when training also failed.
    run_dir = tmp_path / "bandit-smoke" / outcome
    error = (run_dir / "error.txt").read_text("utf-8")
    written = json.loads((run_dir / "summary.json").read_text("utf-8"))
    assert written["status"] == "failure" and written["error"] == error
    assert "cleanup failed" in error
    if outcome != "success":
        assert ("training interrupted" if outcome == "interrupt" else "training failed") in error
