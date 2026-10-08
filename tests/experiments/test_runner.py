import json
from dataclasses import replace
from pathlib import Path

import pytest

from rl_fun.experiments.config import ExperimentConfig
from rl_fun.experiments.runner import run_many


def make_config(tmp_path: Path, workers: int) -> ExperimentConfig:
    return ExperimentConfig.from_dict({
        "name": "batch-bandit",
        "environment": {
            "id": "RLFun/StationaryBandit-v0",
            "kwargs": {"arms": 3, "horizon": 8, "reward_std": 0.0},
        },
        "algorithm": {"id": "bandit_greedy", "kwargs": {}},
        "run": {
            "seeds": [202, 101], "total_steps": 8,
            "workers": workers, "output_root": str(tmp_path),
        },
    })


def test_serial_batch_returns_every_seed(tmp_path):
    # Arrange
    config = make_config(tmp_path, workers=1)

    # Act
    batch = run_many(config)

    # Assert
    assert {run.seed for run in batch.runs} == {101, 202}


def test_parallel_batch_uses_isolated_directories(tmp_path):
    # Arrange
    config = make_config(tmp_path, workers=2)

    # Act
    batch = run_many(config)

    # Assert
    assert len({run.run_dir for run in batch.runs}) == 2


def test_batch_summary_reports_success_count(tmp_path):
    # Arrange
    config = make_config(tmp_path, workers=2)

    # Act
    batch = run_many(config)

    # Assert
    assert batch.success_count == 2


@pytest.mark.parametrize("workers", [1, 2])
def test_batch_sorts_results_by_seed(tmp_path: Path, workers: int) -> None:
    # Arrange
    config = make_config(tmp_path, workers)

    # Act
    batch = run_many(config)

    # Assert
    assert [run.seed for run in batch.runs] == [101, 202]


@pytest.mark.parametrize("workers", [1, 2])
def test_batch_writes_each_runs_generic_metrics(tmp_path: Path, workers: int) -> None:
    # Arrange
    config = make_config(tmp_path, workers)

    # Act
    batch = run_many(config)

    # Assert
    assert all(
        "train/cumulative_reward" in json.loads(
            (run.run_dir / "summary.json").read_text(encoding="utf-8")
        )["metrics"]
        for run in batch.runs
    )


@pytest.mark.parametrize("workers", [1, 2])
def test_failed_seed_still_allows_other_seeds_to_finish(
    tmp_path: Path, workers: int,
) -> None:
    # Arrange
    config = make_config(tmp_path, workers)
    config = replace(
        config, environment=replace(config.environment, id="CartPole-v1", kwargs={}),
    )

    # Act
    batch = run_many(config)

    # Assert
    assert (batch.success_count, batch.failure_count, len(batch.runs)) == (0, 2, 2)


@pytest.mark.parametrize("workers", [1, 2])
def test_failed_runs_keep_independent_error_artifacts(
    tmp_path: Path, workers: int,
) -> None:
    # Arrange
    config = make_config(tmp_path, workers)
    config = replace(
        config, environment=replace(config.environment, id="CartPole-v1", kwargs={}),
    )

    # Act
    batch = run_many(config)

    # Assert
    assert all((run.run_dir / "error.txt").is_file() for run in batch.runs)
