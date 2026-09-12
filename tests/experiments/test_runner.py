from rl_fun.experiments.config import ExperimentConfig
from rl_fun.experiments.runner import run_many


def make_config(tmp_path, workers):
    return ExperimentConfig(
        name="batch-bandit",
        algorithm="random",
        seeds=(101, 202),
        steps=8,
        workers=workers,
        output_root=str(tmp_path),
        parameters={"arms": 3, "reward_std": 0.0},
    )


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
