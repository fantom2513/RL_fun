from pathlib import Path

from rl_fun.experiments.compare import aggregate_batch
from rl_fun.experiments.result import RunSummary
from rl_fun.experiments.runner import BatchSummary


def test_aggregate_batch_calculates_mean_reward():
    # Arrange
    runs = (
        RunSummary("success", 1, Path("one"), 2.0, 1.0, 0.1),
        RunSummary("success", 2, Path("two"), 4.0, 3.0, 0.2),
    )

    # Act
    aggregate = aggregate_batch(BatchSummary(runs))

    # Assert
    assert aggregate["mean_cumulative_reward"] == 3.0
