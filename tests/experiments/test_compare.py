from pathlib import Path

import pytest

from rl_fun.experiments.compare import aggregate_batch
from rl_fun.experiments.result import RunSummary
from rl_fun.experiments.runner import BatchSummary


@pytest.mark.parametrize("metric, mean, std", [("score", 3.0, 1.0), ("other", 15.0, 5.0)])
def test_aggregate_batch_uses_requested_metric(metric: str, mean: float, std: float) -> None:
    # Arrange
    runs = (
        RunSummary("success", 1, Path("one"), {"score": 2.0, "other": 10.0}, 0.1),
        RunSummary("success", 2, Path("two"), {"score": 4.0, "other": 20.0}, 0.2),
        RunSummary("failure", 3, Path("three"), {"score": 100.0, "other": 100.0}, 0.3),
        RunSummary("cancelled", 4, Path("four"), {"score": 200.0}, 0.4),
        RunSummary("success", 5, Path("five"), {}, 0.5),
    )

    # Act
    aggregate = aggregate_batch(BatchSummary(runs), metric)

    # Assert
    assert aggregate == {
        "run_count": 5.0, "success_count": 3.0, "failure_count": 1.0,
        "metric": metric, "sample_count": 2.0, "mean": mean, "std": std,
    }


@pytest.mark.parametrize("runs", [
    (),
    (RunSummary("success", 1, Path("one"), {}, 0.1),),
    (RunSummary("failure", 1, Path("one"), {"missing": 4.0}, 0.1),),
    (RunSummary("cancelled", 1, Path("one"), {"missing": 4.0}, 0.1),),
])
def test_aggregate_batch_rejects_missing_metric(runs: tuple[RunSummary, ...]) -> None:
    # Arrange
    batch = BatchSummary(runs)

    # Act / Assert
    with pytest.raises(ValueError, match="missing"):
        aggregate_batch(batch, "missing")


def test_aggregate_batch_keeps_zero_metric_value() -> None:
    # Arrange
    batch = BatchSummary((RunSummary("success", 1, Path("one"), {"score": 0.0}, 0.1),))

    # Act
    aggregate = aggregate_batch(batch, "score")

    # Assert
    assert (aggregate["sample_count"], aggregate["mean"], aggregate["std"]) == (1.0, 0.0, 0.0)
