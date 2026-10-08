import numpy as np

from rl_fun.experiments.runner import BatchSummary


def aggregate_batch(batch: BatchSummary, metric: str) -> dict[str, float | str]:
    values = [
        run.metrics[metric]
        for run in batch.runs
        if run.status == "success" and metric in run.metrics
    ]
    if not values:
        raise ValueError(f"no successful run contains metric {metric!r}")
    return {
        "run_count": float(len(batch.runs)),
        "success_count": float(batch.success_count),
        "failure_count": float(batch.failure_count),
        "metric": metric,
        "sample_count": float(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }
