import numpy as np

from rl_fun.experiments.runner import BatchSummary


def aggregate_batch(batch: BatchSummary) -> dict[str, float]:
    rewards = [
        run.cumulative_reward
        for run in batch.runs
        if run.status == "success" and run.cumulative_reward is not None
    ]
    regrets = [
        run.cumulative_regret
        for run in batch.runs
        if run.status == "success" and run.cumulative_regret is not None
    ]
    return {
        "run_count": float(len(batch.runs)),
        "success_count": float(batch.success_count),
        "failure_count": float(batch.failure_count),
        "mean_cumulative_reward": float(np.mean(rewards)) if rewards else float("nan"),
        "std_cumulative_reward": float(np.std(rewards)) if rewards else float("nan"),
        "mean_cumulative_regret": float(np.mean(regrets)) if regrets else float("nan"),
    }
