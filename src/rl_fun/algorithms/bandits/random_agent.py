import numpy as np

from rl_fun.algorithms.bandits.types import BanditOutcome, update_estimate
from rl_fun.environments.bandit import StationaryBanditEnv
from rl_fun.tracking.metrics import MetricSink


def run_random(
    env: StationaryBanditEnv,
    steps: int,
    rng: np.random.Generator,
    metrics: MetricSink,
) -> BanditOutcome:
    if steps > env.horizon:
        raise ValueError("steps must not exceed environment horizon")
    estimates = np.zeros(env.arms, dtype=np.float64)
    counts = np.zeros(env.arms, dtype=np.int64)
    cumulative_reward = 0.0
    cumulative_regret = 0.0
    optimal_actions = 0
    for step in range(1, steps + 1):
        action = int(rng.integers(env.arms))
        _, reward, terminated, truncated, info = env.step(action)
        update_estimate(estimates, counts, action, reward)
        cumulative_reward += reward
        cumulative_regret += float(info["instant_regret"])
        optimal_actions += int(action == info["optimal_action"])
        metrics.log(
            step,
            {
                "train/reward": reward,
                "train/cumulative_reward": cumulative_reward,
                "train/cumulative_regret": cumulative_regret,
                "train/optimal_action_rate": optimal_actions / step,
            },
        )
        if terminated or truncated:
            break
    return BanditOutcome(estimates, counts, cumulative_reward, cumulative_regret)
