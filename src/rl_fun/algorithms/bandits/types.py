from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BanditOutcome:
    estimates: np.ndarray
    counts: np.ndarray
    cumulative_reward: float
    cumulative_regret: float


def update_estimate(estimates: np.ndarray, counts: np.ndarray, action: int, reward: float) -> None:
    counts[action] += 1
    estimates[action] += (reward - estimates[action]) / counts[action]


def random_argmax(values: np.ndarray, rng: np.random.Generator) -> int:
    candidates = np.flatnonzero(values == values.max())
    return int(rng.choice(candidates))
