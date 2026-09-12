"""A stationary multi-armed bandit environment."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class StationaryBanditEnv(gym.Env[np.ndarray, int]):
    """Sample Gaussian rewards from fixed arm means for one episode."""

    metadata = {"render_modes": []}

    def __init__(self, arms: int = 10, horizon: int = 1000, reward_std: float = 1.0) -> None:
        if arms < 2:
            raise ValueError("arms must be at least 2")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if reward_std < 0:
            raise ValueError("reward_std must be non-negative")

        self.arms = arms
        self.horizon = horizon
        self.reward_std = reward_std
        self.action_space = gym.spaces.Discrete(arms)
        self.observation_space = gym.spaces.Box(0.0, 0.0, shape=(1,), dtype=np.float32)
        self.arm_means = np.zeros(arms, dtype=np.float64)
        self._step = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, int]]:
        """Start an episode with seedable, fixed means for every arm."""
        super().reset(seed=seed)
        self.action_space.seed(seed)
        self.arm_means = self.np_random.normal(0.0, 1.0, size=self.arms)
        self._step = 0
        return np.zeros(1, dtype=np.float32), {"optimal_action": int(self.arm_means.argmax())}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, int | float]]:
        """Draw a reward for an action and report its instantaneous regret."""
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action}")

        reward = float(self.np_random.normal(self.arm_means[action], self.reward_std))
        regret = float(self.arm_means.max() - self.arm_means[action])
        self._step += 1
        truncated = self._step >= self.horizon
        info = {"optimal_action": int(self.arm_means.argmax()), "instant_regret": regret}
        return np.zeros(1, dtype=np.float32), reward, False, truncated, info
