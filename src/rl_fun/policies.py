"""Policy contracts used for evaluation and interactive rollouts."""

from __future__ import annotations

from typing import Protocol

import gymnasium as gym


class Policy(Protocol):
    """Choose actions after being initialized for an episode."""

    def reset(self, seed: int, action_space: gym.Space) -> None:
        """Prepare the policy for an episode."""

    def act(self, observation: object, deterministic: bool) -> object:
        """Choose an action for an observation."""


class RandomPolicy:
    """Sample actions from the environment-owned seeded action space."""

    def __init__(self) -> None:
        self._action_space: gym.Space | None = None

    def reset(self, seed: int, action_space: gym.Space) -> None:
        """Seed and retain the action space for the next episode."""
        action_space.seed(seed)
        self._action_space = action_space

    def act(self, observation: object, deterministic: bool) -> object:
        """Sample one action from the initialized action space."""
        if self._action_space is None:
            raise RuntimeError("RandomPolicy must be reset before act")
        return self._action_space.sample()
