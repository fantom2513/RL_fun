"""Reusable single-episode Gymnasium rollout."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym

from rl_fun.policies import Policy


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Observable outcome of one completed episode."""

    reward: float
    length: int
    terminated: bool
    truncated: bool
    reached_safety_limit: bool


def rollout_episode(
    env: gym.Env,
    policy: Policy,
    seed: int,
    max_episode_steps: int | None,
) -> EpisodeResult:
    """Run one seeded episode until the environment or safety limit ends it."""
    policy.reset(seed, env.action_space)
    observation, _ = env.reset(seed=seed)
    total_reward = 0.0
    length = 0
    terminated = False
    truncated = False

    while True:
        action = policy.act(observation, deterministic=False)
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        length += 1

        if terminated or truncated:
            return EpisodeResult(total_reward, length, terminated, truncated, False)
        if max_episode_steps is not None and length >= max_episode_steps:
            return EpisodeResult(total_reward, length, False, False, True)
