"""Shared small neural policy and independent greedy evaluation."""

import numpy as np
import torch
from torch import nn

from rl_fun.snake.env import SnakeEnv

ARCHITECTURE = [11, 64, 64, 3]


class Policy(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(11, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3)
        )

    def act(self, observation):
        with torch.inference_mode():
            return int(self(torch.as_tensor(observation, dtype=torch.float32)).argmax())


def evaluate(model, size, seeds, max_steps, control=lambda: True):
    scores, rewards = [], []
    last = None
    for seed in seeds:
        env = SnakeEnv(size)
        observation, _ = env.reset(seed=int(seed))
        total = 0.0
        for _ in range(max_steps):
            if not control():
                return None
            observation, reward, terminated, truncated, _ = env.step(model.act(observation))
            total += reward
            if terminated or truncated:
                break
        scores.append(env.score)
        rewards.append(total)
        last = env.snapshot()
    return {"score": float(np.mean(scores)), "reward": float(np.mean(rewards)), "frame": last}
