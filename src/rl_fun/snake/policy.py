"""Shared small neural policy and independent greedy evaluation."""

import numpy as np
import torch
from torch import nn

from rl_fun.snake.env import SnakeEnv
from rl_fun.snake.frames import emit_frame

ARCHITECTURE = [11, 64, 64, 3]


def architecture(size=16, observation_mode="local"):
    if observation_mode == "egocentric":
        return [5 * (2 * size - 1) ** 2, 128, 128, 3]
    return ARCHITECTURE[:] if observation_mode == "local" else [11 + 2 * size**2, 128, 128, 3]


def evaluation_rank(result):
    return (result["win_rate"], result["occupancy"], result["score"])


class Policy(nn.Sequential):
    def __init__(self, size=16, observation_mode="local"):
        dimensions = architecture(size, observation_mode)
        super().__init__(
            nn.Linear(dimensions[0], dimensions[1]),
            nn.ReLU(),
            nn.Linear(dimensions[1], dimensions[2]),
            nn.ReLU(),
            nn.Linear(dimensions[2], 3),
        )
        self.observation_mode = observation_mode

    def act(self, observation):
        with torch.inference_mode():
            values = self(torch.as_tensor(observation, dtype=torch.float32))
            if getattr(self, "action_adapter", None) == "continuous_scores_argmax":
                values = values.tanh()
            return int(values.argmax())


def evaluate(model, size, seeds, max_steps, control=lambda: True, on_frame=None, env_options=None):
    scores, rewards, outcomes = [], [], []
    last = None
    for episode, seed in enumerate(seeds, start=1):
        options = {
            "observation_mode": getattr(model, "observation_mode", "local"),
            **(env_options or {}),
        }
        env = SnakeEnv(size, **options)
        observation, _ = env.reset(seed=int(seed))
        seen = {(tuple(env.snake), env.food, env.direction)}
        total = 0.0
        eaten = 0
        emit_frame(on_frame, env, episode=episode, ended=False)
        for step in range(max_steps):
            if not control():
                return None
            observation, reward, terminated, truncated, _ = env.step(model.act(observation))
            total += reward
            # A greedy policy cannot escape an identical board with identical food.
            if env.score != eaten:
                eaten = env.score
                seen.clear()
            position = (tuple(env.snake), env.food, env.direction)
            if not (terminated or truncated) and position in seen:
                total -= env.death_penalty
                truncated = env.done = True
                env.reason = "loop"
            seen.add(position)
            if not (terminated or truncated) and step + 1 == max_steps:
                truncated = env.done = True
                env.reason = "max_steps"
            emit_frame(on_frame, env, episode=episode, ended=bool(terminated or truncated))
            if terminated or truncated:
                break
        scores.append(env.score)
        rewards.append(total)
        outcomes.append(
            {
                "seed": int(seed),
                "score": env.score,
                "reward": total,
                "steps": env.steps,
                "occupancy": len(env.snake) / size**2,
                "reason": env.reason,
                "win": env.reason == "win",
            }
        )
        last = env.snapshot()
    if not outcomes:
        raise ValueError("для проверки нужен хотя бы один seed")
    wins = sum(row["win"] for row in outcomes)
    reasons = {}
    for row in outcomes:
        reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1
    return {
        "score": float(np.mean(scores)),
        "reward": float(np.mean(rewards)),
        "frame": last,
        "episodes": len(outcomes),
        "wins": wins,
        "win_rate": wins / len(outcomes),
        "occupancy": float(np.mean([row["occupancy"] for row in outcomes])),
        "max_occupancy": max(row["occupancy"] for row in outcomes),
        "reasons": reasons,
        "outcomes": outcomes,
    }
